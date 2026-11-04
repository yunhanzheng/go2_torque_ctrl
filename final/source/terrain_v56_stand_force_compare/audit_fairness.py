"""Print every effective configuration difference between the V56 arms.

Nothing here builds an environment: every config class is instantiated and the fields the
managers would actually consume are flattened into strings, then diffed.  The point is to
stop guessing from source code which knob differs between the position and torque arms - and,
for the 200 Hz pair, that the *only* extra difference is ``decimation`` / ``sim.render_interval``.

Run on the training box.  This IsaacLab build requires ``SimulationApp`` to be running before
``isaaclab.envs`` is imported, so the kit has to start even though nothing is simulated::

    python audit_fairness.py --source-root /path/to/final/source --headless

``--no_app`` exists for builds that allow deferred imports; on the training box the kit aborts
during Vulkan initialisation, so it has not been exercised end to end there yet.

Add ``--full`` to dump every key for every arm instead of only the differing ones.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields, is_dataclass
from typing import Any


def _scalar(value: Any) -> str:
    """Render one config attribute compactly, expanding nested dataclass configs one level."""
    if isinstance(value, (int, float, bool, str, type(None))):
        return str(value)
    if isinstance(value, (list, tuple)) and len(value) <= 8 and all(
        isinstance(v, (int, float, bool, str)) for v in value
    ):
        return str(value)
    if is_dataclass(value):
        parts = []
        for field in fields(value):
            inner = getattr(value, field.name, None)
            if callable(inner):
                continue
            parts.append(f"{field.name}={_scalar(inner)}")
        return "(" + ", ".join(parts) + ")"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_scalar(v)}" for k, v in value.items()) + "}"
    return type(value).__name__


def _term(entity: Any) -> str:
    """Render an action / event / termination / observation term, dropping the func object."""
    parts = []
    for name, value in vars(entity).items():
        if name.startswith("_") or callable(value):
            continue
        if name == "func":
            parts.append(f"func={getattr(value, '__name__', value)}")
        else:
            parts.append(f"{name}={_scalar(value)}")
    return f"{type(entity).__name__}(" + ", ".join(parts) + ")"


def _manager(holder: Any, out: dict[str, str], kind: str) -> None:
    """Flatten every config term of one manager into ``out`` under ``kind.``."""
    for term_name, term in vars(holder).items():
        if term_name.startswith("_") or not hasattr(term, "__dict__"):
            continue
        if callable(term) or isinstance(term, (int, float, str, bool)):
            continue
        out[f"{kind}.{term_name}"] = _term(term)


def collect(cfg: Any) -> dict[str, str]:
    """Flatten the parts of an env config that can make the arms non-comparable."""
    out: dict[str, str] = {}

    # --- timing -----------------------------------------------------------------------
    for key in ("decimation", "episode_length_s", "action_scale", "metrics_warmup_s", "seed"):
        out[key] = str(getattr(cfg, key, "<absent>"))
    for key in ("dt", "render_interval"):
        out[f"sim.{key}"] = str(getattr(cfg.sim, key, "<absent>"))

    # --- robot asset ------------------------------------------------------------------
    robot = cfg.scene.robot
    out["robot.init_state.pos"] = _scalar(robot.init_state.pos)
    out["robot.init_state.joint_pos"] = _scalar(robot.init_state.joint_pos)
    out["robot.init_state.joint_vel"] = _scalar(robot.init_state.joint_vel)
    out["robot.soft_joint_pos_limit_factor"] = str(robot.soft_joint_pos_limit_factor)
    for act_name, act in robot.actuators.items():
        out[f"actuator.{act_name}"] = _term(act)

    # --- managers ---------------------------------------------------------------------
    _manager(cfg.actions, out, "actions")
    _manager(cfg.events, out, "events")
    _manager(cfg.rewards, out, "rewards")
    _manager(cfg.terminations, out, "terminations")
    _manager(cfg.observations.policy, out, "obs.policy")
    _manager(cfg.commands.base_velocity, out, "command.base_velocity")

    # --- scene ------------------------------------------------------------------------
    out["scene.terrain.terrain_type"] = str(getattr(cfg.scene.terrain, "terrain_type", "<absent>"))
    out["scene.sensors"] = ",".join(sorted(cfg.scene.sensors.keys())) if cfg.scene.sensors else "<none>"
    return out


def main() -> None:
    # ``isaaclab_tasks`` pulls in ``pxr`` through ``isaaclab.envs``, and ``pxr`` only lands on
    # ``sys.path`` once the kit is up, so the app has to start before anything is imported.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=str, required=True, help="Path to final/source.")
    parser.add_argument("--full", action="store_true", help="Print all keys, not only the differing ones.")
    parser.add_argument(
        "--no_app",
        action="store_true",
        help="Skip the kit entirely (needs pxr on sys.path, e.g. via isaaclab.sh -p).",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Imported for its side effect only: it pre-registers the broken ``sa_16dof`` task package as
    # an empty stub, without which ``import isaaclab_tasks`` dies on the training server (same
    # bootstrap as ``evaluate_force.py``).
    sys.path.insert(0, os.path.join(args.source_root, "base_adapter"))
    import cli_args  # noqa: F401, isort: skip

    if args.no_app:
        _report(args)
        return

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        _report(args)
    finally:
        simulation_app.close()


def _report(args: argparse.Namespace) -> None:
    sys.path.insert(0, args.source_root)
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
        UnitreeGo2FlatEnvCfg,
    )

    from terrain_v56_stand_force_compare.config import (
        StandPos200EnvCfg,
        StandPosEnvCfg,
        StandTorque200EnvCfg,
        StandTorqueEnvCfg,
    )

    arms = {
        "stock": collect(UnitreeGo2FlatEnvCfg()),
        "pos": collect(StandPosEnvCfg()),
        "torque": collect(StandTorqueEnvCfg()),
        "pos200": collect(StandPos200EnvCfg()),
        "torque200": collect(StandTorque200EnvCfg()),
    }

    keys = sorted(set().union(*(set(arm) for arm in arms.values())))
    names = list(arms)
    differing = 0
    for key in keys:
        values = {name: arms[name].get(key, "<absent>") for name in names}
        if not args.full and len(set(values.values())) == 1:
            continue
        differing += 1
        print(f"\n[{key}]")
        for name in names:
            print(f"  {name:<10s} {values[name]}")

    print(f"\n{len(keys)} keys compared, {differing} differ across the {len(names)} arms.")
    print(
        "  Expected differences: torque* -> actuator gains 0 + DirectTorqueAction + action_scale;\n"
        "                        *200   -> decimation 1 + sim.render_interval 1 (nothing else)."
    )


if __name__ == "__main__":
    main()
