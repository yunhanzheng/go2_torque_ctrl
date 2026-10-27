"""Smoke test for the V56 standing comparison.  No training, no checkpoint, a few seconds.

Four things are checked per task:

1. wiring - 12 leg actions, 48-D policy observation, the ``feet_forces`` sensor exists and
   ``metrics.py`` can address all four feet by name;
2. the torque path - ``processed_actions == clamp(action_scale * raw_action, +-23.5)`` and the
   articulation's own ``applied_torque`` is the same tensor, i.e. no hidden PD, no filter, no
   saturation beyond the clamp;
3. the plant - the ground carries the robot (``sum(GRF) / m g`` near 1) and nothing is NaN;
4. the logging channel - ``extras["log"]`` carries every force/CoP/actuation key that wandb
   will receive.

Standing with a random torque policy is impossible, so (3) for the torque variant drives the
torque path with the stock Go2 PD torque computed *inside the test* (see
``_stance_pd_action``).  That keeps the experiment free of any PD while still proving that the
feet can carry the robot through a torque command.

    python smoke_test.py                          # all three variants, headless by default
    python smoke_test.py --task Isaac-Go2-V56-StandTorque-v0

Exit code 0 means every assertion passed.
"""
from __future__ import annotations

import argparse
import os
import sys

# --- same path bootstrap as train.py ---------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (_SOURCE_ROOT, os.path.join(_SOURCE_ROOT, "base_adapter"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

# Imported for its side effect only: it pre-registers the broken ``sa_16dof`` task package as an
# empty stub, without which ``import isaaclab_tasks`` dies on the training server (see the
# comment at the top of ``base_adapter/cli_args.py``).
import cli_args  # noqa: F401, isort: skip

TASK_POS = "Isaac-Go2-V56-StandPos-v0"
TASK_TORQUE = "Isaac-Go2-V56-StandTorque-v0"
TASK_TORQUE200 = "Isaac-Go2-V56-StandTorque200-v0"

OBS_DIM = 48
"""Stock Go2 flat observation: 3 + 3 + 3 + 3 base/command terms and 12 + 12 + 12 joint/action."""
NUM_LEG_JOINTS = 12

parser = argparse.ArgumentParser(description="V56 smoke test (no training, no checkpoint).")
parser.add_argument("--task", type=str, default=None, help="Task id; default: all three variants, one process each.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments used for the checks.")
parser.add_argument("--settle_s", type=float, default=1.5, help="Physical seconds before averaging forces.")
parser.add_argument("--check_s", type=float, default=0.4, help="Physical seconds averaged for the force check.")
AppLauncher.add_app_launcher_args(parser)
# The test is headless by nature; an explicit ``--headless`` on the command line still wins.
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

TASKS = (TASK_POS, TASK_TORQUE, TASK_TORQUE200)

if args_cli.task is None:
    # One interpreter per task: SimulationContext is a singleton, so a second ``gym.make`` in
    # the same process would silently reuse the first simulation.
    import subprocess

    failed = []
    for task_id in TASKS:
        child = [
            sys.executable,
            os.path.abspath(__file__),
            "--task",
            task_id,
            "--headless",
            "--num_envs",
            str(args_cli.num_envs),
            "--settle_s",
            str(args_cli.settle_s),
            "--check_s",
            str(args_cli.check_s),
        ]
        if args_cli.device is not None:
            child += ["--device", args_cli.device]
        if subprocess.run(child).returncode != 0:
            failed.append(task_id)
    if failed:
        raise SystemExit(f"V56 smoke test FAILED for: {', '.join(failed)}")
    print("\nAll V56 smoke checks passed.")
    raise SystemExit(0)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import terrain_v56_stand_force_compare.config  # noqa: F401  (registers the three tasks)
from terrain_v56_stand_force_compare.actions import DirectTorqueAction
from terrain_v56_stand_force_compare.metrics import FOOT_NAMES, TORQUE_LIMIT_NM

REQUIRED_LOG_KEYS: tuple[str, ...] = (
    "Force/total_N",
    "Force/total_over_weight",
    *(f"Force/{foot[:2]}_N" for foot in FOOT_NAMES),
    "Force/load_std_frac",
    "Force/asym_LR",
    "Force/asym_FB",
    "Force/front_share",
    "Contact/duty_foot",
    "Contact/duty_all4",
    "CoP/offset_m",
    "CoP/speed_mps",
    "Act/torque_rms",
    "Act/torque_max",
    "Act/sat_frac",
)


def _check(condition: bool, message: str) -> None:
    """Fail loudly with the measured value instead of silently printing a warning."""
    if not condition:
        raise AssertionError(message)


def _worst(tensor: torch.Tensor) -> float:
    return float(tensor.abs().max())


def _check_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, what: str) -> None:
    error = float((actual - expected).abs().max())
    _check(error <= atol, f"{what}: max |error| = {error:.5f} > {atol:.5f}")


def _build_env(task_id: str):
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 7
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    # The check is about physics, not about the noise the policy input carries.
    env_cfg.observations.policy.enable_corruption = False
    return gym.make(task_id, cfg=env_cfg)


def _stance_pd_action(robot, term) -> torch.Tensor:
    """The stock Go2 PD torque, expressed as a *network output* for the torque path.

    Test-only.  ``kd`` is doubled with respect to ``UNITREE_GO2_CFG`` because this loop closes
    at the policy rate (50/200 Hz) instead of the sim rate.  Nothing of this is part of the
    experiment: the experiment has no PD anywhere.
    """
    kp, kd = 25.0, 1.0
    torque = kp * (robot.data.default_joint_pos - robot.data.joint_pos) - kd * robot.data.joint_vel
    return torque / term.cfg.action_scale


def run_task(task_id: str) -> None:
    env = _build_env(task_id)
    unwrapped = env.unwrapped
    term = unwrapped.action_manager.get_term("joint_pos")
    robot = unwrapped.scene["robot"]
    is_torque = isinstance(term, DirectTorqueAction)
    label = "direct torque" if is_torque else "position"
    print(f"\n=== {task_id}  [{label}, {1.0 / unwrapped.step_dt:.0f} Hz] ===", flush=True)

    # --- 1. wiring ---------------------------------------------------------------------
    _check(
        unwrapped.action_manager.total_action_dim == NUM_LEG_JOINTS,
        f"expected {NUM_LEG_JOINTS} leg actions, got {unwrapped.action_manager.total_action_dim}",
    )
    _check("feet_forces" in unwrapped.scene.sensors, "'feet_forces' sensor is missing (see config.py)")
    missing_feet = [name for name in FOOT_NAMES if name not in unwrapped.scene.sensors["feet_forces"].body_names]
    _check(not missing_feet, f"'feet_forces' does not track {missing_feet}")

    legs = robot.actuators["base_legs"]
    if is_torque:
        # The whole point of the variant: the actuator must not add anything of its own.
        _check(_worst(torch.as_tensor(legs.stiffness)) == 0.0, "actuator stiffness is not 0")
        _check(_worst(torch.as_tensor(legs.damping)) == 0.0, "actuator damping is not 0")
        _check(
            float(torch.as_tensor(legs.velocity_limit).min()) > 1.0e2,
            "actuator velocity_limit still clips the torque-speed curve",
        )
    else:
        _check(isinstance(term, JointPositionAction), f"position task uses {type(term).__name__}")

    # --- 2./3. run ---------------------------------------------------------------------
    settle_steps = max(1, round(args_cli.settle_s / unwrapped.step_dt))
    check_steps = max(1, round(args_cli.check_s / unwrapped.step_dt))
    obs, _ = env.reset()
    _check(
        tuple(obs["policy"].shape) == (args_cli.num_envs, OBS_DIM),
        f"policy observation is {tuple(obs['policy'].shape)}, expected {(args_cli.num_envs, OBS_DIM)}",
    )

    ratio_sum, ratio_n, resets = 0.0, 0, 0
    last_log: dict[str, torch.Tensor] = {}
    total_steps = settle_steps + check_steps
    for step in range(total_steps):
        if is_torque:
            action = _stance_pd_action(robot, term)
        else:
            action = torch.zeros((unwrapped.num_envs, NUM_LEG_JOINTS), device=unwrapped.device)
        obs, reward, terminated, truncated, extras = env.step(action)

        if is_torque:
            expected = (term.raw_actions * term.cfg.action_scale).clamp(-TORQUE_LIMIT_NM, TORQUE_LIMIT_NM)
            _check_close(term.processed_actions, expected, 1e-5, f"step {step}: tau != clamp(scale * a)")
            _check(
                _worst(term.processed_actions) <= TORQUE_LIMIT_NM + 1e-5,
                f"step {step}: |tau| = {_worst(term.processed_actions):.2f} N.m exceeds the {TORQUE_LIMIT_NM} clamp",
            )
            # If stiffness/damping/a filter were still in the loop, this is where they show up.
            _check_close(
                robot.data.applied_torque,
                term.processed_actions,
                0.05,
                f"step {step}: actuator output != commanded torque (hidden PD, filter or clip?)",
            )

        _check(
            bool(torch.isfinite(obs["policy"]).all()),
            f"step {step}: non-finite observation",
        )
        _check(bool(torch.isfinite(reward).all()), f"step {step}: non-finite reward")

        resets += int(terminated.sum().item()) + int(truncated.sum().item())
        # ``extras["log"]`` also carries the Episode_* statistics right after a reset, so only
        # the second half of the run (well past the warm-up gate) must have the force keys.
        log = extras.get("log") or {}
        if step < settle_steps:
            continue
        missing_keys = [key for key in REQUIRED_LOG_KEYS if key not in log]
        _check(not missing_keys, f"step {step}: extras['log'] misses {missing_keys}")
        _check(
            all(bool(torch.isfinite(torch.as_tensor(value)).all()) for value in log.values()),
            f"step {step}: non-finite force metric",
        )
        last_log = log
        ratio_sum += float(log["Force/total_over_weight"])
        ratio_n += 1

    # --- 4. did the ground actually carry the robot? ------------------------------------
    _check(ratio_n > 0, "no metric was ever logged - the warm-up gate never opened")
    ratio = ratio_sum / ratio_n
    feet = " ".join(f"{foot[:2]}={float(last_log[f'Force/{foot[:2]}_N']):.1f}N" for foot in FOOT_NAMES)
    print(f"  feet: {feet}", flush=True)
    print(
        f"  sum(GRF)/mg = {ratio:.3f} | CoP offset = {float(last_log['CoP/offset_m']) * 1000:.1f} mm"
        f" | duty(all4) = {float(last_log['Contact/duty_all4']):.2f}"
        f" | |tau|_rms = {float(last_log['Act/torque_rms']):.2f} N.m"
        f" | sat = {float(last_log['Act/sat_frac']) * 100:.1f} %",
        flush=True,
    )
    print(f"  resets during the run: {resets}", flush=True)
    _check(
        0.80 <= ratio <= 1.20,
        f"the feet carry {ratio:.2f} x body weight, expected ~1.0 - is the robot standing?",
    )


def main() -> None:
    run_task(args_cli.task)
    print("\nAll V56 smoke checks passed.")
    sys.stdout.flush()
    # Same teardown hang as in train.py: the render context does not close on this host.
    os._exit(0)


if __name__ == "__main__":
    main()
