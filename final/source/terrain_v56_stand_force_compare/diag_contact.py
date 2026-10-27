"""Print, step by step, what the contact sensors and the termination terms see.

Written to settle one question: the torque arm of V56 ends every episode after exactly ten
policy steps claiming ``base_contact``, while the reported force on the base link reads zero.
This script dumps the raw sensor buffers next to the values the termination manager keys off,
for a handful of steps of one environment, so the two can be compared directly.

    python diag_contact.py --task Isaac-Go2-V56-StandTorque-v0 --checkpoint <model.pt>
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (_SOURCE_ROOT, os.path.join(_SOURCE_ROOT, "base_adapter"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

import cli_args  # noqa: F401, isort: skip

parser = argparse.ArgumentParser(description="Dump contact sensor state and termination terms per step.")
parser.add_argument("--task", type=str, default="Isaac-Go2-V56-StandTorque-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=25)
parser.add_argument("--print_every", type=int, default=1, help="Print every n-th step.")
parser.add_argument(
    "--disable_base_contact",
    action="store_true",
    default=False,
    help="Stop the base-contact term from ending the episode, so the rollout can be watched longer.",
)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata

import gymnasium as gym
import torch

from rsl_rl.runners import OffPolicyRunner, OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import terrain_v56_stand_force_compare.config  # noqa: F401  (registers the tasks)
from terrain_v56_stand_force_compare.eval_common import play_overrides

from isaaclab_tasks.utils.hydra import hydra_task_config


def _stats(sensor) -> str:
    """Largest magnitudes in the raw buffers of one contact sensor."""
    history = getattr(sensor.data, "net_forces_w_history", None)
    if history is None:
        return "history=<none>"
    current = sensor.data.net_forces_w
    return (
        f"hist{tuple(history.shape)} |hist|max={float(history.abs().max()):.1f} "
        f"|now|max={float(current.abs().max()):.1f}"
    )


def _never(env, **kwargs) -> torch.Tensor:
    """Replacement for a DoneTerm: never terminates."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    play_overrides(env_cfg, num_envs=1, seed=args_cli.seed, device=args_cli.device, warmup=0.5, duration=6.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    runner_kwargs = {"log_dir": _HERE, "device": str(env.device)}
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), **runner_kwargs)
    else:
        runner = OffPolicyRunner(env, agent_cfg.to_dict(), **runner_kwargs)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    # --- what each sensor tracks, and what each termination term keys off ---------------------
    for name, sensor in base_env.scene.sensors.items():
        print(f"[SENSOR] {name}: {len(sensor.body_names)} bodies -> {list(sensor.body_names)}", flush=True)
    manager = base_env.termination_manager
    for name in manager.active_terms:
        term_cfg = manager.get_term_cfg(name)
        sensor_cfg = term_cfg.params.get("sensor_cfg")
        if sensor_cfg is not None:
            sensor = base_env.scene.sensors[sensor_cfg.name]
            resolved = [sensor.body_names[i] for i in sensor_cfg.body_ids]
            print(
                f"[TERM] {name}: sensor={sensor_cfg.name} body_ids={list(sensor_cfg.body_ids)} "
                f"-> {resolved} threshold={term_cfg.params.get('threshold')}",
                flush=True,
            )
        else:
            print(f"[TERM] {name}: no sensor_cfg", flush=True)

    env.reset()
    obs = env.get_observations()
    contact = base_env.scene.sensors["contact_forces"]
    feet = base_env.scene.sensors["feet_forces"]
    robot = base_env.scene["robot"]
    print(
        f"[INFO] step_dt={base_env.step_dt:.4f} s  max_episode_length={base_env.max_episode_length} "
        f"  num_envs={base_env.num_envs}",
        flush=True,
    )

    if args_cli.disable_base_contact:
        # A zero return keeps the episode alive; the buffer is read in the same step by
        # ``compute``, so this hides the term without touching the sensors.
        manager.set_term_cfg("base_contact", replace(manager.get_term_cfg("base_contact"), func=_never))
        print("[INFO] base_contact disabled", flush=True)

    header = "step ep_len done base_z  " + "  ".join(f"{n:>9s}" for n in ("terms", "contact_forces", "feet_forces"))
    print("\n" + header, flush=True)
    with torch.inference_mode():
        for step in range(args_cli.steps):
            ep_len = int(base_env.episode_length_buf.item())
            z_before = float(robot.data.root_pos_w[0, 2])
            actions = policy(obs)
            obs, _rew, dones, _extras = env.step(actions)
            if step % args_cli.print_every and not int(dones.item()):
                continue
            fired = " ".join(f"{name}={int(manager.get_term(name).item())}" for name in manager.active_terms)
            print(
                f"{step:4d} {ep_len:6d} {int(dones.item()):4d} "
                f"{z_before:.4f}->{float(robot.data.root_pos_w[0, 2]):.4f}  "
                f"{fired:<40s}  {_stats(contact)}  {_stats(feet)}",
                flush=True,
            )

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
