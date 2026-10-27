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

    header = (
        "step ep_len done base_z  " + "  ".join(f"{n:>9s}" for n in ("terms", "contact_forces", "feet_forces"))
    )
    print("\n" + header, flush=True)
    with torch.inference_mode():
        for step in range(args_cli.steps):
            ep_len = int(base_env.episode_length_buf.item())
            actions = policy(obs)
            obs, _rew, dones, _extras = env.step(actions)
            fired = "".join(
                f"{name}=" f"{int(manager.get_term(name).item())}" for name in manager.active_terms
            )
            print(
                f"{step:4d} {ep_len:6d} {int(dones.item()):4d} "
                f"{float(robot.data.root_pos_w[0, 2]):.4f}  "
                f"{fired:<24s}  {_stats(contact)}  {_stats(feet)}",
                flush=True,
            )

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
