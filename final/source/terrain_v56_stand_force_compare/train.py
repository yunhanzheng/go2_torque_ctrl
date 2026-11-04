"""Train one V56 standing variant with PPO (position control or direct torque).

Four tasks are registered, in two matched pairs: the two arms at the official 50 Hz policy rate
(``stand_pos`` / ``stand_torque``) and the same two at the physics rate (``stand_pos200`` /
``stand_torque200``), which is the only way to give both controllers the same loop bandwidth - see
the module docstring of ``config.py``.

The runner is IsaacLab's stock ``UnitreeGo2FlatPPORunnerCfg`` (network, optimiser, PPO
hyper-parameters); only the experiment name differs, plus the ``num_steps_per_env`` correction
that keeps an iteration at 0.48 s of physical time when the control rate quadruples.  Force
metrics reach wandb/TensorBoard through ``extras["log"]``, see ``metrics.py``.
"""
from __future__ import annotations

import argparse
import os
import sys

# --- Path bootstrap so the script runs from any CWD (session launchers on the GPU host
#     do not export PYTHONPATH).  ``cli_args`` is shared with the other experiments and
#     lives in ``final/source/base_adapter``; the stock Go2 task itself comes from the
#     installed IsaacLab, not from a local copy.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (_SOURCE_ROOT, os.path.join(_SOURCE_ROOT, "base_adapter"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train a V56 standing agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of a recorded video (steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Go2-V56-StandTorque-v0",
    help=(
        "Task id: Isaac-Go2-V56-StandPos-v0 / -StandTorque-v0 (official 50 Hz) and"
        " -StandPos200-v0 / -StandTorque200-v0 (bandwidth-matched pair at 200 Hz)."
    ),
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument(
    "--action_scale",
    type=float,
    default=None,
    help="Override the torque variant's N.m per unit of network output (ignored by the position task).",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OffPolicyRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

# Registers the four V56 tasks (two official-rate arms, two at the physics rate).
import terrain_v56_stand_force_compare.config  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    raise RuntimeError(
        f"Please install rsl-rl-lib >= {RSL_RL_VERSION}. Existing version is: '{installed_version}'"
    )

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # Torque-scale calibration hook.  The position task has no such field, so the flag is
    # ignored there on purpose instead of failing.
    if args_cli.action_scale is not None:
        if hasattr(env_cfg, "action_scale"):
            env_cfg.action_scale = args_cli.action_scale
            env_cfg.actions.joint_pos.action_scale = args_cli.action_scale
            print(f"[INFO] action_scale -> {args_cli.action_scale} N.m per unit output", flush=True)
        else:
            print("[INFO] --action_scale ignored: position-control task", flush=True)

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "OffPolicyRunner":
        runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.add_git_repo_to_log(__file__)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # IsaacLab does not reset inside __init__ and OnPolicyRunner.learn() only calls
    # get_observations(), so the first observations would be the un-reset spawn state.
    env.reset()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # This headless host hangs in the render teardown (ERROR_INCOMPATIBLE_DRIVER): the
    # process spins at 99 % CPU and keeps the GPU.  Results are already written, so exit
    # hard instead of calling env.close()/simulation_app.close().
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
