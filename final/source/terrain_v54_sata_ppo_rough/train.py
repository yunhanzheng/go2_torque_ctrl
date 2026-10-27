# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL for V54 (SATA-aligned: PPO + IsaacLab Go2 rough terrain)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# --- Path bootstrap: make this script runnable from any CWD without a pre-set
# PYTHONPATH (mirrors run_trial.sh). Puts the source tree and the shared adapter
# modules on sys.path so `import cli_args`, `import terrain_v54_sata_ppo_rough.config`,
# and the V51/V52 imports resolve. V51's directory is needed for `sata_alignment`,
# V52's for `sata_rewards`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (
    _SOURCE_ROOT,
    os.path.join(_SOURCE_ROOT, "dependency_chain"),
    os.path.join(_SOURCE_ROOT, "base_adapter"),
    os.path.join(_SOURCE_ROOT, "base_adapter", "initial_stair_source"),
    os.path.join(_SOURCE_ROOT, "terrain_v51_native_terrain"),
    os.path.join(_SOURCE_ROOT, "terrain_v52_sata_ppo_flat"),
    _HERE,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_sac_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--init_from",
    type=str,
    default=None,
    help="Absolute path to a checkpoint used to initialise the policy (cross-experiment warm start, "
    "e.g. V52's model_2999.pt).  Distinct from rsl_rl's --resume/--checkpoint, which look the "
    "checkpoint up under this run's own log root.",
)
parser.add_argument(
    "--init_from_bump",
    type=int,
    default=0,
    help="Value added to runner.current_learning_iteration after --init_from.  Checkpoints "
    "written before the 1-based fix are numbered 0-based (model_2999.pt == 3000 completed "
    "iterations), so pass 1 when warm-starting from such a file.",
)
parser.add_argument(
    "--init_actor_only",
    action="store_true",
    default=False,
    help="With --init_from: load the actor (and the iteration counter) only.  Checkpoints"
    " written before rev H have a 60-D critic while the rev H critic takes 247-D, so their"
    " critic/optimizer weights cannot be reused; the critic then starts from scratch.",
)
parser.add_argument(
    "--growth_step_offset",
    type=int,
    default=None,
    help="Value assigned to env._sata_physics_steps (the counter ``sata_growth_scale`` "
    "actually reads) so SATA's Gompertz curve continues where the source run stopped "
    "(V52/V54: 3000 iters x 24 steps = 72000).  Honoured with --init_from and with --resume.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
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

# Import V54 config to register the task (also registers V52/V51 via its import chain)
import terrain_v54_sata_ppo_rough.config  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# Check RSL-RL version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    raise RuntimeError(
        f"Please install rsl-rl-lib >= {RSL_RL_VERSION}. "
        f"Existing version is: '{installed_version}'"
    )

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "OffPolicyRunner":
        runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # optionally warm-start from a checkpoint of another run (e.g. V52's model_2999.pt)
    if args_cli.init_from:
        print(f"[INFO] Initialising from: {args_cli.init_from}", flush=True)
        if args_cli.init_actor_only:
            # Pre-rev-H checkpoints: the actor's input width did not change (60), the critic's
            # did (60 -> 247), and the Adam state is tied to both, so only the actor and the
            # iteration counter can be carried over.  ``strict`` only guards the loaded keys.
            runner.load(
                args_cli.init_from,
                load_cfg={"actor": True, "critic": False, "optimizer": False,
                          "iteration": True, "rnd": True},
            )
        else:
            runner.load(args_cli.init_from)
        print(f"[INFO] Loaded checkpoint at iteration {runner.current_learning_iteration}", flush=True)
        if args_cli.init_from_bump:
            runner.current_learning_iteration += args_cli.init_from_bump
            print(
                f"[INFO] Legacy 0-based checkpoint: current_learning_iteration := "
                f"{runner.current_learning_iteration}",
                flush=True,
            )

    # Restore SATA's Gompertz clock after a warm start / resume.  ``sata_growth_scale``
    # reads ``env._sata_physics_steps`` (it only falls back to ``common_step_counter`` when
    # that attribute is missing), so *this* is the counter that has to move.  Writing
    # common_step_counter alone left general_scale at ~0.13 (~9 Nm, 113 Hz) and would
    # cripple an already-trained policy.
    if args_cli.growth_step_offset is not None:
        from terrain_v51_native_terrain.sata_alignment import sata_growth_scale

        env.unwrapped._sata_physics_steps = int(args_cli.growth_step_offset)
        # keep IsaacLab's own clock in phase too (interval-event timing, fallback path)
        env.unwrapped.common_step_counter = int(args_cli.growth_step_offset)
        term = env.unwrapped.action_manager.get_term("joint_pos")
        term._update_growth()
        gs = sata_growth_scale(env.unwrapped)
        print(
            f"[INFO] _sata_physics_steps := {args_cli.growth_step_offset} "
            f"(general_scale ~ {gs:.3f}, torque_scale {term.current_torque_limit_scale:.3f}, "
            f"freq {100.0 + 100.0 * gs:.1f} Hz)",
            flush=True,
        )

    # run training
    # SATA calls reset_idx() inside __init__ (legged_robot.py:132), so even its first episode
    # spawns with randomised joints and root state.  IsaacLab does not reset in __init__, and
    # OnPolicyRunner.learn() only calls get_observations(), so reset explicitly here.
    env.reset()
    # rsl_rl's ``init_at_random_ep_len`` randomizes ``episode_length_buf`` only, but SATA's
    # time-out counts physics substeps (``sata_time_out`` reads ``_sata_env_substeps``).  That
    # counter starts at 0 for every env and is only cleared on reset, so without this line all
    # 4096 envs run the 10 s clock in lockstep and time out in the same iteration.  The logged
    # ``Train/mean_reward`` is the mean over the last 100 *finished* episodes, so it then only
    # refreshes a few times per episode and shows up as a staircase.  SATA itself is fine
    # because ``episode_length_buf`` *is* its substep counter and gets randomized directly.
    _env = env.unwrapped
    _env._sata_env_substeps = torch.randint_like(
        _env._sata_env_substeps, high=int(_env.max_episode_length)
    )
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # runner.learn() has already saved the final checkpoint and finished the wandb run.
    # On this Vulkan-less headless box, env.close() / simulation_app.close() hang inside C
    # render-cleanup (ERROR_INCOMPATIBLE_DRIVER): the process never exits, spins at 99% CPU
    # and keeps holding GPU memory. Hard-exit here so training always terminates and frees
    # the GPU. os._exit bypasses the hanging teardown; results are already saved, so nothing
    # is lost. (Confirmed: smoke_test PASSED then hung in sim_app.close() for 20+ min.)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    # main() hard-exits via os._exit(0) right after learn() (see note above), so the usual
    # simulation_app.close() teardown is intentionally omitted -- it hangs on this box.
    main()
