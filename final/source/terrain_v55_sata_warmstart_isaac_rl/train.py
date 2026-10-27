# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train V55: SATA's pretrained actor, IsaacLab's stock Go2 rewards, V54's terrain."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# --- Path bootstrap: make this script runnable from any CWD without a pre-set
# PYTHONPATH (mirrors run_trial.sh). Puts the source tree and the shared adapter
# modules on sys.path so `import cli_args`, `import terrain_v54_sata_ppo_rough.config`,
# and the V51/V52/V54/V55 imports resolve. V51's directory is needed for `sata_alignment`,
# V52's for `sata_rewards`, V54's for the env cfg V55 extends.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (
    _SOURCE_ROOT,
    os.path.join(_SOURCE_ROOT, "dependency_chain"),
    os.path.join(_SOURCE_ROOT, "base_adapter"),
    os.path.join(_SOURCE_ROOT, "base_adapter", "initial_stair_source"),
    os.path.join(_SOURCE_ROOT, "terrain_v51_native_terrain"),
    os.path.join(_SOURCE_ROOT, "terrain_v52_sata_ppo_flat"),
    os.path.join(_SOURCE_ROOT, "terrain_v54_sata_ppo_rough"),
    os.path.join(_SOURCE_ROOT, "terrain_v55_sata_warmstart_isaac_rl"),
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
    "--sata_init",
    type=str,
    default=None,
    help="SATA checkpoint whose actor weights initialise the policy, e.g. "
    ".../SATA/legged_gym/logs/SATA/Oct14_16-33-32_/model_3000.pt.  The per-joint std is NOT "
    "taken (init_noise_std stays 1.0, see --sata_init_std), the critic is NOT loaded (V55 "
    "changes the whole reward function), and neither is the Adam state.  Unlike a rsl_rl "
    "--resume this keeps the iteration counter at 0.",
)
parser.add_argument(
    "--sata_init_std",
    action="store_true",
    default=False,
    help="Also take SATA's saved std (min 1.48 / mean 2.31 / max 3.10) instead of the "
    "IsaacLab default 1.0 (single-variable ablation).",
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

# Import V55 config to register the task (also registers V54/V52/V51 via its import chain)
import terrain_v55_sata_warmstart_isaac_rl.config  # noqa: F401

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
        runner.load(args_cli.init_from)
        print(f"[INFO] Loaded checkpoint at iteration {runner.current_learning_iteration}", flush=True)
        if args_cli.init_from_bump:
            runner.current_learning_iteration += args_cli.init_from_bump
            print(
                f"[INFO] Legacy 0-based checkpoint: current_learning_iteration := "
                f"{runner.current_learning_iteration}",
                flush=True,
            )

    # Warm start from SATA's finished policy.  Only the actor and its per-joint std are
    # copied: the critic was trained against SATA's 25/s reward scale and is meaningless
    # here, so it keeps its fresh initialisation.  No growth-clock offset is needed either
    # -- V55 ramps gs from model_3000's operating point (0.789) up to 1.0 by itself, see
    # v55_growth.py; the inherited Gompertz counter is only kept in sync for logging.
    if args_cli.sata_init:
        from load_sata_actor import load_sata_actor

        load_sata_actor(
            runner.alg.actor,
            args_cli.sata_init,
            load_std=args_cli.sata_init_std,
        )
        _term = env.unwrapped.action_manager.get_term("joint_pos")
        print(
            f"[INFO] V55 warm start: gs {_term.cfg.gs_start:.3f} -> 1.0 over "
            f"{_term.cfg.gs_ramp_steps} policy steps "
            f"({_term.cfg.gs_ramp_steps // 24} iterations at num_steps_per_env=24)",
            flush=True,
        )
        print(
            "[INFO] V55 warm start: critic and Adam state start fresh "
            f"(lr {runner.alg.learning_rate}, {runner.alg.schedule} schedule); "
            "the SATA optimizer_state_dict is deliberately ignored",
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
