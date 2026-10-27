"""Record video with either position control or torque control.

Usage:
    python evaluate_control.py --control position --checkpoint <path>
    python evaluate_control.py --control torque   --checkpoint <path>
"""
import argparse
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

# --- Path bootstrap: runnable from any CWD without a pre-set PYTHONPATH
# (mirrors run_trial.sh). Puts the source tree and shared adapter modules on
# sys.path so `import cli_args` and `import terrain_v51_native_terrain.config` resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (
    _SOURCE_ROOT,
    os.path.join(_SOURCE_ROOT, "dependency_chain"),
    os.path.join(_SOURCE_ROOT, "base_adapter"),
    os.path.join(_SOURCE_ROOT, "base_adapter", "initial_stair_source"),
    _HERE,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate with position or torque control.")
parser.add_argument("--control", type=str, default="position", choices=["position", "torque"],
                    help="Control type: position (JointPositionActionCfg) or torque (JointEffortActionCfg)")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_folder", type=str, default=None, help="Folder to save recorded videos.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Go2-V50-CurriculumRecovery-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_sac_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import logging

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.envs.mdp.actions.actions_cfg import JointEffortActionCfg, JointPositionActionCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

# Import V50 config (model_5099.pt was trained with V50)
import terrain_v50_curriculum_recovery.config  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)
installed_version = metadata.version("rsl-rl-lib")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """Evaluate with specified control type."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed

    # Set terrain seed
    if hasattr(env_cfg.scene.terrain.terrain_generator, 'seed'):
        env_cfg.scene.terrain.terrain_generator.seed = args_cli.seed

    # === Override control type ===
    if args_cli.control == "position":
        env_cfg.actions.joint_pos = JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
        )
        env_cfg.decimation = 4
        env_cfg.sim.render_interval = 4
        print("[INFO] Control: POSITION (JointPositionActionCfg, scale=0.25, decimation=4)")
    else:
        env_cfg.actions.joint_pos = JointEffortActionCfg(
            asset_name="robot", joint_names=[".*"], scale=5.0
        )
        env_cfg.decimation = 1
        env_cfg.sim.render_interval = 1
        print("[INFO] Control: TORQUE (JointEffortActionCfg, scale=5.0, decimation=1)")

    # Viewer config for follow camera
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.0, -2.5, 1.5)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # Video directory
    checkpoint_path = Path(args_cli.checkpoint)
    control_name = args_cli.control
    if args_cli.video_folder:
        video_dir = Path(args_cli.video_folder)
    else:
        video_dir = checkpoint_path.parent / f"control_compare_{control_name}"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Video recording
    video_kwargs = {
        "video_folder": str(video_dir),
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    print(f"[INFO] Video folder: {video_dir}")
    print(f"[INFO] Video length: {args_cli.video_length} steps")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    from rsl_rl.runners import OffPolicyRunner
    log_dir = str(checkpoint_path.parent)
    agent_cfg_dict = agent_cfg.to_dict()
    if "algorithm" in agent_cfg_dict:
        agent_cfg_dict["algorithm"]["replay_buffer_size"] = 1000

    runner = OffPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device="cuda:0")
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)

    print(f"[INFO] Running {args_cli.video_length} steps with {control_name} control...")
    obs = env.get_observations()
    start_time = time.time()

    for step in range(args_cli.video_length):
        with torch.no_grad():
            actions = runner.alg.actor(obs, stochastic_output=False)
        obs, rewards, dones, infos = env.step(actions)
        if step % 100 == 0:
            print(f"  Step {step}/{args_cli.video_length}", flush=True)

    elapsed = time.time() - start_time
    print(f"[INFO] Done in {elapsed:.1f}s — control={control_name}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
