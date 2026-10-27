"""V51 evaluator for video recording with native terrain."""
import argparse
import os
import sys
import time
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
parser = argparse.ArgumentParser(description="Evaluate V51 with video recording.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_folder", type=str, default=None, help="Folder to save recorded videos.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_sac_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments (includes --checkpoint, --resume, --run_name, --logger, --log_project_name, etc.)
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

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

# Import V51 config to register the task
import terrain_v51_native_terrain.config  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

# Check RSL-RL version
installed_version = metadata.version("rsl-rl-lib")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """Evaluate V51 with video recording."""
    # override configurations with CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    
    # Set terrain seed for reproducibility
    if hasattr(env_cfg.scene.terrain.terrain_generator, 'seed'):
        env_cfg.scene.terrain.terrain_generator.seed = args_cli.seed if args_cli.seed else 42
    
    # Configure viewer for follow-camera (third-person view)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.0, -2.5, 1.5)  # Behind and above the robot
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)  # Look at robot center
    
    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    
    # Set up video recording directory
    checkpoint_path = Path(args_cli.checkpoint)
    if args_cli.video_folder:
        video_dir = Path(args_cli.video_folder)
    else:
        video_dir = checkpoint_path.parent / f"V51_seed{args_cli.seed or 42}_video"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # wrap for video recording
    video_kwargs = {
        "video_folder": str(video_dir),
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during evaluation.")
    print(f"[INFO] Video folder: {video_dir}")
    print(f"[INFO] Video length: {args_cli.video_length} steps")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # create runner from rsl-rl
    from rsl_rl.runners import OffPolicyRunner
    log_dir = str(checkpoint_path.parent)
    
    # Override replay buffer size to minimal value for evaluation (avoid OOM)
    agent_cfg_dict = agent_cfg.to_dict()
    if "algorithm" in agent_cfg_dict:
        agent_cfg_dict["algorithm"]["replay_buffer_size"] = 1000  # Minimal size for eval
    
    runner = OffPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device="cuda:0")
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    
    # Run evaluation
    print(f"[INFO] Running evaluation for {args_cli.video_length} steps...")
    obs = env.get_observations()
    
    start_time = time.time()
    for step in range(args_cli.video_length):
        # Use the actor directly for deterministic inference
        with torch.no_grad():
            actions = runner.alg.actor(obs, stochastic_output=False)
        obs, rewards, dones, infos = env.step(actions)
        
        if step % 100 == 0:
            print(f"  Step {step}/{args_cli.video_length}", flush=True)
    
    elapsed = time.time() - start_time
    print(f"[INFO] Evaluation completed in {elapsed:.1f}s")
    
    # close the environment
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
