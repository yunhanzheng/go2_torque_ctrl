# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OffPolicyRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import eth_fixed05_v2  # noqa: F401
import eth_sata_reward_v17_stairs
import json
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Fixed geometry for both checkpoints, evaluated from the central platform.
    direction = os.environ.get('STAIR_DIRECTION', 'down')
    height = float(os.environ.get('STAIR_HEIGHT', '0.05'))
    output = os.environ['STAIR_OUTPUT']
    gen = env_cfg.scene.terrain.terrain_generator
    key = 'pyramid_stairs' if direction == 'down' else 'pyramid_stairs_inv'
    stair = gen.sub_terrains[key]
    stair.step_height_range = (height, height)
    stair.proportion = 1.0
    gen.sub_terrains = {key: stair}
    gen.num_rows = 1
    gen.num_cols = 4
    gen.curriculum = False
    env_cfg.curriculum.terrain_levels = None
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.episode_length_s = 20.0
    env_cfg.events.reset_base.params['pose_range'] = {'x': (0., 0.), 'y': (0., 0.), 'yaw': (0., 0.)}
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.observations.policy.enable_corruption = False
    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    # set viewer camera before creating isaac environment
    env_cfg.viewer.eye = (2.2, -2.2, 1.35)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    print(
        f"[INFO] Viewer camera follows robot: "
        f"eye={env_cfg.viewer.eye}, lookat={env_cfg.viewer.lookat}, "
        f"origin_type={env_cfg.viewer.origin_type}, asset_name={env_cfg.viewer.asset_name}"
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": output + "_video",
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OffPolicyRunner":
        runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    raw = env.unwrapped
    robot = raw.scene['robot']
    feet_ids, _ = robot.find_bodies('.*_foot')
    n = raw.num_envs
    active = torch.ones(n, dtype=torch.bool, device=raw.device)
    success = torch.zeros_like(active)
    failed = torch.zeros_like(active)
    max_x = torch.zeros(n, device=raw.device)
    stable_steps = torch.zeros(n, dtype=torch.long, device=raw.device)
    trace = []
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, extras = env.step(actions)
            local = robot.data.root_pos_w - raw.scene.env_origins
            valid = active & ~dones.bool()
            max_x[valid] = torch.maximum(max_x[valid], local[valid, 0])
            feet_x = robot.data.body_pos_w[:, feet_ids, 0] - raw.scene.env_origins[:, 0:1]
            # Entire robot clears the outer stair edge and remains upright for 0.5 s.
            cleared = (local[:, 0] > 3.5) & (feet_x.amin(dim=1) > 3.05) & (robot.data.projected_gravity_b[:, 2] < -0.85)
            stable_steps = torch.where(valid & cleared, stable_steps + 1, torch.zeros_like(stable_steps))
            reached = valid & (stable_steps >= round(0.5 / dt))
            success |= reached
            failed |= active & dones.bool() & ~extras.get('time_outs', torch.zeros_like(active)).bool()
            if timestep % 20 == 0 or bool(dones[0]):
                trace.append({'step': timestep, 'x': float(local[0,0]), 'z': float(local[0,2]), 'done': bool(dones[0]), 'active': bool(active[0])})
            active &= ~reached & ~dones.bool()
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if timestep >= 4000 or (not args_cli.video and not bool(active.any())):
            break
        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    result = {'checkpoint': resume_path, 'direction': direction, 'height_m': height, 'num_envs': n,
              'steps': timestep, 'dt': dt, 'success': int(success.sum()), 'terminated': int(failed.sum()),
              'other_no_success': int((~success & ~failed).sum()), 'max_x_m': max_x.tolist(),
              'trace_env0': trace, 'criterion': 'first episode; base x > 3.5m, all feet x > 3.05m, gravity_z < -0.85 for 0.5s; no reset'}
    with open(output + '.json', 'w') as f:
        json.dump(result, f, indent=2)
    print('[STAIR_RESULT]', {k:v for k,v in result.items() if k not in ('trace_env0', 'max_x_m')})
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
