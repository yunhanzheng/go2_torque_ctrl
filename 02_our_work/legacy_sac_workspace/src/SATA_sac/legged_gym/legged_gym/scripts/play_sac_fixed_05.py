"""Visualize SAC model_800 at a fixed 0.5 m/s command."""

import isaacgym  # must precede torch
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import class_to_dict, get_args, task_registry
from rsl_rl.runners import OffPolicyRunner


CHECKPOINT = (
    "/data/sac/src/SATA_sac/legged_gym/logs/SAC_stage3_flat_gait/"
    "Aug12_14-33-36_stage3_command_curriculum_seed1/model_800.pt"
)


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs("go2_torque_gait_sac")
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 20
    env_cfg.control.control_type = "T"
    env_cfg.test.use_test = True
    env_cfg.test.checkpoint = 800
    env_cfg.test.vel = torch.tensor([0.5, 0.0, 0.0, 0.0])
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.measure_heights = False
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.loss_action_obs = False

    # Replay is unused during playback; avoid allocating the training buffer.
    train_cfg.algorithm.replay_buffer_size = 16
    train_cfg.algorithm.mini_batch_size = 1
    train_cfg.algorithm.num_mini_batches = 1

    env, _ = task_registry.make_env("go2_torque_gait_sac", args=args, env_cfg=env_cfg)
    runner = OffPolicyRunner(env, class_to_dict(train_cfg), log_dir=None, device=args.rl_device)
    runner.load(CHECKPOINT, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    obs = env.get_observations()

    print("Playing SAC model_800: command vx=0.5 m/s")
    print("Close the simulator window or press Ctrl+C in this terminal to stop.")
    for step in range(10 * int(env.max_episode_length)):
        with torch.no_grad():
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
        robot_pos = env.root_states[0, :3].cpu().numpy()
        env.set_camera(robot_pos + np.array([1.2, 1.2, 0.8]), robot_pos)
        if step % 100 == 0:
            print(
                "step={} command=0.500 vx={:.3f} height={:.3f}".format(
                    step, env.base_lin_vel[0, 0].item(), env.root_states[0, 2].item()
                )
            )


if __name__ == "__main__":
    main()
