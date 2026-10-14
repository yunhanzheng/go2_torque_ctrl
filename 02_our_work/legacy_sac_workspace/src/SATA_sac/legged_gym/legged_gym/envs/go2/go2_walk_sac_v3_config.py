"""Stable-first standard-Go2 SAC walking v3 at 0.25 m/s."""

from legged_gym.envs.go2.go2_walk_sac_v2_config import (
    GO2Walk03SACV2Cfg,
    GO2Walk03SACV2TrainCfg,
)


class GO2Walk025SACV3Cfg(GO2Walk03SACV2Cfg):
    class commands(GO2Walk03SACV2Cfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 10.0

        class ranges:
            lin_vel_x = [0.25, 0.25]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class safety:
        min_base_height = 0.23
        min_forward_speed = 0.10
        max_tilt = 0.55

    class rewards(GO2Walk03SACV2Cfg.rewards):
        tracking_sigma = 0.02
        base_height_target = 0.30

        class scales(GO2Walk03SACV2Cfg.rewards.scales):
            termination = -50.0
            tracking_lin_vel = 4.0
            speed_error = -5.0
            forward_progress = 0.5
            low_speed = -0.75
            base_height = -25.0
            orientation = -10.0
            ang_vel_xy = -1.0
            lin_vel_z = -3.0
            nominal_posture = -1.5
            action_rate = -0.08
            dof_vel = -0.003
            feet_air_time = 0.2


class GO2Walk025SACV3TrainCfg(GO2Walk03SACV2TrainCfg):
    class algorithm(GO2Walk03SACV2TrainCfg.algorithm):
        actor_learning_rate = 8e-5
        critic_learning_rate = 1e-4
        alpha_learning_rate = 5e-5
        alpha = 0.002
        tau = 0.003
        num_mini_batches = 12

    class runner(GO2Walk03SACV2TrainCfg.runner):
        max_iterations = 200
        save_interval = 25
        experiment_name = "SAC_standard_go2_walk_025_v3"
        run_name = "stable_first_fresh_optim_stand500_seed1"
        resume = False


class GO2Walk025SACV3SmokeCfg(GO2Walk025SACV3Cfg):
    class env(GO2Walk025SACV3Cfg.env):
        num_envs = 64


class GO2Walk025SACV3SmokeTrainCfg(GO2Walk025SACV3TrainCfg):
    class algorithm(GO2Walk025SACV3TrainCfg.algorithm):
        replay_buffer_size = 16_384
        num_mini_batches = 2
        mini_batch_size = 256

    class runner(GO2Walk025SACV3TrainCfg.runner):
        max_iterations = 5
        save_interval = 1
        experiment_name = "SAC_standard_go2_walk_025_v3_smoke"
        run_name = "stable_reward_contract_check"
