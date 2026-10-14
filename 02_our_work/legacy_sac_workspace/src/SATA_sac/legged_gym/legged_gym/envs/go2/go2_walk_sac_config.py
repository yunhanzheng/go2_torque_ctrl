"""Independent standard-Go2 SAC Stage B1: flat walking at 0.3 m/s."""

from legged_gym.envs.go2.go2_stand_sac_config import (
    GO2StandSACCfg,
    GO2StandSACTrainCfg,
)


class GO2Walk03SACCfg(GO2StandSACCfg):
    class commands(GO2StandSACCfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 10.0

        class ranges:
            lin_vel_x = [0.3, 0.3]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(GO2StandSACCfg.rewards):
        class scales(GO2StandSACCfg.rewards.scales):
            tracking_lin_vel = 3.0
            tracking_ang_vel = 0.5
            base_height = -8.0
            orientation = -4.0
            feet_air_time = 0.2
            stand_still = 0.0


class GO2Walk03SACTrainCfg(GO2StandSACTrainCfg):
    class runner(GO2StandSACTrainCfg.runner):
        max_iterations = 300
        save_interval = 25
        experiment_name = "SAC_standard_go2_walk_03"
        run_name = "warmstart_stand500_seed1"
        resume = False


class GO2Walk03SACSmokeCfg(GO2Walk03SACCfg):
    class env(GO2Walk03SACCfg.env):
        num_envs = 64


class GO2Walk03SACSmokeTrainCfg(GO2Walk03SACTrainCfg):
    class algorithm(GO2Walk03SACTrainCfg.algorithm):
        replay_buffer_size = 16_384
        num_mini_batches = 2
        mini_batch_size = 256

    class runner(GO2Walk03SACTrainCfg.runner):
        max_iterations = 5
        save_interval = 1
        experiment_name = "SAC_standard_go2_walk_03_smoke"
        run_name = "warmstart_interface_check"
