"""Standard-Go2 SAC v4 with an observable diagonal gait clock."""

from legged_gym.envs.go2.go2_walk_sac_v3_config import (
    GO2Walk025SACV3Cfg,
    GO2Walk025SACV3TrainCfg,
)


class GO2Walk025SACV4Cfg(GO2Walk025SACV3Cfg):
    class env(GO2Walk025SACV3Cfg.env):
        num_observations = 50

    class gait:
        frequency = 1.5
        swing_height = 0.07
        clearance_sigma = 0.035

    class rewards(GO2Walk025SACV3Cfg.rewards):
        # Stage v4 only asks for a basic, sustainable gait. Exact command-speed
        # tracking is deliberately deferred until coordination is reliable.
        tracking_sigma = 0.10

        class scales(GO2Walk025SACV3Cfg.rewards.scales):
            termination = -50.0
            tracking_lin_vel = 0.5
            speed_error = 0.0
            forward_progress = 2.0
            low_speed = -0.25
            gait_contact = 3.0
            swing_clearance = 1.5
            feet_slip = -0.5
            contact_count_error = -0.5
            base_height = -25.0
            orientation = -10.0
            action_rate = -0.10


class GO2Walk025SACV4TrainCfg(GO2Walk025SACV3TrainCfg):
    class runner(GO2Walk025SACV3TrainCfg.runner):
        max_iterations = 250
        save_interval = 25
        experiment_name = "SAC_standard_go2_walk_025_v4_gait"
        run_name = "clocked_diagonal_fresh_optim_stand500_seed1"


class GO2Walk025SACV4SmokeCfg(GO2Walk025SACV4Cfg):
    class env(GO2Walk025SACV4Cfg.env):
        num_envs = 64


class GO2Walk025SACV4SmokeTrainCfg(GO2Walk025SACV4TrainCfg):
    class algorithm(GO2Walk025SACV4TrainCfg.algorithm):
        replay_buffer_size = 16_384
        num_mini_batches = 2
        mini_batch_size = 256

    class runner(GO2Walk025SACV4TrainCfg.runner):
        max_iterations = 5
        save_interval = 1
        experiment_name = "SAC_standard_go2_walk_025_v4_smoke"
        run_name = "clock_and_foot_state_check"
