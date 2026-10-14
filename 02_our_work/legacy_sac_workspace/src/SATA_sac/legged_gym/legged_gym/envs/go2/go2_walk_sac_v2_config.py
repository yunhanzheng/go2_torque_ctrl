"""Independent standard-Go2 SAC walking v2 at 0.3 m/s."""

from legged_gym.envs.go2.go2_walk_sac_config import GO2Walk03SACCfg, GO2Walk03SACTrainCfg


class GO2Walk03SACV2Cfg(GO2Walk03SACCfg):
    class safety:
        min_base_height = 0.22
        min_forward_speed = 0.15

    class rewards(GO2Walk03SACCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.03
        base_height_target = 0.30

        class scales(GO2Walk03SACCfg.rewards.scales):
            termination = -20.0
            tracking_lin_vel = 5.0
            tracking_ang_vel = 0.5
            forward_progress = 2.0
            low_speed = -2.0
            base_height = -20.0
            orientation = -5.0
            nominal_posture = -1.0
            feet_air_time = 0.5
            torques = -0.0002
            action_rate = -0.02


class GO2Walk03SACV2TrainCfg(GO2Walk03SACTrainCfg):
    class runner(GO2Walk03SACTrainCfg.runner):
        max_iterations = 400
        save_interval = 25
        experiment_name = "SAC_standard_go2_walk_03_v2"
        run_name = "reward_fix_warmstart_stand500_seed1"
        resume = False


class GO2Walk03SACV2SmokeCfg(GO2Walk03SACV2Cfg):
    class env(GO2Walk03SACV2Cfg.env):
        num_envs = 64


class GO2Walk03SACV2SmokeTrainCfg(GO2Walk03SACV2TrainCfg):
    class algorithm(GO2Walk03SACV2TrainCfg.algorithm):
        replay_buffer_size = 16_384
        num_mini_batches = 2
        mini_batch_size = 256

    class runner(GO2Walk03SACV2TrainCfg.runner):
        max_iterations = 5
        save_interval = 1
        experiment_name = "SAC_standard_go2_walk_03_v2_smoke"
        run_name = "reward_contract_check"
