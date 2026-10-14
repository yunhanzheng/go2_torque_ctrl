"""Independent standard-Go2 SAC Stage A: flat-ground standing."""

from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.envs.go2.go2_config import GO2RoughCfg


class GO2StandSACCfg(GO2RoughCfg):
    class env(GO2RoughCfg.env):
        num_envs = 256
        # 48 proprioceptive values; terrain-height samples are disabled below.
        num_observations = 48
        episode_length_s = 10

    class terrain(GO2RoughCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False

    class commands(GO2RoughCfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 10.0

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(GO2RoughCfg.domain_rand):
        # First establish whether SAC itself can stand; robustness comes later.
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class noise(GO2RoughCfg.noise):
        add_noise = False

    class normalization(GO2RoughCfg.normalization):
        clip_actions = 1.0

    class rewards(GO2RoughCfg.rewards):
        only_positive_rewards = False
        base_height_target = 0.30

        class scales(GO2RoughCfg.rewards.scales):
            termination = -10.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.2
            orientation = -5.0
            torques = -0.0002
            dof_vel = -0.001
            dof_acc = -2.5e-7
            base_height = -10.0
            feet_air_time = 0.0
            collision = -1.0
            action_rate = -0.02
            stand_still = -1.0


class GO2StandSACTrainCfg(BaseConfig):
    seed = 1
    runner_class_name = "OffPolicyRunner"

    class algorithm:
        hidden_dims = [256, 256]
        activation = "silu"
        normalize_obs = True
        init_noise_std = 0.30
        action_limit = 1.0
        replay_buffer_size = 524_288
        num_learning_epochs = 1
        num_mini_batches = 16
        mini_batch_size = 1024
        actor_learning_rate = 3e-4
        critic_learning_rate = 3e-4
        alpha_learning_rate = 3e-4
        gamma = 0.99
        tau = 0.005
        alpha = 0.05
        auto_alpha = True
        target_entropy_scale = 1.0
        max_grad_norm = 1.0
        policy_frequency = 1
        n_steps = 1

    class runner:
        num_steps_per_env = 12
        max_iterations = 500
        save_interval = 25
        experiment_name = "SAC_standard_go2_stand"
        run_name = "flat_zero_speed_seed1"
        resume = False
        load_run = -1
        checkpoint = -1


class GO2StandSACSmokeCfg(GO2StandSACCfg):
    class env(GO2StandSACCfg.env):
        num_envs = 64


class GO2StandSACSmokeTrainCfg(GO2StandSACTrainCfg):
    class algorithm(GO2StandSACTrainCfg.algorithm):
        hidden_dims = [128, 128]
        replay_buffer_size = 16_384
        num_mini_batches = 2
        mini_batch_size = 256

    class runner(GO2StandSACTrainCfg.runner):
        num_steps_per_env = 6
        max_iterations = 3
        save_interval = 1
        experiment_name = "SAC_standard_go2_stand_smoke"
        run_name = "interface_check"
