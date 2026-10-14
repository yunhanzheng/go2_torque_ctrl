"""SAC configuration matching the frozen Stage-3 flat Go2 environment."""

from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.envs.go2.go2_torque_gait_config import GO2TorqueGaitCfg


class GO2TorqueGaitSACCfg(GO2TorqueGaitCfg):
    """Environment is inherited unchanged from the Stage-3 PPO task."""


class GO2TorqueGaitSACTrainCfg(BaseConfig):
    seed = 1
    runner_class_name = "OffPolicyRunner"

    class algorithm:
        hidden_dims = [512, 256, 128]
        activation = "silu"
        normalize_obs = True
        init_noise_std = 0.15
        # 23.5 Nm task limit / action_scale 5.0. Validate against PPO action
        # statistics before the formal comparison.
        action_limit = 4.7
        replay_buffer_size = 1_000_000
        num_learning_epochs = 1
        num_mini_batches = 50
        mini_batch_size = 4096
        actor_learning_rate = 2e-4
        critic_learning_rate = 2e-4
        alpha_learning_rate = 2e-5
        gamma = 0.97
        tau = 0.003
        alpha = 0.001
        auto_alpha = True
        target_entropy_scale = 0.167
        max_grad_norm = 1.0
        policy_frequency = 1
        n_steps = 5

    class runner:
        num_steps_per_env = 24
        max_iterations = 800
        save_interval = 50
        experiment_name = "SAC_stage3_flat_gait"
        run_name = "stage3_command_curriculum_seed1"
        resume = False
        load_run = -1
        checkpoint = -1


class GO2TorqueGaitSACSmokeCfg(GO2TorqueGaitSACCfg):
    class env(GO2TorqueGaitSACCfg.env):
        num_envs = 64


class GO2TorqueGaitSACSmokeTrainCfg(GO2TorqueGaitSACTrainCfg):
    class algorithm(GO2TorqueGaitSACTrainCfg.algorithm):
        hidden_dims = [128, 64]
        replay_buffer_size = 32_768
        num_mini_batches = 2
        mini_batch_size = 256

    class runner(GO2TorqueGaitSACTrainCfg.runner):
        num_steps_per_env = 6
        max_iterations = 3
        save_interval = 1
        experiment_name = "SAC_stage3_smoke"
        run_name = "transition_and_update_smoke"


class GO2TorqueGaitSACPilotCfg(GO2TorqueGaitSACCfg):
    class env(GO2TorqueGaitSACCfg.env):
        num_envs = 256


class GO2TorqueGaitSACPilotTrainCfg(GO2TorqueGaitSACTrainCfg):
    class algorithm(GO2TorqueGaitSACTrainCfg.algorithm):
        replay_buffer_size = 262_144
        num_mini_batches = 8
        mini_batch_size = 1024

    class runner(GO2TorqueGaitSACTrainCfg.runner):
        num_steps_per_env = 12
        max_iterations = 300
        save_interval = 50
        experiment_name = "SAC_stage3_pilot"
        run_name = "256env_action4p7_300iter_seed1"
