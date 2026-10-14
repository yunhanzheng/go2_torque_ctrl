"""Stage A v3: residual SAC around a physics-based standing controller."""

from legged_gym.envs.go2.go2_torque_stand_sac_config import GO2TorqueStandSACCfg, GO2TorqueStandSACTrainCfg


class GO2TorqueStandResidualSACCfg(GO2TorqueStandSACCfg):
    class env(GO2TorqueStandSACCfg.env):
        num_envs = 256

    class init_state(GO2TorqueStandSACCfg.init_state):
        # Start Stage A from the normal Go2 standing geometry.  The inherited
        # SATA torque pose (z=0.10, thigh=1.45, calf=-2.5) is a deep crouch
        # intended for the PPO start-up curriculum, not SAC cold-start standing.
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    class safety:
        min_base_height = 0.18

    class residual_support:
        stiffness = 40.0
        damping = 1.0

    class rewards(GO2TorqueStandSACCfg.rewards):
        class scales(GO2TorqueStandSACCfg.rewards.scales):
            residual_action = -0.05


class GO2TorqueStandResidualSACTrainCfg(GO2TorqueStandSACTrainCfg):
    class algorithm(GO2TorqueStandSACTrainCfg.algorithm):
        action_limit = 1.0
        init_noise_std = 0.15
        replay_buffer_size = 524_288
        num_mini_batches = 32
        mini_batch_size = 2048

    class runner(GO2TorqueStandSACTrainCfg.runner):
        max_iterations = 500
        save_interval = 50
        experiment_name = "SAC_stageA_stand_v3_residual"
        run_name = "normal_pose_pd40_residual1p0_h018_seed1"
