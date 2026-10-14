"""Stage A: learn stable standing before forward locomotion."""

from legged_gym.envs.go2.go2_torque_gait_sac_config import (
    GO2TorqueGaitSACCfg,
    GO2TorqueGaitSACTrainCfg,
)


class GO2TorqueStandSACCfg(GO2TorqueGaitSACCfg):
    class env(GO2TorqueGaitSACCfg.env):
        num_envs = 256

    class commands(GO2TorqueGaitSACCfg.commands):
        heading_command = False
        resampling_time = 5

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class safety:
        # A learnable first gate; raise only after the policy reliably stands.
        min_base_height = 0.18

    class rewards(GO2TorqueGaitSACCfg.rewards):
        base_height_target = 0.30

        class scales(GO2TorqueGaitSACCfg.rewards.scales):
            forward = 0.0
            moving_y = 0.0
            moving_yaw = 0.0
            gait_contact = 0.0
            swing_clearance = 0.0
            feet_slip = -0.2
            base_height_error = -20.0
            nominal_posture = -2.0
            stand_still = -1.0
            termination = -10.0


class GO2TorqueStandSACTrainCfg(GO2TorqueGaitSACTrainCfg):
    class algorithm(GO2TorqueGaitSACTrainCfg.algorithm):
        action_limit = 2.0
        replay_buffer_size = 262_144
        num_mini_batches = 16
        mini_batch_size = 1024

    class runner(GO2TorqueGaitSACTrainCfg.runner):
        num_steps_per_env = 12
        max_iterations = 300
        save_interval = 50
        experiment_name = "SAC_stageA_stand"
        run_name = "stand_h018_action2p0_seed1_pilot"


class GO2TorqueStandSACV2Cfg(GO2TorqueStandSACCfg):
    class safety:
        # Direct torque control starts without gravity compensation. Allow the
        # learner to experience recovery/support torques before tightening this.
        min_base_height = 0.10


class GO2TorqueStandSACV2TrainCfg(GO2TorqueStandSACTrainCfg):
    class algorithm(GO2TorqueStandSACTrainCfg.algorithm):
        action_limit = 4.7
        init_noise_std = 0.5
        replay_buffer_size = 524_288
        num_mini_batches = 32
        mini_batch_size = 2048

    class runner(GO2TorqueStandSACTrainCfg.runner):
        max_iterations = 500
        save_interval = 50
        experiment_name = "SAC_stageA_stand_v2"
        run_name = "stand_h010_action4p7_sigma0p5_seed1"
