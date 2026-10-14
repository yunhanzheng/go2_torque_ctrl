"""Today's independent flat-ground training task at a fixed 0.5 m/s."""

from legged_gym.envs.go2.go2_torque_gait_config import (
    GO2TorqueGaitCfg,
    GO2TorqueGaitCfgPPO,
)


class GO2TorqueFixedSpeedCfg(GO2TorqueGaitCfg):
    class commands(GO2TorqueGaitCfg.commands):
        heading_command = False
        resampling_time = 5

        class ranges:
            lin_vel_x = [0.5, 0.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]


class GO2TorqueFixedSpeedCfgPPO(GO2TorqueGaitCfgPPO):
    class runner(GO2TorqueGaitCfgPPO.runner):
        # A timestamp plus this unique run name creates a new directory.
        experiment_name = "SATA_flat_gait_coordination"
        run_name = "aug07_flat_fixed_0p5_from_zero_seed1"
        max_iterations = 1000
