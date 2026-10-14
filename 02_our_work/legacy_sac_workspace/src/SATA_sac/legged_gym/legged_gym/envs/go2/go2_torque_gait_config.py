"""Stage-3 flat-ground SATA configuration with gait coordination."""

from legged_gym.envs.go2.go2_torque_flat_config import (
    GO2TorqueFlatCfg,
    GO2TorqueFlatCfgPPO,
)


class GO2TorqueGaitCfg(GO2TorqueFlatCfg):
    class env(GO2TorqueFlatCfg.env):
        num_observations = 62

    class gait:
        frequency = 1.5
        swing_height = 0.065
        clearance_sigma = 0.025

    class rewards(GO2TorqueFlatCfg.rewards):
        class scales(GO2TorqueFlatCfg.rewards.scales):
            feet_air_time = 0.0
            gait_contact = 3.0
            swing_clearance = 1.5
            feet_slip = -0.5


class GO2TorqueGaitCfgPPO(GO2TorqueFlatCfgPPO):
    class runner(GO2TorqueFlatCfgPPO.runner):
        experiment_name = "SATA_flat_gait_coordination"
        run_name = "stage3_diagonal_gait"
        max_iterations = 3000

