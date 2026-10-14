"""Fixed 0.5 m/s task with hard minimum body-height termination."""

from legged_gym.envs.go2.go2_torque_fixed_speed_config import (
    GO2TorqueFixedSpeedCfg,
    GO2TorqueFixedSpeedCfgPPO,
)


class GO2TorqueHeightTerminationCfg(GO2TorqueFixedSpeedCfg):
    class safety:
        min_base_height = 0.22

    class rewards(GO2TorqueFixedSpeedCfg.rewards):
        base_height_target = 0.30

        class scales(GO2TorqueFixedSpeedCfg.rewards.scales):
            termination = -10.0


class GO2TorqueHeightTerminationCfgPPO(GO2TorqueFixedSpeedCfgPPO):
    class runner(GO2TorqueFixedSpeedCfgPPO.runner):
        experiment_name = "SATA_flat_fixed_speed_height_termination"
        run_name = "aug07_fixed_0p5_height_term_from_zero_seed1"
        max_iterations = 1000
