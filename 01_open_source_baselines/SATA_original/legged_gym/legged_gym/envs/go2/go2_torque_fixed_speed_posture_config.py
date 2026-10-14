"""Fixed 0.5 m/s flat task with an explicit standing-height constraint."""

from legged_gym.envs.go2.go2_torque_fixed_speed_config import (
    GO2TorqueFixedSpeedCfg,
    GO2TorqueFixedSpeedCfgPPO,
)


class GO2TorqueFixedSpeedPostureCfg(GO2TorqueFixedSpeedCfg):
    class rewards(GO2TorqueFixedSpeedCfg.rewards):
        base_height_target = 0.30

        class scales(GO2TorqueFixedSpeedCfg.rewards.scales):
            # _reward_base_height is squared height error, so a negative scale
            # penalizes crouching while preserving all baseline rewards.
            base_height = -10.0


class GO2TorqueFixedSpeedPostureCfgPPO(GO2TorqueFixedSpeedCfgPPO):
    class runner(GO2TorqueFixedSpeedCfgPPO.runner):
        experiment_name = "SATA_flat_fixed_speed_posture"
        run_name = "aug07_fixed_0p5_posture_from_zero_seed1"
        max_iterations = 1000
