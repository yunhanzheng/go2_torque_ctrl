"""Original fixed-speed task plus a soft four-leg standing-posture prior."""

from legged_gym.envs.go2.go2_torque_fixed_speed_config import (
    GO2TorqueFixedSpeedCfg,
    GO2TorqueFixedSpeedCfgPPO,
)


class GO2TorqueNominalPostureCfg(GO2TorqueFixedSpeedCfg):
    class rewards(GO2TorqueFixedSpeedCfg.rewards):
        class scales(GO2TorqueFixedSpeedCfg.rewards.scales):
            nominal_posture = -1.0


class GO2TorqueNominalPostureCfgPPO(GO2TorqueFixedSpeedCfgPPO):
    class runner(GO2TorqueFixedSpeedCfgPPO.runner):
        experiment_name = "SATA_flat_fixed_speed_nominal_posture"
        run_name = "aug07_fixed_0p5_nominal_posture_from_zero_seed1"
        max_iterations = 1000
