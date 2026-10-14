"""ETH-SAC v14: stronger diagonal coordination with gentle clearance shaping."""

import gymnasium as gym
import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from eth_sata_reward_v10 import (
    FOOT_NAMES,
    Go2EthSACSATARewardAllLegsEnvCfg,
    Go2EthSACSATARewardAllLegsRunnerCfg,
    sata_all_leg_gait_contact,
    sata_worst_swing_leg_clearance,
)


def bounded_high_kick_error(
    env,
    asset_cfg: SceneEntityCfg,
    maximum_height: float = 0.14,
    error_scale: float = 0.06,
):
    """Gentle bounded guardrail, active only for clearly excessive foot height."""

    height = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    excess = torch.relu(height - maximum_height) / error_scale
    return torch.clamp(excess.square().amax(dim=1), max=4.0)


@configclass
class Go2EthSACSATARewardStrongGaitEnvCfg(Go2EthSACSATARewardAllLegsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        ordered_feet = SceneEntityCfg("robot", body_names=FOOT_NAMES, preserve_order=True)
        ordered_sensor = SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True)
        self.rewards.gait_contact = RewTerm(
            func=sata_all_leg_gait_contact,
            weight=5.0,
            params={"sensor_cfg": ordered_sensor, "command_name": "base_velocity", "frequency": 1.5},
        )
        self.rewards.swing_clearance = RewTerm(
            func=sata_worst_swing_leg_clearance,
            weight=2.0,
            params={
                "asset_cfg": ordered_feet,
                "command_name": "base_velocity",
                "frequency": 1.5,
                "swing_height": 0.065,
                "sigma": 0.030,
            },
        )
        self.rewards.excessive_foot_height = RewTerm(
            func=bounded_high_kick_error,
            weight=-0.03,
            params={"asset_cfg": ordered_feet, "maximum_height": 0.14, "error_scale": 0.06},
        )


@configclass
class Go2EthSACSATARewardStrongGaitRunnerCfg(Go2EthSACSATARewardAllLegsRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "eth_sac_isaaclab_sata_reward_strong_gait_v14"
        self.max_iterations = 300


gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-ETHSAC-SATAReward-StrongGait-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2EthSACSATARewardStrongGaitEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2EthSACSATARewardStrongGaitRunnerCfg",
    },
)
