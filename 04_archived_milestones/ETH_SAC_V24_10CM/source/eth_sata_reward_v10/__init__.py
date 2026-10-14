"""ETH-SAC SATA reward with upright safety and all-leg gait participation."""

import gymnasium as gym
import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

from eth_sata_reward_v8 import _apply_sata_stage3_rewards, _desired_contacts
from eth_sata_reward_v9 import (
    Go2EthSACSATARewardUprightEnvCfg,
    Go2EthSACSATARewardUprightRunnerCfg,
)


FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def sata_all_leg_gait_contact(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    frequency: float = 1.5,
):
    """Reward contact timing while making the worst leg matter.

    The previous all-leg mean allowed one inactive leg to be hidden by the
    other three.  This keeps the same [0, 1] range but blends the mean with the
    minimum per-leg score, so full reward requires all four legs to follow the
    requested diagonal contact clock.
    """

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    actual = forces > 1.0
    desired = _desired_contacts(env, frequency)
    correct = (actual == desired).float()
    moving = (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    return 0.5 * (correct.mean(dim=1) + correct.amin(dim=1)) * moving


def sata_worst_swing_leg_clearance(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    frequency: float = 1.5,
    swing_height: float = 0.065,
    sigma: float = 0.025,
):
    """Score the worse of the two commanded swing legs.

    Both diagonal legs must now lift: a stationary RR leg can no longer be
    compensated by a high-scoring FL leg.  The output remains in [0, 1].
    """

    robot = env.scene[asset_cfg.name]
    desired_swing = ~_desired_contacts(env, frequency)
    height = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    score = torch.exp(-torch.square((height - swing_height) / sigma))
    swing_scores = score.masked_fill(~desired_swing, 1.0)
    moving = (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    return swing_scores.amin(dim=1) * moving


@configclass
class Go2EthSACSATARewardAllLegsEnvCfg(Go2EthSACSATARewardUprightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Re-apply for clarity when this task is inspected independently, then
        # override only the two aggregation rules responsible for gait collapse.
        _apply_sata_stage3_rewards(self)
        ordered_feet_sensor = SceneEntityCfg(
            "contact_forces", body_names=FOOT_NAMES, preserve_order=True
        )
        ordered_feet_asset = SceneEntityCfg(
            "robot", body_names=FOOT_NAMES, preserve_order=True
        )
        self.rewards.gait_contact = RewTerm(
            func=sata_all_leg_gait_contact,
            weight=3.0,
            params={
                "sensor_cfg": ordered_feet_sensor,
                "command_name": "base_velocity",
                "frequency": 1.5,
            },
        )
        self.rewards.swing_clearance = RewTerm(
            func=sata_worst_swing_leg_clearance,
            weight=1.5,
            params={
                "asset_cfg": ordered_feet_asset,
                "command_name": "base_velocity",
                "frequency": 1.5,
                "swing_height": 0.065,
                "sigma": 0.025,
            },
        )


@configclass
class Go2EthSACSATARewardAllLegsRunnerCfg(Go2EthSACSATARewardUprightRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "eth_sac_isaaclab_sata_reward_all_legs_v10"
        self.max_iterations = 1000


gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-ETHSAC-SATAReward-AllLegs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2EthSACSATARewardAllLegsEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2EthSACSATARewardAllLegsRunnerCfg",
    },
)
