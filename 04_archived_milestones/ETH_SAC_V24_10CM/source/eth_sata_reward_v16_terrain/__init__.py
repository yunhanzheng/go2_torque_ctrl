"""SAC v15 fine-tuning on Isaac Lab rough terrains.

This task deliberately keeps the v15 62-D SATA actor observation so the
selected v15 actor can be transferred without changing the network input.
The height scanner is used only to make foot-clearance rewards relative to
the local terrain; it is not exposed to the policy.
"""

from copy import deepcopy

import gymnasium as gym
import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

from eth_sata_reward_v10 import FOOT_NAMES, sata_all_leg_gait_contact
from eth_sata_reward_v15 import (
    Go2EthSACSATARewardStableEnvCfg,
    Go2EthSACSATARewardStableRunnerCfg,
)


def _foot_clearance_above_terrain(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Estimate each foot's clearance from the nearest height-scanner ray hit."""

    feet_xy = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, :2]
    ray_hits = env.scene.sensors["height_scanner"].data.ray_hits_w
    # The Go2 scanner has 187 rays.  This temporary distance matrix is small
    # compared with the simulator state and avoids assumptions about ray order.
    nearest_ray = torch.cdist(feet_xy, ray_hits[..., :2]).argmin(dim=-1)
    ground_z = torch.gather(ray_hits[..., 2], 1, nearest_ray)
    foot_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    return foot_z - ground_z


def terrain_relative_swing_clearance(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    frequency: float = 1.5,
    swing_height: float = 0.065,
    sigma: float = 0.030,
):
    """v15 worst-swing-leg reward measured above local terrain."""

    phase = 2.0 * torch.pi * frequency * env.episode_length_buf * env.step_dt
    first_diagonal = torch.sin(phase) >= 0.0
    desired_swing = torch.stack(
        (~first_diagonal, first_diagonal, first_diagonal, ~first_diagonal), dim=1
    )
    clearance = _foot_clearance_above_terrain(env, asset_cfg)
    score = torch.exp(-torch.square((clearance - swing_height) / sigma))
    swing_scores = score.masked_fill(~desired_swing, 1.0)
    moving = (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    return swing_scores.amin(dim=1) * moving


def terrain_relative_high_kick_error(
    env,
    asset_cfg: SceneEntityCfg,
    maximum_height: float = 0.14,
    error_scale: float = 0.06,
):
    """v15 high-kick guardrail measured above local terrain."""

    excess = torch.relu(_foot_clearance_above_terrain(env, asset_cfg) - maximum_height) / error_scale
    return torch.clamp(excess.square().amax(dim=1), max=4.0)


@configclass
class Go2EthSACSATARewardTerrainEnvCfg(Go2EthSACSATARewardStableEnvCfg):
    """The v15 task on Go2-scaled rough terrain, with unchanged actor input."""

    def __post_init__(self):
        super().__post_init__()

        rough = UnitreeGo2RoughEnvCfg()
        self.scene.terrain = deepcopy(rough.scene.terrain)
        self.scene.height_scanner = deepcopy(rough.scene.height_scanner)
        self.curriculum.terrain_levels = deepcopy(rough.curriculum.terrain_levels)

        # Keep the scanner as reward-only privileged geometry.  In particular,
        # do not add height_scan to the 62-D v15 policy observation.
        ordered_feet = SceneEntityCfg("robot", body_names=FOOT_NAMES, preserve_order=True)
        ordered_sensor = SceneEntityCfg(
            "contact_forces", body_names=FOOT_NAMES, preserve_order=True
        )
        self.rewards.gait_contact = RewTerm(
            func=sata_all_leg_gait_contact,
            weight=5.0,
            params={"sensor_cfg": ordered_sensor, "command_name": "base_velocity", "frequency": 1.5},
        )
        self.rewards.swing_clearance = RewTerm(
            func=terrain_relative_swing_clearance,
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
            func=terrain_relative_high_kick_error,
            weight=-0.03,
            params={"asset_cfg": ordered_feet, "maximum_height": 0.14, "error_scale": 0.06},
        )


@configclass
class Go2EthSACSATARewardTerrainRunnerCfg(Go2EthSACSATARewardStableRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "eth_sac_isaaclab_sata_reward_terrain_v16"


gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-ETHSAC-SATAReward-Terrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2EthSACSATARewardTerrainEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2EthSACSATARewardTerrainRunnerCfg",
    },
)
