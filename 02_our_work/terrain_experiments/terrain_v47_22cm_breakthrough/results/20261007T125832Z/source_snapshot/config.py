"""Focused 20--22 cm curriculum with exact-height consolidation anchors."""
from copy import deepcopy

import gymnasium as gym
import torch
from isaaclab.utils import configclass

from eth_sata_reward_v16_terrain import terrain_relative_swing_clearance
from terrain_v46_23cm_from_20cm.config import (
    V46TwentyToTwentyThreeCmEnvCfg,
    V46TwentyToTwentyThreeCmRunnerCfg,
)


def focused_twenty_two_cm_clearance(
    env, asset_cfg, command_name, frequency=1.5, swing_height=0.09, sigma=0.045
):
    """Raise the terrain-relative target slightly for the 22 cm frontier."""
    hits = env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits)
    high = hits.masked_fill(~valid, -torch.inf).amax(dim=-1)
    low = hits.masked_fill(~valid, torch.inf).amin(dim=-1)
    relief = torch.nan_to_num(high - low, nan=0.0, posinf=0.0, neginf=0.0)
    target = swing_height + 0.65 * relief.clamp(0.0, 0.22)
    return terrain_relative_swing_clearance(
        env, asset_cfg, command_name, frequency, target[:, None], sigma
    )


@configclass
class V47TwentyTwoCmEnvCfg(V46TwentyToTwentyThreeCmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrains = self.scene.terrain.terrain_generator.sub_terrains

        terrains["stairs_up"].step_height_range = (0.20, 0.22)
        terrains["stairs_up"].step_width = 0.30

        anchor21 = deepcopy(terrains["stairs_up"])
        anchor21.step_height_range = (0.21, 0.21)
        terrains["stairs_up_21_anchor"] = anchor21
        anchor22 = deepcopy(terrains["stairs_up"])
        anchor22.step_height_range = (0.22, 0.22)
        terrains["stairs_up_22_anchor"] = anchor22

        proportions = {
            "plane": 0.04,
            "stairs_up": 0.45,
            "stairs_up_22_anchor": 0.15,
            "stairs_up_21_anchor": 0.20,
            "stairs_up_20_anchor": 0.10,
            "stairs_up_18_anchor": 0.00,
            "stairs_down": 0.02,
            "slope_up": 0.005,
            "slope_down": 0.005,
            "boxes": 0.015,
            "random_rough": 0.015,
        }
        for name, proportion in proportions.items():
            terrains[name].proportion = proportion

        # Start the main curriculum at no more than about 20.9 cm.
        self.scene.terrain.max_init_terrain_level = 4
        self.rewards.swing_clearance.func = focused_twenty_two_cm_clearance
        self.rewards.swing_clearance.weight = 6.0
        self.rewards.swing_clearance.params["swing_height"] = 0.09
        self.rewards.swing_clearance.params["sigma"] = 0.045


@configclass
class V47TwentyTwoCmRunnerCfg(V46TwentyToTwentyThreeCmRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v47_22cm_breakthrough"
        self.save_interval = 5
        self.algorithm.class_name = "terrain_v47_22cm_breakthrough.models:V47GuardedTransferSAC"
        self.algorithm.actor_learning_rate = 3.0e-7
        self.algorithm.critic_learning_rate = 5.0e-5
        self.algorithm.policy_frequency = 5
        self.algorithm.auto_alpha = False


gym.register(
    id="Isaac-Go2-V47-22cmBreakthrough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V47TwentyTwoCmEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V47TwentyTwoCmRunnerCfg",
    },
)
