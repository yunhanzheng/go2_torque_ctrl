"""Focused 21.5--23 cm curriculum with fixed 22/23 cm anchors."""
from copy import deepcopy

import gymnasium as gym
import torch
from isaaclab.utils import configclass

from eth_sata_reward_v16_terrain import terrain_relative_swing_clearance
from terrain_v47_22cm_breakthrough.config import V47TwentyTwoCmEnvCfg, V47TwentyTwoCmRunnerCfg


def focused_twenty_three_cm_clearance(
    env, asset_cfg, command_name, frequency=1.5, swing_height=0.09, sigma=0.045
):
    hits = env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits)
    high = hits.masked_fill(~valid, -torch.inf).amax(dim=-1)
    low = hits.masked_fill(~valid, torch.inf).amin(dim=-1)
    relief = torch.nan_to_num(high - low, nan=0.0, posinf=0.0, neginf=0.0)
    target = swing_height + 0.65 * relief.clamp(0.0, 0.23)
    return terrain_relative_swing_clearance(
        env, asset_cfg, command_name, frequency, target[:, None], sigma
    )


@configclass
class V48TwentyThreeCmEnvCfg(V47TwentyTwoCmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        terrains["stairs_up"].step_height_range = (0.215, 0.23)
        terrains["stairs_up"].step_width = 0.30

        anchor23 = deepcopy(terrains["stairs_up"])
        anchor23.step_height_range = (0.23, 0.23)
        terrains["stairs_up_23_anchor"] = anchor23

        proportions = {
            "plane": 0.03,
            "stairs_up": 0.47,
            "stairs_up_23_anchor": 0.15,
            "stairs_up_22_anchor": 0.25,
            "stairs_up_21_anchor": 0.05,
            "stairs_up_20_anchor": 0.00,
            "stairs_up_18_anchor": 0.00,
            "stairs_down": 0.02,
            "slope_up": 0.005,
            "slope_down": 0.005,
            "boxes": 0.01,
            "random_rough": 0.01,
        }
        for name, proportion in proportions.items():
            terrains[name].proportion = proportion

        # Main levels initially reach about 22.2 cm, then curriculum promotes to 23 cm.
        self.scene.terrain.max_init_terrain_level = 4
        self.rewards.swing_clearance.func = focused_twenty_three_cm_clearance
        self.rewards.swing_clearance.weight = 6.0
        self.rewards.swing_clearance.params["swing_height"] = 0.09
        self.rewards.swing_clearance.params["sigma"] = 0.045


@configclass
class V48TwentyThreeCmRunnerCfg(V47TwentyTwoCmRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v48_23cm_final"
        self.save_interval = 5
        self.algorithm.class_name = "terrain_v48_23cm_final.models:V48GuardedTransferSAC"
        self.algorithm.actor_learning_rate = 3.0e-7
        self.algorithm.critic_learning_rate = 5.0e-5
        self.algorithm.policy_frequency = 5
        self.algorithm.auto_alpha = False


gym.register(
    id="Isaac-Go2-V48-23cmFinal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V48TwentyThreeCmEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V48TwentyThreeCmRunnerCfg",
    },
)
