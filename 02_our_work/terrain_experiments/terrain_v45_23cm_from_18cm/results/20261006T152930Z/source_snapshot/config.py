"""12--23 cm curriculum with a fixed 18 cm anti-forgetting anchor."""
from copy import deepcopy

import gymnasium as gym
import torch
from isaaclab.utils import configclass

from eth_sata_reward_v16_terrain import terrain_relative_swing_clearance
from terrain_v44_18cm_from_v38.config import V44EighteenCmEnvCfg, V44EighteenCmRunnerCfg


def twenty_three_cm_clearance(
    env, asset_cfg, command_name, frequency=1.5, swing_height=0.09, sigma=0.045
):
    """Scale the target to 22.8 cm at 23 cm terrain relief."""
    hits = env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits)
    high = hits.masked_fill(~valid, -torch.inf).amax(dim=-1)
    low = hits.masked_fill(~valid, torch.inf).amin(dim=-1)
    relief = torch.nan_to_num(high - low, nan=0.0, posinf=0.0, neginf=0.0)
    target = swing_height + 0.60 * relief.clamp(0.0, 0.23)
    return terrain_relative_swing_clearance(
        env, asset_cfg, command_name, frequency, target[:, None], sigma
    )


@configclass
class V45TwentyThreeCmEnvCfg(V44EighteenCmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrains = self.scene.terrain.terrain_generator.sub_terrains

        terrains["stairs_up"].step_height_range = (0.12, 0.23)
        terrains["stairs_up"].step_width = 0.30
        terrains["stairs_down"].step_height_range = (0.10, 0.18)
        terrains["stairs_down"].step_width = 0.30

        anchor = deepcopy(terrains["stairs_up"])
        anchor.step_height_range = (0.18, 0.18)
        anchor.step_width = 0.30
        terrains["stairs_up_18_anchor"] = anchor

        proportions = {
            "plane": 0.15,
            "stairs_up": 0.45,
            "stairs_up_18_anchor": 0.20,
            "stairs_down": 0.08,
            "slope_up": 0.01,
            "slope_down": 0.01,
            "boxes": 0.05,
            "random_rough": 0.05,
        }
        for name, proportion in proportions.items():
            terrains[name].proportion = proportion

        # Main curriculum levels 0--5 cover approximately 12--18.1 cm.
        self.scene.terrain.max_init_terrain_level = 5
        self.rewards.swing_clearance.func = twenty_three_cm_clearance
        self.rewards.swing_clearance.weight = 5.0
        self.rewards.swing_clearance.params["swing_height"] = 0.09
        self.rewards.swing_clearance.params["sigma"] = 0.045
        self.rewards.excessive_foot_height.params["maximum_height"] = 0.30
        self.rewards.excessive_foot_height.weight = -0.02
        assert self.observations.policy.height_scan is None


@configclass
class V45TwentyThreeCmRunnerCfg(V44EighteenCmRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v45_23cm_from_18cm"
        self.save_interval = 5
        self.algorithm.class_name = "terrain_v45_23cm_from_18cm.models:V45GuardedTransferSAC"
        self.algorithm.actor_learning_rate = 2.0e-7
        self.algorithm.critic_learning_rate = 5.0e-5
        self.algorithm.policy_frequency = 5
        self.algorithm.auto_alpha = False


gym.register(
    id="Isaac-Go2-V45-23cmFrom18cm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V45TwentyThreeCmEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V45TwentyThreeCmRunnerCfg",
    },
)
