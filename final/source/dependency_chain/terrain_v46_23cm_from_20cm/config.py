"""18--23 cm curriculum with fixed 20 cm and 18 cm retention anchors."""
from copy import deepcopy

import gymnasium as gym
from isaaclab.utils import configclass

from terrain_v45_23cm_from_18cm.config import V45TwentyThreeCmEnvCfg, V45TwentyThreeCmRunnerCfg


@configclass
class V46TwentyToTwentyThreeCmEnvCfg(V45TwentyThreeCmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrains = self.scene.terrain.terrain_generator.sub_terrains

        # A narrow curriculum: levels 0--9 span 18--23 cm.
        terrains["stairs_up"].step_height_range = (0.18, 0.23)
        terrains["stairs_up"].step_width = 0.30

        anchor20 = deepcopy(terrains["stairs_up"])
        anchor20.step_height_range = (0.20, 0.20)
        terrains["stairs_up_20_anchor"] = anchor20

        # Keep exact-height anchors while most environments advance toward 23 cm.
        proportions = {
            "plane": 0.07,
            "stairs_up": 0.50,
            "stairs_up_20_anchor": 0.25,
            "stairs_up_18_anchor": 0.10,
            "stairs_down": 0.03,
            "slope_up": 0.01,
            "slope_down": 0.01,
            "boxes": 0.015,
            "random_rough": 0.015,
        }
        for name, proportion in proportions.items():
            terrains[name].proportion = proportion

        # Initialize the main curriculum no higher than about 20.2 cm, then promote
        # successful traversals through the remaining levels to 23 cm.
        self.scene.terrain.max_init_terrain_level = 4


@configclass
class V46TwentyToTwentyThreeCmRunnerCfg(V45TwentyThreeCmRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v46_23cm_from_20cm"
        self.save_interval = 5
        self.algorithm.class_name = "terrain_v46_23cm_from_20cm.models:V46GuardedTransferSAC"
        self.algorithm.actor_learning_rate = 3.0e-7
        self.algorithm.critic_learning_rate = 5.0e-5
        self.algorithm.policy_frequency = 5
        self.algorithm.auto_alpha = False


gym.register(
    id="Isaac-Go2-V46-23cmFrom20cm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V46TwentyToTwentyThreeCmEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V46TwentyToTwentyThreeCmRunnerCfg",
    },
)
