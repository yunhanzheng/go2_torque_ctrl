"""Four-terrain curriculum with a stronger straight-line flat anchor."""
import gymnasium as gym

from isaaclab.utils import configclass

from terrain_v35_straight_finetune.config import V35StraightEnvCfg, V35StraightRunnerCfg


@configclass
class V36TerrainEnvCfg(V35StraightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        # 30% flat anchor. The remaining 70% is split equally among the four
        # paper terrain families; up/down variants share their family weight.
        terrains["plane"].proportion = 0.30
        terrains["stairs_up"].proportion = 0.0875
        terrains["stairs_down"].proportion = 0.0875
        terrains["slope_up"].proportion = 0.0875
        terrains["slope_down"].proportion = 0.0875
        terrains["boxes"].proportion = 0.175
        terrains["random_rough"].proportion = 0.175
        self.scene.terrain.max_init_terrain_level = 1


@configclass
class V36TerrainRunnerCfg(V35StraightRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v36_straight_four_terrain"
        self.save_interval = 25
        self.algorithm.actor_learning_rate = 5.0e-6


gym.register(
    id="Isaac-Go2-V36-StraightFourTerrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V36TerrainEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V36TerrainRunnerCfg",
    },
)

