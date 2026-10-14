import gymnasium as gym
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from stair_v24.config import TenCmEnvCfg, TenCmRunnerCfg


def terrain_catalog():
    """The three non-stair terrain families from Isaac Lab Rough, scaled for Go2."""
    return {
        # One third of the columns: the Go2-specific ranges from rough_env_cfg.py.
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=0.45,
            grid_height_range=(0.025, 0.10),
            platform_width=2.0,
        ),
        # Together these two directions occupy one third of the columns.
        "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
    }


@configclass
class OfficialThreeEnvCfg(TenCmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        generator = self.scene.terrain.terrain_generator
        generator.size = (8.0, 8.0)
        generator.num_rows = 10
        generator.num_cols = 24
        generator.horizontal_scale = 0.1
        generator.vertical_scale = 0.005
        generator.slope_threshold = 0.75
        generator.sub_terrains = terrain_catalog()
        # Begin on rows 0--2, including nearly-flat low-difficulty slopes.
        self.scene.terrain.max_init_terrain_level = 2


@configclass
class OfficialThreeRunnerCfg(TenCmRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v26_official3_from_v24"


gym.register(
    id="Isaac-Go2-Terrain-V26-Official3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:OfficialThreeEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:OfficialThreeRunnerCfg",
    },
)
