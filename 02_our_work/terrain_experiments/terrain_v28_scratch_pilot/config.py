from copy import deepcopy

import gymnasium as gym
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_sac_cfg import UnitreeGo2RoughSACRunnerCfg


def balanced_catalog(source):
    """Twenty percent each: plane, stairs, slopes, boxes and rough surface."""
    stairs_up = deepcopy(source["pyramid_stairs_inv"] if "pyramid_stairs_inv" in source else source["stairs_up"])
    stairs_down = deepcopy(source["pyramid_stairs"] if "pyramid_stairs" in source else source["stairs_down"])
    for stairs in (stairs_up, stairs_down):
        stairs.proportion = 0.10
        stairs.step_height_range = (0.05, 0.12)
        stairs.step_width = 0.4
        stairs.platform_width = 2.0
        stairs.border_width = 1.0
        stairs.holes = False

    slope_up = deepcopy(source["hf_pyramid_slope_inv"] if "hf_pyramid_slope_inv" in source else source["slope_up"])
    slope_down = deepcopy(source["hf_pyramid_slope"] if "hf_pyramid_slope" in source else source["slope_down"])
    for slope in (slope_up, slope_down):
        slope.proportion = 0.10
        slope.slope_range = (0.0, 0.30)
        slope.platform_width = 2.0

    boxes = deepcopy(source["boxes"])
    boxes.proportion = 0.20
    boxes.grid_height_range = (0.025, 0.10)

    rough = deepcopy(source["random_rough"])
    rough.proportion = 0.20
    rough.noise_range = (0.01, 0.06)
    rough.noise_step = 0.01

    return {
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        "stairs_up": stairs_up,
        "stairs_down": stairs_down,
        "slope_up": slope_up,
        "slope_down": slope_down,
        "boxes": boxes,
        "random_rough": rough,
    }


@configclass
class ScratchMultiTerrainEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        generator = self.scene.terrain.terrain_generator
        generator.size = (8.0, 8.0)
        generator.num_rows = 10
        generator.num_cols = 20
        generator.sub_terrains = balanced_catalog(generator.sub_terrains)
        # A from-scratch policy begins only on rows 0--2, then advances by curriculum.
        self.scene.terrain.max_init_terrain_level = 2


@configclass
class ScratchMultiTerrainRunnerCfg(UnitreeGo2RoughSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v28_scratch_multiterrain"
        self.save_interval = 50
        # Keep the height-scan replay practical on the 32 GB host.
        self.algorithm.replay_buffer_size = 1_000_000


gym.register(
    id="Isaac-Go2-Terrain-V28-ScratchMulti-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:ScratchMultiTerrainEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:ScratchMultiTerrainRunnerCfg",
    },
)
