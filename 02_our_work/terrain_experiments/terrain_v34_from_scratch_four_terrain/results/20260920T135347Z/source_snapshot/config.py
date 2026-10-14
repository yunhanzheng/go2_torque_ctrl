"""Strict from-scratch 62-D SAC: flat foundation followed by four terrains."""
from copy import deepcopy
import math

import gymnasium as gym
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

from stair_v24.config import TenCmEnvCfg, TenCmRunnerCfg


def terrain_catalog():
    """Four target families plus a 10% flat anti-forgetting anchor."""
    source = UnitreeGo2RoughEnvCfg().scene.terrain.terrain_generator.sub_terrains
    stairs_up = deepcopy(source["pyramid_stairs_inv"])
    stairs_down = deepcopy(source["pyramid_stairs"])
    for stairs in (stairs_up, stairs_down):
        stairs.proportion = 0.1125
        stairs.step_height_range = (0.05, 0.12)
        stairs.step_width = 0.4
        stairs.platform_width = 2.0
        stairs.border_width = 1.0
        stairs.holes = False

    slope_up = deepcopy(source["hf_pyramid_slope_inv"])
    slope_down = deepcopy(source["hf_pyramid_slope"])
    for slope in (slope_up, slope_down):
        slope.proportion = 0.1125
        slope.slope_range = (0.0, 0.30)
        slope.platform_width = 2.0

    boxes = deepcopy(source["boxes"])
    boxes.proportion = 0.225
    boxes.grid_width = 0.45
    boxes.grid_height_range = (0.025, 0.10)
    boxes.platform_width = 2.0

    rough = deepcopy(source["random_rough"])
    rough.proportion = 0.225
    rough.noise_range = (0.01, 0.06)
    rough.noise_step = 0.01
    rough.border_width = 0.25

    return {
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        "stairs_up": stairs_up,
        "stairs_down": stairs_down,
        "slope_up": slope_up,
        "slope_down": slope_down,
        "boxes": boxes,
        "random_rough": rough,
    }


@configclass
class V34BaseEnvCfg(TenCmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_standing_envs = 0.0
        command.rel_heading_envs = 0.0
        command.ranges.lin_vel_x = (0.5, 0.5)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False


@configclass
class V34FlatEnvCfg(V34BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        generator = self.scene.terrain.terrain_generator
        generator.sub_terrains = {"plane": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
        generator.num_rows = 1
        generator.num_cols = 1
        generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None


@configclass
class V34MixedEnvCfg(V34BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        generator = self.scene.terrain.terrain_generator
        generator.size = (8.0, 8.0)
        generator.num_rows = 10
        generator.num_cols = 40
        generator.curriculum = True
        generator.sub_terrains = terrain_catalog()
        self.scene.terrain.max_init_terrain_level = 2


@configclass
class V34RunnerCfg(TenCmRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 50
        self.actor.init_noise_std = 0.02
        self.actor.log_std_max = math.log(0.05)
        self.algorithm.class_name = "terrain_v34_from_scratch_four_terrain.models:CenteredStartSAC"
        self.algorithm.actor_learning_rate = 1.0e-5
        self.algorithm.critic_learning_rate = 2.0e-4
        self.algorithm.replay_buffer_size = 5_000_000


@configclass
class V34FlatRunnerCfg(V34RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v34_scratch_flat_foundation"
        self.actor.log_std_max = math.log(0.02)


@configclass
class V34MixedRunnerCfg(V34RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v34_scratch_four_terrain"


gym.register(
    id="Isaac-Go2-V34-ScratchFlat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V34FlatEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V34FlatRunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2-V34-ScratchFourTerrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V34MixedEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V34MixedRunnerCfg",
    },
)

