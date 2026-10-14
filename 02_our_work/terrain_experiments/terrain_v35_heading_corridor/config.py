"""Heading-locked, corridor-constrained continuation for the four terrain families."""
from copy import deepcopy
import math

import gymnasium as gym
import isaaclab.terrains as terrain_gen
import torch
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from terrain_v34_from_scratch_four_terrain.config import (
    V34MixedEnvCfg,
    V34MixedRunnerCfg,
    terrain_catalog as v34_terrain_catalog,
)


def world_forward_tracking(env, command_name="base_velocity", sigma=0.25):
    """Track +X world velocity so a rotated body cannot exploit body-frame velocity reward."""
    velocity = env.scene["robot"].data.root_lin_vel_w[:, 0]
    command = env.command_manager.get_command(command_name)[:, 0]
    raw = torch.exp(-torch.abs(velocity - command) / sigma)
    standing = torch.exp(-torch.abs(command) / sigma)
    return ((raw - standing) / (1.0 - standing + 1.0e-6)).clamp(0.0, 1.0)


def world_lateral_velocity_l2(env):
    return env.scene["robot"].data.root_lin_vel_w[:, 1].square()


def heading_error_l2(env):
    heading = env.scene["robot"].data.heading_w
    wrapped = torch.atan2(torch.sin(heading), torch.cos(heading))
    return wrapped.square()


def cross_track_error_l2(env):
    local_y = env.scene["robot"].data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return local_y.square()


def outside_route_corridor(env, max_abs_y=0.75, grace_steps=25):
    local_y = env.scene["robot"].data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return (env.episode_length_buf >= grace_steps) & (local_y.abs() > max_abs_y)


def heading_terrain_catalog():
    terrains = deepcopy(v34_terrain_catalog())
    terrains["plane"].proportion = 0.20
    terrains["stairs_up"].proportion = 0.10
    terrains["stairs_down"].proportion = 0.10
    terrains["slope_up"].proportion = 0.10
    terrains["slope_down"].proportion = 0.10
    terrains["boxes"].proportion = 0.20
    terrains["random_rough"].proportion = 0.20
    return terrains


@configclass
class V35HeadingCorridorEnvCfg(V34MixedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.heading_command = True
        command.rel_heading_envs = 1.0
        # Introduce heading feedback gently: the transferred actor was trained
        # with zero yaw-rate commands, so a strong controller destabilizes it.
        command.heading_control_stiffness = 0.3
        command.ranges.heading = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.3, 0.3)

        generator = self.scene.terrain.terrain_generator
        generator.sub_terrains = heading_terrain_catalog()
        self.scene.terrain.max_init_terrain_level = 1

        self.rewards.track_lin_vel_xy_exp = RewardTermCfg(
            func=world_forward_tracking,
            weight=20.0,
            params={"command_name": "base_velocity", "sigma": 0.25},
        )
        self.rewards.moving_y = RewardTermCfg(func=world_lateral_velocity_l2, weight=-4.0)
        self.rewards.heading_error = RewardTermCfg(func=heading_error_l2, weight=-4.0)
        self.rewards.cross_track_error = RewardTermCfg(func=cross_track_error_l2, weight=-4.0)
        self.terminations.outside_route_corridor = TerminationTermCfg(
            func=outside_route_corridor,
            params={"max_abs_y": 0.75, "grace_steps": 25},
        )


@configclass
class V35HeadingFlatEnvCfg(V35HeadingCorridorEnvCfg):
    """Plane-only heading correction before introducing terrain relief."""
    def __post_init__(self):
        super().__post_init__()
        # Preserve the transferred policy's familiar zero-yaw command while
        # its critic learns the new world-frame/corridor objective. Heading
        # feedback is introduced only in the following mixed-terrain stage.
        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.ranges.ang_vel_z = (0.0, 0.0)
        generator = self.scene.terrain.terrain_generator
        generator.sub_terrains = {"plane": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
        generator.num_rows = 1
        generator.num_cols = 1
        generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None


@configclass
class V35StairBoxesEnvCfg(V35HeadingCorridorEnvCfg):
    """Focused continuation for the two families still failing strict evaluation."""
    def __post_init__(self):
        super().__post_init__()
        terrains = heading_terrain_catalog()
        proportions = {
            "plane": 0.10,
            "stairs_up": 0.20,
            "stairs_down": 0.20,
            "slope_up": 0.05,
            "slope_down": 0.05,
            "boxes": 0.25,
            "random_rough": 0.15,
        }
        for name, proportion in proportions.items():
            terrains[name].proportion = proportion
        self.scene.terrain.terrain_generator.sub_terrains = terrains
        # Begin every focused run at the original range's easiest row; the
        # built-in curriculum promotes successful robots toward 12 cm steps.
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class V35StairsUpFocusEnvCfg(V35HeadingCorridorEnvCfg):
    """Advance the proven 5 cm uphill gait toward the middle of the original range."""
    def __post_init__(self):
        super().__post_init__()
        terrains = heading_terrain_catalog()
        proportions = {
            "plane": 0.10,
            "stairs_up": 0.50,
            "stairs_down": 0.10,
            "slope_up": 0.025,
            "slope_down": 0.025,
            "boxes": 0.15,
            "random_rough": 0.10,
        }
        for name, proportion in proportions.items():
            terrains[name].proportion = proportion
        self.scene.terrain.terrain_generator.sub_terrains = terrains
        self.scene.terrain.max_init_terrain_level = 0
        self.rewards.swing_clearance.weight = 4.0


@configclass
class V35HeadingCorridorRunnerCfg(V34MixedRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v35_heading_corridor"
        self.algorithm.actor_learning_rate = 1.0e-5
        self.algorithm.critic_learning_rate = 1.0e-4
        self.actor.init_noise_std = 0.03
        self.actor.log_std_max = math.log(0.03)


@configclass
class V35HeadingFlatRunnerCfg(V35HeadingCorridorRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v35_heading_flat"
        self.actor.init_noise_std = 0.02
        self.actor.log_std_max = math.log(0.02)


@configclass
class V35StairBoxesRunnerCfg(V35HeadingCorridorRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v35_stair_boxes"


@configclass
class V35StairsUpFocusRunnerCfg(V35HeadingCorridorRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v35_stairs_up_focus"


gym.register(
    id="Isaac-Go2-V35-HeadingCorridor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V35HeadingCorridorEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V35HeadingCorridorRunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2-V35-HeadingFlat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V35HeadingFlatEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V35HeadingFlatRunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2-V35-StairBoxes-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V35StairBoxesEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V35StairBoxesRunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2-V35-StairsUpFocus-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V35StairsUpFocusEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V35StairsUpFocusRunnerCfg",
    },
)
