"""Strict from-scratch 62-D SAC: flat foundation followed by four terrains."""
from copy import deepcopy
import math

import gymnasium as gym
import isaaclab.terrains as terrain_gen
import torch
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

from stair_v24.config import TenCmEnvCfg, TenCmRunnerCfg


def zero_baseline_forward_tracking(env, command_name="base_velocity", sigma=0.25):
    """Give exactly zero reward for standing still and one at the command speed."""
    velocity = env.scene["robot"].data.root_lin_vel_b[:, 0]
    command = env.command_manager.get_command(command_name)[:, 0]
    raw = torch.exp(-torch.abs(velocity - command) / sigma)
    standing = torch.exp(-torch.abs(command) / sigma)
    return ((raw - standing) / (1.0 - standing + 1.0e-6)).clamp(0.0, 1.0)


def lateral_velocity_l2(env):
    return env.scene["robot"].data.root_lin_vel_b[:, 1].square()


def yaw_velocity_error_l2(env, command_name="base_velocity"):
    yaw = env.scene["robot"].data.root_ang_vel_b[:, 2]
    command = env.command_manager.get_command(command_name)[:, 2]
    return (yaw - command).square()


def stationary_velocity_tracking(env, sigma=0.20):
    """Dense balance-stage reward with maximum only at zero planar velocity."""
    velocity = env.scene["robot"].data.root_lin_vel_b[:, :2]
    return torch.exp(-velocity.square().sum(dim=1) / (sigma * sigma))


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
        # From-scratch Gaussian exploration at 200 Hz changes the command every
        # 5 ms and knocked every robot down at the one-second safety boundary.
        # Hold each sampled action for 20 ms, matching the official 50 Hz task.
        self.decimation = 4
        self.sim.render_interval = self.decimation
        # The inherited SATA reward gives large positive yaw/lateral/upright
        # rewards while standing still.  That produced a verified 0/32 policy
        # in the first V34 attempt.  Make forward progress the dominant signal
        # and express yaw/lateral objectives as costs instead of survival pay.
        self.rewards.track_lin_vel_xy_exp = RewardTermCfg(
            func=zero_baseline_forward_tracking,
            weight=20.0,
            params={"command_name": "base_velocity", "sigma": 0.25},
        )
        self.rewards.track_ang_vel_z_exp = RewardTermCfg(
            func=yaw_velocity_error_l2,
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.moving_y = RewardTermCfg(func=lateral_velocity_l2, weight=-2.0)
        # Video review of the 0.3 m/s policy showed a stable but conservative
        # gait. Raise the nominal flat-ground swing target by only 1 cm.
        self.rewards.swing_clearance.params["swing_height"] = 0.075


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
class V34StandEnvCfg(V34FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.rewards.track_lin_vel_xy_exp = RewardTermCfg(
            func=stationary_velocity_tracking, weight=10.0, params={"sigma": 0.20}
        )


@configclass
class V34SlowEnvCfg(V34FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Gait rewards are enabled only when ||command|| > 0.1.  Keep the
        # first walking stage safely above that strict threshold.
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.15)


@configclass
class V34MediumEnvCfg(V34FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.3)


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
        self.actor.init_noise_std = 0.05
        self.actor.log_std_max = math.log(0.05)
        self.algorithm.class_name = "terrain_v34_from_scratch_four_terrain.models:CenteredStartSAC"
        self.algorithm.actor_learning_rate = 2.5e-5
        self.algorithm.critic_learning_rate = 2.0e-4
        self.algorithm.replay_buffer_size = 5_000_000
        self.algorithm.gamma = 0.96
        self.algorithm.n_steps = 5


@configclass
class V34FlatRunnerCfg(V34RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v34_scratch_flat_foundation"
        self.actor.init_noise_std = 0.05
        self.actor.log_std_max = math.log(0.05)
        self.algorithm.actor_learning_rate = 5.0e-5


@configclass
class V34StandRunnerCfg(V34RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v34_scratch_stage0_stand"
        self.actor.init_noise_std = 0.01
        self.actor.log_std_max = math.log(0.01)
        self.algorithm.actor_learning_rate = 5.0e-5


@configclass
class V34SlowRunnerCfg(V34RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v34_scratch_stage1_slow"
        self.actor.init_noise_std = 0.02
        self.actor.log_std_max = math.log(0.02)
        self.algorithm.actor_learning_rate = 5.0e-5


@configclass
class V34MediumRunnerCfg(V34RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v34_scratch_stage2_medium"
        self.actor.init_noise_std = 0.03
        self.actor.log_std_max = math.log(0.03)
        self.algorithm.actor_learning_rate = 5.0e-5


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
    id="Isaac-Go2-V34-ScratchStand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V34StandEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V34StandRunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2-V34-ScratchSlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V34SlowEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V34SlowRunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2-V34-ScratchMedium-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V34MediumEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V34MediumRunnerCfg",
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
