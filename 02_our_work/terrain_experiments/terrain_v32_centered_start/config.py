import gymnasium as gym
import math
import torch

import isaaclab.terrains as terrain_gen
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, RewardTermCfg
from isaaclab.utils import configclass

from terrain_v28_scratch_pilot.config import ScratchMultiTerrainEnvCfg, ScratchMultiTerrainRunnerCfg
from terrain_v32_centered_start.buffer import SparseHistory


def official_frame(env):
    """The official 235-D Go2 Rough observation, composed in its original order."""
    robot = env.scene["robot"]
    scanner = env.scene.sensors["height_scanner"]
    height_scan = scanner.data.pos_w[:, 2:3] - scanner.data.ray_hits_w[..., 2] - 0.5
    height_scan = torch.nan_to_num(height_scan, nan=0.0, posinf=1.0, neginf=-1.0).clip(-1.0, 1.0)
    frame = torch.cat(
        (
            robot.data.root_lin_vel_b,
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            env.command_manager.get_command("base_velocity"),
            robot.data.joint_pos - robot.data.default_joint_pos,
            robot.data.joint_vel,
            env.action_manager.action,
            height_scan,
        ),
        dim=-1,
    )
    if frame.shape[-1] != 235:
        raise RuntimeError(f"Expected 235-D official frame, got {frame.shape[-1]}")
    return frame


class OfficialHistoryObservation(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if abs(env.step_dt - 0.02) > 1e-9:
            raise ValueError(f"History offsets assume 50 Hz policy execution, got dt={env.step_dt}")
        self.buffer = SparseHistory(env.num_envs, env.device)

    def reset(self, env_ids=None):
        self.buffer.reset(env_ids)

    def __call__(self, env):
        return self.buffer.update(official_frame(env), env.common_step_counter)


@configclass
class HistoryBaseEnvCfg(ScratchMultiTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Stage 1: learn one ordinary forward gait first.  Keep every terrain
        # family present, but spawn all robots on the easiest curriculum row.
        self.scene.terrain.max_init_terrain_level = 0
        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_standing_envs = 0.0
        command.rel_heading_envs = 0.0
        command.ranges.lin_vel_x = (0.5, 0.5)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.ranges.heading = (0.0, 0.0)
        # Give a from-scratch SAC policy an explicit reason to remain upright,
        # and discourage the violent leg motion seen in V29/V30.
        self.rewards.alive = RewardTermCfg(func=mdp.is_alive, weight=0.5)
        self.rewards.flat_orientation_l2.weight = -0.25
        self.rewards.action_rate_l2.weight = -0.02
        # Replace the individual official terms with one ordered five-frame pack.
        self.observations.policy.base_lin_vel = ObservationTermCfg(func=OfficialHistoryObservation)
        self.observations.policy.base_ang_vel = None
        self.observations.policy.projected_gravity = None
        self.observations.policy.velocity_commands = None
        self.observations.policy.joint_pos = None
        self.observations.policy.joint_vel = None
        self.observations.policy.actions = None
        self.observations.policy.height_scan = None
        self.observations.policy.enable_corruption = False


@configclass
class HistoryFlatEnvCfg(HistoryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        generator = self.scene.terrain.terrain_generator
        generator.sub_terrains = {"plane": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
        generator.num_rows = 1
        generator.num_cols = 1
        generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.rewards.feet_air_time.weight = 0.0


@configclass
class HistoryMixedEnvCfg(HistoryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class HistoryRunnerCfg(ScratchMultiTerrainRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actor.class_name = "terrain_v32_centered_start.models:HistoryActor"
        self.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        self.actor.init_noise_std = 0.02
        self.actor.log_std_max = math.log(0.10)
        self.algorithm.class_name = "terrain_v32_centered_start.models:CenteredStartSAC"
        # The 1175-D current+history replay is much wider than the official 235-D replay.
        self.algorithm.replay_buffer_size = 250_000


@configclass
class HistoryFlatRunnerCfg(HistoryRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v32_history_flat_foundation"
        self.algorithm.gamma = 0.96


@configclass
class HistoryMixedRunnerCfg(HistoryRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v32_history_multiterrain"


gym.register(
    id="Isaac-Go2-Terrain-V32-HistoryFlat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:HistoryFlatEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:HistoryFlatRunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2-Terrain-V32-HistoryMixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:HistoryMixedEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:HistoryMixedRunnerCfg",
    },
)
