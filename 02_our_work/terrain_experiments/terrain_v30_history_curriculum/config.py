import gymnasium as gym
import torch

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from terrain_v28_scratch_pilot.config import ScratchMultiTerrainEnvCfg, ScratchMultiTerrainRunnerCfg
from terrain_v30_history_curriculum.buffer import SparseHistory


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
class HistoryCurriculumEnvCfg(ScratchMultiTerrainEnvCfg):
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
class HistoryCurriculumRunnerCfg(ScratchMultiTerrainRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v30_history_curriculum_multiterrain"
        self.actor.class_name = "terrain_v30_history_curriculum.models:HistoryActor"
        self.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        # The 1175-D current+history replay is much wider than the official 235-D replay.
        self.algorithm.replay_buffer_size = 250_000


gym.register(
    id="Isaac-Go2-Terrain-V30-HistoryCurriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:HistoryCurriculumEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:HistoryCurriculumRunnerCfg",
    },
)
