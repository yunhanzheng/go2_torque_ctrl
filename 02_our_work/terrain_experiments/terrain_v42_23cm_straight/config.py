"""Reach 23 cm by changing stair height only and enforcing straight promotion."""
import gymnasium as gym
import torch

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from terrain_v41_height_bridge.config import V41HeightBridgeEnvCfg, V41HeightBridgeRunnerCfg


def heading_error_l2(env):
    heading = torch.atan2(torch.sin(env.scene["robot"].data.heading_w), torch.cos(env.scene["robot"].data.heading_w))
    return heading.square()


def lateral_position_l2(env):
    lateral = env.scene["robot"].data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return lateral.square()


def straight_terrain_levels(env, env_ids, asset_cfg=None):
    """Advance only after straight, upright traversal; demote clear failures."""
    robot = env.scene["robot"]
    terrain = env.scene.terrain
    local = robot.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    heading = torch.atan2(torch.sin(robot.data.heading_w[env_ids]), torch.cos(robot.data.heading_w[env_ids]))
    upright = robot.data.projected_gravity_b[env_ids, 2] < -0.85

    move_up = local[:, 0] > terrain.cfg.terrain_generator.size[0] / 2
    move_up &= local[:, 1].abs() < 0.25
    move_up &= heading.abs() < 0.0872665
    move_up &= upright

    move_down = (local[:, 0] < 2.0) | (local[:, 1].abs() > 0.50)
    move_down |= heading.abs() > 0.174533
    move_down |= ~upright
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


@configclass
class V42TwentyThreeCmEnvCfg(V41HeightBridgeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        for name in ("stairs_up", "stairs_down"):
            terrains[name].step_height_range = (0.05, 0.23)
            terrains[name].step_width = 0.40

        proportions = {
            "plane": 0.30,
            "stairs_up": 0.35,
            "stairs_down": 0.15,
            "slope_up": 0.025,
            "slope_down": 0.025,
            "boxes": 0.075,
            "random_rough": 0.075,
        }
        for name, proportion in proportions.items():
            terrains[name].proportion = proportion

        self.scene.terrain.max_init_terrain_level = 1
        self.curriculum.terrain_levels.func = straight_terrain_levels
        self.rewards.track_ang_vel_z_exp = RewardTermCfg(func=heading_error_l2, weight=-5.0)
        self.rewards.moving_y = RewardTermCfg(func=lateral_position_l2, weight=-3.0)
        assert self.observations.policy.height_scan is None


@configclass
class V42TwentyThreeCmRunnerCfg(V41HeightBridgeRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v42_23cm_straight"
        self.save_interval = 10
        self.algorithm.class_name = "terrain_v42_23cm_straight.models:V42GuardedTransferSAC"
        self.algorithm.actor_learning_rate = 1.0e-8
        self.algorithm.critic_learning_rate = 5.0e-5
        self.algorithm.policy_frequency = 10
        self.algorithm.auto_alpha = False


gym.register(
    id="Isaac-Go2-V42-23cmStraight-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V42TwentyThreeCmEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V42TwentyThreeCmRunnerCfg",
    },
)
