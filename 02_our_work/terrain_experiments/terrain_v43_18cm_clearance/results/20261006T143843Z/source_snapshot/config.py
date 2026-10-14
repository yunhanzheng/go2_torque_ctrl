"""Recover stair ability up to 18 cm with a geometry-scaled swing target."""
import gymnasium as gym
import torch

from isaaclab.utils import configclass

from eth_sata_reward_v16_terrain import terrain_relative_swing_clearance
from terrain_v42_23cm_straight.config import V42TwentyThreeCmEnvCfg, V42TwentyThreeCmRunnerCfg


def eighteen_cm_clearance(
    env, asset_cfg, command_name, frequency=1.5, swing_height=0.075, sigma=0.045
):
    """Raise the target continuously from 7.5 cm to 21.9 cm at 18 cm relief."""
    hits = env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits)
    highest = hits.masked_fill(~valid, -torch.inf).amax(dim=-1)
    lowest = hits.masked_fill(~valid, torch.inf).amin(dim=-1)
    relief = torch.nan_to_num(highest - lowest, nan=0.0, posinf=0.0, neginf=0.0)
    target = swing_height + 0.80 * relief.clamp(0.0, 0.18)
    return terrain_relative_swing_clearance(
        env, asset_cfg, command_name, frequency, target[:, None], sigma
    )


def climb_first_terrain_levels(env, env_ids, asset_cfg=None):
    """Promote physical traversal first; leave paper-level straightness to evaluation."""
    robot = env.scene["robot"]
    terrain = env.scene.terrain
    local = robot.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    heading = torch.atan2(
        torch.sin(robot.data.heading_w[env_ids]), torch.cos(robot.data.heading_w[env_ids])
    )
    upright = robot.data.projected_gravity_b[env_ids, 2] < -0.85

    move_up = local[:, 0] > terrain.cfg.terrain_generator.size[0] / 2
    move_up &= local[:, 1].abs() < 0.50
    move_up &= heading.abs() < 0.261799  # 15 degrees
    move_up &= upright

    move_down = (local[:, 0] < 2.0) | (local[:, 1].abs() > 0.80)
    move_down |= heading.abs() > 0.523599  # 30 degrees
    move_down |= ~upright
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


@configclass
class V43EighteenCmEnvCfg(V42TwentyThreeCmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        for name in ("stairs_up", "stairs_down"):
            terrains[name].step_height_range = (0.05, 0.18)
            terrains[name].step_width = 0.40

        proportions = {
            "plane": 0.20,
            "stairs_up": 0.55,
            "stairs_down": 0.15,
            "slope_up": 0.015,
            "slope_down": 0.015,
            "boxes": 0.035,
            "random_rough": 0.035,
        }
        for name, proportion in proportions.items():
            terrains[name].proportion = proportion

        self.scene.terrain.max_init_terrain_level = 1
        self.curriculum.terrain_levels.func = climb_first_terrain_levels
        self.rewards.swing_clearance.func = eighteen_cm_clearance
        self.rewards.swing_clearance.weight = 5.0
        self.rewards.swing_clearance.params["swing_height"] = 0.075
        self.rewards.swing_clearance.params["sigma"] = 0.045
        self.rewards.excessive_foot_height.params["maximum_height"] = 0.26
        self.rewards.excessive_foot_height.weight = -0.02
        assert self.observations.policy.height_scan is None


@configclass
class V43EighteenCmRunnerCfg(V42TwentyThreeCmRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v43_18cm_clearance"
        self.save_interval = 10
        self.algorithm.class_name = "terrain_v43_18cm_clearance.models:V43GuardedTransferSAC"
        self.algorithm.actor_learning_rate = 2.0e-7
        self.algorithm.critic_learning_rate = 5.0e-5
        self.algorithm.policy_frequency = 5
        self.algorithm.auto_alpha = False


gym.register(
    id="Isaac-Go2-V43-18cmClearance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V43EighteenCmEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V43EighteenCmRunnerCfg",
    },
)
