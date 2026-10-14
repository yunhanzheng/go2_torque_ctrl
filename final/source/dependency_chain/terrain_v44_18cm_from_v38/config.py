"""A narrow 10--18 cm continuation from V38 model 3798."""
import gymnasium as gym
import torch

from isaaclab.utils import configclass

from eth_sata_reward_v16_terrain import terrain_relative_swing_clearance
from terrain_v38_official_stairs.config import V38OfficialStairsEnvCfg, V38OfficialStairsRunnerCfg


def moderate_height_scaled_clearance(
    env, asset_cfg, command_name, frequency=1.5, swing_height=0.09, sigma=0.04
):
    """Preserve V38's gait while raising the target smoothly to 18 cm."""
    hits = env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits)
    high = hits.masked_fill(~valid, -torch.inf).amax(dim=-1)
    low = hits.masked_fill(~valid, torch.inf).amin(dim=-1)
    relief = torch.nan_to_num(high - low, nan=0.0, posinf=0.0, neginf=0.0)
    target = swing_height + 0.50 * relief.clamp(0.0, 0.18)
    return terrain_relative_swing_clearance(
        env, asset_cfg, command_name, frequency, target[:, None], sigma
    )


def traversal_curriculum(env, env_ids, asset_cfg=None):
    """Advance on real traversal without confusing drift with inability to climb."""
    robot = env.scene["robot"]
    terrain = env.scene.terrain
    local = robot.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    upright = robot.data.projected_gravity_b[env_ids, 2] < -0.85
    move_up = (local[:, 0] > terrain.cfg.terrain_generator.size[0] / 2) & upright
    move_up &= local[:, 1].abs() < 0.80
    move_down = (local[:, 0] < 2.0) | (local[:, 1].abs() > 1.20) | ~upright
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


@configclass
class V44EighteenCmEnvCfg(V38OfficialStairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        for name in ("stairs_up", "stairs_down"):
            terrains[name].step_height_range = (0.10, 0.18)
            terrains[name].step_width = 0.30

        proportions = {
            "plane": 0.15,
            "stairs_up": 0.65,
            "stairs_down": 0.10,
            "slope_up": 0.015,
            "slope_down": 0.015,
            "boxes": 0.035,
            "random_rough": 0.035,
        }
        for name, proportion in proportions.items():
            terrains[name].proportion = proportion

        # Levels 0--5 correspond to roughly 10--14.4 cm at initialization.
        self.scene.terrain.max_init_terrain_level = 5
        self.curriculum.terrain_levels.func = traversal_curriculum
        self.rewards.swing_clearance.func = moderate_height_scaled_clearance
        self.rewards.swing_clearance.weight = 5.0
        self.rewards.swing_clearance.params["swing_height"] = 0.09
        self.rewards.swing_clearance.params["sigma"] = 0.04
        self.rewards.excessive_foot_height.params["maximum_height"] = 0.24
        self.rewards.excessive_foot_height.weight = -0.02
        assert self.observations.policy.height_scan is None


@configclass
class V44EighteenCmRunnerCfg(V38OfficialStairsRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v44_18cm_from_v38"
        self.save_interval = 10
        self.algorithm.class_name = "terrain_v44_18cm_from_v38.models:V44GuardedTransferSAC"
        self.algorithm.actor_learning_rate = 5.0e-7
        self.algorithm.critic_learning_rate = 5.0e-5
        self.algorithm.policy_frequency = 5
        self.algorithm.auto_alpha = False


gym.register(
    id="Isaac-Go2-V44-18cmFromV38-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V44EighteenCmEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V44EighteenCmRunnerCfg",
    },
)
