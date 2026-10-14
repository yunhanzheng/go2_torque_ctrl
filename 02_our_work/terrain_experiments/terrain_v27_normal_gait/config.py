import gymnasium as gym
import torch
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from eth_sata_reward_v16_terrain import terrain_relative_swing_clearance
from terrain_v26_official3.config import OfficialThreeEnvCfg, OfficialThreeRunnerCfg


def terrain_catalog():
    return {
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0 / 6.0,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0 / 6.0,
            grid_width=0.45,
            grid_height_range=(0.025, 0.10),
            platform_width=2.0,
        ),
        "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.125,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.125,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        # From the central spawn platform, inverted pyramid is the ascent direction.
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0 / 12.0,
            step_height_range=(0.075, 0.105),
            step_width=0.4,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0 / 12.0,
            step_height_range=(0.075, 0.105),
            step_width=0.4,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    }


def normal_clearance(env, asset_cfg, command_name, frequency=1.5, swing_height=0.065, sigma=0.030):
    """Restore the V23-style 10 cm maximum instead of V24's 12 cm target."""
    hits = env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits)
    relief = hits.masked_fill(~valid, -torch.inf).amax(-1) - hits.masked_fill(~valid, torch.inf).amin(-1)
    relief = torch.nan_to_num(relief, nan=0.0, posinf=0.0, neginf=0.0)
    target = swing_height + 0.035 * (relief / 0.08).clamp(0.0, 1.0)
    return terrain_relative_swing_clearance(env, asset_cfg, command_name, frequency, target[:, None], sigma)


@configclass
class NormalGaitEnvCfg(OfficialThreeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        generator = self.scene.terrain.terrain_generator
        generator.num_rows = 10
        generator.num_cols = 24
        generator.sub_terrains = terrain_catalog()
        self.scene.terrain.max_init_terrain_level = 2

        self.rewards.swing_clearance.func = normal_clearance
        self.rewards.swing_clearance.params["sigma"] = 0.030
        # Earlier and stronger guardrail against the 12--14 cm high-kick habit.
        self.rewards.excessive_foot_height.params["maximum_height"] = 0.115
        self.rewards.excessive_foot_height.params["error_scale"] = 0.04
        self.rewards.excessive_foot_height.weight = -0.10


@configclass
class NormalGaitRunnerCfg(OfficialThreeRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v27_normal_gait_from_v26"


gym.register(
    id="Isaac-Go2-Terrain-V27-NormalGait-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:NormalGaitEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:NormalGaitRunnerCfg",
    },
)
