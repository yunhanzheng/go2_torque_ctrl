"""Stair-focused SAC continuation that preserves the 62-D proprioceptive actor."""
import gymnasium as gym

from isaaclab.utils import configclass

from terrain_v36_straight_four_terrain.config import V36TerrainEnvCfg, V36TerrainRunnerCfg


@configclass
class V37StairsEnvCfg(V36TerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrains = self.scene.terrain.terrain_generator.sub_terrains

        # Concentrate experience on the failed family while retaining every
        # previously demonstrated terrain as an anti-forgetting anchor.
        proportions = {
            "plane": 0.15,
            "stairs_up": 0.50,
            "stairs_down": 0.15,
            "slope_up": 0.025,
            "slope_down": 0.025,
            "boxes": 0.075,
            "random_rough": 0.075,
        }
        for name, proportion in proportions.items():
            terrains[name].proportion = proportion

        # First establish a reliable 10 cm policy.  Do not expose 12 cm until
        # 10 cm passes the matched 32-environment evaluation.
        for name in ("stairs_up", "stairs_down"):
            terrains[name].step_height_range = (0.05, 0.10)
            terrains[name].step_width = 0.40
        self.scene.terrain.max_init_terrain_level = 1

        # The inherited observation remains the exact 62-D proprioceptive
        # vector.  Height geometry is not added to the actor or critic input.
        assert self.observations.policy.height_scan is None

        # Retain the reward-only local-clearance calculation already used by
        # the successful V24 stair policy; strengthen it modestly for stairs.
        self.rewards.swing_clearance.weight = 4.0


@configclass
class V37StairsRunnerCfg(V36TerrainRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v37_stairs_focus"
        self.save_interval = 25
        self.algorithm.actor_learning_rate = 2.0e-6


gym.register(
    id="Isaac-Go2-V37-StairsFocus-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V37StairsEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V37StairsRunnerCfg",
    },
)

