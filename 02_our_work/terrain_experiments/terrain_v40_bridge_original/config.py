"""Conservative bridge from V37 toward IsaacLab's original stair geometry."""
import gymnasium as gym

from isaaclab.utils import configclass

from terrain_v37_stairs_focus.config import V37StairsEnvCfg, V37StairsRunnerCfg


@configclass
class V40BridgeOriginalEnvCfg(V37StairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrains = self.scene.terrain.terrain_generator.sub_terrains

        # V37: 5--10 cm, 40 cm wide.  Original IsaacLab: 5--23 cm,
        # 30 cm wide.  Change both dimensions only halfway in this bridge.
        for name in ("stairs_up", "stairs_down"):
            terrains[name].step_height_range = (0.05, 0.12)
            terrains[name].step_width = 0.35

        # Keep substantial easy/retention experience.  V38 used the same
        # stair-heavy mix for 1000 iterations and drifted; this stage is short
        # and checkpoint-gated instead.
        proportions = {
            "plane": 0.20,
            "stairs_up": 0.42,
            "stairs_down": 0.15,
            "slope_up": 0.035,
            "slope_down": 0.035,
            "boxes": 0.06,
            "random_rough": 0.10,
        }
        for name, proportion in proportions.items():
            terrains[name].proportion = proportion

        self.scene.terrain.max_init_terrain_level = 1
        assert self.observations.policy.height_scan is None


@configclass
class V40BridgeOriginalRunnerCfg(V37StairsRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v40_bridge_original"
        self.save_interval = 25
        self.algorithm.actor_learning_rate = 1.0e-6


gym.register(
    id="Isaac-Go2-V40-BridgeOriginal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V40BridgeOriginalEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V40BridgeOriginalRunnerCfg",
    },
)
