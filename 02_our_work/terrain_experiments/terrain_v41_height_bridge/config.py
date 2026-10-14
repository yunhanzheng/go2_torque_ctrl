"""Height-only bridge: keep V37's 40 cm tread and extend height to 12 cm."""
import gymnasium as gym

from isaaclab.utils import configclass

from terrain_v37_stairs_focus.config import V37StairsEnvCfg, V37StairsRunnerCfg


@configclass
class V41HeightBridgeEnvCfg(V37StairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrains = self.scene.terrain.terrain_generator.sub_terrains

        # Change exactly one geometric variable from V37. Keeping the 40 cm
        # tread separates height adaptation from the width failure seen in V40.
        for name in ("stairs_up", "stairs_down"):
            terrains[name].step_height_range = (0.05, 0.12)
            terrains[name].step_width = 0.40

        # Strong retention anchors while the critic adapts to the taller steps.
        proportions = {
            "plane": 0.25,
            "stairs_up": 0.35,
            "stairs_down": 0.15,
            "slope_up": 0.05,
            "slope_down": 0.05,
            "boxes": 0.075,
            "random_rough": 0.075,
        }
        for name, proportion in proportions.items():
            terrains[name].proportion = proportion

        self.scene.terrain.max_init_terrain_level = 1
        assert self.observations.policy.height_scan is None


@configclass
class V41HeightBridgeRunnerCfg(V37StairsRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v41_height_bridge"
        self.save_interval = 5
        self.algorithm.class_name = "terrain_v41_height_bridge.models:GuardedHeightTransferSAC"
        self.algorithm.actor_learning_rate = 2.0e-7
        self.algorithm.critic_learning_rate = 5.0e-5


gym.register(
    id="Isaac-Go2-V41-HeightBridge-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V41HeightBridgeEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V41HeightBridgeRunnerCfg",
    },
)
