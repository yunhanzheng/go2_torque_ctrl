"""Continue V37 on IsaacLab's original 5--23 cm, 30 cm-wide stairs."""
import gymnasium as gym

from isaaclab.utils import configclass

from terrain_v37_stairs_focus.config import V37StairsEnvCfg, V37StairsRunnerCfg


@configclass
class V38OfficialStairsEnvCfg(V37StairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrains = self.scene.terrain.terrain_generator.sub_terrains

        # Match IsaacLab ROUGH_TERRAINS_CFG's unscaled stair geometry.
        for name in ("stairs_up", "stairs_down"):
            terrains[name].step_height_range = (0.05, 0.23)
            terrains[name].step_width = 0.30

        # Begin at the lower curriculum rows. Promotion is governed by the
        # inherited terrain-level curriculum rather than exposing 23 cm at reset.
        self.scene.terrain.max_init_terrain_level = 1

        # Preserve V37's 62-D proprioceptive observation. The scanner remains
        # reward-only and is never concatenated into the SAC policy observation.
        assert self.observations.policy.height_scan is None


@configclass
class V38OfficialStairsRunnerCfg(V37StairsRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v38_official_stairs"
        self.save_interval = 50
        self.algorithm.actor_learning_rate = 2.0e-6


gym.register(
    id="Isaac-Go2-V38-OfficialStairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V38OfficialStairsEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V38OfficialStairsRunnerCfg",
    },
)

