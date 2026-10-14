"""Balanced multi-terrain fine-tuning with a fixed 23 cm upstairs anchor."""
import gymnasium as gym
from isaaclab.utils import configclass

from terrain_v48_23cm_final.config import V48TwentyThreeCmEnvCfg, V48TwentyThreeCmRunnerCfg


@configclass
class V49Generalize23CmEnvCfg(V48TwentyThreeCmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrains = self.scene.terrain.terrain_generator.sub_terrains

        # Keep the exact 23 cm geometry learned by V48.  Only rebalance how
        # often each family is sampled; do not weaken the frontier itself.
        proportions = {
            "plane": 0.10,
            "stairs_up": 0.25,
            "stairs_up_23_anchor": 0.15,
            "stairs_up_22_anchor": 0.10,
            "stairs_up_21_anchor": 0.05,
            "stairs_up_20_anchor": 0.00,
            "stairs_up_18_anchor": 0.00,
            "stairs_down": 0.10,
            "slope_up": 0.075,
            "slope_down": 0.075,
            "boxes": 0.05,
            "random_rough": 0.05,
        }
        assert abs(sum(proportions.values()) - 1.0) < 1.0e-9
        for name, proportion in proportions.items():
            terrains[name].proportion = proportion

        # V48's adaptive clearance reward already scales with scanned terrain
        # relief, so it is useful on both the fixed stair and mixed terrains.
        # Start in the middle of each curriculum rather than at maximum relief.
        self.scene.terrain.max_init_terrain_level = 4


@configclass
class V49Generalize23CmRunnerCfg(V48TwentyThreeCmRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v49_generalize_23cm"
        self.save_interval = 10
        self.algorithm.class_name = "terrain_v49_generalize_23cm.models:V49GuardedTransferSAC"
        # Preserve V48's deliberately slow actor adaptation.  The critic can
        # learn the new terrain distribution more quickly before actor updates.
        self.algorithm.actor_learning_rate = 3.0e-7
        self.algorithm.critic_learning_rate = 5.0e-5
        self.algorithm.policy_frequency = 5
        self.algorithm.auto_alpha = False


gym.register(
    id="Isaac-Go2-V49-Generalize23cm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V49Generalize23CmEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V49Generalize23CmRunnerCfg",
    },
)
