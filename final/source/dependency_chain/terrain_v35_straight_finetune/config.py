"""Straight-line fine-tuning without changing V34 observations or networks."""
import gymnasium as gym

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from terrain_v34_from_scratch_four_terrain.config import (
    V34MixedEnvCfg,
    V34MixedRunnerCfg,
    yaw_velocity_error_l2,
)


@configclass
class V35StraightEnvCfg(V34MixedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        # Keep the existing 3-D velocity command inside the 62-D observation.
        # The command manager converts world-heading error into a bounded yaw-rate
        # command, so the policy receives correction information without a new
        # observation or any network surgery.
        command.heading_command = True
        command.rel_heading_envs = 1.0
        command.heading_control_stiffness = 0.5
        command.ranges.heading = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.5, 0.5)
        # Slightly strengthen command tracking, but leave V34's gait/posture
        # rewards untouched to preserve the user's preferred motion.
        self.rewards.track_ang_vel_z_exp = RewardTermCfg(
            func=yaw_velocity_error_l2,
            weight=-3.0,
            params={"command_name": "base_velocity"},
        )


@configclass
class V35StraightRunnerCfg(V34MixedRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v35_straight_finetune"
        self.save_interval = 25
        self.algorithm.actor_learning_rate = 1.0e-5


gym.register(
    id="Isaac-Go2-V35-StraightFourTerrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V35StraightEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V35StraightRunnerCfg",
    },
)

