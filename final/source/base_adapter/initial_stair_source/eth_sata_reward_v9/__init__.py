"""ETH-SAC with SATA Stage-3 rewards and an upright safety envelope.

The reward terms are intentionally identical to v8.  This variant fixes the
failure mode where SAC earns velocity/gait reward while crawling on its torso
by terminating states that cannot represent normal quadruped locomotion.
"""

import gymnasium as gym
import torch

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from eth_fixed05_v2 import (
    Go2EthOfficialFixed05Policy200HzSATAObsEnvCfg,
    Go2EthOfficialFixed05Policy200HzTimeAlignedSATAObsSACRunnerCfg,
)
from eth_sata_reward_v8 import _apply_sata_stage3_rewards


def invalid_upright_state(
    env,
    minimum_height: float = 0.22,
    minimum_up_projection: float = 0.60,
    grace_steps: int = 200,
) -> torch.Tensor:
    """Terminate a low torso or a body tilt greater than about 53 degrees.

    For an upright robot, projected gravity's body-frame z component is -1.
    Using root height and gravity makes this independent of collision-body
    naming and prevents missed torso-contact events from being exploited.
    """

    robot = env.scene["robot"]
    too_low = robot.data.root_pos_w[:, 2] < minimum_height
    not_upright = robot.data.projected_gravity_b[:, 2] > -minimum_up_projection
    # Early SAC exploration is noisy enough to knock the robot down before it
    # sees useful recovery transitions.  One physical second (200 steps at
    # 200 Hz) supplies that learning signal; after it, crawling is still never
    # allowed to continue collecting a full episode of velocity reward.
    grace_elapsed = env.episode_length_buf >= grace_steps
    return grace_elapsed & (too_low | not_upright)


@configclass
class Go2EthSACSATARewardUprightEnvCfg(Go2EthOfficialFixed05Policy200HzSATAObsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_sata_stage3_rewards(self)
        self.terminations.invalid_upright_state = DoneTerm(
            func=invalid_upright_state,
            params={
                "minimum_height": 0.22,
                "minimum_up_projection": 0.60,
                "grace_steps": 200,
            },
        )


@configclass
class Go2EthSACSATARewardUprightRunnerCfg(
    Go2EthOfficialFixed05Policy200HzTimeAlignedSATAObsSACRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "eth_sac_isaaclab_sata_reward_upright_v9"
        self.max_iterations = 1000


gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-ETHSAC-SATAReward-Upright-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2EthSACSATARewardUprightEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2EthSACSATARewardUprightRunnerCfg",
    },
)
