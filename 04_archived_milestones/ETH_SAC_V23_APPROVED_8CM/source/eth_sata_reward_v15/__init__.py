"""ETH-SAC v15: conservative stabilization fine-tune from the selected v14 actor."""

import gymnasium as gym

from isaaclab.utils import configclass

from eth_sata_reward_v14 import (
    Go2EthSACSATARewardStrongGaitEnvCfg,
    Go2EthSACSATARewardStrongGaitRunnerCfg,
)


@configclass
class Go2EthSACSATARewardStableEnvCfg(Go2EthSACSATARewardStrongGaitEnvCfg):
    """Keep the v14 task and rewards unchanged for a controlled comparison."""


@configclass
class Go2EthSACSATARewardStableRunnerCfg(Go2EthSACSATARewardStrongGaitRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "eth_sac_isaaclab_sata_reward_stable_v15"
        self.max_iterations = 300
        self.save_interval = 25
        self.algorithm.actor_learning_rate = 5.0e-5
        self.algorithm.critic_learning_rate = 2.0e-4
        self.algorithm.policy_frequency = 2
        self.algorithm.target_entropy_scale = 0.12


gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-ETHSAC-SATAReward-Stable-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2EthSACSATARewardStableEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2EthSACSATARewardStableRunnerCfg",
    },
)
