import gymnasium as gym
import torch
from isaaclab.utils import configclass
from stair_v14_transfer.config import TransferEnvCfg, TransferRunnerCfg
from eth_sata_reward_v16_terrain import terrain_relative_swing_clearance


def adaptive_clearance(env, asset_cfg, command_name, frequency=1.5, swing_height=0.065, sigma=0.035):
    hits = env.scene.sensors['height_scanner'].data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits)
    high = hits.masked_fill(~valid, -torch.inf).amax(-1)
    low = hits.masked_fill(~valid, torch.inf).amin(-1)
    relief = torch.nan_to_num(high-low, nan=0., posinf=0., neginf=0.)
    target = swing_height + 0.035 * (relief / 0.05).clamp(0., 1.)
    return terrain_relative_swing_clearance(env, asset_cfg, command_name, frequency,
                                          target[:, None], sigma)


@configclass
class StairLearningEnvCfg(TransferEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        gen.sub_terrains['pyramid_stairs'].proportion = 1/6
        gen.sub_terrains['pyramid_stairs_inv'].proportion = 7/12
        for key in ('pyramid_stairs', 'pyramid_stairs_inv'):
            gen.sub_terrains[key].step_height_range = (0.05, 0.095)
        self.scene.terrain.max_init_terrain_level = 2
        self.rewards.swing_clearance.func = adaptive_clearance
        self.rewards.swing_clearance.params['sigma'] = 0.035


@configclass
class StairLearningRunnerCfg(TransferRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'stairs_v23_from_v22'
        self.algorithm.actor_learning_rate = 2.5e-5


gym.register(id='Isaac-Go2-Stairs-V23-v0', entry_point='isaaclab.envs:ManagerBasedRLEnv',
             disable_env_checker=True, kwargs={'env_cfg_entry_point': f'{__name__}:StairLearningEnvCfg',
             'rsl_rl_sac_cfg_entry_point': f'{__name__}:StairLearningRunnerCfg'})
