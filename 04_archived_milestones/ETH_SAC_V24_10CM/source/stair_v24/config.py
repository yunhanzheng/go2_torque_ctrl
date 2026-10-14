import gymnasium as gym
import torch
from isaaclab.utils import configclass
from stair_v23.config import StairLearningEnvCfg, StairLearningRunnerCfg
from eth_sata_reward_v16_terrain import terrain_relative_swing_clearance


def higher_clearance(env, asset_cfg, command_name, frequency=1.5, swing_height=0.065, sigma=0.035):
    hits = env.scene.sensors['height_scanner'].data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits)
    relief = hits.masked_fill(~valid, -torch.inf).amax(-1) - hits.masked_fill(~valid, torch.inf).amin(-1)
    relief = torch.nan_to_num(relief, nan=0., posinf=0., neginf=0.)
    target = swing_height + 0.055 * (relief / 0.08).clamp(0., 1.)
    return terrain_relative_swing_clearance(env, asset_cfg, command_name, frequency, target[:, None], sigma)


@configclass
class TenCmEnvCfg(StairLearningEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for key in ('pyramid_stairs', 'pyramid_stairs_inv'):
            self.scene.terrain.terrain_generator.sub_terrains[key].step_height_range = (0.075, 0.12)
        self.scene.terrain.max_init_terrain_level = 2
        self.rewards.swing_clearance.func = higher_clearance


@configclass
class TenCmRunnerCfg(StairLearningRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'stairs_v24_from_v23'


gym.register(id='Isaac-Go2-Stairs-V24-v0', entry_point='isaaclab.envs:ManagerBasedRLEnv',
             disable_env_checker=True, kwargs={'env_cfg_entry_point': f'{__name__}:TenCmEnvCfg',
             'rsl_rl_sac_cfg_entry_point': f'{__name__}:TenCmRunnerCfg'})
