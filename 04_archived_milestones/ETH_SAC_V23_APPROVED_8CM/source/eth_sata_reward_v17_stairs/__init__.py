"""Low-stair curriculum transferred from v16; unchanged blind actor."""
import gymnasium as gym
import torch
from isaaclab.utils import configclass
from eth_sata_reward_v16_terrain import Go2EthSACSATARewardTerrainEnvCfg, Go2EthSACSATARewardTerrainRunnerCfg


def base_clearance(env):
    robot = env.scene['robot']
    hits = env.scene.sensors['height_scanner'].data.ray_hits_w
    distance = (hits[:, :, :2] - robot.data.root_pos_w[:, None, :2]).square().sum(-1)
    distance = distance.masked_fill(~torch.isfinite(hits).all(-1), float('inf'))
    nearest = distance.argmin(-1, keepdim=True)
    ground = hits[:, :, 2].gather(1, nearest).squeeze(1)
    # Missing terrain hits must not become a successful upright state.
    return torch.nan_to_num(robot.data.root_pos_w[:, 2] - ground, nan=0., posinf=0., neginf=0.)


def relative_upright(env, minimum_height=0.22, minimum_up_projection=0.60, grace_steps=200):
    low = base_clearance(env) < minimum_height
    tilted = env.scene['robot'].data.projected_gravity_b[:, 2] > -minimum_up_projection
    return (env.episode_length_buf >= grace_steps) & (low | tilted)


def relative_head_height(env, target_height=0.3):
    head_up = -env.scene['robot'].data.projected_gravity_b[:, 0].clip(min=0.)
    return 2. * base_clearance(env).clip(max=target_height) + head_up


@configclass
class StairEnvCfg(Go2EthSACSATARewardTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrain = self.scene.terrain
        gen = terrain.terrain_generator
        gen.size = (8.0, 8.0)
        gen.num_rows = 10
        gen.num_cols = 10
        gen.curriculum = True
        gen.sub_terrains = {key: gen.sub_terrains[key] for key in ('pyramid_stairs', 'pyramid_stairs_inv')}
        for cfg in gen.sub_terrains.values():
            cfg.proportion = 0.5
            cfg.step_height_range = (0.025, 0.08)
            cfg.step_width = 0.4
            cfg.platform_width = 2.0
            cfg.border_width = 1.0
        terrain.max_init_terrain_level = 2
        self.events.reset_base.params['pose_range'] = {'x': (-0.1, 0.1), 'y': (-0.1, 0.1), 'yaw': (-0.1, 0.1)}
        self.rewards.swing_clearance.params['swing_height'] = 0.09
        self.rewards.swing_clearance.params['sigma'] = 0.04
        self.rewards.excessive_foot_height.params['maximum_height'] = 0.18
        self.terminations.invalid_upright_state.func = relative_upright
        self.rewards.head_height.func = relative_head_height


@configclass
class StairRunnerCfg(Go2EthSACSATARewardTerrainRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'eth_sac_isaaclab_sata_reward_stairs_v17'


gym.register(
    id='Isaac-Velocity-Rough-Unitree-Go2-ETHSAC-SATAReward-Stairs-v0',
    entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
    kwargs={'env_cfg_entry_point': f'{__name__}:StairEnvCfg',
            'rsl_rl_sac_cfg_entry_point': f'{__name__}:StairRunnerCfg'},
)
