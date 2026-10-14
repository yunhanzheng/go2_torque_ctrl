import gymnasium as gym
from isaaclab.utils import configclass
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from eth_sata_reward_v17_stairs import StairEnvCfg, StairRunnerCfg


@configclass
class TransferEnvCfg(StairEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        gen.num_cols = 12
        gen.sub_terrains['pyramid_stairs'].proportion = 0.25
        gen.sub_terrains['pyramid_stairs_inv'].proportion = 0.50
        gen.sub_terrains['plane'] = MeshPlaneTerrainCfg(proportion=0.25)
        # Retain V14 clearance targets, measured relative to local terrain.
        self.rewards.swing_clearance.params.update(swing_height=0.065, sigma=0.030)
        self.rewards.excessive_foot_height.params['maximum_height'] = 0.14


@configclass
class TransferRunnerCfg(StairRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'stairs_v22_from_v14'
        self.save_interval = 50


gym.register(id='Isaac-Go2-Stairs-V22-v0', entry_point='isaaclab.envs:ManagerBasedRLEnv',
             disable_env_checker=True, kwargs={'env_cfg_entry_point': f'{__name__}:TransferEnvCfg',
             'rsl_rl_sac_cfg_entry_point': f'{__name__}:TransferRunnerCfg'})
