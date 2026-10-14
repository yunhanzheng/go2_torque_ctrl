import numpy as np
import torch
import trimesh
import gymnasium as gym
from copy import deepcopy
import isaaclab.terrains as t
from isaaclab.utils import configclass
from isaaclab.terrains.trimesh import mesh_terrains
from isaaclab.terrains.height_field import hf_terrains
from stair_v24.config import TenCmEnvCfg, TenCmRunnerCfg


def gap_with_floor(difficulty, cfg):
    meshes, origin = mesh_terrains.gap_terrain(difficulty, cfg)
    meshes.append(trimesh.creation.box((cfg.size[0], cfg.size[1], .1),
                  trimesh.transformations.translation_matrix((cfg.size[0]/2, cfg.size[1]/2, -.55))))
    return meshes, origin


def platform_from_ground(difficulty, cfg):
    meshes, origin = mesh_terrains.box_terrain(difficulty, cfg)
    origin = np.array([origin[0]-2., origin[1], 0.])
    return meshes, origin


def rough_with_difficulty(difficulty, cfg):
    local = deepcopy(cfg)
    amplitude = .01 + .05*difficulty
    local.noise_range = (-amplitude, amplitude)
    return hf_terrains.random_uniform_terrain(difficulty, local)


def terrain_catalog():
    return {
        'flat':t.MeshPlaneTerrainCfg(proportion=2.),
        'rough':t.HfRandomUniformTerrainCfg(proportion=1., function=rough_with_difficulty, noise_range=(-.06,.06), noise_step=.01, downsampled_scale=.2, border_width=.25),
        'grid':t.MeshRandomGridTerrainCfg(proportion=1.,grid_width=.4,grid_height_range=(.02,.12),platform_width=1.6),
        'slope_up':t.HfInvertedPyramidSlopedTerrainCfg(proportion=1.,slope_range=(.05,.25),platform_width=1.6,border_width=.25),
        'slope_down':t.HfPyramidSlopedTerrainCfg(proportion=1.,slope_range=(.05,.25),platform_width=1.6,border_width=.25),
        'wave':t.HfWaveTerrainCfg(proportion=1.,amplitude_range=(.01,.08),num_waves=4,border_width=.25),
        'stones':t.HfSteppingStonesTerrainCfg(proportion=1.,stone_height_max=.03,stone_width_range=(.35,.55),stone_distance_range=(.12,.28),holes_depth=-.5,platform_width=1.6),
        'gap':t.MeshGapTerrainCfg(proportion=1.,function=gap_with_floor,gap_width_range=(.05,.22),platform_width=1.6),
        'pit':t.MeshPitTerrainCfg(proportion=1.,pit_depth_range=(.04,.14),platform_width=1.6,double_pit=False),
        'platform':t.MeshBoxTerrainCfg(proportion=1.,function=platform_from_ground,box_height_range=(.04,.14),platform_width=1.5,double_box=False),
        'rails':t.MeshRailsTerrainCfg(proportion=1.,rail_thickness_range=(.04,.10),rail_height_range=(.04,.12),platform_width=1.6),
        'discrete':t.HfDiscreteObstaclesTerrainCfg(proportion=1.,obstacle_width_range=(.3,.7),obstacle_height_range=(.02,.12),num_obstacles=40,platform_width=1.6,border_width=.25),
        'ring':t.MeshFloatingRingTerrainCfg(proportion=1.,ring_width_range=(.1,.35),ring_height_range=(.28,.45),ring_thickness=.08,platform_width=1.6),
        'star':t.MeshStarTerrainCfg(proportion=1.,num_bars=5,bar_width_range=(.4,.7),bar_height_range=(.03,.12),platform_width=1.6),
    }


def type_mask(env, names):
    cache = getattr(env,'_v25_type_masks',{})
    tag = tuple(sorted(names))
    if tag in cache:
        return cache[tag]
    gen = env.cfg.scene.terrain.terrain_generator
    keys = list(gen.sub_terrains)
    if len(keys)==1:
        result = torch.full((env.num_envs,), keys[0] in names, device=env.device, dtype=torch.bool)
        cache[tag] = result; env._v25_type_masks = cache
        return result
    weights = np.array([v.proportion for v in gen.sub_terrains.values()]); weights /= weights.sum()
    samples = np.arange(gen.num_cols)/gen.num_cols+.001
    ids = np.searchsorted(np.cumsum(weights),samples)
    lookup = torch.tensor([keys[i] in names for i in ids],device=env.device)
    result = lookup[env.scene.terrain.terrain_types]
    cache[tag] = result; env._v25_type_masks = cache
    return result


def ground_at(env, xy):
    hits = env.scene.sensors['height_scanner'].data.ray_hits_w
    valid = torch.isfinite(hits).all(-1)
    safe = torch.nan_to_num(hits, nan=0.,posinf=0.,neginf=0.)
    distance = torch.cdist(xy,safe[...,:2]).masked_fill(~valid[:,None,:],float('inf'))
    ground = safe[...,2].gather(1,distance.argmin(-1))
    # The only overhang type has a known flat floor: never interpret its ceiling as ground.
    return torch.where(type_mask(env,{'ring'})[:,None],env.scene.env_origins[:,2:3],ground)


def clearance(env, asset_cfg):
    feet = env.scene[asset_cfg.name].data.body_pos_w[:,asset_cfg.body_ids]
    return feet[...,2]-ground_at(env,feet[...,:2])


def swing_reward(env, asset_cfg, command_name, frequency=1.5,swing_height=.065,sigma=.035):
    phase=2*torch.pi*frequency*env.episode_length_buf*env.step_dt
    first=torch.sin(phase)>=0
    swing=torch.stack((~first,first,first,~first),dim=1)
    hits=env.scene.sensors['height_scanner'].data.ray_hits_w[...,2]
    valid=torch.isfinite(hits)
    relief=hits.masked_fill(~valid,-torch.inf).amax(-1)-hits.masked_fill(~valid,torch.inf).amin(-1)
    relief=torch.nan_to_num(relief,nan=0.,posinf=0.,neginf=0.)
    relief=torch.where(type_mask(env,{'ring'}),0.,relief)
    target=swing_height+.055*(relief/.08).clamp(0,1)
    score=torch.exp(-((clearance(env,asset_cfg)-target[:,None])/sigma).square())
    moving=(env.command_manager.get_command(command_name)[:,:2].norm(dim=1)>.1).float()
    return score.masked_fill(~swing,1.).amin(-1)*moving


def high_kick(env,asset_cfg,maximum_height=.14,error_scale=.06):
    return ((clearance(env,asset_cfg)-maximum_height).clamp(min=0)/error_scale).square().amax(-1).clamp(max=4.)


def base_clearance(env):
    pos=env.scene['robot'].data.root_pos_w
    return pos[:,2]-ground_at(env,pos[:,None,:2]).squeeze(1)


def upright(env,minimum_height=.22,minimum_up_projection=.60,grace_steps=200):
    robot=env.scene['robot']
    bad=(base_clearance(env)<minimum_height)|(robot.data.projected_gravity_b[:,2]>-minimum_up_projection)
    fallen=type_mask(env,{'gap','stones'}) & ((robot.data.root_pos_w[:,2]-env.scene.env_origins[:,2])<-.05)
    return ((env.episode_length_buf>=grace_steps)&bad)|fallen


def head_height(env,target_height=.3):
    head_up=-env.scene['robot'].data.projected_gravity_b[:,0].clip(min=0.)
    return 2*base_clearance(env).clip(max=target_height)+head_up


@configclass
class MixedEnvCfg(TenCmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        gen=self.scene.terrain.terrain_generator
        gen.size=(10.,10.);gen.num_rows=10;gen.num_cols=30
        gen.horizontal_scale=.1;gen.vertical_scale=.005
        gen.sub_terrains=terrain_catalog()
        self.scene.terrain.max_init_terrain_level=2
        self.rewards.swing_clearance.func=swing_reward
        self.rewards.excessive_foot_height.func=high_kick
        self.rewards.head_height.func=head_height
        self.terminations.invalid_upright_state.func=upright


@configclass
class MixedRunnerCfg(TenCmRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name='terrain_v25_from_v24'


gym.register(id='Isaac-Go2-Terrain-V25-v0',entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
 kwargs={'env_cfg_entry_point':f'{__name__}:MixedEnvCfg','rsl_rl_sac_cfg_entry_point':f'{__name__}:MixedRunnerCfg'})
