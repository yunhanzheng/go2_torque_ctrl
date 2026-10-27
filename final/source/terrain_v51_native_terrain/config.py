"""V51: SATA dynamics ported into IsaacLab, keeping SAC + V51 rewards + V51 terrain.

Goal: make the environment match /home/ubuntu22/data/SATA go2_torque as closely
as the IsaacLab framework allows, while deliberately KEEPING:
- SAC algorithm (not SATA's PPO)
- IsaacLab / IsaacSim framework (not IsaacGym)
- V51 reward configuration (not SATA's 9 rewards)
- V51 IsaacLab native terrain + curriculum (not SATA's slope trimesh)

Aligned to SATA (see sata_alignment.py):
- TG torque model: action_scale=5 -> tanh activation -> hill model -> motor fatigue
- Gompertz growth curriculum (general_scale) driving torque-limit + command ranges
- 60-D SATA observation with real motor fatigue (no gait clock)
- Per-channel observation noise + 10% action/observation loss
- Commands: lin_vel_x[-0.5,1.5], y[-0.5,0.5], yaw[-1.5,1.5], resampling 5s
- decimation=1 (200 Hz), episode_length_s=10, sim.dt=0.005
- Domain randomization: friction/mass/COM/push (already matched)

Only approximation: SATA ramps control frequency 100->200 Hz; IsaacLab decimation
is fixed at construction, so it is held at 200 Hz (SATA's final frequency).
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence
from typing import TYPE_CHECKING

import gymnasium as gym
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

from sata_alignment import (
    SATA_JOINT_NAMES,
    SataTorqueActionCfg,
    SataVelocityCommandCfg,
    sata_observation_60d_noisy,
)
from terrain_v50_curriculum_recovery.config import (
    V50CurriculumRecoveryEnvCfg,
    V50CurriculumRecoveryRunnerCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def forward_fan_swing_clearance(
    env,
    asset_cfg,
    command_name,
    frequency=1.5,
    swing_height=0.09,
    sigma=0.045,
    fan_radius=0.3,
):
    """Swing clearance with per-foot forward fan target height.
    
    For each foot, look at a 180-degree forward fan (radius=fan_radius).
    Target height = max(terrain_ahead - terrain_below, 0) + swing_height.
    If terrain ahead is all lower, target = swing_height (0.09m).
    """
    robot = env.scene["robot"]
    feet_xy = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]  # [N, 4, 2]
    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]     # [N, 4]
    
    # Robot yaw angle (from base orientation)
    yaw = robot.data.root_quat_w[:, 3]  # w component, but we need actual yaw
    # Use atan2 for proper yaw extraction
    qx, qy, qz, qw = robot.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    
    # Height scanner ray hits
    ray_hits = env.scene.sensors["height_scanner"].data.ray_hits_w  # [N, num_rays, 3]
    ray_xy = ray_hits[..., :2]  # [N, num_rays, 2]
    ray_z = ray_hits[..., 2]    # [N, num_rays]
    valid = torch.isfinite(ray_z)
    
    # For each foot, compute target height from forward fan
    N = feet_xy.shape[0]
    num_feet = feet_xy.shape[1]
    num_rays = ray_xy.shape[1]
    
    # Relative position of rays w.r.t. each foot: [N, 4, num_rays, 2]
    feet_xy_exp = feet_xy.unsqueeze(2)  # [N, 4, 1, 2]
    ray_xy_exp = ray_xy.unsqueeze(1)    # [N, 1, num_rays, 2]
    rel_xy = ray_xy_exp - feet_xy_exp   # [N, 4, num_rays, 2]
    
    # Rotate to foot's forward frame (using robot yaw)
    cos_yaw = torch.cos(-yaw)  # [N]
    sin_yaw = torch.sin(-yaw)  # [N]
    cos_yaw = cos_yaw.view(N, 1, 1).expand(N, num_feet, num_rays)
    sin_yaw = sin_yaw.view(N, 1, 1).expand(N, num_feet, num_rays)
    
    rot_x = cos_yaw * rel_xy[..., 0] - sin_yaw * rel_xy[..., 1]
    rot_y = sin_yaw * rel_xy[..., 0] + cos_yaw * rel_xy[..., 1]
    
    # Distance from foot
    dist = torch.sqrt(rot_x ** 2 + rot_y ** 2)  # [N, 4, num_rays]
    
    # Fan mask: within radius AND in forward half (rot_y > 0) AND valid
    fan_mask = (dist < fan_radius) & (rot_y > 0.0) & valid.unsqueeze(1).expand(N, num_feet, num_rays)
    
    # Find max terrain z in fan for each foot
    ray_z_exp = ray_z.unsqueeze(1).expand(N, num_feet, num_rays)  # [N, 4, num_rays]
    masked_z = ray_z_exp.masked_fill(~fan_mask, -torch.inf)
    max_ahead_z = masked_z.amax(dim=-1)  # [N, 4]
    
    # Find terrain below each foot (nearest ray)
    nearest_ray = torch.cdist(feet_xy, ray_xy).argmin(dim=-1)  # [N, 4]
    ground_z = torch.gather(ray_z, 1, nearest_ray)  # [N, 4]
    
    # Height difference (clamped to >= 0)
    height_diff = torch.relu(max_ahead_z - ground_z)  # [N, 4]
    height_diff = torch.nan_to_num(height_diff, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Per-foot target height, then take max across all feet
    target = swing_height + height_diff  # [N, 4]
    target = target.amax(dim=-1, keepdim=True).expand(-1, 4)  # [N, 4] uniform max target
    
    # Compute swing clearance score with per-foot targets
    phase = 2.0 * torch.pi * frequency * env.episode_length_buf * env.step_dt
    first_diagonal = torch.sin(phase) >= 0.0
    desired_swing = torch.stack(
        (~first_diagonal, first_diagonal, first_diagonal, ~first_diagonal), dim=1
    )
    
    clearance = foot_z - ground_z  # [N, 4]
    # Score: 1.0 when clearance >= target, penalize when below target
    deficit = torch.relu(target - clearance)  # [N, 4], 0 when above target
    score = torch.exp(-torch.square(deficit / sigma))  # [N, 4]
    swing_scores = score.masked_fill(~desired_swing, 1.0)
    
    moving = (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    return swing_scores.amin(dim=1) * moving


def forward_fan_excessive_foot_height(
    env,
    asset_cfg,
    error_scale=0.06,
    fan_radius=0.2,
):
    """Excessive foot height with dynamic target from forward fan.
    
    Same target as swing_clearance: max(forward fan height diff) + 0.09m.
    Penalize when foot clearance exceeds this target.
    """
    robot = env.scene["robot"]
    feet_xy = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    
    # Robot yaw
    qx, qy, qz, qw = robot.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    
    # Height scanner ray hits
    ray_hits = env.scene.sensors["height_scanner"].data.ray_hits_w
    ray_xy = ray_hits[..., :2]
    ray_z = ray_hits[..., 2]
    valid = torch.isfinite(ray_z)
    
    N = feet_xy.shape[0]
    num_feet = feet_xy.shape[1]
    num_rays = ray_xy.shape[1]
    
    # Compute forward fan target (same logic as swing_clearance)
    feet_xy_exp = feet_xy.unsqueeze(2)
    ray_xy_exp = ray_xy.unsqueeze(1)
    rel_xy = ray_xy_exp - feet_xy_exp
    
    cos_yaw = torch.cos(-yaw).view(N, 1, 1).expand(N, num_feet, num_rays)
    sin_yaw = torch.sin(-yaw).view(N, 1, 1).expand(N, num_feet, num_rays)
    
    rot_x = cos_yaw * rel_xy[..., 0] - sin_yaw * rel_xy[..., 1]
    rot_y = sin_yaw * rel_xy[..., 0] + cos_yaw * rel_xy[..., 1]
    dist = torch.sqrt(rot_x ** 2 + rot_y ** 2)
    
    fan_mask = (dist < fan_radius) & (rot_y > 0.0) & valid.unsqueeze(1).expand(N, num_feet, num_rays)
    ray_z_exp = ray_z.unsqueeze(1).expand(N, num_feet, num_rays)
    masked_z = ray_z_exp.masked_fill(~fan_mask, -torch.inf)
    max_ahead_z = masked_z.amax(dim=-1)
    
    nearest_ray = torch.cdist(feet_xy, ray_xy).argmin(dim=-1)
    ground_z = torch.gather(ray_z, 1, nearest_ray)
    
    height_diff = torch.relu(max_ahead_z - ground_z)
    height_diff = torch.nan_to_num(height_diff, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Target: same as swing_clearance (max across all feet + 0.09)
    target = 0.09 + height_diff
    target = target.amax(dim=-1, keepdim=True).expand(-1, num_feet)
    
    # Penalize excess above target
    clearance = foot_z - ground_z
    excess = torch.relu(clearance - target) / error_scale
    return torch.clamp(excess.square().amax(dim=1), max=4.0)


@configclass
class V51NativeTerrainEnvCfg(V50CurriculumRecoveryEnvCfg):
    """V50 with IsaacLab native rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        
        # === Replace V49 terrain with IsaacLab native ROUGH_TERRAINS_CFG ===
        rough = UnitreeGo2RoughEnvCfg()
        self.scene.terrain = deepcopy(rough.scene.terrain)
        self.scene.height_scanner = deepcopy(rough.scene.height_scanner)
        
        # === Use IsaacLab native terrain curriculum ===
        self.curriculum.terrain_levels = deepcopy(rough.curriculum.terrain_levels)
        self.curriculum.terrain_levels.func = mdp.terrain_levels_vel
        self.scene.terrain.max_init_terrain_level = 6
        
        # Fixed seed for reproducible terrain generation
        self.scene.terrain.terrain_generator.seed = 42
        
        # === Match SATA spawn height and joint pose ===
        # SATA: pos = [0.0, 0.0, 0.10] relative to terrain origin
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.10)
        # SATA joint angles (radians)
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            ".*_thigh_joint": 1.45,
            ".*_calf_joint": -2.5,
        }
        
        # === Match SATA TG torque control (tanh activation + hill + fatigue + growth) ===
        self.actions.joint_pos = SataTorqueActionCfg(
            asset_name="robot",
            joint_names=list(SATA_JOINT_NAMES),
            preserve_order=True,
            scale=1.0,          # action_scale is applied inside the TG model
            action_scale=5.0,
            torque_limit=23.5,
            activation_process=True,
            hill_model=True,
            motor_fatigue=True,
            loss_rate=0.1,
        )
        # SATA decimation=1 -> 200 Hz control (fixed; SATA 100->200Hz ramp not portable)
        self.decimation = 1

        # === Make the legs a pure torque source (SATA go2_torque.py:62-97) ===
        # SATA's _process_dof_props returns the URDF properties untouched: no PD gains, no
        # armature, no friction -- the TG model in _compute_torques is the only thing that
        # ever limits the joint torque.  IsaacLab's UNITREE_GO2_CFG instead brings
        # stiffness=25/damping=0.5 *and* DCMotor's four-quadrant torque-speed clip
        # (actuator_pd.py:293-320), which would shave a second hill model on top of SATA's
        # own.  Drop the gains and open the effort range so the TG term owns the limit:
        # SATA peaks at 23.5 * (1 + |q_dot| / 30.1) when braking against joint motion, i.e.
        # above its nominal 23.5 Nm ceiling, which a 23.5 clip would truncate.
        _legs = self.scene.robot.actuators["base_legs"]
        _legs.stiffness = 0.0
        _legs.damping = 0.0
        _legs.effort_limit = 1.0e3
        _legs.saturation_effort = 1.0e3
        self.sim.render_interval = 1
        self.sim.dt = 0.005
        # SATA episode_length_s = 10 (was 20)
        self.episode_length_s = 10.0

        # === Match SATA commands (growth-scaled ranges, resampling 5s) ===
        self.commands.base_velocity = SataVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 5.0),
            heading_command=False,
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            ranges=SataVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 1.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.5, 1.5),
                heading=(-3.14, 3.14),
            ),
        )
        
        # === Match SATA domain randomization ===
        # Friction: SATA [0.5, 1.25]
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.25)
        # Base mass: SATA [-1, 5] kg
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 5.0)
        # COM shift: SATA x±0.2, y±0.1, z±0.1 (base_com was None in Go2 rough config)
        self.events.base_com = EventTermCfg(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.2, 0.2), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
            },
        )
        # Base reset: SATA only XY yaw, no velocity randomization
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        # Disable joint randomization (SATA doesn't have it)
        self.events.reset_robot_joints = None
        # Periodic push: SATA every 4s, max 1.5 m/s linear, 1.0 rad/s angular
        self.events.push_robot = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 4.0),
            params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "yaw": (-1.0, 1.0)}},
        )
        
        # === Match SATA 60-D observation (real fatigue + per-channel noise + loss) ===
        self.observations.policy.base_lin_vel = ObsTerm(
            func=sata_observation_60d_noisy,
            params={"loss_rate": 0.1, "noise_level": 1.5},
        )
        
        # === Disable redundant/conflicting rewards ===
        # feet_slide: redundant with feet_slip (both penalize foot sliding)
        self.rewards.feet_slide = None
        # base_height_l2: conflicts with head_height (both control height)
        self.rewards.base_height_l2 = None
        
        # === Override swing_clearance with forward fan target ===
        # Per-foot 180-degree forward fan (radius=0.2m) for target height
        # Score: 1.0 when clearance >= target, penalize when below target
        ordered_feet = SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        self.rewards.swing_clearance.func = forward_fan_swing_clearance
        self.rewards.swing_clearance.weight = 6.0
        self.rewards.swing_clearance.params = {
            "asset_cfg": ordered_feet,
            "command_name": "base_velocity",
            "frequency": 1.5,
            "swing_height": 0.09,
            "sigma": 0.045,
            "fan_radius": 0.2,
        }
        
        # === Override excessive_foot_height with same dynamic target ===
        # Penalize when foot clearance exceeds swing_clearance target
        self.rewards.excessive_foot_height.func = forward_fan_excessive_foot_height
        self.rewards.excessive_foot_height.weight = -0.03
        self.rewards.excessive_foot_height.params = {
            "asset_cfg": ordered_feet,
            "error_scale": 0.06,
            "fan_radius": 0.2,
        }


@configclass
class V51NativeTerrainRunnerCfg(V50CurriculumRecoveryRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v51_native_terrain"
        self.save_interval = 20


gym.register(
    id="Isaac-Go2-V51-NativeTerrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V51NativeTerrainEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V51NativeTerrainRunnerCfg",
    },
)
