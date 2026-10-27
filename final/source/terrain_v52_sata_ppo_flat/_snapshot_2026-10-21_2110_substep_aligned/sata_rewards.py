"""SATA ``go2_torque`` reward functions, ported verbatim to IsaacLab managers.

Source of truth (local checkout):
  /home/ubuntu22/data/SATA/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque.py
      _reward_forward / _reward_head_height / _reward_moving_y / _reward_moving_yaw /
      _reward_motor_fatigue / _reward_roll / _reward_lin_vel_z / _reward_soft_dof_pos_limits
  /home/ubuntu22/data/SATA/legged_gym/legged_gym/envs/base/legged_robot.py
      _reward_dof_acc and the soft-limit computation in _process_dof_props

These 9 terms are what the V52 "SATA-identical / PPO / flat-ground" verification run
trains on.  Four of them (forward, head_height, moving_y, moving_yaw) read the Gompertz
``general_scale`` live from the SataTorqueAction term, exactly as SATA's growth
curriculum couples reward shaping to training progress.

SATA reward scales (GO2TorqueCfg.rewards.scales):
    forward=+10, head_height=+5, moving_y=+5, moving_yaw=+5,
    soft_dof_pos_limits=-5, motor_fatigue=-0.05, dof_acc=-1e-6, roll=-5, lin_vel_z=-5
Constants: tracking_sigma=0.25, base_height_target=0.3, soft_dof_pos_limit=0.9
"""
from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

# Action term registered by the V51/V52 config: ``self.actions.joint_pos = SataTorqueActionCfg(...)``.
_ACTION_TERM = "joint_pos"


def _sata_action(env):
    """The SataTorqueAction instance carrying general_scale / motor_fatigue / torques_action."""
    return env.action_manager.get_term(_ACTION_TERM)


def _growth(env) -> float:
    """SATA ``general_scale`` (Gompertz growth curve), a scalar in [0, 1]."""
    return float(_sata_action(env).general_scale)


def _ground_height_below_base(env) -> torch.Tensor:
    """Mean terrain height under the base, from the height scanner.

    Mirrors SATA's ``measured_heights`` used by _reward_head_height.  Invalid rays
    (NaN/inf) are excluded from the mean so a missing hit never fakes a tall base.
    """
    hits = env.scene.sensors["height_scanner"].data.ray_hits_w  # [N, R, 3]
    z = hits[..., 2]
    finite = torch.isfinite(z)
    z_safe = torch.where(finite, z, torch.zeros_like(z))
    count = finite.sum(dim=1).clamp(min=1)
    return z_safe.sum(dim=1) / count  # [N]


# --- 1. forward (+10): exp(-|vx - target| / sigma), growth-coupled target ---
def sata_forward(
    env,
    command_name: str = "base_velocity",
    sigma: float = 0.25,
    lin_vel_x_lo: float = -0.5,
    lin_vel_x_hi: float = 1.5,
) -> torch.Tensor:
    """Verbatim SATA _reward_forward.

    SATA source (go2_torque.py:367-376):

        abs(vx - (hi + lo)/2 * max(1 - 2*gs, 0) - cmd_x * min(2*gs, 1))

    The parentheses after ``vx -`` close right after ``lin_vel_x[0]``, so the command
    term sits **outside** them and lands with a *plus* sign in the target velocity:
    ``target = center * max(1 - 2*gs, 0) + cmd_x * min(2*gs, 1)``, center = 0.5.
    (Reading it as ``vx - (A - cmd)`` flips the sign of the whole command and trains a
    reversed gait - the policy then walks backwards for a positive command.)

    Verified at runtime against SATA's own play: cmd_x=+1.2, vx=+1.222, gs=0.789 ->
    ``_reward_forward`` = exp(-|1.222 - 1.2| / 0.25) = 0.9156.
    """
    gs = _growth(env)
    vx = env.scene["robot"].data.root_lin_vel_b[:, 0]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    center = 0.5 * (lin_vel_x_hi + lin_vel_x_lo)
    target = center * max(1.0 - gs * 2.0, 0.0) + cmd_x * min(gs * 2.0, 1.0)
    return torch.exp(-torch.abs(vx - target) / sigma)


# --- 2. head_height (+5): base_height * (1 + gs) + head_up ---
def sata_head_height(env, base_height_target: float = 0.3) -> torch.Tensor:
    """Verbatim SATA _reward_head_height."""
    gs = _growth(env)
    robot = env.scene["robot"]
    base_height = (robot.data.root_pos_w[:, 2] - _ground_height_below_base(env)).clip(max=base_height_target)
    clip_min = min(0.0, -0.2 * (1.5 - gs * 2.0))
    head_up = -(robot.data.projected_gravity_b[:, 0].clip(min=clip_min))
    return base_height * (1.0 + gs) + head_up


# --- 3. moving_y (+5): exp(-|vy - cmd_y| / sigma) * gs ---
def sata_moving_y(env, command_name: str = "base_velocity", sigma: float = 0.25) -> torch.Tensor:
    """Verbatim SATA _reward_moving_y (lateral-velocity tracking, gated by growth)."""
    gs = _growth(env)
    vy = env.scene["robot"].data.root_lin_vel_b[:, 1]
    cmd_y = env.command_manager.get_command(command_name)[:, 1]
    return torch.exp(-torch.abs(vy - cmd_y) / sigma) * gs


# --- 4. moving_yaw (+5): exp(-|wz - cmd_yaw| / sigma) * gs ---
def sata_moving_yaw(env, command_name: str = "base_velocity", sigma: float = 0.25) -> torch.Tensor:
    """Verbatim SATA _reward_moving_yaw (yaw-rate tracking, gated by growth)."""
    gs = _growth(env)
    wz = env.scene["robot"].data.root_ang_vel_b[:, 2]
    cmd_yaw = env.command_manager.get_command(command_name)[:, 2]
    return torch.exp(-torch.abs(wz - cmd_yaw) / sigma) * gs


# --- 5. motor_fatigue (-0.05): sum(fatigue * |torques_action|) ---
def sata_motor_fatigue(env) -> torch.Tensor:
    """Verbatim SATA _reward_motor_fatigue."""
    at = _sata_action(env)
    return (at.motor_fatigue * at.torques_action.abs()).sum(dim=1)


# --- 6. roll (-5): |projected_gravity_y| ---
def sata_roll(env) -> torch.Tensor:
    """Verbatim SATA _reward_roll."""
    return env.scene["robot"].data.projected_gravity_b[:, 1].abs()


# --- 7. lin_vel_z (-5): base_lin_vel_z^2 ---
def sata_lin_vel_z(env) -> torch.Tensor:
    """Verbatim SATA _reward_lin_vel_z."""
    return env.scene["robot"].data.root_lin_vel_b[:, 2].square()


# --- 8. soft_dof_pos_limits (-5): sum of excursions beyond the 90% soft bounds ---
def sata_soft_dof_pos_limits(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), soft_limit_frac: float = 0.9
) -> torch.Tensor:
    """Verbatim SATA _reward_soft_dof_pos_limits.

    Soft bounds = center +/- 0.5 * range * frac, with frac = 0.9 (GO2RoughCfg.rewards).
    Returns the summed positive excursion outside those bounds (a cost; weight is -5).
    """
    robot = env.scene[asset_cfg.name]
    pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.joint_pos_limits[:, asset_cfg.joint_ids]  # [N, J, 2]
    lower, upper = limits[..., 0], limits[..., 1]
    mid = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * soft_limit_frac
    soft_lo, soft_hi = mid - half, mid + half
    out = -(pos - soft_lo).clip(max=0.0) + (pos - soft_hi).clip(min=0.0)
    return out.sum(dim=1)


# --- 9. dof_acc (-1e-6): sum(((last_vel - vel)/dt)^2) == sum(joint_acc^2) ---
def sata_dof_acc(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Verbatim SATA _reward_dof_acc (the finite-difference sign is squared away)."""
    robot = env.scene[asset_cfg.name]
    return robot.data.joint_acc[:, asset_cfg.joint_ids].square().sum(dim=1)


## ---------------------------------------------------------------------------------
# SATA go2_torque._push_robots (go2_torque.py:275-281), scaled by general_scale.
## ---------------------------------------------------------------------------------
def sata_push_robots(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                     max_vel_xy: float = 1.5, max_vel_ang: float = 1.0) -> None:
    """Random base-velocity impulse; magnitude scales with general_scale.

    Verbatim GO2Torque._push_robots (go2_torque.py:272-282): **all three** linear
    components get ``U(-1, 1) * max_push_vel_xy * gs`` (``root_states[:, 7:10]``) and
    all three angular components get ``U(-1, 1) * max_push_vel_ang * gs``
    (``root_states[:, 10:13]``), i.e. the whole 6-D root velocity is overwritten.
    """
    gs = _growth(env)
    robot = env.scene[asset_cfg.name]
    # EventManager calls interval terms with env_ids=None (meaning all environments);
    # indexing with [None] would add a leading dim, so branch explicitly.
    if env_ids is None:
        vel = robot.data.root_vel_w.clone()
    else:
        vel = robot.data.root_vel_w[env_ids].clone()
    vel[:, 0:3] = (2.0 * torch.rand_like(vel[:, 0:3]) - 1.0) * max_vel_xy * gs
    vel[:, 3:6] = (2.0 * torch.rand_like(vel[:, 3:6]) - 1.0) * max_vel_ang * gs
    robot.write_root_velocity_to_sim(vel, env_ids=env_ids)


## ---------------------------------------------------------------------------------
# SATA GO2Torque.check_termination (go2_torque.py:99-107) + asset.terminate_after_contacts_on
## ---------------------------------------------------------------------------------
def sata_dof_pos_limit(env, margin: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """SATA terminates when any joint leaves the URDF limit by more than ``margin``.

    GO2Torque.check_termination compares against ``termination_dof_pos_limits``
    (= ``dof_pos_limits +/- 0.05``, go2_torque.py:87-88,100-104).
    """
    robot = env.scene[asset_cfg.name]
    pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.joint_pos_limits[:, asset_cfg.joint_ids]
    return ((pos > limits[..., 1] + margin) | (pos < limits[..., 0] - margin)).any(dim=1)


def sata_body_flipped(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """SATA terminates when the base z-axis points below the horizon.

    ``GO2Torque.check_termination``: ``reset |= projected_gravity[:, 2] > 0``.
    """
    return env.scene[asset_cfg.name].data.projected_gravity_b[:, 2] > 0.0
