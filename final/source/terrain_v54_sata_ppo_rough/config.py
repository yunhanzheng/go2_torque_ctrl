"""V54 - V52's SATA-aligned setup on IsaacLab's official Go2 rough terrain.

Purpose
-------
Isolate the *terrain* factor against V52 (SATA's 20 % smooth slope + 80 % rough
slope).  Everything else is taken from V52 unchanged: the 60-D observation, the TG
torque model, the Gompertz growth curriculum, SATA joint limits, the termination set
(sata_time_out / dof_pos_limit / fell_over, no invalid_upright_state; ``base_contact``
was restored on 2026-10-24, see "2026-10-24 rev B" below),
the SATA spawn state (joints default * U(0.95, 1.05), xy +-1 m, yaw 0, zero velocity),
the 10 s episode, PPO (OnPolicyRunner) and the nine growth-coupled rewards.

Only change vs V52
------------------
- terrain: SATA's ``sata_smooth_slope`` / ``sata_rough_slope`` tiles -> IsaacLab's
  official Go2 rough ``ROUGH_TERRAINS_CFG`` (8x8 m tiles, 10x20 grid, pyramid stairs
  0.05-0.23 m, inverted stairs, random grid boxes, uniform rough, sloped pyramids),
  including the Go2-specific boxes/random_rough scaling from ``UnitreeGo2RoughEnvCfg``.
- terrain curriculum: ``terrain_levels_top_freeze`` (IsaacLab's stock uphill *and* downhill
  motion, with the last row frozen) plus ``generator.curriculum = True`` (V52 had both off).
  Difficulty rises with the row index, and an env moves one row per episode; ``max_init_terrain_level
  = 5`` scatters the initial envs over rows 0-5 (5.0-15.8 cm risers) so nobody starts on a
  23 cm step.  Rev E gated the yaw command on the level; rev G restored the demotion half.

Reward table (reverted 2026-10-24)
----------------------------------
The reward table is back to the one the ``2026-10-22_13-52-21_V54_sata_ppo_rough_
20261022T055209Z`` run trained with: SATA's nine growth-coupled terms plus
``ang_vel_xy_l2`` (-0.05), ``action_rate_l2`` (-0.01) and ``feet_air_time`` (with
``track_air_time=True`` on the contact sensor).  Four of the five terms switched on on
2026-10-23 (``feet_slip``, ``gait_contact``, ``swing_clearance``,
``excessive_foot_height``) are switched off again, so the stance gate that multiplied two
of them is dormant; ``undesired_contacts`` came back at a new weight in rev B and is off
again in rev C, so the table is now the 2026-10-22 one term for term.
All their functions stay at module level.

The previous table -- all five terms live, each gated by the stance diagonal -- is kept
byte-for-byte in ``config_stancegate_backup_20261023.py.bak``
(md5 9cb47b6f1a7318b2c88031d547edcf50); copy that file back over ``config.py`` to restore
it.  The ``2026-10-23_15-44-26_V54_sata_ppo_rough_20261023T074413Z`` training run and its
checkpoints belong to that earlier table.

2026-10-24 rev B
----------------
Two changes on top of the 2026-10-22 table, aimed at the stair failure mode that run
showed (``terrain_levels`` 2.49 -> 0.38, ``fell_over`` 14 % -> 27 %):

- ``undesired_contacts``: back on at -1.0 (thighs/calves, 1 N).  Legs brushing a riser had
  no penalty at all in the 2026-10-22 table.  Switched off again in rev C.
- ``base_contact``: restored to IsaacLab's stock termination (``illegal_contact`` on
  "base", 1 N).  V52 had cleared it to match SATA.  ``fell_over`` still fires at 90 deg,
  so this only adds "the torso actually touched the ground".

``feet_air_time`` stays at IsaacLab's stock 0.01 / 0.5 s -- the value the 2026-10-22 table
already carried.  It was raised to 10.0 / 0.2 s during this revision and rolled back the
same day: with a 0.5 s threshold every landing pays ``t_air - 0.5 < 0`` (Go2 swings for
0.15-0.3 s), so weight 10 only turns that into a -0.10/s toll per landing (14 % of the
0.73/s ``forward`` measured in that run) whose useful gradient is a third of its size.  At
the stock weight the term shapes nothing -- over the whole 2026-10-22 run it never went
positive (10 537 points, min -9.3e-4, max -2.5e-5).

2026-10-24 rev C
----------------
``undesired_contacts`` is switched off again, so the reward table is exactly the one the
``2026-10-22_13-52-21`` run trained with: SATA's nine growth-coupled terms plus
``ang_vel_xy_l2`` (-0.05), ``action_rate_l2`` (-0.01) and ``feet_air_time`` (0.01 / 0.5 s).
``base_contact`` stays on from rev B.

2026-10-24 rev D
----------------
A second task on the same table, ``Isaac-Go2-V54-PitFirst-v0``: the stock 6-sub-terrain
terrain is untouched, but ``PIT_ROWS`` extra rows of 5 cm inverted-pyramid stairs are
prepended to the grid and every env is allocated there first.  That extra rung is called
**level -1** (stock levels 0-9 keep their numbers).  An env leaves level -1 the moment it
walks 4 m -- per env, not at a fixed iteration -- lands on stock level 0, and is floored
there afterwards, so nothing can slide back into the pit.  Rows ``PIT_ROWS + r`` reproduce
stock level ``r`` difficulty for difficulty.  See ``pit_terrain.py`` and
:class:`V54PitFirstEnvCfg`.  V54 itself is unchanged: same task id, same 10-row grid,
same checkpoints.

Why the pit takes a whole block of rows (10 x 20 = 200 tiles) instead of one: the importer
spreads envs by (row, column), so a single pit row would hold all 4096 envs on its 20 tiles.

2026-10-25 rev E -- the curriculum could not be climbed, only sat on
--------------------------------------------------------------------
``2026-10-24_21-46-46_V54_pit_first_20261024T134633Z`` finished at
``Curriculum/terrain_levels`` 0.356 after 15000 iterations, and the pit run before it was no
better.  The cause is a mismatch between the command distribution and the promotion test,
not the policy: ``terrain_levels_vel`` promotes on the **net displacement** from the spawn
point (``size[0] / 2`` = 4 m) and demotes when that displacement is less than
``|v_cmd| * episode_length_s``.  SATA's sampler emits ``ang_vel_z`` up to 1.5 rad/s, i.e. a
0.33 m radius circle at 0.5 m/s, and a 10 s episode holds two independently sampled commands
(resampling every 5 s).  A Monte-Carlo over the full SATA ranges with *perfect* tracking
gives P(promote) 27 %, P(demote) 56 % per episode -- the curriculum is dominated by
demotions, so it parks at row 0.  The demotion bar is also taller than the promotion bar
(7.9 m vs 4 m at full speed), so an episode can be judged too short to keep its row and too
short to earn the next one.

Two changes, both in the command/curriculum plumbing rather than the reward table:

1. ``SataVelocityCommandCfg.level_scaled_yaw`` -- the yaw range grows with the env's
   terrain level: 0 in the pit and on rows 0-1, SATA's full +-1.5 rad/s on the last row.
   Straight walking first (P(promote) ~67 %), turning learned once the policy is high
   enough that the ``sata_yaw`` reward is still meaningful.
2. The last row is a fixed point (``terrain_levels_top_freeze`` /
   ``terrain_levels_pit_first``): reaching the top simply closes the curriculum for that env
   instead of the stock "solved the last level -> random row".  Steady state: the batch ends
   up on the hardest rows with SATA's full command range.  Rev G put the demotion half back
   (see below); only the top stays frozen.

Both apply to ``Isaac-Go2-V54-SataPpoRough-v0`` and ``Isaac-Go2-V54-PitFirst-v0``; V51/V52/V53
are untouched (``level_scaled_yaw`` defaults to False).  Runs before rev E are not directly
comparable: their curriculum sat at row 0.

2026-10-25 rev G -- demotion is back, only the top is frozen
-----------------------------------------------------------
Rev E removed the demotion half *everywhere*, which is not what this curriculum is for: an env
that cannot hold its row should fall back one.  Rev G keeps the intended behaviour and makes
the stock rule usable instead:

1. ``_stock_move_flags`` in ``pit_terrain.py`` is the stock pair of tests -- promote on 4 m of
   net displacement, demote when it is under *half* of the displacement the command asked for
   (``|v_cmd_xy| * max_episode_length_s * 0.5``; note the stock norm is x/y only, ``ang_vel_z``
   is not in it).  Both are gated by the t=0 mask, which the stock version lacks.
2. ``terrain_levels_top_freeze`` / ``terrain_levels_pit_first`` clamp the last row, so an env
   that clears it stays there.  This is now the *only* difference to ``terrain_levels_vel``.
3. Demotion has a floor: on the pit task it is level 0 (row ``PIT_ROWS``), never the pit again.

The yaw gate from rev E is unchanged (``k = level / level_ref``, ``level_ref = 9``), so the
climb is mostly straight walking *and* the top row samples exactly like stock SATA
(``+-1.5 rad/s`` uniform, times the Gompertz ``gs``).

2026-10-25 rev H -- privileged critic (height map)
--------------------------------------------------
Actor and critic used to share one group: both read the same noisy 60-D tensor, so the
critic had no information the policy lacked.  Rev H splits them:

1. ``policy`` unchanged -- ``sata_observation_60d_noisy`` (per-channel uniform noise +
   10 % observation loss), 60 dims.
2. ``critic`` = :class:`CriticObsCfg` -- the same 60 quantities from the clean
   ``sata_observation_60d`` *plus* the 187-D height map (``mdp.height_scan`` on the
   ``height_scanner`` that the reward terms already use: 1.6 x 1.0 m, 0.1 m grid, 187 rays,
   ``clip=(-1, 1)``).  247 dims, ``enable_corruption = False``.
3. ``obs_groups = {"actor": ["policy"], "critic": ["critic"]}`` in the runner cfg.  Only
   the critic grows: the actor network, its input width and therefore the deployed policy are
   unchanged, so a rev H run stays usable on the real robot exactly like rev G.

Both tasks get it (``V54PitFirstEnvCfg`` / ``V54PitFirstRunnerCfg`` inherit).  A checkpoint
from an earlier rev still loads: the actor never touches the new group.
"""
from __future__ import annotations

import copy
import math
import os
import sys

# --- Path bootstrap: make this module importable from any CWD (mirrors run_trial.sh).
# V52's directory is added because we import its config and reuse its ``sata_rewards``.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (
    _SOURCE_ROOT,
    os.path.join(_SOURCE_ROOT, "dependency_chain"),
    os.path.join(_SOURCE_ROOT, "base_adapter"),
    os.path.join(_SOURCE_ROOT, "base_adapter", "initial_stair_source"),
    os.path.join(_SOURCE_ROOT, "terrain_v51_native_terrain"),
    os.path.join(_SOURCE_ROOT, "terrain_v52_sata_ppo_flat"),
    _HERE,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gymnasium as gym
import torch
from isaaclab.managers import (
    CurriculumTermCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from terrain_v52_sata_ppo_flat.config import V52SataPpoFlatEnvCfg, V52SataPpoFlatRunnerCfg
from terrain_v51_native_terrain.sata_alignment import sata_observation_60d
from sata_rewards import _growth, sata_forward
from pit_terrain import (
    PIT_ROWS,
    PitFirstTerrainGenerator,
    log_episode_stats_pit,
    log_episode_stats_stock,
    terrain_levels_pit_first,
    terrain_levels_top_freeze,
)


# --- Head height: distance to the local ground line, plus pitch against it -----------
# The reference is no longer "vertical clearance to the nearest ray" but the line through
# the terrain under the front feet and the terrain under the rear feet, placed in the
# vertical plane of the body's fore-aft axis:
#   * every foot samples the height-scanner ray closest in xy to it.  The grid is welded
#     to the base, so a ray cannot be cast from the (swinging) foot itself; the nearest
#     cell is at most half a grid step (~5 cm) away.  No disc averaging.
#   * the two points sit a *nominal* body length apart.  Using the instantaneous foot
#     positions instead injects stride-phase noise: the spacing swings 0.2-0.5 m, which
#     would modulate the slope once per step.
#   * ``dist`` is the perpendicular distance from the base to that line.  It keeps
#     SATA's ``base_height_target`` cap and its ``(1 + gs)`` growth scaling.
#   * ``head_up`` keeps SATA's shaper verbatim and only moves its zero: the shaper sees
#     ``theta_b - slope`` instead of ``theta_b``, i.e. the relative gravity projection is
#     ``-sin(lag)`` where it used to be ``g_x``.  On flat ground the two coincide bit for
#     bit.  Consequences: pitching the nose *up relative to the terrain* earns up to
#     ``0.2 * (1.5 - 2 gs)`` (fading to zero by ``gs = 0.75``, exactly as in SATA), while
#     diving *below* the line costs ``sin(lag)`` with no cap, again as in SATA.  On a
#     23 cm staircase that is a standing ``sin(31 deg) = -0.5``, so this term also pulls
#     the body toward the stair angle.
_HIP_X = 0.1934  # Go2 URDF: FL/FR hips at +x, RL/RR at -x
_BODY_LENGTH = 2.0 * _HIP_X  # 0.3868 m between the front and rear ground points


def foot_ground_height(env, feet_cfg) -> torch.Tensor:
    """Terrain height under every foot: the height-scanner ray nearest in xy."""
    robot = env.scene["robot"]
    feet_xy = robot.data.body_pos_w[:, feet_cfg.body_ids, :2]     # [N, F, 2]
    hits = env.scene.sensors["height_scanner"].data.ray_hits_w    # [N, R, 3]
    ray_xy, ray_z = hits[..., :2], hits[..., 2]
    valid = torch.isfinite(ray_z)                                 # [N, R]
    dist = torch.cdist(feet_xy, ray_xy).masked_fill(~valid[:, None, :], float("inf"))
    z = torch.gather(ray_z, 1, dist.argmin(dim=-1))               # [N, F]
    # A foot whose nearest ray missed (past the patch border) falls back to the mean
    # valid reading under the base, so one missing hit cannot tilt the reference line.
    count = valid.sum(dim=1).clamp(min=1)
    grid_mean = torch.where(valid, ray_z, torch.zeros_like(ray_z)).sum(dim=1) / count
    return torch.where(torch.isfinite(z), z, grid_mean[:, None])


def _lowpass(env, key: str, value: torch.Tensor, tau: float) -> torch.Tensor:
    """First-order exponential moving average with time constant ``tau`` (seconds).

    The scanner is welded to the base, so a foot's ground sample jumps by a whole riser
    the moment its nearest ray crosses a step edge.  On a 23 cm staircase that swings the
    reference line between 0 and 31 deg, and the line height at the base by 0.115 m, at
    1-3 Hz -- an artefact of the sampling rather than a property of the robot's posture.
    Left raw it injects a 0.26 rad square wave into the reward that no action can
    influence.  Filtering the ground heights (not the angle) smooths the slope and the
    distance derived from them at once.  Rows that just reset are re-seeded so no
    episode inherits the previous one's terrain.
    """
    store = env.__dict__.setdefault("_reward_lowpass", {})
    if tau <= 0.0:  # off switch: ``slope_tau=0`` returns the raw estimate
        return value
    buf = store.get(key)
    if buf is None or buf.shape != value.shape:
        buf = store[key] = value.clone()
        return buf
    fresh = env.episode_length_buf <= 1
    buf[fresh] = value[fresh]
    buf.add_((value - buf) * (1.0 - math.exp(-env.step_dt / tau)))
    return buf


def relative_head_height(
    env, feet_cfg, target_height: float = 0.3, slope_tau: float = 0.3
) -> torch.Tensor:
    """SATA's head_height against the local ground line instead of against level."""
    gs = _growth(env)
    robot = env.scene["robot"]
    # Low-pass the two ground heights: one filter smooths both the slope and ``dist``.
    z = _lowpass(env, "head_height_ground_z", foot_ground_height(env, feet_cfg), slope_tau)
    z_front, z_rear = z[:, :2].mean(dim=1), z[:, 2:].mean(dim=1)
    slope = torch.atan((z_front - z_rear) / _BODY_LENGTH)         # + = uphill
    # s = 0 is halfway between the two points, so the line sits at their mean height.
    dist = (robot.data.root_pos_w[:, 2] - 0.5 * (z_front + z_rear)) * torch.cos(slope)
    # SATA's shaper with its zero on the ground line: ``-sin(lag)`` replaces ``g_x``,
    # so nose-up relative to the terrain earns up to ``-clip_min`` and diving below the
    # line costs without bound.  Identical to SATA on flat ground.
    base_pitch = torch.asin((-robot.data.projected_gravity_b[:, 0]).clamp(-1.0, 1.0))
    g_rel = -torch.sin(base_pitch - slope)
    head_up = -g_rel.clip(min=min(0.0, -0.2 * (1.5 - 2.0 * gs)))
    return dist.clip(max=target_height) * (1.0 + gs) + head_up


# --- Vertical-velocity penalty, gated on flat ground --------------------------------
# SATA's ``lin_vel_z`` assumes the base never has to climb: on a 23 cm step the base
# *must* move vertically, so a blanket ``v_z^2`` penalty fights the terrain.  Gate it on
# the local relief under each foot -- the same ``amax - amin`` probe used by
# ``stair_v24.higher_clearance`` and ``terrain_v48.focused_twenty_three_cm_clearance``
# -- but measured per foot, so a single planted foot on a riser already exempts the step.
def foot_terrain_relief(env, feet_cfg, radius: float = 0.15) -> torch.Tensor:
    """Max-minus-min height-scanner reading within ``radius`` of every foot."""
    robot = env.scene["robot"]
    feet_xy = robot.data.body_pos_w[:, feet_cfg.body_ids, :2]   # [N, F, 2]
    hits = env.scene.sensors["height_scanner"].data.ray_hits_w  # [N, R, 3]
    ray_xy, ray_z = hits[..., :2], hits[..., 2]
    valid = torch.isfinite(ray_z)                               # [N, R]

    dist = torch.cdist(feet_xy, ray_xy)                         # [N, F, R]
    inside = (dist < radius) & valid.unsqueeze(1)               # [N, F, R]
    z = ray_z.unsqueeze(1).expand_as(inside)
    high = z.masked_fill(~inside, -torch.inf).amax(dim=-1)      # [N, F]
    low = z.masked_fill(~inside, torch.inf).amin(dim=-1)        # [N, F]
    relief = high - low
    # A foot without valid rays under it is not *known* to be on flat ground, so it must
    # not unlock the penalty.
    return torch.where(torch.isfinite(relief), relief, torch.full_like(relief, torch.inf))


def on_flat_ground(
    env, feet_cfg, radius: float = 0.15, max_relief: float = 0.01
) -> torch.Tensor:
    """Boolean ``[N]`` mask: every foot stands on locally flat ground."""
    return (foot_terrain_relief(env, feet_cfg, radius) < max_relief).all(dim=1)


def lin_vel_z_on_flat(
    env, asset_cfg, feet_cfg, radius: float = 0.15, max_relief: float = 0.01
) -> torch.Tensor:
    """``v_z^2`` only where every foot stands on locally flat ground."""
    on_flat = on_flat_ground(env, feet_cfg, radius, max_relief).float()
    return env.scene[asset_cfg.name].data.root_lin_vel_b[:, 2].square() * on_flat


# --- V51's five foot/contact/clearance terms -------------------------------------------
# ``undesired_contacts``, ``feet_slip``, ``gait_contact``, ``swing_clearance`` and
# ``excessive_foot_height`` are V51's versions, not V54's earlier substitutions.  Two
# differences from the PGTT-inspired ``feet_slip`` V54 briefly carried:
#   * the gate is the contact force's *z* component at the current step (V51/v8
#     ``sata_feet_slip``), not a norm over a 3-step history;
#   * the diagonal clock is no longer a fixed 1.5 Hz: its frequency follows the commanded
#     speed so that one full gait cycle advances the robot ``stride_length`` metres (0.25 m,
#     i.e. two 0.125 m steps).  V51's constant 1.5 Hz forced one cadence on every episode,
#     which is what made ``gait_contact`` fight ``forward``.
#
# ONE clock, two consumers: ``gait_contact`` scores the contact pattern and
# ``swing_clearance`` masks the swing diagonal, so both must run on the same phase.  They
# read their parameters from this single dict and share one accumulator (``_gait_phase``);
# editing the numbers here re-times both terms at once, there is no second copy to forget.
GAIT_STRIDE_LENGTH = 0.25  # metres travelled by one full cycle (two 0.125 m steps)
GAIT_MIN_FREQUENCY = 0.8   # Hz floor: a near-zero command still ticks instead of freezing
GAIT_MAX_FREQUENCY = 4.0   # Hz ceiling: 0.25 m/cycle up to 1.0 m/s, longer stride above
_GAIT_CLOCK = {
    "command_name": "base_velocity",
    "stride_length": GAIT_STRIDE_LENGTH,
    "min_frequency": GAIT_MIN_FREQUENCY,
    "max_frequency": GAIT_MAX_FREQUENCY,
}


def _gait_frequency(
    speed,
    stride_length: float = GAIT_STRIDE_LENGTH,
    min_frequency: float = GAIT_MIN_FREQUENCY,
    max_frequency: float = GAIT_MAX_FREQUENCY,
) -> torch.Tensor:
    """Cycle frequency that makes one full gait cycle advance ``stride_length`` metres.

    ``f = v / stride_length``: at 0.25 m per cycle that is 2.0 Hz at 0.5 m/s and 4.0 Hz at
    1.0 m/s.  Both clamps are physical: below 0.8 Hz the swing phase lasts over 0.6 s and
    the robot can stand on one diagonal indefinitely; above 4 Hz the leg has under 125 ms to
    lift and plant 0.09 m.  The cap means the stride stretches past 0.25 m as soon as the
    command exceeds 1.0 m/s (1.5 m/s would need 6 Hz).
    """
    return (speed / stride_length).clamp(min=min_frequency, max=max_frequency)


def _gait_phase(
    env, command_name, stride_length: float = GAIT_STRIDE_LENGTH,
    min_frequency: float = GAIT_MIN_FREQUENCY, max_frequency: float = GAIT_MAX_FREQUENCY,
):
    """Shared speed-locked gait phase in [0, 2*pi), plus this step's frequency.

    Integrated rather than evaluated as ``2*pi*f*t``: ``f`` follows the command, so the
    closed form would snap the phase by ``2*pi*(f_new - f_old)*t`` every time the command
    is resampled (every 5 s).  There is exactly one accumulator per env, advanced once per
    physics substep -- the reward manager computes once per substep and ``gait_contact``
    and ``swing_clearance`` are called inside that same compute, so the second caller hits
    the cache and reads the identical phase.  The accumulator is deliberately not reset at
    episode end: episodes then start on varied diagonals instead of always on FL+RR.
    """
    token = getattr(env, "_sim_step_counter", None)
    cache = getattr(env, "_v54_gait_phase_cache", None)
    if token is not None and cache is not None and cache[0] == token:
        return cache[1], cache[2]

    speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    freq = _gait_frequency(speed, stride_length, min_frequency, max_frequency)
    acc = getattr(env, "_v54_gait_phase_acc", None)
    if acc is None or acc.shape[0] != env.num_envs:
        acc = torch.zeros(env.num_envs, device=env.device)
    acc = torch.remainder(acc + 2.0 * torch.pi * freq * env.physics_dt, 2.0 * torch.pi)
    env._v54_gait_phase_acc = acc
    if token is not None:
        env._v54_gait_phase_cache = (token, acc, freq)
    return acc, freq


def _desired_contacts(phase: torch.Tensor) -> torch.Tensor:
    """Open-loop diagonal split over [FL, FR, RL, RR], swapping every half period."""
    first_diagonal = torch.sin(phase) >= 0.0
    desired = torch.zeros((phase.shape[0], 4), dtype=torch.bool, device=phase.device)
    desired[:, 0] = first_diagonal
    desired[:, 3] = first_diagonal
    desired[:, 1] = ~first_diagonal
    desired[:, 2] = ~first_diagonal
    return desired


def _stance_gate(env, sensor_cfg, desired_stance: torch.Tensor) -> torch.Tensor:
    """Fraction of the feet that *should* carry weight and actually are.

    ``desired_stance`` is the diagonal the clock assigns to stance, so the denominator is
    always 2 and the gate reads 1.0 when both feet press harder than 1 N, 0.5 when only one
    does, and 0.0 when the robot is airborne.

    This is the structural fix, not a reweighting.  ``gait_contact`` on its own pays 0.25
    for a crouch with all four feet in the air (two legs match ``desired`` just by being
    off the ground), and ``swing_clearance`` masks the stance diagonal to 1.0, so a robot
    that simply holds its feet up collected both without travelling.  Both terms are
    multiplied by this gate, which is why "hold everything in the air" now scores 0.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    loaded = (forces > 1.0) & desired_stance
    required = desired_stance.sum(dim=1).clamp(min=1)
    return loaded.sum(dim=1).float() / required.float()


def gait_contact(
    env, sensor_cfg, command_name, stride_length: float = GAIT_STRIDE_LENGTH,
    min_frequency: float = GAIT_MIN_FREQUENCY, max_frequency: float = GAIT_MAX_FREQUENCY,
) -> torch.Tensor:
    """V51 ``sata_all_leg_gait_contact``, paced by the speed-locked clock.

    The mean alone let one idle leg hide behind the other three; blending in the per-leg
    minimum makes full reward require all four legs to follow the requested diagonal.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    actual = forces > 1.0
    phase, _ = _gait_phase(env, command_name, stride_length, min_frequency, max_frequency)
    desired = _desired_contacts(phase)
    correct = (actual == desired).float()
    moving = (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    gate = _stance_gate(env, sensor_cfg, ~desired)
    return 0.5 * (correct.mean(dim=1) + correct.amin(dim=1)) * moving * gate


def feet_slip(env, feet_cfg, sensor_cfg, threshold: float = 1.0) -> torch.Tensor:
    """V51 ``sata_feet_slip``: squared horizontal foot speed over the loaded feet."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > threshold
    robot = env.scene[feet_cfg.name]
    vel_xy = robot.data.body_lin_vel_w[:, feet_cfg.body_ids, :2]
    return (vel_xy.pow(2).sum(dim=-1) * contacts.float()).sum(dim=1)


def _forward_fan_target(env, asset_cfg, fan_radius, swing_height):
    """Forward-fan ``(target, clearance)`` for the four feet, shared by both terms.

    ``swing_clearance`` scores clearance *below* the target and ``excessive_foot_height``
    penalises clearance *above* it, so both need exactly the same two tensors.  The reward
    manager calls the two terms back to back inside one ``compute_reward``, so the first
    caller fills a one-slot cache on the env and the second reuses it: the 187-ray
    transform and the fan mask now run once per physics step instead of twice.

    ``target`` is already broadcast over the feet (max across the four fans, so a riser in
    front of one foot lifts the whole gait); ``clearance`` stays per foot (foot z minus the
    terrain under *that* foot).
    """
    # Per-physics-step token: ``_sim_step_counter`` advances with every ``sim.step`` and
    # the reward manager computes once per substep, so a hit can only happen inside one
    # compute.  The body ids are part of the key because the manager resolves each term's
    # ``SceneEntityCfg`` separately (the two params dicts hold different objects), so
    # ``id(asset_cfg)`` would never match; the ids are what the geometry actually reads.
    token = getattr(env, "_sim_step_counter", None)
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_key = (asset_cfg.name, body_ids.start, body_ids.stop)
    else:
        body_key = tuple(int(i) for i in body_ids)
    key = (fan_radius, swing_height, body_key)
    cache = getattr(env, "_v54_fan_target_cache", None)
    if token is not None and cache is not None and cache[0] == token and cache[1] == key:
        return cache[2], cache[3]

    robot = env.scene["robot"]
    feet_xy = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]  # [N, 4, 2]
    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]    # [N, 4]

    # Robot heading.  V51 hand-rolled this as ``qx, qy, qz, qw = root_quat_w.unbind(-1)``,
    # i.e. it read IsaacLab's (w, x, y, z) buffer as (x, y, z, w); the expression it then
    # evaluated, ``atan2(2(w x + y z), 1 - 2(x^2 + y^2))``, is the *roll* formula.  A purely
    # yawed robot therefore read 0 deg and the fan never turned with the body.  Use the
    # library helper, which returns (roll, pitch, yaw) in the buffer's own convention.
    yaw = euler_xyz_from_quat(robot.data.root_quat_w)[2]  # [N]

    ray_hits = env.scene.sensors["height_scanner"].data.ray_hits_w  # [N, num_rays, 3]
    ray_xy = ray_hits[..., :2]
    ray_z = ray_hits[..., 2]
    valid = torch.isfinite(ray_z)

    N, num_feet = feet_xy.shape[0], feet_xy.shape[1]
    num_rays = ray_xy.shape[1]

    # Relative position of every ray w.r.t. every foot, rotated into the body frame.
    rel_xy = ray_xy.unsqueeze(1) - feet_xy.unsqueeze(2)  # [N, 4, num_rays, 2]
    cos_yaw = torch.cos(-yaw).view(N, 1, 1).expand(N, num_feet, num_rays)
    sin_yaw = torch.sin(-yaw).view(N, 1, 1).expand(N, num_feet, num_rays)
    rot_x = cos_yaw * rel_xy[..., 0] - sin_yaw * rel_xy[..., 1]
    rot_y = sin_yaw * rel_xy[..., 0] + cos_yaw * rel_xy[..., 1]
    dist = torch.sqrt(rot_x ** 2 + rot_y ** 2)  # [N, 4, num_rays]

    # Fan mask: within radius AND ahead of the foot AND valid.  ``rot_x`` is the body's
    # fore-aft axis; V51 tested ``rot_y``, which selects the *left* half-plane.
    fan_mask = (dist < fan_radius) & (rot_x > 0.0) & valid.unsqueeze(1).expand(N, num_feet, num_rays)

    # Highest terrain inside the fan, and the terrain directly under the foot.
    masked_z = ray_z.unsqueeze(1).expand(N, num_feet, num_rays).masked_fill(~fan_mask, -torch.inf)
    max_ahead_z = masked_z.amax(dim=-1)  # [N, 4]
    nearest_ray = torch.cdist(feet_xy, ray_xy).argmin(dim=-1)  # [N, 4]
    ground_z = torch.gather(ray_z, 1, nearest_ray)             # [N, 4]

    height_diff = torch.relu(max_ahead_z - ground_z)  # [N, 4]
    height_diff = torch.nan_to_num(height_diff, nan=0.0, posinf=0.0, neginf=0.0)

    target = (swing_height + height_diff).amax(dim=-1, keepdim=True).expand(-1, num_feet)
    clearance = foot_z - ground_z  # [N, 4]

    if token is not None:
        env._v54_fan_target_cache = (token, key, target, clearance)
    return target, clearance


def forward_fan_swing_clearance(
    env,
    asset_cfg,
    sensor_cfg,
    command_name,
    stride_length=GAIT_STRIDE_LENGTH,
    min_frequency=GAIT_MIN_FREQUENCY,
    max_frequency=GAIT_MAX_FREQUENCY,
    swing_height=0.09,
    sigma=0.045,
    fan_radius=0.3,
):
    """V51 swing clearance: Gaussian credit for reaching the fan target while swinging."""
    target, clearance = _forward_fan_target(env, asset_cfg, fan_radius, swing_height)
    deficit = torch.relu(target - clearance)  # 0 once the foot is at or above target
    score = torch.exp(-torch.square(deficit / sigma))  # [N, 4]

    # Same clock ``gait_contact`` scores against: the diagonal that is not in stance is
    # the one this term rewards, so stance feet cannot farm the clearance bonus.
    phase, _ = _gait_phase(env, command_name, stride_length, min_frequency, max_frequency)
    first_diagonal = torch.sin(phase) >= 0.0
    desired_swing = torch.stack(
        (~first_diagonal, first_diagonal, first_diagonal, ~first_diagonal), dim=1
    )
    swing_scores = score.masked_fill(~desired_swing, 1.0)

    moving = (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    gate = _stance_gate(env, sensor_cfg, ~desired_swing)
    return swing_scores.amin(dim=1) * moving * gate


def forward_fan_excessive_foot_height(env, asset_cfg, error_scale=0.06, fan_radius=0.2):
    """V51 guardrail: quadratic penalty for clearance above the same fan target."""
    # 0.09 is V51's fixed ``swing_height``; keep it in the key so a differing
    # ``swing_height`` on the other term recomputes instead of silently sharing.
    target, clearance = _forward_fan_target(env, asset_cfg, fan_radius, 0.09)
    excess = torch.relu(clearance - target) / error_scale
    return torch.clamp(excess.square().amax(dim=1), max=4.0)


@configclass
class CriticObsCfg(ObsGroup):
    """Privileged critic: the 60-D SATA state *without* corruption, plus the 187-D height map.

    ``policy`` carries SATA's own corruption -- per-channel uniform noise and a 10 % chance of
    repeating the previous frame (``sata_observation_60d_noisy``) -- because that is what a
    deployed controller would see.  The critic is never deployed, only trained, so it gets

    * the same 60 quantities noise- and loss-free (``sata_observation_60d``), and
    * the terrain the policy has no sense for: a 1.6 x 1.0 m grid at 0.1 m resolution cast
      straight down from the base, i.e. 17 x 11 = **187** rays.  Same term and clip as the
      stock Go2 rough config (``mdp.height_scan``, ``clip=(-1, 1)``).  ``height_scan``
      subtracts its ``offset=0.5`` (the nominal sensor mount) from the base height, so flat
      ground under the base reads about -0.2 and a 23 cm riser about +0.03.

    247 = 60 + 187.
    """

    base_lin_vel = ObsTerm(func=sata_observation_60d, clip=(-5.0, 5.0))
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        # Privileged: nothing here is corrupted.  (This flag is not what de-noises the 60-D
        # term -- the policy's corruption lives inside ``sata_observation_60d_noisy`` -- it
        # only keeps a future ``noise=Unoise(...)`` from silently re-enabling it.)
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class V54SataPpoRoughEnvCfg(V52SataPpoFlatEnvCfg):
    """V52 (9 SATA rewards + PPO + SATA dynamics) on IsaacLab's official Go2 rough terrain."""

    def __post_init__(self):
        super().__post_init__()

        # === Terrain: swap V52's SATA slope tiles for IsaacLab's official Go2 rough ===
        # Deep-copy the module-level generator so this experiment never mutates it
        # (V48/V49 edit sub_terrains on the instance they inherit).
        generator = copy.deepcopy(ROUGH_TERRAINS_CFG)
        # Go2-specific terrain scaling, identical to UnitreeGo2RoughEnvCfg.__post_init__.
        generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        generator.sub_terrains["random_rough"].noise_step = 0.01
        # Official difficulty ordering.  With ``curriculum=True`` the generator maps the
        # row index linearly onto ``difficulty_range``:
        #     difficulty = (row + U(0, 1)) / num_rows * (upper - lower) + lower,
        # so row 0 is a 5.0-6.8 cm riser and row 9 is 21.2-23 cm.  Rev E promotes an env one
        # row after it walks 4 m (net displacement, ``size[0] / 2``) and drops it a row when it
        # covers less than half of what the command asked for; the last row is a fixed point,
        # so clearing it closes the curriculum for that env.
        # ``max_init_terrain_level = 5`` starts every env in rows 0-5, i.e. 5.0-15.8 cm, so
        # the first episode never puts a 23 cm riser in front of an untrained policy.
        generator.curriculum = True
        # Fixed seed: reproducible terrain across runs (official leaves it None).
        generator.seed = 42
        self.scene.terrain.terrain_generator = generator
        self.scene.terrain.max_init_terrain_level = 5

        # === Privileged critic: clean 60-D state + 187-D height map (V54 rev H) ===
        # ``policy`` is untouched.  A second group just has to exist; rsl_rl's
        # ``resolve_obs_groups`` maps the ``critic`` set onto the group of the same name on its
        # own, but the runner cfg spells the mapping out so the shapes are visible in the log.
        self.observations.critic = CriticObsCfg()

        # === Terrain-level curriculum: stock uphill/downhill, last row frozen (V54 rev G) ===
        # Pure logging, but registered *before* ``terrain_levels``: promotion rewrites
        # ``env_origins`` for the very envs whose final displacement is logged, so the logger
        # has to run first (the ``__dict__`` shuffle below forces that order).
        self.curriculum.by_type = CurriculumTermCfg(func=log_episode_stats_stock)
        self.curriculum.terrain_levels = CurriculumTermCfg(func=terrain_levels_top_freeze)
        # Curriculum terms run in config-attribute order (``CurriculumManager._prepare_terms``
        # walks ``cfg.__dict__``), and ``terrain_levels`` already exists in the parent config, so
        # a plain assignment (or a ``pop`` + re-insert) always lands after it.  Rebuild the dict
        # with the logger first.
        _terms = self.curriculum.__dict__
        _logger_term = _terms.pop("by_type")
        _rest = dict(_terms)
        _terms.clear()
        _terms.update({"by_type": _logger_term, **_rest})

        # === Command range: V51's ranges, but the yaw range follows the terrain level ===
        # Kept at [(-0.5, 1.5), (-0.5, 0.5), (-1.5, 1.5)] on purpose: the command distribution
        # is a curriculum decision, not a reward one.
        #
        # Rev E (2026-10-25): only *when* the yaw range opens is new.  The curriculum
        # promotes on the net displacement from the spawn point (4 m = size[0] / 2), and a
        # full +-1.5 rad/s yaw command walks the episode in a circle -- radius 0.33 m at
        # 0.5 m/s -- instead of away from the tile centre.  Monte-Carlo over SATA's full
        # ranges with perfect tracking: P(promote) 27 %, P(demote) 56 % per episode, so the
        # 2026-10-24_21-46-46 run parked at ``Curriculum/terrain_levels`` 0.356 for 15000
        # iterations.  The yaw range therefore grows with the level: 0 on the bottom rows
        # (straight walking, P(promote) ~67 %) and SATA's full +-1.5 on the last row, where
        # the ``sata_yaw`` reward is still exercised instead of wasting the whole run.
        self.commands.base_velocity.level_scaled_yaw = True
        # Full SATA turn rate on the last row -- which is now a fixed point: the curriculum
        # closes there (``terrain_levels_top_freeze``) instead of re-placing the env.
        self.commands.base_velocity.level_ref = 9.0

        # === Spawn pose: inherited from V52 unchanged ===
        # V52 already installs SATA ``go2_torque``'s spawn state (joints
        # ``default * U(0.95, 1.05)``, xy +-1 m, yaw 0, zero lin/ang velocity).
        # Do not re-add V51's wider ``U(0.5, 1.5)`` joint range: it straightens the legs and
        # drives the feet ~18 cm under the floor at SATA's 0.10 m spawn height.

        # === Three reward terms adapted for stairs (2026-10-22) ===
        # 1. Joint limits back to the stock IsaacLab/Unitree range.  SATA's positive-only
        #    thigh window (front [0, 1.5], rear [0, 2.0]) puts the default stance (1.45 rad)
        #    *outside* the 90 % soft bound (1.425) and leaves the calf only 0.128 rad before
        #    its hard stop -- a 23 cm step needs far more travel than that.  The
        #    ``soft_dof_pos_limits`` term itself is untouched (weight -5); only the bounds
        #    move.  Clearing the startup event lets the USD limits through unchanged.
        self.events.sata_joint_limits = None
        # 2. ``head_height``: reference is the local ground line (terrain under the front
        #    feet vs terrain under the rear feet), not a world-level plane, so the same
        #    term works on slopes and on stairs.  No flat-ground gate: the line *is* the
        #    terrain.  Weight, the 0.3 m cap and the ``(1 + gs)`` scaling are unchanged.
        _joints = SceneEntityCfg("robot")
        _feet = SceneEntityCfg(
            "robot",
            body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            preserve_order=True,
        )
        # 1b. ``forward``: V52's +10, the value the 2026-10-22 run used.  (The 2026-10-23
        #     run raised it to +20 once ``swing_clearance`` came back; that change is not
        #     part of this table.)  Curve unchanged: sigma 0.25, lo/hi -0.5/1.5.
        self.rewards.forward = RewardTermCfg(
            func=sata_forward,
            weight=10.0,
            params={
                "command_name": "base_velocity",
                "sigma": 0.25,
                # SATA's own lo/hi.  Keep these equal to
                # ``commands.base_velocity.ranges.lin_vel_x``: SATA shrinks the target
                # toward ``0.5 * (hi + lo)`` at low growth, so a mismatch aims the policy
                # away from the sampled command.
                "lin_vel_x_lo": -0.5,
                "lin_vel_x_hi": 1.5,
            },
        )
        self.rewards.head_height = RewardTermCfg(
            func=relative_head_height,
            weight=5.0,
            params={"feet_cfg": _feet, "target_height": 0.3, "slope_tau": 0.3},
        )
        # 3. ``lin_vel_z``: v_z^2 at weight -5, but only where all four feet stand on
        #    locally flat ground (per-foot relief < 1 cm).  Rough terrains and stairs
        #    force the base to move vertically, so the blanket penalty is a pure drag
        #    there; on flat ground it still suppresses the usual bobbing.
        self.rewards.lin_vel_z = None
        self.rewards.lin_vel_z_l2 = RewardTermCfg(
            func=lin_vel_z_on_flat,
            weight=-5.0,
            params={"asset_cfg": _joints, "feet_cfg": _feet, "radius": 0.15, "max_relief": 0.01},
        )
        # 4. The last two SATA copies collapse onto the stock IsaacLab functions:
        #    ``joint_pos_limits`` is the *same* probe/clip/sum against
        #    ``data.soft_joint_pos_limits``, which UNITREE_GO2_CFG builds with
        #    ``soft_joint_pos_limit_factor=0.9`` -- exactly SATA's ``soft_limit_frac``.
        #    ``joint_acc_l2`` is ``sum(joint_acc^2)``, which is what ``sata_dof_acc``
        #    already reads.  Both therefore keep their SATA weights (-5 and -1e-6) with
        #    zero change in behaviour.
        self.rewards.soft_dof_pos_limits = RewardTermCfg(
            func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": _joints},
        )
        self.rewards.dof_acc = RewardTermCfg(
            func=mdp.joint_acc_l2, weight=-1e-6, params={"asset_cfg": _joints},
        )

        # === Three stock regularisers from the official Go2 rough lineup ===
        # V52 flattened all inherited reward terms to None, so these start from scratch.
        # 5. ``ang_vel_xy_l2``: penalises roll/pitch *rate*.  ``roll`` only sees the angle,
        #    so the controller can oscillate through level at high angular speed for free.
        self.rewards.ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
        # 6. ``action_rate_l2``: pure first-difference regulariser on the joint targets.
        #    SATA dropped it entirely, which is why its policy relies on ``motor_fatigue``
        #    to stay smooth.  Back to -0.01.  The 2026-10-23 run had raised it to -0.03 after
        #    measuring raw 96 at -0.01 (only -0.96/s at 200 Hz, while the policy buzzed at
        #    ~1.7 sigma per 5 ms step); it is reverted together with the rest of this table.
        self.rewards.action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
        # ``feet_air_time``: IsaacLab's stock Go2 setting (rough_env_cfg.py L52-53 -- weight
        # 0.01, threshold 0.5 s, sensor on the four feet).  It is knowingly a no-op here:
        # Go2's swing phase is 0.15-0.3 s, so every landing satisfies ``t_air < 0.5`` and pays
        # a negative amount; end to end the 2026-10-22 run measured it at -0.0002 per episode
        # (min -9.3e-4, max -2.5e-5 over 10 537 points, never positive).  Raising the weight
        # without lowering the threshold only enlarges that per-landing toll (weight 10 =
        # -0.10/s, 14 % of ``forward``) instead of buying foot lift -- that is why the
        # 0.2 s / 10.0 experiment of 2026-10-24 rev B was rolled back.  Still gated by a live
        # velocity command and settled only on the landing step.
        self.scene.feet_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=True,
            debug_vis=False,
        )
        _foot_contact = SceneEntityCfg(
            "feet_contact",
            body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            preserve_order=True,
        )
        self.rewards.feet_air_time = RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.01,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": _foot_contact,
                "threshold": 0.5,
            },
        )
        # 7. ``undesired_contacts`` (off again in rev C): thighs and calves brushing a riser
        #    stay free, as they were in the 2026-10-22 table.  Rev B had it at -1.0 / 1 N to
        #    stop the policy from scraping up the steps; switched off on request, so the
        #    table is now term-for-term the 2026-10-22 one.  ``feet_contact`` is kept because
        #    ``feet_air_time`` reads it (``track_air_time``).
        self.rewards.undesired_contacts = None

        # === The five terms the 2026-10-22 run did not have, all off ===
        # ``undesired_contacts`` came back in rev B and is off again in rev C; these four
        # keep their stance-gated implementations at module level (dormant).  The previous,
        # all-on table is in ``config_stancegate_backup_20261023.py.bak``.
        self.rewards.feet_slip = None
        self.rewards.gait_contact = None
        self.rewards.swing_clearance = None
        self.rewards.excessive_foot_height = None

        # === Terminations: ``base_contact`` restored to IsaacLab's default (2026-10-24) ===
        # Stock definition from ``velocity_env_cfg.py``: ``illegal_contact`` on "base",
        # 1 N, over the sensor's 3-frame (15 ms) history.  V52 had cleared it to match
        # SATA's termination set, which left torso contact entirely free.  ``fell_over``
        # keeps SATA's 90 deg, so this only adds "the torso actually touched ground".
        # Note: V52's ``self.scene.contact_forces.body_names = "base"`` line is inert
        # (``ContactSensorCfg`` has no such field); the body filter lives in this cfg.
        self.terminations.base_contact = TerminationTermCfg(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
                "threshold": 1.0,
            },
        )


@configclass
class V54SataPpoRoughRunnerCfg(V52SataPpoFlatRunnerCfg):
    """Same PPO hyperparameters as V52 (SATA GO2TorqueCfgPPO), new experiment name."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 15000
        # One checkpoint every 1000 iterations (~31 min): 15 files (~72 MB) for the curve.
        self.save_interval = 1000
        self.experiment_name = "terrain_v54_sata_ppo_rough"
        # Asymmetric actor-critic: the actor sees the noisy 60-D ``policy`` group, the critic
        # sees the clean 60-D state plus the 187-D height map (``CriticObsCfg``, 60 + 187 = 247).
        # Without this the stock fallback would hand both networks ``policy``.
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        # ``init_noise_std`` (1.0) and ``entropy_coef`` (0.01) are rsl_rl's defaults, i.e.
        # V52's values.  A 0.5 / 0.003 pair was drafted while the reward table was being
        # rewritten but never ran; this round keeps the explorer at the original setting.


@configclass
class V54PitFirstEnvCfg(V54SataPpoRoughEnvCfg):
    """Rev D: the stock V54 terrain, plus a 5 cm inverted-stair pit every env has to clear.

    The extra rung is called **level -1**: stock levels 0-9 keep their numbers and the pit sits
    one below level 0, so ``Curriculum/terrain_levels`` starts at -1 and climbs to 9.  Grid rows
    stay non-negative (``terrain_levels`` indexes ``terrain_origins``); the mapping lives in
    ``pit_terrain.pit_display_levels``.

    The stock grid is reused rather than rebuilt (``super().__post_init__`` already produced
    the Go2-scaled 6-sub-terrain generator); this only widens it by ``PIT_ROWS`` rows and
    swaps the generator class for one that fills those rows with the pit.  All envs are then
    allocated there by ``max_init_terrain_level = PIT_ROWS - 1``: rows 0-9 are the pit, and
    since the block's 200 tiles are identical copies spread over 10 rows, the ~20 env/tile
    spacing is the same as the stock grid's.

    Promotion is ``terrain_levels_pit_first``: 4 m walked at level -1 -> stock level 0 (row
    ``PIT_ROWS``, 5.0-6.8 cm), then the stock +-1 level motion with a floor at 0.  Nothing
    returns to level -1 once it has left.
    """

    def __post_init__(self):
        super().__post_init__()
        # Deep-copy V54's own generator: its Go2 scaling of ``boxes`` / ``random_rough`` and
        # the 6 sub-terrain proportions stay exactly as they are.
        generator = copy.deepcopy(self.scene.terrain.terrain_generator)
        generator.num_rows = 2 * PIT_ROWS  # PIT_ROWS pit rows on top + the stock 10
        generator.class_type = PitFirstTerrainGenerator
        self.scene.terrain.terrain_generator = generator
        # Every env starts on a pit row; the block is wide enough that its density matches
        # the stock grid's (~20 env per 8x8 m tile).
        self.scene.terrain.max_init_terrain_level = PIT_ROWS - 1
        self.curriculum.terrain_levels = CurriculumTermCfg(func=terrain_levels_pit_first)
        # Same per-type episode statistics as the stock task, but in display levels: the pit
        # rows form their own ``pit`` bucket instead of being folded into a terrain type.
        self.curriculum.by_type = CurriculumTermCfg(func=log_episode_stats_pit)
        # Yaw commands are gated on the *display* level (V54 rev E), so the pit rows have to be
        # subtracted first: inside the pit the level is -1 and the command stays straight,
        # which is what lets the 4 m displacement test fire at all.
        self.commands.base_velocity.level_offset = float(PIT_ROWS)


@configclass
class V54PitFirstRunnerCfg(V54SataPpoRoughRunnerCfg):
    """Same PPO hyperparameters as V54; its own experiment name so the logs do not mix."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v54_pit_first"


gym.register(
    id="Isaac-Go2-V54-SataPpoRough-v0",
    entry_point="terrain_v51_native_terrain.sata_alignment:SataControlFreqEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V54SataPpoRoughEnvCfg",
        # Point both entry-point keys at the PPO runner so train.py works whether it
        # is invoked with --agent rsl_rl_sac_cfg_entry_point (V51/V52 default) or
        # --agent rsl_rl_cfg_entry_point (the usual PPO key).
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V54SataPpoRoughRunnerCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:V54SataPpoRoughRunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2-V54-PitFirst-v0",
    entry_point="terrain_v51_native_terrain.sata_alignment:SataControlFreqEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V54PitFirstEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V54PitFirstRunnerCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:V54PitFirstRunnerCfg",
    },
)
