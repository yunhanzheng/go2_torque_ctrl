"""V52 - SATA-identical verification run: PPO + SATA terrain + SATA's 9 rewards.

Purpose
-------
This is a *separate* experiment that sanity-checks the SATA dynamics port living in
V51's ``sata_alignment.py``.  It deliberately reproduces the original SATA
``go2_torque`` recipe as closely as IsaacLab allows:

  - algorithm : SAC  -> PPO           (SATA's original; rsl_rl OnPolicyRunner)
  - terrain   : SATA's trimesh, i.e. 20 % smooth slope + 80 % rough slope tiles
                (8 m x 8 m, 10 x 20 of them, no curriculum) - see ``sata_terrain.py``
  - rewards   : V51's 16 -> SATA's 9  (forward/head_height/moving_y/moving_yaw/
                                       motor_fatigue/roll/lin_vel_z/
                                       soft_dof_pos_limits/dof_acc, growth-coupled)

Everything else is *reused from V51* (and therefore from SATA): the TG torque model,
Gompertz growth curriculum, 60-D observation with per-dimension noise, the growth
-scaled velocity command, decimation=1 (200 Hz), episode length 10 s, and the domain
randomization.  V51 (SAC + native rough terrain + 16 rewards) is left completely
untouched - this file only reads from it.

Note: the module / task id still say "flat" (they are also the log directory name), but
the terrain is SATA's sloped/rough one.  Flat ground only appears in ``evaluate_video.py``
because SATA's ``play.py`` switches to ``mesh_type = 'plane'``.
"""
from __future__ import annotations

import os
import sys

# --- Path bootstrap: make this module importable from any CWD (mirrors run_trial.sh).
# Adds the V51 directory too, because we reuse its ``sata_alignment`` (V51's config
# imports it as a bare top-level module).  Importing terrain_v51_native_terrain.config
# is side-effect-safe: that package's __init__.py is a comment only.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (
    _SOURCE_ROOT,
    os.path.join(_SOURCE_ROOT, "dependency_chain"),
    os.path.join(_SOURCE_ROOT, "base_adapter"),
    os.path.join(_SOURCE_ROOT, "base_adapter", "initial_stair_source"),
    os.path.join(_SOURCE_ROOT, "terrain_v51_native_terrain"),
    _HERE,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gymnasium as gym
import torch
from isaaclab.envs.mdp import reset_joints_by_scale
from isaaclab.managers import (
    EventTermCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
)

from terrain_v51_native_terrain.config import V51NativeTerrainEnvCfg
from terrain_v51_native_terrain.sata_alignment import sata_observation_60d_noisy, sata_time_out

from sata_terrain import SataSlopedTerrainCfg

from sata_rewards import (
    sata_body_flipped,
    sata_dof_acc,
    sata_dof_pos_limit,
    sata_forward,
    sata_head_height,
    sata_lin_vel_z,
    sata_motor_fatigue,
    sata_moving_y,
    sata_moving_yaw,
    sata_push_robots,
    sata_randomize_rigid_body_props,
    sata_roll,
    sata_soft_dof_pos_limits,
)


# SATA go2_torque.urdf joint position limits (rad).  The stock IsaacLab Go2 USD ships
# the wide Unitree thigh range [-1.5708, 3.1416]; SATA narrows the thigh to a
# positive-only, standing-compatible window (front [0, 1.5], rear [0, 2.0]) that acts as
# an exploration "exoskeleton" - saturated TG torques cannot splay the thighs and tip the
# robot before the policy has learned anything.  hip/calf already agree with the USD but
# are set explicitly for fidelity.
_SATA_JOINT_LIMITS = {
    "hip": (-1.0472, 1.0472),
    "front_thigh": (0.0, 1.5),
    "rear_thigh": (0.0, 2.0),
    "calf": (-2.7227, -0.83776),
}


def apply_sata_joint_limits(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Startup event: clamp the Go2 USD joints to SATA's go2_torque.urdf limits.

    ``write_joint_position_limit_to_sim`` pushes the limits into PhysX (hard constraint)
    and refreshes ``data.joint_pos_limits`` + the soft limits, so the growth-coupled
    ``soft_dof_pos_limits`` reward follows SATA automatically.
    """
    robot = env.scene[asset_cfg.name]
    limits = robot.data.default_joint_pos_limits.clone()  # (num_envs, num_joints, 2)
    orig_thigh = None
    for i, name in enumerate(robot.joint_names):
        if "hip" in name:
            lo, hi = _SATA_JOINT_LIMITS["hip"]
        elif "thigh" in name:
            key = "front_thigh" if name.startswith(("FL", "FR")) else "rear_thigh"
            lo, hi = _SATA_JOINT_LIMITS[key]
            if orig_thigh is None:
                orig_thigh = robot.data.default_joint_pos_limits[0, i].tolist()
        elif "calf" in name:
            lo, hi = _SATA_JOINT_LIMITS["calf"]
        else:
            continue
        limits[:, i, 0] = lo
        limits[:, i, 1] = hi
    robot.write_joint_position_limit_to_sim(limits, warn_limit_violation=False)
    if orig_thigh is not None:
        print(f"[V52 SATA-align] thigh limit USD {orig_thigh} -> SATA front[0,1.5]/rear[0,2.0]", flush=True)


@configclass
class V52SataPpoFlatEnvCfg(V51NativeTerrainEnvCfg):
    """V51 dynamics/observation/command, but flat ground + PPO + SATA's 9 rewards."""

    def __post_init__(self):
        super().__post_init__()

        # === Enforce SATA go2_torque.urdf joint limits (esp. narrow thigh) physically ===
        self.events.sata_joint_limits = EventTermCfg(
            func=apply_sata_joint_limits,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # === SATA asset flag: self_collisions = 0 is NOT reproducible here ===
        # GO2TorqueCfg.asset sets ``self_collisions = 0``.  Turning it off on the IsaacLab Go2
        # *USD* makes the robot markedly less stable (measured: 124 fall-terminations per
        # 600x4 steps vs 49 with collisions on), because the USD collision volumes are tuned
        # for self collisions being enabled.  Keep the engine default and record the gap.
        self.scene.robot.self_collisions = True

        # === SATA terminations: dof-limit breach + tipped over + time_out ===
        # GO2Torque.check_termination (go2_torque.py:99-107) is exactly:
        #   reset |= any(dof_pos > URDF_limit + 0.05) | any(dof_pos < URDF_limit - 0.05)
        #   reset |= projected_gravity[:, 2] > 0        (base inverted)
        #   reset |= episode_length_buf > max_episode_length
        # plus ``asset.terminate_after_contacts_on = ["Head"]``, which is dead code in Isaac
        # Gym: the Head_lower sphere already sits 7 mm below the floor at the 0.10 m spawn
        # height and the engine reports no force for that penetration, so it never fires.
        # Measured: with zero torque SATA's episode survives ~124 substeps and ends on the
        # dof/inversion conditions, never on a contact - therefore "fall = torso touch" is
        # *not* part of SATA's termination set.  Adding ``base_contact`` here (as V50/V51
        # did) truncated every early episode at ~17 policy steps and made the training
        # metric incomparable with SATA's (1611 substeps at iteration 195).
        # The inherited ETH-SAC ``invalid_upright_state`` envelope is disabled for the same
        # reason as before: its 0.22 m height floor hard-capped every episode at 200 steps.
        self.terminations.invalid_upright_state = None
        # --- 21:13 cadence kept, torso-contact termination removed (2026-10-21) -----
        # ``...20261021T131323Z`` kept the control-period cadence but still inherited
        # IsaacLab's ``base_contact`` termination (``illegal_contact`` on "base",
        # threshold 1.0).  It fired on only 1.9 % of episodes (``fell_over`` covers the
        # rest), and SATA has no torso-contact termination, so it is disabled here.
        # Re-enable by deleting the next line.
        self.terminations.base_contact = None
        # The base-only contact sensor is kept for diagnostics (evaluate_video.py prints it).
        self.scene.contact_forces.body_names = "base"
        self.terminations.dof_pos_limit = TerminationTermCfg(
            func=sata_dof_pos_limit, params={"margin": 0.05},
        )
        self.terminations.fell_over = TerminationTermCfg(func=sata_body_flipped)
        # SATA's time-out is counted in physics substeps (legged_robot.py:114,142-148), so an
        # episode lasts 10 s of simulated time whatever decimation the growth curve produces.
        # IsaacLab's ``mdp.time_out`` counts policy steps instead, which stretches episodes by
        # up to 2x while the control frequency is still 100 Hz.
        self.terminations.time_out = TerminationTermCfg(func=sata_time_out, time_out=True)

        # === SATA domain_rand mass randomization (go2_torque.py:109-135) ===
        # GO2Torque overrides ``_process_rigid_body_props`` completely: base mass
        # += U(-1,5), then *every* body (base included -> base shifted twice) += U(-1/16, 5/16),
        # and the base CoM/inertia follow the added point mass through the parallel-axis
        # theorem.  IsaacLab's ``randomize_rigid_body_mass`` re-reads ``default_mass`` on
        # each call (events.py:368), so the base would be *overwritten* instead of shifted
        # twice; both inherited terms must be removed and replaced by one combined event.
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.sata_rigid_body_props = EventTermCfg(
            func=sata_randomize_rigid_body_props,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "added_mass_range": (-1.0, 5.0),
                "com_range": ((-0.2, 0.2), (-0.1, 0.1), (-0.1, 0.1)),
            },
        )

        # === SATA domain_rand.push_robots (go2_torque.py:272-282) ===
        # push_interval_s=4 -> one velocity impulse every 4 s that overwrites the whole
        # 6-D root velocity: U(-1,1)*1.5*gs on lin x/y/z, U(-1,1)*1.0*gs on ang x/y/z.
        # V51's ``push_robot`` (push_by_setting_velocity, no growth scaling) must be
        # disabled or the robot would be pushed twice per interval.
        self.events.push_robot = None
        self.events.push_robots = EventTermCfg(
            func=sata_push_robots,
            mode="interval",
            interval_range_s=(4.0, 4.0),
            is_global_time=True,
        )

        # === Spawn state: SATA's _reset_dofs / _reset_root_states (go2_torque.py:137-174) ===
        #   joints      = default * U(0.95, 1.05)   <- go2_torque OVERRIDES the base 0.5:1.5 range
        #   xy          += U(-1, 1) m, yaw = 0      (base_init_state rot = [0, 0, 0, 1])
        #   lin/ang vel  = 0                        (root_states[:, 7:13] = 0)
        # The narrow joint range keeps the spawn in SATA's low crouch; the base 0.5:1.5 range
        # stretches the legs straight, which drives the feet ~18 cm under the floor.
        self.events.reset_robot_joints = EventTermCfg(
            func=reset_joints_by_scale,
            mode="reset",
            params={"position_range": (0.95, 1.05), "velocity_range": (0.0, 0.0)},
        )
        self.events.reset_base.params["pose_range"] = {
            "x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }

        # === SATA go2_torque terrain, not flat ground (go2_torque_config.py:66-90) ===
        # Training uses ``mesh_type = 'trimesh'`` with ``terrain_proportions = [0.2, 0.8, 0, 0, 0]``
        # -> 20 % smooth slope + 80 % rough slope, 8 m x 8 m tiles, 10 rows x 20 cols, and
        # ``curriculum = False`` so ``Terrain.randomized_terrain`` draws the type, the sign and
        # the difficulty of every tile independently.  Only ``play.py`` switches to 'plane';
        # ``evaluate_video.py`` reproduces that on its own.
        generator = self.scene.terrain.terrain_generator
        generator.sub_terrains = {
            "sata_smooth_slope": SataSlopedTerrainCfg(proportion=0.2, rough=False, negative_prob=0.5),
            "sata_rough_slope": SataSlopedTerrainCfg(proportion=0.8, rough=True, negative_prob=0.375),
        }
        generator.size = (8.0, 8.0)
        generator.num_rows = 10
        generator.num_cols = 20
        generator.curriculum = False
        generator.border_width = 25.0  # SATA terrain.border_size
        generator.horizontal_scale = 0.1
        generator.vertical_scale = 0.005
        generator.slope_threshold = 0.75  # SATA terrain.slope_treshold
        # The SATA draw happens *inside* the sub-terrain function, so the cache key (which only
        # sees the cfg) would collapse all tiles with one difficulty into a single mesh.
        generator.use_cache = False
        generator.seed = None  # SATA draws the terrain from the unseeded global RNG
        self.scene.terrain.max_init_terrain_level = None  # -> num_rows - 1, as in SATA
        self.curriculum.terrain_levels = None  # SATA terrain.curriculum = False

        # === Observations: exactly SATA's 60-D vector ===
        # V51 only replaced ``base_lin_vel`` with the 60-D SATA tensor and left the seven
        # inherited terms (base_ang_vel / gravity / commands / joint_pos / joint_vel /
        # actions / height_scan) in place, so the policy actually saw 60+57+187 dims with
        # the same information duplicated three times.  SATA feeds exactly 60 dims.
        for _name, _term in list(vars(self.observations.policy).items()):
            if isinstance(_term, ObsTerm):
                setattr(self.observations.policy, _name, None)
        self.observations.policy.base_lin_vel = ObsTerm(
            func=sata_observation_60d_noisy,
            params={"loss_rate": 0.1, "noise_level": 1.5},
            clip=(-5.0, 5.0),  # SATA normalization.clip_observations = 5
        )

        # === Replace V51's 16 rewards with SATA go2_torque's 9 (verbatim) ===
        # Disable every inherited RewardTermCfg first, then attach the SATA terms.
        for _name, _val in list(vars(self.rewards).items()):
            if isinstance(_val, RewardTermCfg):
                setattr(self.rewards, _name, None)

        joints = SceneEntityCfg("robot")  # all 12 joints; the sums are order-independent
        # SATA's ``_prepare_reward_function`` multiplies every non-zero scale by ``dt``
        # (legged_robot.py:200-208: ``self.reward_scales[key] *= self.dt``), so one SATA
        # substep pays ``term * weight * dt`` - which is exactly what IsaacLab's RewardManager
        # adds (``term * weight * dt`` with ``dt = step_dt = sim.dt * decimation``,
        # manager_based_rl_env.py:207).  Both frameworks therefore agree with SATA's **raw**
        # scales; do not add an extra factor here.
        self.rewards.forward = RewardTermCfg(
            func=sata_forward,
            weight=10.0,
            params={"command_name": "base_velocity", "sigma": 0.25,
                    "lin_vel_x_lo": -0.5, "lin_vel_x_hi": 1.5},
        )
        self.rewards.head_height = RewardTermCfg(
            func=sata_head_height, weight=5.0, params={"base_height_target": 0.3},
        )
        self.rewards.moving_y = RewardTermCfg(
            func=sata_moving_y, weight=5.0,
            params={"command_name": "base_velocity", "sigma": 0.25},
        )
        self.rewards.moving_yaw = RewardTermCfg(
            func=sata_moving_yaw, weight=5.0,
            params={"command_name": "base_velocity", "sigma": 0.25},
        )
        self.rewards.soft_dof_pos_limits = RewardTermCfg(
            func=sata_soft_dof_pos_limits, weight=-5.0,
            params={"asset_cfg": joints, "soft_limit_frac": 0.9},
        )
        self.rewards.motor_fatigue = RewardTermCfg(func=sata_motor_fatigue, weight=-0.05)
        self.rewards.dof_acc = RewardTermCfg(
            func=sata_dof_acc, weight=-1e-6, params={"asset_cfg": joints},
        )
        self.rewards.roll = RewardTermCfg(func=sata_roll, weight=-5.0)
        self.rewards.lin_vel_z = RewardTermCfg(func=sata_lin_vel_z, weight=-5.0)


@configclass
class V52SataPpoFlatRunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    """SATA GO2TorqueCfgPPO hyperparameters.

    The IsaacLab default Go2 PPO config already matches SATA's LeggedRobotCfgPPO
    term-for-term (num_steps_per_env=24, [512,256,128] elu, lr=1e-3 adaptive,
    desired_kl=0.01, gamma=0.99, lam=0.95, clip=0.2, entropy=0.01).  Only the
    iteration budget is raised to SATA's 3000.
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.save_interval = 50
        self.experiment_name = "terrain_v52_sata_ppo_flat"


gym.register(
    id="Isaac-Go2-V52-SataPpoFlat-v0",
    entry_point="terrain_v51_native_terrain.sata_alignment:SataControlFreqEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V52SataPpoFlatEnvCfg",
        # Point both entry-point keys at the PPO runner so train.py works whether it
        # is invoked with --agent rsl_rl_sac_cfg_entry_point (V51 default) or
        # --agent rsl_rl_cfg_entry_point (the usual PPO key).
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V52SataPpoFlatRunnerCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:V52SataPpoFlatRunnerCfg",
    },
)
