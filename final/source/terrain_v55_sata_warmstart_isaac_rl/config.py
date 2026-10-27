"""V55 - SATA's pretrained Go2 policy, retrained on this task with IsaacLab's stock rewards.

Purpose
-------
Every previous run in this repository (V51-V54) inherited SATA's nine growth-coupled
rewards, whose positive terms alone are worth ~25 reward/s against ~2.26/s for the whole
IsaacLab Go2 table.  V55 asks the complementary question: **with the reward signal reduced to
IsaacLab's official Go2 rough table, can a SATA-pretrained policy still be pushed onto the
23 cm stair?**  It is therefore a warm start, not a from-scratch run:

- actor/critic initialisation: ``SATA/logs/SATA/Oct14_16-33-32_/model_3000.pt``
  (actor [512,256,128] ELU, 60-D observation, 12 torque outputs -- all identical to ours, so
  the actor weights load directly; the critic is re-initialised because the reward function
  changed completely), see ``--sata_init`` in ``train.py`` / ``load_sata_actor.py``.
- environment: V54 unchanged -- official Go2 rough terrain with the official
  ``terrain_levels_vel`` curriculum, SATA spawn state, 10 s episodes, 23 cm risers at the
  top rows, TG torque actuation, 60-D observation, the four terminations
  (``sata_time_out`` / ``base_contact`` / ``dof_pos_limit`` / ``fell_over``).
- rewards: the eleven terms of ``velocity_env_cfg.py`` / ``go2/rough_env_cfg.py`` and
  nothing else; ``undesired_contacts`` is left off (Go2 rough's default) by decision, so
  calf scraping is free.
- growth clock: no longer SATA's Gompertz, but the bounded ramp in ``v55_growth.py``
  (0.789 -> 1.0 over 500 iterations, then constant).  The reward side of gs is gone by
  construction; gs now only sets control frequency, torque ceiling and command range.

What is *not* changed vs V54
----------------------------
Observation (60-D), action space (12 torques through the same tanh/hill/fatigue model),
network sizes, ``num_steps_per_env`` (24), terrain generator, curriculum, episode length,
termination set, spawn state, command term (``SataVelocityCommand``, whose ranges are
scaled by gs).  That is what makes V54 -> V55 a single-factor comparison: rewards + gs clock.

Known consequences of the stock table (documented, not fixed)
-------------------------------------------------------------
- ``lin_vel_z_l2`` (-2.0) is no longer flat-gated: on stairs it punishes the vertical motion
  that climbing requires (~-0.18/s at V54's typical |vz|).
- ``undesired_contacts`` is ``None``: nothing discourages shin/knee contact with a 23 cm
  step, which is exactly how V40-V49 runs learned to "climb".
- ``dof_pos_limits`` is 0.0 and ``flat_orientation_l2`` is 0.0: present for fidelity with the
  official table, but they contribute no gradient.
- ``feet_air_time`` (+0.01, 0.5 s) is close to a no-op at 200 Hz.
"""
from __future__ import annotations

import math
import os
import sys

# --- Path bootstrap: make this module importable from any CWD (mirrors run_trial.sh).
# The V51 directory is needed for ``sata_alignment`` (SATA joint names, TG action cfg),
# V52's for its config chain, V54's for the base env cfg we extend.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (
    _SOURCE_ROOT,
    os.path.join(_SOURCE_ROOT, "dependency_chain"),
    os.path.join(_SOURCE_ROOT, "base_adapter"),
    os.path.join(_SOURCE_ROOT, "base_adapter", "initial_stair_source"),
    os.path.join(_SOURCE_ROOT, "terrain_v51_native_terrain"),
    os.path.join(_SOURCE_ROOT, "terrain_v52_sata_ppo_flat"),
    os.path.join(_SOURCE_ROOT, "terrain_v54_sata_ppo_rough"),
    _HERE,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gymnasium as gym
# The velocity task's mdp re-exports ``isaaclab.envs.mdp`` and adds the task-local terms --
# notably ``feet_air_time``, which does *not* live in ``isaaclab.envs.mdp``.  Same import as V54.
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from terrain_v51_native_terrain.sata_alignment import SATA_JOINT_NAMES
from terrain_v54_sata_ppo_rough.config import V54SataPpoRoughEnvCfg, V54SataPpoRoughRunnerCfg

try:  # normal case: imported as part of the package
    from .v55_growth import V55TorqueActionCfg
except ImportError:  # imported as a bare module (scripts that run inside this directory)
    from v55_growth import V55TorqueActionCfg


@configclass
class V55SataWarmstartIsaacRlEnvCfg(V54SataPpoRoughEnvCfg):
    """V54's terrain/observation/terminations with the stock Go2 reward table."""

    def __post_init__(self):
        super().__post_init__()

        # ==================================================================================
        # 1. Actuation: identical TG model, new growth clock (0.789 -> 1.0, then frozen).
        #    Every non-growth parameter repeats V51's block verbatim, because the action
        #    term's dynamics are part of what the pretrained actor was trained against.
        # ==================================================================================
        self.actions.joint_pos = V55TorqueActionCfg(
            asset_name="robot",
            joint_names=list(SATA_JOINT_NAMES),
            preserve_order=True,
            scale=1.0,           # action_scale is applied inside the TG model
            action_scale=5.0,
            torque_limit=23.5,
            activation_process=True,
            hill_model=True,
            # The fatigue buffer is *actuator state* (it is one of the 60 observation
            # channels), not a reward input in V55.  Keep it, or the observation would
            # silently change distribution against the pretrained actor.
            motor_fatigue=True,
            loss_rate=0.1,
            gs_start=0.7891,
            gs_ramp_steps=12000,
        )

        # ==================================================================================
        # 2. Rewards: IsaacLab Go2 rough, verbatim (velocity_env_cfg.py L231-262 overridden by
        #    go2/rough_env_cfg.py L52-61).  First drop every inherited term -- V52/V54 carry
        #    SATA's nine gs-coupled rewards plus six extras -- then re-attach eleven.
        # ==================================================================================
        for _name, _value in list(vars(self.rewards).items()):
            if isinstance(_value, RewardTermCfg):
                setattr(self.rewards, _name, None)

        _foot_contact = SceneEntityCfg(
            "feet_contact",
            body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            preserve_order=True,
        )

        # --- tracking (the only two positive, dense terms; ~2.25/s combined at full score)
        # std = sqrt(0.25) = 0.5, i.e. the official ``std`` parameter for the exp kernel.
        self.rewards.track_lin_vel_xy_exp = RewardTermCfg(
            func=mdp.track_lin_vel_xy_exp,
            weight=1.5,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )
        self.rewards.track_ang_vel_z_exp = RewardTermCfg(
            func=mdp.track_ang_vel_z_exp,
            weight=0.75,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )

        # --- penalties (identical weights to go2/rough_env_cfg.py)
        self.rewards.lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
        self.rewards.ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
        self.rewards.dof_torques_l2 = RewardTermCfg(func=mdp.joint_torques_l2, weight=-2e-4)
        self.rewards.dof_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
        self.rewards.action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)

        # --- gait shaping: stock 0.01 / 0.5 s, on the four feet (sensor built by V54)
        self.rewards.feet_air_time = RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.01,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": _foot_contact,
                "threshold": 0.5,
            },
        )

        # --- present at 0.0 in the official table, kept so the term list matches it exactly
        self.rewards.flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=0.0)
        self.rewards.dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=0.0)

        # --- off by decision (Go2 rough default).  Calf scraping on the 23 cm riser
        #     therefore costs nothing; this is one of the three "traps" noted in the module
        #     docstring and the first thing to turn back on if V55 learns to shin-climb.
        self.rewards.undesired_contacts = None


@configclass
class V55SataWarmstartIsaacRlRunnerCfg(V54SataPpoRoughRunnerCfg):
    """V54's PPO with the stock IsaacLab numbers -- nothing retuned for the warm start.

    lr 1e-3 / adaptive / desired_kl 0.01 / entropy 0.01 / init_noise_std 1.0 / [512, 256, 128]
    / elu are exactly ``UnitreeGo2RoughPPORunnerCfg`` (config/go2/agents/rsl_rl_ppo_cfg.py);
    V54 does not touch them either.  So V55 differs from V52/V54 only in the budget, the name
    and the warm start itself.  ``init_noise_std`` 1.0 is kept as is: ``--sata_init`` copies the
    actor weights only (pass ``--sata_init_std`` to also take SATA's std, min 1.48 / mean 2.31
    / max 3.10, as a single-variable ablation).

    The optimizer is NOT restored from the SATA checkpoint -- see load_sata_actor.py.
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 15000
        # A checkpoint every 1000 iterations (15 files x ~4.7 MB ~ 70 MB per run).
        self.save_interval = 1000
        self.experiment_name = "terrain_v55_sata_warmstart_isaac_rl"


gym.register(
    id="Isaac-Go2-V55-SataWarmstartIsaacRl-v0",
    entry_point="terrain_v55_sata_warmstart_isaac_rl.v55_growth:V55ControlFreqEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V55SataWarmstartIsaacRlEnvCfg",
        # Both keys point at the PPO runner so train.py works with either agent flag.
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V55SataWarmstartIsaacRlRunnerCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:V55SataWarmstartIsaacRlRunnerCfg",
    },
)
