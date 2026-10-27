"""V56 - position control vs direct torque on a flat standing task, read through the GRF.

Baseline is IsaacLab's own ``UnitreeGo2FlatEnvCfg`` (Go2, plane, no height scan, 50 Hz,
48-D observation, stock Go2 reward table, stock PPO runner), *not* the V50/V54 chain.
Everything that differs from that baseline is listed here, nothing else is touched:

 1. ``commands.base_velocity`` ranges -> all zero, ``heading_command=False``.  The task is
    "stand still"; the stock ``track_lin_vel_xy_exp`` / ``track_ang_vel_z_exp`` terms then
    reward zero velocity, so no reward term has to be rewritten.
 2. ``episode_length_s`` 20 s -> 5 s.  Standing does not need 20 s, and PPO cannot start
    aggregating episode statistics until the first episodes end.
 3. two reward weights: ``+base_height_l2`` (-2.0, the stock flat table has no height term
    and the policy would otherwise be free to crouch) and ``dof_pos_limits`` 0.0 -> -1.0
    (a torque policy that explores its way into a joint stop should feel it).
 4. one extra sensor ``feet_forces`` (all robot bodies, 4 are addressed by name in
    ``metrics.py``) next to the stock ``contact_forces``, which stays untouched.
 5. the torque variant replaces the action term and zeroes the actuator PD gains; the
    position variant keeps ``JointPositionActionCfg(scale=0.25)`` and the stock DCMotor.
 6. the 200 Hz ablation only changes ``decimation`` 4 -> 1.

Deliberately kept from the stock config: robot asset, terrain, physics material, domain
randomisation (friction 0.8/0.6, base mass -1..3 kg, spawn x/y +-0.5 m, yaw +-pi), noise
on the observations, termination set (time-out + base contact), 48-D observation layout,
and the PPO network/algorithm of ``UnitreeGo2FlatPPORunnerCfg``.  Only the iteration
budget is raised (300 -> 1000), because a torque policy has to discover a stance from
white noise.

Consequence worth knowing when reading the numbers: ``add_base_mass`` is active, so the
denominator of ``Force/total_over_weight`` is the *nominal* weight and the ratio carries a
few percent of spread that both variants share.
"""
from __future__ import annotations

import gymnasium as gym

from isaaclab.managers import RewardTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

from .actions import DirectTorqueActionCfg
from .metrics import FOOT_NAMES


@configclass
class StandTaskEnvCfg(UnitreeGo2FlatEnvCfg):
    """Stock Go2 flat task turned into "stand still on the plane"."""

    metrics_warmup_s: float = 0.25
    """Seconds ignored per environment after a reset, converted to policy steps by the
    environment.  The robot spawns at z = 0.4 m and touches down inside that window, so the
    impact force never enters ``Force/total_over_weight``.  Expressed in seconds because the
    200 Hz ablation has four times as many steps per second."""

    def __post_init__(self):
        super().__post_init__()

        # --- 1. zero command, no heading randomisation -------------------------------
        velocities = self.commands.base_velocity
        velocities.heading_command = False
        velocities.rel_heading_envs = 0.0
        velocities.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        )

        # --- 2. shorter episodes -----------------------------------------------------
        self.episode_length_s = 5.0

        # --- 3. the only two reward edits --------------------------------------------
        self.rewards.base_height_l2 = RewardTermCfg(
            func=mdp.base_height_l2,
            weight=-2.0,
            params={"target_height": 0.30},
        )
        self.rewards.dof_pos_limits.weight = -1.0
        # ``feet_air_time`` is already a no-op at zero command (the stock term gates on
        # |v_cmd| > 0.1), so it stays in the table untouched.

        # --- 4. force sensor for the metrics ----------------------------------------
        # The stock ``contact_forces`` sensor is left as is: the base-contact termination
        # and ``feet_air_time`` resolve their own body filters against it.  ``update_period``
        # is left at its default 0.0 (= refresh on every physics step), so the value that
        # ``metrics.py`` reads is the last substep of the policy step, not a stale sample.
        self.scene.feet_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=0,
            track_air_time=False,
            debug_vis=False,
        )


@configclass
class StandPosEnvCfg(StandTaskEnvCfg):
    """Position control (reference): stock ``JointPositionActionCfg(scale=0.25)`` + PD."""


@configclass
class StandTorqueEnvCfg(StandTaskEnvCfg):
    """Direct torque: ``tau = clamp(action_scale * a, +-23.5 N.m)``, no actuator PD."""

    action_scale: float = 10.0
    """N.m per unit of network output; see ``actions.py`` for how to calibrate it."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos = DirectTorqueActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=1.0,  # the real scaling lives in ``action_scale`` inside the term
            action_scale=self.action_scale,
            torque_limit=23.5,
        )

        # The stock Go2 actuator is a DCMotor with stiffness 25 / damping 0.5 and a
        # four-quadrant torque-speed clip at velocity_limit=30 rad/s.  For "the network
        # output is the torque" both have to go: the PD gain would add a hidden position
        # loop, and the clip would shave ~17 % of the torque at 5 rad/s.  The torque
        # ceiling is enforced by the action term's clamp instead.
        legs = self.scene.robot.actuators["base_legs"]
        legs.stiffness = 0.0
        legs.damping = 0.0
        legs.effort_limit = 1.0e3
        legs.saturation_effort = 1.0e3
        legs.velocity_limit = 1.0e3


@configclass
class StandTorque200EnvCfg(StandTorqueEnvCfg):
    """Bandwidth ablation: everything of the torque variant, but 200 Hz instead of 50 Hz.

    Only ``decimation`` changes; ``metrics_warmup_s`` is in seconds, so the force metrics stay
    comparable across the two control rates.
    """

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.sim.render_interval = 1


@configclass
class StandPosRunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1000
        self.save_interval = 100
        self.experiment_name = "v56_stand_pos"


@configclass
class StandTorqueRunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1000
        self.save_interval = 100
        self.experiment_name = "v56_stand_torque"


@configclass
class StandTorque200RunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1000
        self.save_interval = 100
        self.experiment_name = "v56_stand_torque200"


_ENTRY_POINT = "terrain_v56_stand_force_compare.metrics:StandForceEnv"

gym.register(
    id="Isaac-Go2-V56-StandPos-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:StandPosEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:StandPosRunnerCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:StandPosRunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2-V56-StandTorque-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:StandTorqueEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:StandTorqueRunnerCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:StandTorqueRunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2-V56-StandTorque200-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:StandTorque200EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:StandTorque200RunnerCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:StandTorque200RunnerCfg",
    },
)

__all__ = [
    "FOOT_NAMES",
    "StandPosEnvCfg",
    "StandPosRunnerCfg",
    "StandTorque200EnvCfg",
    "StandTorque200RunnerCfg",
    "StandTorqueEnvCfg",
    "StandTorqueRunnerCfg",
]
