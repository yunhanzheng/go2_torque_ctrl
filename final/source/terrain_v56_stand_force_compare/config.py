"""V56 - position control vs direct torque on a flat standing task, read through the GRF.

Baseline is IsaacLab's own ``UnitreeGo2FlatEnvCfg`` (Go2, plane, no height scan, 50 Hz,
48-D observation, stock Go2 reward table, stock PPO runner), *not* the V50/V54 chain.
Everything that differs from that baseline is listed here, nothing else is touched:

 1. ``commands.base_velocity`` ranges -> all zero, ``heading_command=False``.  The task is
    "stand still"; the stock ``track_lin_vel_xy_exp`` / ``track_ang_vel_z_exp`` terms then
    reward zero velocity, so no reward term has to be rewritten.
 2. no reward edits at all.  An earlier revision added ``base_height_l2`` (-2.0 against a
    target of 0.30 m) and set ``dof_pos_limits`` to -1.0.  Both are gone: the height term
    judged an *absolute* body height, which the spawn geometry does not define (the nominal
    joint angles leave the rear soles 9.2 mm above the front ones - see ``reference.py``), and
    a hand-picked target is exactly the kind of prior that made the position arm the winner
    before the comparison started.  ``dof_pos_limits`` is left at the official 0.0, and
    ``episode_length_s`` at the official 20 s.
 3. instead of a reward, "am I still standing?" is measured base-relatively and logged as
    ``Ref/*`` by ``metrics.py``: joint drift from the episode's own first frame, tilt from
    projected gravity, and the base-to-foot extension.  Ground-free, controller-free, identical
    for both arms.
 4. one extra sensor ``feet_forces`` (all robot bodies, 4 are addressed by name in
    ``metrics.py``) next to the stock ``contact_forces``, which stays untouched.
 5. the torque variant replaces the action term and zeroes the actuator PD gains, but keeps
    the stock DCMotor *envelope* (effort_limit 23.5 N.m, velocity_limit 30 rad/s); the position
    variant keeps ``JointPositionActionCfg(scale=0.25)`` and the same DCMotor untouched.
 6. four tasks: the two official-rate arms (``decimation = 4``, ``sim.dt = 0.005``, i.e. 50 Hz
    policy over a 200 Hz physics loop) and the bandwidth-matched pair (``decimation = 1``, 200 Hz
    policy) whose runners quadruple ``num_steps_per_env`` so an iteration still covers 0.48 s.

Why there are two pairs.  The one axis that cannot be equalised by configuration is the control
loop bandwidth: the position arm's PD is evaluated by the actuator inside *every* physics step
(200 Hz), while the torque arm's controller - the policy - runs once per policy step and its
torque is zero-order held for the rest of them.  At 50 Hz that is a 20 ms hold against a 5 ms
one; at 200 Hz both close at 5 ms and the difference disappears.

That equivalence is measured, not assumed.  Driving the torque path with the stock PD law
(kp 25 / kd 0.5) reproduces the position arm **digit for digit** at 200 Hz - dq_max 0.292 rad,
tilt 0.018 rad, d_ell -38.4 mm, CoP 6.6 mm, |tau|rms 3.95 N.m - while the same law at 50 Hz
lands at dq_max 0.263, tilt 0.010, d_ell -40.8 mm, CoP 10.2 mm (``diag_torque_hold.py``).  So the
200/200 pair is the controlled comparison (same information, same rate, same hold, same budget,
difficulty left free) and the 50/50 pair is the deployment-shaped appendix whose gap is bounded
by exactly those 20 ms.

Deliberately kept from the stock config: robot asset, terrain, physics material, domain
randomisation (friction 0.8/0.6, base mass -1..3 kg, spawn x/y +-0.5 m, yaw +-pi), noise
on the observations, termination set (time-out + base contact), 48-D observation layout,
control rate, and the whole PPO recipe of ``UnitreeGo2FlatPPORunnerCfg`` - 300 iterations,
``save_interval`` 50, three 128-unit hidden layers, KL-adaptive learning rate.  Neither arm
gets a longer budget or a different network: a torque policy that needs more iterations than
the baseline would be a result, not a setting.

The spawn pose itself is untouched (stock ``(0, 0, 0.4)``, default joint angles).  It was
measured (``diag_spawn_pose.py``) and found *not* to be a four-foot-level stance: with the
nominal angles the rear soles sit 9.2 mm above the front ones, so no pure z translation lets
all four feet touch.  Nothing is corrected for it - the reference frame is the first frame of
the episode itself (``reference.py``), which makes the offset part of the reference instead of
an error.

Consequence worth knowing when reading the numbers: ``add_base_mass`` is active, so the
denominator of ``Force/total_over_weight`` is the *nominal* weight and the ratio carries a
few percent of spread that both variants share.
"""
from __future__ import annotations

import gymnasium as gym

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
    impact force never enters ``Force/total_over_weight``.  Kept in seconds rather than steps so
    the gate keeps its meaning in wall-clock time if the control rate ever changes."""

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

        # --- 2. episodes and rewards stay exactly as IsaacLab ships them -----------------
        # ``episode_length_s`` 20 s, ``dof_pos_limits`` weight 0.0, no ``base_height_l2`` term.
        # See the module docstring: the height judgement moved to the base-relative ``Ref/*``
        # metrics, which do not need an absolute target the spawn geometry cannot define.

        # ``feet_air_time`` is already a no-op at zero command (the stock term gates on
        # |v_cmd| > 0.1), so it stays in the table untouched.

        # --- 3. force sensor for the metrics -----------------------------------------
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
        # four-quadrant torque-speed clip at velocity_limit=30 rad/s.  Only the *gains* go:
        # a non-zero stiffness would add a hidden position loop on top of the torque command.
        #
        # The envelope (effort_limit 23.5 N.m, saturation_effort 23.5 N.m, velocity_limit
        # 30 rad/s) deliberately stays stock.  It is actuator physics, not control law, and
        # the position variant runs into exactly the same clip on its PD output; widening it
        # here would give the torque variant a stronger motor than the baseline it is
        # compared against.  The 23.5 N.m ceiling is additionally enforced by the action
        # term's clamp, so a saturating action cannot exceed it either way.
        legs = self.scene.robot.actuators["base_legs"]
        legs.stiffness = 0.0
        legs.damping = 0.0


@configclass
class StandPosRunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """Stock flat PPO runner, only renamed so the two arms land in separate folders.

    ``max_iterations`` (300), ``save_interval`` (50), the network, the optimiser and the PPO
    hyper-parameters are left at the values IsaacLab ships, and both arms inherit the same
    ones.  ``train.py --max_iterations`` raises the budget from the command line for a longer
    run; the config itself never hands one arm more iterations than the other.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "v56_stand_pos"


@configclass
class StandTorqueRunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """Identical runner to :class:`StandPosRunnerCfg` - the algorithm is not a variable here."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "v56_stand_torque"


# --- the bandwidth-matched pair ---------------------------------------------------------------
# Everything above runs at the official rate.  The pair below runs at the physics rate instead,
# which is the only way to give both controllers the same loop bandwidth (see module docstring).

PHYSICS_RATE_STEPS = 96
"""``num_steps_per_env`` that keeps one iteration at 0.48 s of physical time at 200 Hz.

``num_steps_per_env`` counts *policy* steps: the stock 24 at 50 Hz is 0.48 s, while 24 steps at
200 Hz is only 0.12 s.  Quadrupling it keeps the iteration budget comparable in seconds rather
than in steps - without it the 200 Hz arms would train on a quarter of the trajectory per update
and the "same budget" claim would be false.
"""


@configclass
class StandPos200EnvCfg(StandPosEnvCfg):
    """Position arm with the policy loop at the physics rate (``decimation`` 4 -> 1)."""

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.sim.render_interval = 1


@configclass
class StandTorque200EnvCfg(StandTorqueEnvCfg):
    """Torque arm at the physics rate: 200 Hz policy, 200 Hz physics, 5 ms hold."""

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.sim.render_interval = 1


@configclass
class StandPos200RunnerCfg(StandPosRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = PHYSICS_RATE_STEPS
        self.experiment_name = "v56_stand_pos200"


@configclass
class StandTorque200RunnerCfg(StandTorqueRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = PHYSICS_RATE_STEPS
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
    id="Isaac-Go2-V56-StandPos200-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:StandPos200EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:StandPos200RunnerCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:StandPos200RunnerCfg",
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
    "PHYSICS_RATE_STEPS",
    "StandPos200EnvCfg",
    "StandPos200RunnerCfg",
    "StandPosEnvCfg",
    "StandPosRunnerCfg",
    "StandTorque200EnvCfg",
    "StandTorque200RunnerCfg",
    "StandTorqueEnvCfg",
    "StandTorqueRunnerCfg",
]
