"""SATA go2_torque dynamics ported into IsaacLab (manager-based).

Source of truth: /home/ubuntu22/data/SATA/legged_gym/legged_gym/envs/go2/go2_torque/

This module replicates the parts of SATA that define the robot's *dynamics and
task interface*, while the surrounding V51 experiment keeps:
  - SAC algorithm (unchanged)
  - IsaacLab / IsaacSim framework (unchanged)
  - V51 reward configuration (unchanged)
  - V51 IsaacLab native terrain + curriculum (unchanged)

Ported from SATA:
  1. Gompertz growth curriculum: general_scale = exp(-exp(-k*(step - x0))).
     Drives command ranges and the torque-limit ramp exactly like SATA.
  2. TG torque model: action_scale -> tanh muscle activation (1st-order filter)
     -> hill model (velocity-dependent) -> motor fatigue accumulation.
  3. Command ranges scaled by general_scale (SATA _resample_commands).
  4. 60-D SATA observation (real motor fatigue, no gait clock) with per-channel
     uniform noise and 10% action/observation loss.

Control frequency (replicated):
  SATA ramps its *control frequency* 100 -> 200 Hz through the growth curve by
  changing how many physics substeps run per policy step (go2_torque.py:201-210).
  ``SataControlFreqEnv`` below varies IsaacLab's ``decimation`` per step while
  carrying the remainder, reproducing SATA's average policy frequency
  (112.8 Hz at gs~0.128 up to 200 Hz at gs=1).

Joint ordering everywhere below is SATA's: FL/FR/RL/RR, each hip->thigh->calf.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

# SATA leg ordering: FL, FR, RL, RR; within each leg hip -> thigh -> calf.
SATA_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

# Hill-model velocity normalizers, taken from go2_torque.urdf joint ``velocity``
# limits (legged_robot reads props["velocity"] into dof_vel_limits): hip/thigh
# = 30.1, calf = 20.07, in SATA_JOINT_NAMES order.  SATA also overrides the
# per-joint torque limit to a constant 23.5 Nm (go2_torque.py:315), which is why
# ``torque_limit`` below is a scalar rather than the URDF effort (23.7 / 35.55).
# We hardcode these instead of reading ``asset.data.joint_vel_limits`` because the
# IsaacLab Go2 USD may report different velocity limits, which would silently
# change the hill-model torque scaling.
SATA_DOF_VEL_LIMITS = (
    30.1, 30.1, 20.07,   # FL: hip, thigh, calf
    30.1, 30.1, 20.07,   # FR
    30.1, 30.1, 20.07,   # RL
    30.1, 30.1, 20.07,   # RR
)


def sata_growth_scale(env, k: float = 3e-5, x0: float = 24000.0) -> float:
    """SATA Gompertz growth curve evaluated at the global step counter.

    SATA increments ``step_count`` once per *physics substep* (it is called inside
    ``_compute_torques``, go2_torque.py:220).  The action term accumulates the actual number
    of substeps in ``_sata_physics_steps`` (it is bumped in ``_update_growth``); using the raw
    policy counter would make the curriculum run ~1.8x slower than SATA.

    Play mode (``_sata_freeze_step_count >= 0``) reproduces
    ``GO2Torque._update_growth_scale`` (go2_torque.py:179-181): when
    ``control_type == "T"`` or ``test.use_test`` the counter is pinned to
    ``num_steps_per_env * test.checkpoint`` (= 24 * 3000 = 72000), which makes
    ``general_scale`` a constant 0.789 instead of restarting from 0.
    """
    frozen = int(getattr(env, "_sata_freeze_step_count", -1))
    if frozen >= 0:
        step_count = float(frozen)
    else:
        step_count = float(getattr(env, "_sata_physics_steps", env.common_step_counter))
    return math.exp(-math.exp(-k * (step_count - x0)))


##
# 1. TG torque action term.
##


class SataTorqueAction(JointAction):
    """SATA 'TG' (torque-with-growth) actuation model.

    Mirrors GO2Torque._compute_torques:
        actions_scaled = raw_action * action_scale
        torque_limits  = current_torque_limit_scale * 23.5
        activation     = tanh(actions_scaled / torque_limits)   (1st-order filtered)
        torque         = activation * torque_limits * (1 - sign(activation)*vel/vel_limit)
        motor_fatigue += |torque| * dt ;  motor_fatigue *= 0.9

    Note: SATA's rear-leg scale ``r_leg_scaled`` is identically 1.0
    (max_rear == start_rear == 1.0), so the rear-leg branch is a no-op and is
    omitted here.
    """

    cfg: "SataTorqueActionCfg"

    def __init__(self, cfg: "SataTorqueActionCfg", env) -> None:
        super().__init__(cfg, env)
        n = self._num_joints
        dev = self.device
        # physics dt = policy dt / decimation (fatigue integration step)
        self._sim_dt = self._env.step_dt / self._env.cfg.decimation
        # state buffers (SATA: motor_fatigue / activation_sign / torques)
        self.motor_fatigue = torch.zeros(self.num_envs, n, device=dev)
        self.activation_sign = torch.zeros(self.num_envs, n, device=dev)
        self.torques = torch.zeros(self.num_envs, n, device=dev)
        self.torques_action = torch.zeros(self.num_envs, n, device=dev)
        # constant per-joint torque limit (SATA: 23.5 Nm)
        self._torque_limits = torch.full((n,), float(cfg.torque_limit), device=dev)
        # constant per-joint hill-model velocity normalizer (SATA go2_torque.urdf)
        self._dof_vel_limits = torch.tensor(SATA_DOF_VEL_LIMITS, device=dev, dtype=torch.float).clamp(min=1e-6).unsqueeze(0)
        # runtime scalars
        self.general_scale = 0.0
        self.current_torque_limit_scale = cfg.start_torque_scale

    def _update_growth(self) -> None:
        # SATA increments ``step_count`` inside ``_update_growth_scale`` (go2_torque.py:177-185),
        # which ``_compute_torques`` calls once per physics substep.  IsaacLab calls
        # ``apply_action()`` once per substep as well (manager_based_rl_env.py:185), so the clock
        # advances at exactly the same place instead of once per policy step.
        self._env._sata_physics_steps += 1
        self.general_scale = sata_growth_scale(self._env, self.cfg.growth_k, self.cfg.growth_x0)
        gs = self.general_scale
        self.current_torque_limit_scale = gs * (self.cfg.max_torque_scale - self.cfg.start_torque_scale) \
            + self.cfg.start_torque_scale

    def apply_actions(self) -> None:
        self._update_growth()
        # scale raw network output by action_scale (=5), matching SATA
        self.torques_action = self._raw_actions * self.cfg.action_scale
        limits = (self.current_torque_limit_scale * self._torque_limits).unsqueeze(0)  # [1, n]

        # --- muscle activation (tanh) with 1st-order filter ---
        if self.cfg.activation_process:
            current = torch.tanh(self.torques_action / limits)
            activation = (current - self.activation_sign) * 0.6 + self.activation_sign
        else:
            activation = self.torques_action / limits

        # --- action loss: with prob loss_rate keep previous activation ---
        if self.cfg.loss_rate > 0.0:
            keep = torch.rand(self.num_envs, 1, device=self.device) > self.cfg.loss_rate
            self.activation_sign = torch.where(keep, activation, self.activation_sign)
        else:
            self.activation_sign = activation

        # --- hill model: torque drops with joint velocity ---
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]
        vel_limits = self._dof_vel_limits  # SATA URDF values, not the USD's
        if self.cfg.hill_model:
            self.torques = self.activation_sign * limits * (
                1.0 - torch.sign(self.activation_sign) * joint_vel / vel_limits
            )
        else:
            self.torques = self.activation_sign * limits

        # --- motor fatigue accumulation ---
        if self.cfg.motor_fatigue:
            self.motor_fatigue += torch.abs(self.torques) * self._sim_dt
            self.motor_fatigue *= 0.9
        else:
            self.motor_fatigue.zero_()

        # Neutralize the DCMotor PD path.  Go2's actuator is a DCMotorCfg with
        # stiffness=25/damping=0.5 and IdealPDActuator.compute returns
        # stiffness*error_pos + damping*error_vel + effort.  SATA applies pure
        # torque (set_dof_actuation_force), so drive the pos/vel targets to the
        # current state => error_pos=error_vel=0 => only our TG torque is applied.
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._asset.set_joint_position_target(joint_pos, joint_ids=self._joint_ids)
        self._asset.set_joint_velocity_target(joint_vel, joint_ids=self._joint_ids)
        self._asset.set_joint_effort_target(self.torques, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.activation_sign[ids] = 0.0
        if self.cfg.motor_fatigue:
            hi = 0.2 * self.general_scale  # SATA: uniform(0, 0.2*general_scale)
            if hi > 0.0:
                self.motor_fatigue[ids] = torch.rand_like(self.motor_fatigue[ids]) * hi
            else:
                self.motor_fatigue[ids] = 0.0
        else:
            self.motor_fatigue[ids] = 0.0


@configclass
class SataTorqueActionCfg(JointActionCfg):
    """Configuration for :class:`SataTorqueAction`."""

    class_type: type[ActionTerm] = SataTorqueAction

    # NOTE: joint_names must be SATA order with preserve_order=True so that the
    # internal torque/fatigue buffers line up with the 60-D observation.
    action_scale: float = 5.0
    """SATA control.action_scale."""
    torque_limit: float = 23.5
    """SATA per-joint torque limit (Nm)."""
    activation_process: bool = True
    """SATA control.activation_process (tanh + 1st-order filter)."""
    hill_model: bool = True
    """SATA control.hill_model (velocity-dependent torque)."""
    motor_fatigue: bool = True
    """SATA control.motor_fatigue."""
    loss_rate: float = 0.1
    """SATA domain_rand.loss_rate (fraction of steps that keep previous activation)."""
    # growth curriculum (SATA cfg.growth)
    start_torque_scale: float = 0.3
    max_torque_scale: float = 1.0
    growth_k: float = 3e-5
    growth_x0: float = 24000.0


##
# 2. Velocity command scaled by the growth curve.
##


class SataVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand whose ranges grow with SATA's general_scale.

    Reproduces GO2Torque._resample_commands:
        x:   uniform(max(center - half*gs, lo), min(center + half*gs, hi))
        y:   uniform(lo_y*gs, hi_y*gs)
        yaw: uniform(lo_yaw*gs, hi_yaw*gs)
    With gs->0 at start this collapses to x=0.5 m/s, y=0, yaw=0 (the fixed
    straight-walk command V51 used before); with gs->1 it reaches the full
    SATA ranges.
    """

    cfg: "SataVelocityCommandCfg"

    def compute(self, dt: float) -> None:
        """Resample as usual, then pin ``cfg.test.vel`` when ``use_test`` is set.

        Mirrors GO2Torque._post_physics_step_callback (go2_torque.py:320-329):
        ``super()._post_physics_step_callback()`` resamples first and then
        ``self.commands[:] = self.cfg.test.vel`` overwrites every env, and the
        ``heading_command`` block runs **after** that overwrite while play.py sets
        ``heading_command = True``.  So in play mode the yaw command is not
        ``test_vel[2]`` but the heading servo ``0.5 * wrap_to_pi(target - heading)``
        clipped to ``+-ranges.ang_vel_z * general_scale``.
        """
        super().compute(dt)
        if self.cfg.use_test:
            self.vel_command_b[:, 0] = self.cfg.test_vel[0]
            self.vel_command_b[:, 1] = self.cfg.test_vel[1]
            self.vel_command_b[:, 2] = self.cfg.test_vel[2]
            if self.cfg.heading_command:
                gs = sata_growth_scale(self._env, self.cfg.growth_k, self.cfg.growth_x0)
                heading_error = wrap_to_pi(
                    self.cfg.test_vel[3] - self._env.scene["robot"].data.heading_w
                )
                lo, hi = self.cfg.ranges.ang_vel_z
                self.vel_command_b[:, 2] = torch.clip(
                    0.5 * heading_error, min=lo * gs, max=hi * gs
                )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        gs = sata_growth_scale(self._env, self.cfg.growth_k, self.cfg.growth_x0)
        rng = self.cfg.ranges
        r = torch.empty(len(env_ids), device=self.device)

        x_center = 0.5 * (rng.lin_vel_x[1] + rng.lin_vel_x[0])
        x_half = 0.5 * (rng.lin_vel_x[1] - rng.lin_vel_x[0])
        # SATA uses x_cmd_diff = 2.0 around the centre => half-width = 2*x_half*gs.
        x_lo = max(x_center - 2.0 * x_half * gs, rng.lin_vel_x[0])
        x_hi = min(x_center + 2.0 * x_half * gs, rng.lin_vel_x[1])
        if x_hi < x_lo:
            x_lo, x_hi = x_hi, x_lo
        self.vel_command_b[env_ids, 0] = r.uniform_(x_lo, x_hi)
        self.vel_command_b[env_ids, 1] = r.uniform_(rng.lin_vel_y[0] * gs, rng.lin_vel_y[1] * gs)
        self.vel_command_b[env_ids, 2] = r.uniform_(rng.ang_vel_z[0] * gs, rng.ang_vel_z[1] * gs)

        # standing envs (rel_standing_envs is 0 for SATA go2_torque)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs


@configclass
class SataVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`SataVelocityCommand`."""

    class_type: type = SataVelocityCommand

    growth_k: float = 3e-5
    growth_x0: float = 24000.0

    # SATA cfg.test (go2_torque_config.py:168-171).  When ``use_test`` is True the
    # policy receives the fixed ``test_vel`` every step, exactly like play.py.
    use_test: bool = False
    """SATA cfg.test.use_test."""
    test_vel: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """SATA cfg.test.vel = [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]."""


##
# 3. Control-frequency ramp (SATA go2_torque.py:199-210).
##


class SataControlFreqEnv(ManagerBasedRLEnv):
    """``ManagerBasedRLEnv`` whose policy frequency follows SATA's growth curve.

    SATA's loop (``GO2Torque.step``, go2_torque.py:199-212) simulates physics substeps
    until one control period has elapsed, then carries the remainder::

        while current_dt * current_freq < 1:    # simulate one 200 Hz substep
            current_dt += dt
        current_dt %= 1 / current_freq

    IsaacLab's ``decimation`` is an integer fixed at construction, so the equivalent is to
    run the same loop numerically and hand the resulting substep count to IsaacLab::

        n = 0
        while accum * current_freq < 1:   # == SATA's ``while current_dt * current_freq < 1``
            accum += physics_dt
            n += 1
        accum %= 1 / current_freq
        cfg.decimation = n

    ``accum`` plays the role of SATA's ``current_dt``, so the per-step decimation sequence
    (2,1,2,1,1,... at 150 Hz -> 1.335 substeps/step) is identical, not just the long-run
    average frequency.

    Reward/termination cadence matches SATA too: ``GO2Torque.step`` calls
    ``post_physics_step()`` *inside* the substep loop, so ``check_termination``,
    ``compute_reward``, ``reset_idx`` and ``compute_observations`` all run once per 5 ms
    physics substep while the torque is held for the whole control period.
    :meth:`step` therefore does not delegate to ``ManagerBasedRLEnv.step`` (which evaluates
    reward/termination once per control period), see :meth:`_sata_substep_step`.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        # SATA's time-out counts physics substeps (legged_robot.py:114), not policy steps.
        # Created before ``super().__init__`` because the first ``_reset_idx`` runs there.
        self._sata_env_substeps = torch.zeros(
            cfg.scene.num_envs, dtype=torch.long, device=cfg.sim.device
        )
        super().__init__(cfg, render_mode, **kwargs)
        self._sata_ctrl_accum = 0.0
        # SATA's step_count counts physics substeps, not policy steps.
        self._sata_physics_steps = 0
        # >= 0 pins the growth curve (SATA play: num_steps_per_env * test.checkpoint).
        self._sata_freeze_step_count = -1

    def step(self, action):
        gs = sata_growth_scale(self, 3e-5, 24000.0)
        freq = 100.0 + 100.0 * gs
        # GO2Torque.step (go2_torque.py:201-210), verbatim: simulate 200 Hz substeps until
        # one control period is covered, then keep the overshoot as the next period's head
        # start.  Verified against SATA's loop: identical n sequence at 100/120/133/150/
        # 178.9/199/200 Hz (2,2,1,... / 2,1,2,1,1,... / 1,1,1,...).
        n = 0
        while self._sata_ctrl_accum * freq < 1.0:
            self._sata_ctrl_accum += self.physics_dt
            n += 1
            if n > 8:  # unreachable (<= 2 substeps at 100 Hz); guards a NaN growth scale
                break
        n = max(1, n)
        self._sata_ctrl_accum %= 1.0 / freq
        self.cfg.decimation = n
        return self._sata_substep_step(action, n)

    def _sata_substep_step(self, action, n: int):
        """``ManagerBasedRLEnv.step`` with SATA's post-physics step at 200 Hz.

        SATA (``go2_torque.py:199-210``) runs the whole post-processing block per physics
        substep: ``episode_length_buf += 1`` -> ``check_termination`` -> ``compute_reward``
        -> ``reset_idx`` -> ``compute_observations``.  The policy's torque is held across
        the 1-2 substeps of a control period.  Consequences copied here:

        * the reward is sampled ``n`` times with ``dt = physics_dt`` and summed, instead of
          one sample scaled by ``step_dt`` (identical for linear terms, ~1% for the
          exponential ones, and doubles the sampling rate of ``dof_acc``);
        * terminations and resets can happen in the middle of a control period, so the
          remaining substeps already run on the freshly reset state.

        Deliberate difference: SATA's ``check_termination`` zeroes ``reset_buf`` on every
        substep, which erases the flag written by ``reset_idx`` and therefore hides
        mid-period time-outs from the runner (their ``Train/mean_episode_length`` exceeds
        the 2000-substep cap).  The masks are OR-ed here instead so PPO still sees every
        episode end.
        """
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # SATA bumps its counter before the substep's termination check; IsaacLab's counter
        # is in policy steps, so one increment per control period is the equivalent.
        self.episode_length_buf += 1
        self.common_step_counter += 1

        rew = torch.zeros(self.num_envs, device=self.device)
        reset_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terminated_any = torch.zeros_like(reset_any)
        timeout_any = torch.zeros_like(reset_any)

        for _ in range(n):
            self._sim_step_counter += 1
            # set actions into buffers (the torque model advances every substep)
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

            # ---- SATA post_physics_step(), once per physics substep (legged_robot.py:114-136)
            self._sata_env_substeps += 1
            dones = self.termination_manager.compute()
            rew += self.reward_manager.compute(dt=self.physics_dt)
            reset_any |= dones
            terminated_any |= self.termination_manager.terminated
            timeout_any |= self.termination_manager.time_outs
            reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self.recorder_manager.record_pre_reset(reset_env_ids)
                self._reset_idx(reset_env_ids)
                if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                    for _ in range(self.cfg.num_rerenders_on_reset):
                        self.sim.render()
                self.recorder_manager.record_post_reset(reset_env_ids)

        self.reward_buf = rew
        self.reset_buf = reset_any
        self.reset_terminated = terminated_any
        self.reset_time_outs = timeout_any

        # -- update command (SATA's _post_physics_step_callback, once per control period)
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)
        if len(self.recorder_manager.active_terms) > 0:
            self.recorder_manager.record_post_step()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids):
        if env_ids is None:
            self._sata_env_substeps[:] = 0
        else:
            self._sata_env_substeps[env_ids] = 0
        super()._reset_idx(env_ids)


def sata_time_out(env, episode_length_s: float | None = None) -> torch.Tensor:
    """SATA ``time_out_buf``: ``episode_length_buf > max_episode_length``.

    ``legged_robot.check_termination`` (legged_robot.py:142-148) compares the step counter
    with ``max_episode_length = ceil(episode_length_s / dt)`` (legged_robot.py:791), and
    that counter is incremented once per **physics substep** (legged_robot.py:114).
    IsaacLab's ``episode_length_buf`` counts policy steps, so the substep counter kept by
    :class:`SataControlFreqEnv` is used instead: an episode lasts 10 s of simulated time no
    matter what decimation the growth curve currently produces.
    """
    seconds = env.cfg.episode_length_s if episode_length_s is None else episode_length_s
    limit = int(math.ceil(seconds / env.physics_dt))
    return env._sata_env_substeps > limit


##
# 4. 60-D SATA observation (real fatigue) + per-channel noise + obs loss.
##


def sata_observation_60d(env) -> torch.Tensor:
    """SATA go2_torque compute_observations, exact 60-D order.

    [ base_lin_vel*2 | base_ang_vel*0.25 | gravity | dof_pos_rel | dof_vel*0.05
      | command*(2,2,0.25) | torques | motor_fatigue ]  = 3+3+3+12+12+3+12+12 = 60

    ``torques`` and ``motor_fatigue`` are the live TG-model buffers from
    :class:`SataTorqueAction` (SATA joint order), not IsaacLab placeholders.
    """
    robot = env.scene["robot"]
    command = env.command_manager.get_command("base_velocity")
    joint_ids, _ = robot.find_joints(list(SATA_JOINT_NAMES), preserve_order=True)
    joint_pos = robot.data.joint_pos[:, joint_ids]
    default_joint_pos = robot.data.default_joint_pos[:, joint_ids]
    joint_vel = robot.data.joint_vel[:, joint_ids]

    action_term = env.action_manager.get_term("joint_pos")
    torques = action_term.torques
    fatigue = action_term.motor_fatigue

    return torch.cat(
        (
            robot.data.root_lin_vel_b * 2.0,
            robot.data.root_ang_vel_b * 0.25,
            robot.data.projected_gravity_b,
            (joint_pos - default_joint_pos),
            joint_vel * 0.05,
            command * command.new_tensor((2.0, 2.0, 0.25)),
            torques,
            fatigue,
        ),
        dim=-1,
    )


def sata_observation_60d_noisy(
    env, loss_rate: float = 0.1, noise_level: float = 1.5, add_noise: bool = True
) -> torch.Tensor:
    """60-D SATA observation with per-channel uniform noise and observation loss.

    Per-channel noise reproduces GO2Torque._get_noise_scale_vec (noise_scales *
    noise_level * obs_scale).  Command / torque channels get zero noise, exactly
    as in SATA.  Observation loss keeps the previous frame with prob ``loss_rate``.
    ``add_noise=False`` mirrors SATA's play.py (``cfg.noise.add_noise = False``).
    """
    obs = sata_observation_60d(env)

    # per-channel noise magnitudes (SATA noise_scales x noise_level x obs_scales)
    nv = torch.zeros(60, device=obs.device)
    nv[0:3] = 0.1 * noise_level * 2.0      # lin_vel:  scale 0.1, obs_scale 2.0
    nv[3:6] = 0.2 * noise_level * 0.25     # ang_vel:  scale 0.2, obs_scale 0.25
    nv[6:9] = 0.2 * noise_level            # gravity:  scale 0.2
    nv[9:21] = 0.01 * noise_level * 1.0    # dof_pos:  scale 0.01, obs_scale 1.0
    nv[21:33] = 1.5 * noise_level * 0.05   # dof_vel:  scale 1.5, obs_scale 0.05
    nv[33:36] = 0.0                        # command:  no noise
    nv[36:48] = 0.0                        # torques:  no noise
    nv[48:60] = 0.5 * noise_level / 10.0   # fatigue:  scale 0.5 / 10
    if add_noise:
        obs = obs + (2.0 * torch.rand_like(obs) - 1.0) * nv

    # observation loss: with prob loss_rate keep the previous frame
    if loss_rate > 0.0:
        prev = getattr(env, "_sata_prev_obs", None)
        if prev is None or prev.shape != obs.shape:
            prev = obs
        keep = torch.rand(env.num_envs, 1, device=obs.device) > loss_rate
        obs = torch.where(keep, obs, prev)
        env._sata_prev_obs = obs.clone()
    return obs
