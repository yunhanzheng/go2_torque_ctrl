"""Ground-reaction-force metrics for the position-vs-torque standing comparison.

Everything is computed inside the environment, once per policy step, and published through
``extras["log"]``.  rsl_rl's logger appends that dict every step (``logger.py:106``) and
averages every key over one iteration, so wandb/TensorBoard receive one scalar per key per
iteration without touching the third-party stack.  The dict is *merged* into the one the
managers already put there, so ``Episode_Reward/*`` and ``Episode_Termination/*`` survive.

Design notes
------------
* The four feet are addressed by name (``FOOT_NAMES``), never by USD ordering, so the
  sensor and the robot can reorder their bodies independently.
* Statistics are warm-up gated per environment instead of globally: right after a reset the
  robot is still falling onto the plane (UNITREE_GO2_CFG spawns at z = 0.4 m) and its
  impact force would otherwise bias ``Force/total_over_weight`` upward.
* No GPU->CPU synchronisation and no data-dependent branch: the warm-up mask is a float
  weight, so the kernels stay on the device.
"""
from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

FOOT_NAMES: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
"""Force-metric body order: front-left, front-right, rear-left, rear-right."""

CONTACT_THRESHOLD_N: float = 5.0
"""A foot counts as loaded above this normal force (N) - 14 % of its share of weight."""

TORQUE_LIMIT_NM: float = 23.5
"""Torque used for the saturation metric.  Identical for both controllers: it is Go2's
``DCMotorCfg.effort_limit``, the physical ceiling the position controller runs into, and
the explicit clamp of :class:`DirectTorqueAction`."""


def foot_ids(env) -> tuple[list[int], list[int]]:
    """Sensor and robot-body indices of the four feet, resolved by name and cached on ``env``.

    Shared by the training metrics, the force evaluation and the PD identification so that
    all three read the same number for "the force under FL_foot".
    """
    cached = getattr(env, "_v56_foot_ids", None)
    if cached is not None:
        return cached

    sensor = env.scene.sensors.get("feet_forces")
    if sensor is None:
        raise RuntimeError("V56 needs the 'feet_forces' contact sensor, see config.py.")
    sensor_bodies = list(sensor.body_names)
    missing = [name for name in FOOT_NAMES if name not in sensor_bodies]
    if missing:
        raise RuntimeError(f"'feet_forces' does not track {missing}; it tracks {sensor_bodies}.")

    robot_bodies = list(env.scene["robot"].data.body_names)
    missing = [name for name in FOOT_NAMES if name not in robot_bodies]
    if missing:
        raise RuntimeError(f"the robot has no feet named {missing}; it has {robot_bodies}.")

    ids = (
        [sensor_bodies.index(name) for name in FOOT_NAMES],
        [robot_bodies.index(name) for name in FOOT_NAMES],
    )
    setattr(env, "_v56_foot_ids", ids)
    return ids


def foot_normal_forces(env) -> torch.Tensor:
    """``[num_envs, 4]`` compression-only normal force, ordered FL / FR / RL / RR.

    A foot being *pulled* (adhesion) is a solver artefact, not a support force, hence the
    lower clamp.
    """
    sensor_ids, _ = foot_ids(env)
    return env.scene.sensors["feet_forces"].data.net_forces_w[:, sensor_ids, 2].clamp(min=0.0)


class StandForceEnv(ManagerBasedRLEnv):
    """``ManagerBasedRLEnv`` that logs four-foot ground reaction forces and CoP.

    Registered as the entry point of every V56 task, so the standing task and the metric
    set are identical for the position-controlled and the torque-controlled variant.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        self._setup_force_metrics()

    def _setup_force_metrics(self) -> None:
        self._robot_foot_ids = foot_ids(self)[1]

        # Weight reference for the "does the ground carry the robot?" check.  Read once:
        # the add_base_mass event randomises the base link, and this metric is meant to be
        # a drift detector, not a mass estimator.
        robot = self.scene["robot"]
        self._weight_n = float(robot.root_physx_view.get_masses()[0].sum()) * 9.81
        # Warm-up in policy steps, derived from a duration so that the 200 Hz ablation (four
        # times the steps per second) gates exactly the same amount of physical time.
        warmup_s = float(getattr(self.cfg, "metrics_warmup_s", 0.25))
        self._warmup_steps = int(round(warmup_s / self.step_dt))
        self._prev_cop_xy: torch.Tensor | None = None

    def step(self, action):
        obs, rewards, terminated, truncated, extras = super().step(action)
        log = self._compute_metrics()
        if log is not None:
            # Merge, never replace: on a step with resets ``extras["log"]`` was just refilled
            # by ``_reset_idx`` with the ``Episode_Reward/*`` / ``Episode_Termination/*`` of the
            # finished episodes, and rsl_rl's logger iterates over the keys of the *first* step
            # of the iteration only - dropping them would silently remove those curves.
            extras.setdefault("log", {}).update(log)
        return obs, rewards, terminated, truncated, extras

    def _compute_metrics(self) -> dict[str, torch.Tensor] | None:
        robot = self.scene["robot"]

        force_z = foot_normal_forces(self)  # [N, 4], compression only
        total = force_z.sum(dim=1)

        foot_xy = robot.data.body_pos_w[:, self._robot_foot_ids, :2]
        cop_xy = (force_z.unsqueeze(-1) * foot_xy).sum(dim=1) / total.clamp(min=1e-3).unsqueeze(-1)
        base_xy = robot.data.root_pos_w[:, :2]
        torque = robot.data.applied_torque

        # Warm-up weight, per environment: 0 while the robot is still landing after a reset.
        # An all-zero mask (the first steps of a fresh run, or a run in which every episode just
        # restarted) means "nothing to report yet": the keys are left out for that step and the
        # previous value is repeated, exactly like any other metric that is not recomputed.
        # Emitting zeros instead would drag the curve towards 0 by the warm-up fraction.
        # The single device->host read costs microseconds against a multi-millisecond step.
        w = (self.episode_length_buf >= self._warmup_steps).float()
        w_sum = w.sum()
        if float(w_sum) == 0.0:
            self._prev_cop_xy = None
            return None
        total_safe = total.clamp(min=1e-3)

        log = {
            # --- load level ---
            "Force/total_N": (total * w).sum() / w_sum,
            "Force/total_over_weight": (total / self._weight_n * w).sum() / w_sum,
            "Force/FL_N": (force_z[:, 0] * w).sum() / w_sum,
            "Force/FR_N": (force_z[:, 1] * w).sum() / w_sum,
            "Force/RL_N": (force_z[:, 2] * w).sum() / w_sum,
            "Force/RR_N": (force_z[:, 3] * w).sum() / w_sum,
            # --- how evenly is it shared ---
            "Force/load_std_frac": (force_z.std(dim=1) / total_safe * w).sum() / w_sum,
            "Force/asym_LR": (
                (force_z[:, 0] + force_z[:, 2] - force_z[:, 1] - force_z[:, 3]).abs() / total_safe * w
            ).sum()
            / w_sum,
            "Force/asym_FB": (
                (force_z[:, 0] + force_z[:, 1] - force_z[:, 2] - force_z[:, 3]).abs() / total_safe * w
            ).sum()
            / w_sum,
            "Force/front_share": ((force_z[:, 0] + force_z[:, 1]) / total_safe * w).sum() / w_sum,
            # --- contact timing ---
            "Contact/duty_foot": ((force_z > CONTACT_THRESHOLD_N).float().mean(dim=1) * w).sum() / w_sum,
            "Contact/duty_all4": ((force_z > CONTACT_THRESHOLD_N).all(dim=1).float() * w).sum() / w_sum,
            # --- centre of pressure (approximate: foot origins, not contact patches) ---
            "CoP/offset_m": ((cop_xy - base_xy).norm(dim=-1) * w).sum() / w_sum,
            # --- actuation, comparable across controllers via the applied torque ---
            "Act/torque_rms": (((torque.square().mean(dim=1)) * w).sum() / w_sum).sqrt(),
            # Not a run-wide peak: rsl_rl averages every key over one iteration, so this is the
            # mean of the per-step maximum.  ``Act/sat_frac`` is the quantity to compare.
            "Act/torque_max": torque.abs().amax(),
            "Act/sat_frac": ((torque.abs() >= 0.98 * TORQUE_LIMIT_NM).float().mean(dim=1) * w).sum() / w_sum,
        }

        if self._prev_cop_xy is not None:
            log["CoP/speed_mps"] = ((cop_xy - self._prev_cop_xy).norm(dim=-1) * w).sum() / w_sum / self.step_dt
        else:
            log["CoP/speed_mps"] = torch.zeros((), device=self.device)
        self._prev_cop_xy = cop_xy.detach()

        return log

    def reset(self, *args, **kwargs):
        self._prev_cop_xy = None
        return super().reset(*args, **kwargs)
