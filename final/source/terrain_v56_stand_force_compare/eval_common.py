"""Shared low-level pieces of the V56 evaluation scripts.

Three entry points read the same physics and must agree on every number:

    metrics.py         publishes them as training curves (iteration means, warm-up gated)
    evaluate_force.py  records one controller per step into a CSV, a JSON and figures
    identify_pd.py     records the network plus two PD replays and compares the three

So the definition of "the normal force under FL_foot", of ``sum(F_z)/mg``, of the CoP and
of the deterministic-play overrides lives here exactly once.  :mod:`metrics` owns the
per-step force helpers (they run on the device during training); this module owns the
numpy-side statistics and the evaluation-only setup.
"""
from __future__ import annotations

import numpy as np
import torch

from terrain_v56_stand_force_compare.metrics import (
    CONTACT_THRESHOLD_N,
    FOOT_NAMES,
    TORQUE_LIMIT_NM,
    foot_ids,
    foot_normal_forces,
)

GRAVITY = 9.81
"""Matches the simulation's gravity (``-9.81`` along z)."""

FOOT_LABELS = tuple(foot[:2] for foot in FOOT_NAMES)
"""``("FL", "FR", "RL", "RR")`` - the short names used in column headers and plots."""

SNAPSHOT_KEYS = (
    "f_feet",
    "f_total",
    "cop_x",
    "cop_y",
    "cop_offset",
    "base_z",
    "roll",
    "pitch",
    "tau_rms",
    "sat_count",
    "contact",
    "q",
    "qd",
    "tau",
    "grav",
    "ang_vel",
)
"""Keys produced by :func:`force_snapshot`, one array of ``[num_envs, ...]`` per step."""


REFERENCE_KEYS = ("ref_dq_rms", "ref_dq_max", "ref_tilt", "ref_d_ell", "ref_ell_0")
"""Keys produced by :func:`reference_snapshot`, one ``[num_envs]`` array per step."""


def force_snapshot(env) -> dict[str, np.ndarray]:
    """Per-step force / pose / torque / joint-state read-out, ``[num_envs, ...]`` per key.

    Deliberately *not* ``StandForceEnv``'s iteration means: an evaluation needs the raw
    per-step signal.  The joint state (``q``, ``qd``, ``tau``) and the base tilt are carried
    along so that :mod:`identify_pd` can fit a controller to the same recording.
    """
    robot = env.scene["robot"]
    _, robot_ids = foot_ids(env)

    force_z = foot_normal_forces(env)  # [N, 4]
    total = force_z.sum(dim=1)
    foot_xy = robot.data.body_pos_w[:, robot_ids, :2]
    cop = (force_z.unsqueeze(-1) * foot_xy).sum(dim=1) / total.clamp(min=1e-3).unsqueeze(-1)
    cop_offset = (cop - robot.data.root_pos_w[:, :2]).norm(dim=-1)

    # Roll / pitch from the measured gravity direction: independent of the yaw convention.
    gravity_b = robot.data.projected_gravity_b
    roll = torch.atan2(gravity_b[:, 1], -gravity_b[:, 2])
    pitch = torch.atan2(-gravity_b[:, 0], torch.sqrt(gravity_b[:, 1].square() + gravity_b[:, 2].square()))
    torque = robot.data.applied_torque

    return {
        "f_feet": force_z.cpu().numpy(),
        "f_total": total.cpu().numpy(),
        "cop_x": cop[:, 0].cpu().numpy(),
        "cop_y": cop[:, 1].cpu().numpy(),
        "cop_offset": cop_offset.cpu().numpy(),
        "base_z": robot.data.root_pos_w[:, 2].cpu().numpy(),
        "roll": roll.cpu().numpy(),
        "pitch": pitch.cpu().numpy(),
        "tau_rms": torque.square().mean(dim=1).sqrt().cpu().numpy(),
        "sat_count": (torque.abs() >= 0.98 * TORQUE_LIMIT_NM).sum(dim=1).cpu().numpy(),
        "contact": (force_z > CONTACT_THRESHOLD_N).cpu().numpy(),
        "q": robot.data.joint_pos.cpu().numpy(),
        "qd": robot.data.joint_vel.cpu().numpy(),
        "tau": torque.cpu().numpy(),
        "grav": gravity_b.cpu().numpy(),
        "ang_vel": robot.data.root_ang_vel_b.cpu().numpy(),
    }


def reference_snapshot(env) -> dict[str, np.ndarray]:
    """Per-step base-relative tracking error, ``[num_envs]`` per key.

    The same quantities :mod:`metrics` publishes as ``Ref/*`` during training, read raw instead of
    averaged over an iteration.  They are relative to the frame the episode started in, so a run at
    a different control rate, a different spawn height or a different plane elevation is still
    comparable - unlike ``base_z``, which is an absolute world coordinate.
    """
    err = env.reference.tracking_error()
    return {
        "ref_dq_rms": err["dq_rms"].cpu().numpy(),
        "ref_dq_max": err["dq_max"].cpu().numpy(),
        "ref_tilt": err["tilt"].cpu().numpy(),
        "ref_d_ell": err["d_ell"].cpu().numpy(),
        "ref_ell_0": env.reference.ell_0.cpu().numpy(),
    }


def summarise_reference(rows: dict[str, np.ndarray]) -> dict[str, float]:
    """Statistics of the base-relative error over every environment and recorded step.

    ``d_ell`` is signed (negative = the body sank towards its own feet), so both the mean and the
    extreme are reported; ``tilt`` is an angle, so it is reported in degrees next to the radians
    the plots use.
    """
    tilt = rows["ref_tilt"]
    d_ell = rows["ref_d_ell"]
    return {
        "dq_rms_rad": round(float(rows["ref_dq_rms"].mean()), 4),
        "dq_max_rad": round(float(rows["ref_dq_max"].mean()), 4),
        "dq_max_worst_rad": round(float(rows["ref_dq_max"].max()), 4),
        "tilt_mean_rad": round(float(tilt.mean()), 4),
        "tilt_mean_deg": round(float(np.degrees(tilt).mean()), 3),
        "tilt_max_deg": round(float(np.degrees(tilt).max()), 3),
        "d_ell_mean_mm": round(1e3 * float(d_ell.mean()), 2),
        "d_ell_worst_mm": round(1e3 * float(d_ell.min()), 2),
        "ell_0_m": round(float(rows["ref_ell_0"].mean()), 4),
    }


def play_overrides(env_cfg, *, num_envs: int, seed: int, device, warmup: float, duration: float) -> None:
    """Remove everything that would make two rollouts of the *same* controller differ.

    Kept: the robot asset, the plane, the physics step, the control rate and the observation
    layout - i.e. the plant and the controller.  Removed: friction/material randomisation, the
    random base mass (which would change the weight that normalises the force), external
    pushes and the random spawn pose/velocity.
    """
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    if device is not None:
        env_cfg.sim.device = device

    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")},
    }
    env_cfg.observations.policy.enable_corruption = False
    # Long enough that no robot truncates in the middle of the recording.
    env_cfg.episode_length_s = warmup + duration + 5.0


def psd(signal: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided periodogram with a Hann window (no SciPy dependency)."""
    signal = np.asarray(signal, dtype=float)
    signal = signal - signal.mean()
    window = np.hanning(signal.size)
    spectrum = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    power = 2.0 * np.abs(spectrum) ** 2 / (sample_rate * np.sum(window**2))
    return freqs, power


def summarise_force(
    rows: dict[str, np.ndarray], weight: float, step_dt: float, torque_limit: float = TORQUE_LIMIT_NM
) -> dict[str, dict]:
    """The comparison numbers, averaged over every environment and every recorded step.

    ``rows`` holds ``[T, N, ...]`` arrays as produced by :func:`force_snapshot`.  The same
    function feeds ``evaluate_force.py``'s ``summary.json`` and the three-way replay
    comparison of ``identify_pd.py``, so a difference between two runs is a difference of
    the controllers, never of the statistics.
    """
    feet = rows["f_feet"]  # [T, N, 4]
    total = rows["f_total"]  # [T, N]
    eps = 1e-3
    load = feet.reshape(-1, 4)
    load_share = load.sum(axis=0) / max(load.sum(), eps)
    cop_xy = np.stack([rows["cop_x"], rows["cop_y"]], axis=-1)  # [T, N, 2]
    cop_step = np.linalg.norm(np.diff(cop_xy, axis=0), axis=-1)  # [T-1, N]

    return {
        "force": {
            "total_mean_N": round(float(total.mean()), 3),
            "total_std_N": round(float(total.std()), 3),
            "total_over_weight_mean": round(float((total / weight).mean()), 4),
            "total_over_weight_std": round(float((total / weight).std()), 4),
            "per_foot_mean_N": {f: round(float(feet[:, :, i].mean()), 3) for i, f in enumerate(FOOT_LABELS)},
            "per_foot_std_N": {f: round(float(feet[:, :, i].std()), 3) for i, f in enumerate(FOOT_LABELS)},
            "load_share": {f: round(float(load_share[i]), 4) for i, f in enumerate(FOOT_LABELS)},
            "load_std_frac": round(float((load.std(axis=1) / np.maximum(load.sum(axis=1), eps)).mean()), 4),
            "asym_LR": round(
                float(((feet[:, :, 0] + feet[:, :, 2] - feet[:, :, 1] - feet[:, :, 3]) / np.maximum(total, eps)).mean()),
                4,
            ),
            "asym_FB": round(
                float(((feet[:, :, 0] + feet[:, :, 1] - feet[:, :, 2] - feet[:, :, 3]) / np.maximum(total, eps)).mean()),
                4,
            ),
            "front_share": round(float(((feet[:, :, 0] + feet[:, :, 1]) / np.maximum(total, eps)).mean()), 4),
        },
        "contact": {
            "duty_per_foot": {f: round(float(rows["contact"][:, :, i].mean()), 4) for i, f in enumerate(FOOT_LABELS)},
            "duty_all4": round(float(rows["contact"].all(axis=2).mean()), 4),
        },
        "cop": {
            "offset_mean_m": round(float(rows["cop_offset"].mean()), 5),
            "offset_std_m": round(float(rows["cop_offset"].std()), 5),
            "speed_mean_mps": round(float(cop_step.mean() / step_dt), 5),
            "drift_m": round(float(np.linalg.norm(cop_xy[-1] - cop_xy[0], axis=-1).mean()), 5),
        },
        "pose": {
            "base_z_mean_m": round(float(rows["base_z"].mean()), 4),
            "base_z_std_m": round(float(rows["base_z"].std()), 4),
            "roll_rms_rad": round(float(np.sqrt((rows["roll"] ** 2).mean())), 5),
            "pitch_rms_rad": round(float(np.sqrt((rows["pitch"] ** 2).mean())), 5),
        },
        "actuation": {
            "tau_rms_Nm": round(float(rows["tau_rms"].mean()), 4),
            "tau_rms_std_Nm": round(float(rows["tau_rms"].std()), 4),
            "sat_frac": round(float((rows["sat_count"] > 0).mean()), 5),
            "torque_limit_Nm": torque_limit,
        },
    }
