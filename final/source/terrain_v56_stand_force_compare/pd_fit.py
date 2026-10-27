"""Least-squares identification of a joint PD law from ``(q, qdot, tau)`` samples.

Pure numpy, no IsaacLab: the estimator is the part that can be verified offline, against
synthetic data with a known ``Kp`` / ``Kd`` / ``q*``.  :mod:`identify_pd` supplies the
samples from the simulator and consumes :func:`fit_models` / :func:`gains_from_fits`.

The five models
---------------
``pd3``      ``tau_j ~ [q_j, qdot_j, 1]``            free Kp, Kd, q*   <- the deployable form
``pd2``      ``tau_j ~ [q*_def - q_j, -qdot_j]``     no intercept, q* = the given reference pose
``pd_base``  ``pd3`` + projected gravity and base rate
``pd_hist``  ``pd3`` + the torque of the previous step
``pd_cross`` all joints' ``q`` and ``qdot`` (every joint's torque from every joint's state)

``pd3`` answers the question; the other four are ablations that separate "this is a PD" from
"this happens to look like one along the trajectory we visited" (``pd_base``, ``pd_hist``) and
from "the joints are coupled" (``pd_cross``).
"""
from __future__ import annotations

import csv

import numpy as np

FIT_MODELS = ("pd3", "pd2", "pd_base", "pd_hist", "pd_cross")
"""``pd3`` is the deployable one; the others only explain why it does or does not fit."""

MODEL_NOTES = {
    "pd3": "tau ~ [q, qdot, 1] - free Kp, Kd, q*",
    "pd2": "tau ~ [q*_ref - q, -qdot] - no intercept",
    "pd_base": "pd3 + base tilt / base rate (not available to a joint PD)",
    "pd_hist": "pd3 + previous torque (one-step memory)",
    "pd_cross": "all joints' q and qdot (joint coupling)",
}


def fit(design: np.ndarray, target: np.ndarray) -> dict:
    """Ordinary least squares plus the two sums every ablation needs (``sse`` and ``sst``).

    ``r2`` is left to the caller so that it can be pooled over joints as
    ``1 - sum(sse) / sum(sst)``, which is the variance-weighted R2 of the whole robot's torque.
    """
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coef
    sse = float(residual @ residual)
    return {
        "coef": [float(c) for c in coef],
        "sse": sse,
        "sst": float(((target - target.mean()) ** 2).sum()),
        "rms_Nm": float(np.sqrt(sse / max(target.size, 1))),
    }


def r2(fit_result: dict) -> float:
    """R2 of a single joint."""
    return 1.0 - fit_result["sse"] / max(fit_result["sst"], 1e-12)


def pool_rows(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Flatten ``[T, N, ...]`` to ``[T*N, ...]`` and add the one-step-lagged torque.

    The lagged torque is the linear analogue of the policy's own action feedback (the
    observation carries the previous action), which is what ``pd_hist`` measures.
    """
    num_joints = rows["q"].shape[-1]
    pool = {
        "q": rows["q"].reshape(-1, num_joints).astype(np.float64),
        "qd": rows["qd"].reshape(-1, num_joints).astype(np.float64),
        "tau": rows["tau"].reshape(-1, num_joints).astype(np.float64),
        "grav": rows["grav"].reshape(-1, 3).astype(np.float64),
        "ang_vel": rows["ang_vel"].reshape(-1, 3).astype(np.float64),
    }
    pool["tau_prev"] = np.concatenate([pool["tau"][:1], pool["tau"][:-1]], axis=0)
    return pool


def fit_models(pool: dict[str, np.ndarray], q_ref: np.ndarray) -> dict[str, dict]:
    """Fit all five models to every joint.

    ``pooled`` aggregates SSE/SST over the whole robot, so a joint with large torques weighs
    more than a joint that barely moves.
    """
    num_joints = pool["q"].shape[1]
    ones = np.ones_like(pool["q"][:, 0])
    targets = [pool["tau"][:, j] for j in range(num_joints)]

    designs: dict[str, list[np.ndarray]] = {model: [] for model in FIT_MODELS}
    for j in range(num_joints):
        q, qd = pool["q"][:, j], pool["qd"][:, j]
        designs["pd3"].append(np.stack([q, qd, ones], axis=1))
        designs["pd2"].append(np.stack([q_ref[j] - q, -qd], axis=1))
        designs["pd_base"].append(
            np.stack(
                [
                    q,
                    qd,
                    ones,
                    pool["grav"][:, 0],
                    pool["grav"][:, 1],
                    pool["ang_vel"][:, 0],
                    pool["ang_vel"][:, 1],
                ],
                axis=1,
            )
        )
        designs["pd_hist"].append(np.stack([q, qd, ones, pool["tau_prev"][:, j]], axis=1))
        designs["pd_cross"].append(np.stack([*pool["q"].T, *pool["qd"].T, ones], axis=1))

    fits: dict[str, dict] = {}
    for model in FIT_MODELS:
        per_joint = [fit(designs[model][j], targets[j]) for j in range(num_joints)]
        sse = sum(result["sse"] for result in per_joint)
        sst = sum(result["sst"] for result in per_joint)
        fits[model] = {
            "note": MODEL_NOTES[model],
            "num_params_per_joint": designs[model][0].shape[1],
            "per_joint": per_joint,
            "pooled": {
                "r2": 1.0 - sse / max(sst, 1e-12),
                "rms_Nm": float(np.sqrt(sse / max(sum(target.size for target in targets), 1))),
            },
        }
    return fits


def gains_from_fits(fits: dict[str, dict], num_joints: int) -> dict[str, np.ndarray]:
    """Turn the affine ``pd3`` coefficients into ``Kp, Kd, q*``; read ``pd2``'s two gains.

    ``pd3`` fits ``tau = a q + b qdot + c``, so ``Kp = -a``, ``Kd = -b`` and ``q* = c / Kp``.
    ``q*`` degenerates when ``Kp`` is ~0 (a joint that does not use position feedback), hence
    the guard: such a joint is reported as NaN and must be read from ``Kd`` alone.
    """
    kp = np.zeros(num_joints)
    kd = np.zeros(num_joints)
    q_star = np.full(num_joints, np.nan)
    kp2 = np.zeros(num_joints)
    kd2 = np.zeros(num_joints)
    for j in range(num_joints):
        a, b, c = fits["pd3"]["per_joint"][j]["coef"]
        kp[j], kd[j] = -a, -b
        if abs(kp[j]) > 1e-6:
            q_star[j] = c / kp[j]
        kp2[j], kd2[j] = fits["pd2"]["per_joint"][j]["coef"]
    return {"kp": kp, "kd": kd, "q_star": q_star, "kp2": kp2, "kd2": kd2}


def write_gains_csv(path: str, joint_names, q_ref, gains: dict, fits: dict) -> None:
    """One row per joint: the fitted gains, the reference pose and the R2 of every model."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "joint",
                "kp_fit",
                "kd_fit",
                "q_star_fit",
                "q_ref",
                "kp_pd2",
                "kd_pd2",
                "r2_pd3",
                "rms_pd3_Nm",
                "r2_pd2",
                "r2_pd_base",
                "r2_pd_hist",
                "r2_pd_cross",
            ]
        )
        for j, name in enumerate(joint_names):
            writer.writerow(
                [
                    name,
                    f"{gains['kp'][j]:.3f}",
                    f"{gains['kd'][j]:.4f}",
                    f"{gains['q_star'][j]:.5f}",
                    f"{float(q_ref[j]):.5f}",
                    f"{gains['kp2'][j]:.3f}",
                    f"{gains['kd2'][j]:.4f}",
                    f"{r2(fits['pd3']['per_joint'][j]):.4f}",
                    f"{fits['pd3']['per_joint'][j]['rms_Nm']:.3f}",
                    f"{r2(fits['pd2']['per_joint'][j]):.4f}",
                    f"{r2(fits['pd_base']['per_joint'][j]):.4f}",
                    f"{r2(fits['pd_hist']['per_joint'][j]):.4f}",
                    f"{r2(fits['pd_cross']['per_joint'][j]):.4f}",
                ]
            )
