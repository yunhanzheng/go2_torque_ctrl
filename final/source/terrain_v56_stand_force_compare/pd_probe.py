"""Read the equivalent PD gains of a torque policy while it plays, one step at a time.

A torque network never writes its law down, but the law is differentiable.  At every control
step take the Jacobian of the commanded torque with respect to the joint positions and
velocities the policy sees, and read the gains straight off the diagonal:

    Kp_j(t) = - d tau_j / d q_j          Kd_j(t) = - d tau_j / d qdot_j
    q*_j(t) = q_j(t) + (tau_j(t) + Kd_j qdot_j) / Kp_j(t)

The damping term is added back on purpose: ``q*`` is defined by
``tau = -Kp (q - q*) - Kd qdot``, so dropping the ``Kd qdot`` while solving for ``q*``
would make the target drift with joint speed and read as a target error that is not there.

This is not an identification problem and nothing is fitted: one sample is a one-step
sensitivity taken at the exact state the policy is in.  What that buys is a time series, and
the interesting question is not the value of Kp at one instant but whether it stays put --
a network whose gains are constant to within a few percent can be replaced by a fixed PD,
one whose gains wander cannot.

Two caveats that come with the definition, both visible in the artefacts rather than
hidden here:

* Torque saturation kills the derivative (``clamp`` has a zero gradient outside the box),
  so saturated joints are reported as NaN instead of a misleading ``Kp = 0``.
* The observation contains the previous action, so the policy has one step of memory.  The
  Jacobian measures the direct dependence on q and qdot only; a gain that appears late is
  the second-order effect of that feedback and is not included.

The probe is a callable wrapped around the policy: it returns the same action the policy
would have returned (so the rollout is unchanged) plus the instantaneous gains.
"""
from __future__ import annotations

import csv
import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
import torch

EPS = 1e-6
"""Below this a gain counts as zero and ``q* = q + tau / Kp`` is reported as NaN."""


@dataclass(frozen=True)
class ObsLayout:
    """Where the joints sit inside the flattened policy observation."""

    joint_pos: slice
    joint_vel: slice
    pos_scale: float
    vel_scale: float

    @property
    def n_joints(self) -> int:
        return self.joint_pos.stop - self.joint_pos.start


def _base_env(env):
    return getattr(env, "unwrapped", env)


def resolve_obs_layout(env, group: str = "policy") -> ObsLayout:
    """Locate ``joint_pos`` / ``joint_vel`` in the observation without hard-coded offsets.

    The order of the terms is the order the observation manager concatenates them in, which
    is the configuration order with the disabled terms (``height_scan`` on the flat task)
    already dropped, so it is read from the manager rather than guessed.
    """
    base = _base_env(env)
    manager = base.observation_manager
    names = list(manager.active_terms[group])
    dims = [int(np.prod(dim)) for dim in manager.group_obs_term_dim[group]]
    if len(names) != len(dims):
        raise RuntimeError(f"observation manager reported {len(names)} names but {len(dims)} dims")

    offsets = {}
    cursor = 0
    for name, dim in zip(names, dims):
        offsets[name] = (cursor, dim)
        cursor += dim

    group_cfg = getattr(base.cfg.observations, group)

    def locate(needle: str) -> tuple[int, int, float]:
        for name in names:
            if name == needle or needle in name:
                start, dim = offsets[name]
                term_cfg = getattr(group_cfg, name)
                scale = getattr(term_cfg, "scale", None)
                if scale is None:
                    return start, dim, 1.0
                value = np.asarray(scale, dtype=float)
                if value.size != 1:
                    raise RuntimeError(
                        f"observation term '{name}' uses a per-element scale ({value.size} values); "
                        "the probe expects one scalar per joint group"
                    )
                return start, dim, float(value)
        raise RuntimeError(f"no observation term matching '{needle}' in group '{group}' (have: {names})")

    pos_start, pos_dim, pos_scale = locate("joint_pos")
    vel_start, vel_dim, vel_scale = locate("joint_vel")
    if pos_dim != vel_dim:
        raise RuntimeError(f"joint_pos has {pos_dim} entries but joint_vel has {vel_dim}")

    return ObsLayout(
        joint_pos=slice(pos_start, pos_start + pos_dim),
        joint_vel=slice(vel_start, vel_start + vel_dim),
        pos_scale=pos_scale,
        vel_scale=vel_scale,
    )


def torque_term(env, term_name: str = "joint_pos"):
    """The action term that turns the network output into a torque, or ``None``.

    ``None`` means the task is not a direct-torque variant and the probe is meaningless:
    on ``StandPos`` the torque comes out of the actuator PD, so the equivalent gains are
    simply the configured 25 / 0.5 and there is nothing to measure.
    """
    term = _base_env(env).action_manager.get_term(term_name)
    if not hasattr(term.cfg, "torque_limit"):
        return None
    return term


def _actions_of(output):
    """The bare action tensor inside whatever ``act_inference`` returned.

    rsl_rl 4 hands back a ``TensorDict`` keyed by ``actions``; older releases hand back the
    tensor itself.  Only a plain tensor can be run through ``torch.autograd.grad``.
    """
    keys = getattr(output, "keys", None)
    if keys is not None and "actions" in keys():
        return output["actions"]
    return output


class PdProbe:
    """Wraps a policy: same action out, plus the instantaneous equivalent PD gains."""

    def __init__(self, policy, env, *, term_name: str = "joint_pos", stock_kp: float = 25.0, stock_kd: float = 0.5):
        base = _base_env(env)
        term = torque_term(env, term_name)
        if term is None:
            raise RuntimeError("PdProbe needs a direct-torque action term (see config.StandTorqueEnvCfg)")

        self.policy = policy
        self.layout = resolve_obs_layout(env)
        self.n_joints = self.layout.n_joints
        self.action_scale = float(getattr(term.cfg, "action_scale", 1.0))
        self.torque_limit = float(getattr(term.cfg, "torque_limit", float("inf")))
        self.stock_kp = float(stock_kp)
        self.stock_kd = float(stock_kd)

        robot = base.scene["robot"]
        self.joint_names = list(robot.data.joint_names)
        if len(self.joint_names) != self.n_joints:
            raise RuntimeError(f"{self.n_joints} observation slots for {len(self.joint_names)} joints")
        self.default_joint_pos = robot.data.default_joint_pos[0].detach().cpu().numpy().astype(float)
        self.step_dt = float(base.step_dt)

    # -- one control step ------------------------------------------------------------------
    def __call__(self, obs: torch.Tensor):
        """Return ``(action, record)`` for this step; ``action`` is what ``env.step`` expects."""
        # ``env.step`` runs inside ``torch.inference_mode``; a tensor cloned there carries the
        # inference flag and silently refuses to build a graph, so the mode has to be lifted
        # before the copy is made.
        with torch.inference_mode(False), torch.enable_grad():
            x = obs.detach().clone().requires_grad_(True)
            output = self.policy(x)
            # rsl_rl >= 4 answers with a TensorDict, and ``torch.autograd.grad`` refuses to
            # differentiate one, so the bare action tensor is what the Jacobian is taken on.
            tau = (_actions_of(output) * self.action_scale).clamp(-self.torque_limit, self.torque_limit)
            jacobian = self._jacobian(x, tau)
            with torch.inference_mode(True):
                record = self._record(x.detach(), tau.detach(), jacobian.detach())
                # Hand back whatever the policy returned, detached, so this stays a drop-in
                # replacement for calling the policy directly.
                return output.detach() if hasattr(output, "detach") else output, record

    def _jacobian(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """d tau / d obs, shape ``(num_envs, num_joints, obs_dim)``.

        One backward pass per joint: the actor is a 48 -> 12 map, so twelve passes cost less
        than the physics step they sit in front of.
        """
        rows = []
        last = tau.shape[1] - 1
        for joint in range(tau.shape[1]):
            (grad,) = torch.autograd.grad(tau[:, joint].sum(), x, retain_graph=joint < last)
            rows.append(grad)
        return torch.stack(rows, dim=1)

    def _record(self, obs: torch.Tensor, tau: torch.Tensor, jacobian: torch.Tensor) -> dict[str, np.ndarray]:
        layout = self.layout
        index = torch.arange(self.n_joints, device=jacobian.device)
        pos_index = layout.joint_pos.start + index
        vel_index = layout.joint_vel.start + index

        # tau already contains ``action_scale``, so the derivative is in N.m per unit of the
        # observation.  The observation is ``scale * q``, hence d(obs)/dq = scale and going back
        # to N.m per rad means multiplying by the scale (dividing would be off by 1/scale**2 --
        # invisible on joint_pos, whose scale is 1.0, and a factor of 400 on joint_vel).
        kp = -jacobian[:, index, pos_index] * layout.pos_scale
        kd = -jacobian[:, index, vel_index] * layout.vel_scale
        kp_full = -jacobian[:, :, layout.joint_pos] * layout.pos_scale

        q_rel = obs[:, layout.joint_pos] / layout.pos_scale
        q_dot = obs[:, layout.joint_vel] / layout.vel_scale

        # Outside the torque box the gradient is zero for a reason that has nothing to do with
        # stiffness, so a saturated joint gets NaN rather than a Kp of 0.
        saturated = tau.abs() >= self.torque_limit * (1.0 - EPS)
        nan = torch.full_like(kp, float("nan"))
        # tau + Kd qdot is the part of the torque law that the position error has to explain;
        # without the Kd term the target would slide with velocity.
        q_star_rel = torch.where(kp.abs() > EPS, q_rel + (tau + kd * q_dot) / kp, nan)
        kp = kp.masked_fill(saturated, float("nan"))
        kd = kd.masked_fill(saturated, float("nan"))
        q_star_rel = q_star_rel.masked_fill(saturated, float("nan"))

        return {
            "kp": kp.cpu().numpy(),
            "kd": kd.cpu().numpy(),
            "kp_full": kp_full.cpu().numpy(),
            "tau": tau.cpu().numpy(),
            "q_rel": q_rel.cpu().numpy(),
            "q_dot": q_dot.cpu().numpy(),
            "q_star_rel": q_star_rel.cpu().numpy(),
            "saturated": saturated.cpu().numpy(),
        }


def stack_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """``[step][key] -> key[step, env, joint]``."""
    return {key: np.stack([record[key] for record in records]) for key in records[0]}


def _nan_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean / std / valid count along axis 0, tolerating all-NaN columns."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
    count = np.isfinite(values).sum(axis=0)
    return mean, std, count


def summarise_gains(stacked: dict[str, np.ndarray], joint_names: list[str], *, stock_kp: float, stock_kd: float) -> dict:
    """Per-joint gain statistics plus the one number that decides replaceability.

    ``fixed_pd_r2`` rebuilds the torque with the *time-averaged* gains and the time-averaged
    target, ``tau_hat = kp_bar (q*_bar - q) - kd_bar q_dot``, and scores it against the torque
    the network actually asked for.  It answers "could one constant PD have stood in for this
    network during this rollout", which is exactly what the per-step gains are there to show.
    """
    kp, kd = stacked["kp"], stacked["kd"]
    tau, q_rel, q_dot, q_star = stacked["tau"], stacked["q_rel"], stacked["q_dot"], stacked["q_star_rel"]
    n_joints = kp.shape[-1]

    flat_kp = kp.reshape(-1, n_joints)
    flat_kd = kd.reshape(-1, n_joints)
    kp_mean, kp_std, kp_count = _nan_stats(flat_kp)
    kd_mean, kd_std, kd_count = _nan_stats(flat_kd)
    q_star_mean, q_star_std, _ = _nan_stats(q_star.reshape(-1, n_joints))

    joint_entries = []
    for joint in range(n_joints):
        kp_cv = float(kp_std[joint] / abs(kp_mean[joint])) if kp_mean[joint] and np.isfinite(kp_mean[joint]) else float("nan")
        kd_cv = float(kd_std[joint] / abs(kd_mean[joint])) if kd_mean[joint] and np.isfinite(kd_mean[joint]) else float("nan")
        joint_entries.append(
            {
                "joint": joint_names[joint],
                "kp_mean": float(kp_mean[joint]),
                "kp_std": float(kp_std[joint]),
                "kp_cv": kp_cv,
                "kd_mean": float(kd_mean[joint]),
                "kd_std": float(kd_std[joint]),
                "kd_cv": kd_cv,
                "q_star_rel_mean": float(q_star_mean[joint]),
                "q_star_rel_std": float(q_star_std[joint]),
                "samples": int(kp_count[joint]),
                "saturated_fraction": float(1.0 - kp_count[joint] / max(flat_kp.shape[0], 1)),
            }
        )

    # --- fixed-PD replay score ---------------------------------------------------------
    tau_flat = tau.reshape(-1, n_joints)
    q_flat = q_rel.reshape(-1, n_joints)
    qd_flat = q_dot.reshape(-1, n_joints)
    tau_hat = kp_mean * (q_star_mean - q_flat) - kd_mean * qd_flat
    valid = np.isfinite(tau_flat) & np.isfinite(tau_hat)
    residual = (tau_flat - tau_hat)[valid]
    target = tau_flat[valid]
    if residual.size:
        sse = float(np.sum(residual**2))
        sst = float(np.sum((target - target.mean()) ** 2))
        fixed_pd_r2 = 1.0 - sse / sst if sst > 0.0 else float("nan")
        fixed_pd_rms = float(np.sqrt(np.mean(residual**2)))
    else:
        fixed_pd_r2 = fixed_pd_rms = float("nan")

    # --- coupling: is the gain matrix diagonal? ----------------------------------------
    kp_full = stacked["kp_full"].reshape(-1, n_joints, n_joints)
    coupling = np.nanmean(np.abs(kp_full), axis=0)
    off_diagonal = coupling[~np.eye(n_joints, dtype=bool)]
    diagonal = np.diag(coupling)
    diag_mean = float(np.nanmean(diagonal))
    off_mean = float(np.nanmean(off_diagonal))
    # A perfectly diagonal gain matrix has an off-diagonal mean of exactly zero; the floor keeps
    # the ratio finite so that summary.json stays valid JSON.
    dominance = diag_mean / max(off_mean, 1e-9)

    finite_kp = flat_kp[np.isfinite(flat_kp)]
    finite_kd = flat_kd[np.isfinite(flat_kd)]
    return {
        "joints": joint_entries,
        "kp_all_mean": float(np.nanmean(finite_kp)) if finite_kp.size else float("nan"),
        "kp_all_std": float(np.nanstd(finite_kp)) if finite_kp.size else float("nan"),
        "kd_all_mean": float(np.nanmean(finite_kd)) if finite_kd.size else float("nan"),
        "kd_all_std": float(np.nanstd(finite_kd)) if finite_kd.size else float("nan"),
        "saturated_fraction": float(np.mean(~np.isfinite(flat_kp))),
        "fixed_pd_r2": fixed_pd_r2,
        "fixed_pd_rms_Nm": fixed_pd_rms,
        "coupling_dominance": dominance,
        "coupling_off_diagonal_mean": off_mean,
        "stock": {"kp": stock_kp, "kd": stock_kd},
        "kp_over_stock": float(np.nanmean(finite_kp) / stock_kp) if finite_kp.size and stock_kp else float("nan"),
    }


# -- artefacts -----------------------------------------------------------------------------
def write_csv(path: str, stacked: dict[str, np.ndarray], joint_names: list[str], step_dt: float) -> None:
    steps, envs, n_joints = stacked["kp"].shape
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["env", "step", "t_s", "joint", "kp", "kd", "tau", "q_rel", "q_dot", "q_star_rel", "saturated"])
        for env_id in range(envs):
            for step in range(steps):
                for joint in range(n_joints):
                    writer.writerow(
                        [
                            env_id,
                            step,
                            f"{step * step_dt:.4f}",
                            joint_names[joint],
                            f"{stacked['kp'][step, env_id, joint]:.4f}",
                            f"{stacked['kd'][step, env_id, joint]:.4f}",
                            f"{stacked['tau'][step, env_id, joint]:.4f}",
                            f"{stacked['q_rel'][step, env_id, joint]:.5f}",
                            f"{stacked['q_dot'][step, env_id, joint]:.4f}",
                            f"{stacked['q_star_rel'][step, env_id, joint]:.5f}",
                            int(stacked["saturated"][step, env_id, joint]),
                        ]
                    )


def write_json(path: str, summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def print_table(summary: dict) -> None:
    print("\n[PD] equivalent gains, mean +- std over steps x envs", flush=True)
    print(f"  {'joint':<20}{'Kp':>16}{'Kd':>16}{'q* (deg)':>13}{'sat %':>8}", flush=True)
    for entry in summary["joints"]:
        print(
            f"  {entry['joint']:<20}"
            f"{entry['kp_mean']:>9.2f} +- {entry['kp_std']:<4.2f}"
            f"{entry['kd_mean']:>9.3f} +- {entry['kd_std']:<4.3f}"
            f"{np.degrees(entry['q_star_rel_mean']):>13.2f}"
            f"{entry['saturated_fraction'] * 100:>8.2f}",
            flush=True,
        )
    print(
        f"  -> Kp {summary['kp_all_mean']:.2f} +- {summary['kp_all_std']:.2f} N.m/rad "
        f"(stock {summary['stock']['kp']:.1f}, ratio {summary['kp_over_stock']:.2f})",
        flush=True,
    )
    print(
        f"  -> Kd {summary['kd_all_mean']:.3f} +- {summary['kd_all_std']:.3f} N.m.s/rad "
        f"(stock {summary['stock']['kd']:.2f})",
        flush=True,
    )
    print(
        f"  -> one constant PD: R^2 {summary['fixed_pd_r2']:.3f}, rms {summary['fixed_pd_rms_Nm']:.2f} N.m; "
        f"coupling diagonal/off-diagonal {summary['coupling_dominance']:.1f}x",
        flush=True,
    )
    print(f"  -> saturated joints {summary['saturated_fraction'] * 100:.1f}% of samples", flush=True)


def write_plots(out_dir: str, label: str, stacked: dict[str, np.ndarray], summary: dict, step_dt: float) -> list[str]:
    """Three PNGs: gains over time, gain distribution, coupling matrix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, envs, n_joints = stacked["kp"].shape
    time_s = np.arange(steps) * step_dt
    entries = summary["joints"]
    kp_mean = np.array([entry["kp_mean"] for entry in entries])
    kp_std = np.array([entry["kp_std"] for entry in entries])
    kd_mean = np.array([entry["kd_mean"] for entry in entries])
    kd_std = np.array([entry["kd_std"] for entry in entries])
    names = [entry["joint"] for entry in entries]
    written = []

    # --- 1. gains over time -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for joint in range(n_joints):
        axes[0].plot(time_s, stacked["kp"][:, :, joint], lw=0.8, alpha=0.65)
        axes[1].plot(time_s, stacked["kd"][:, :, joint], lw=0.8, alpha=0.65)
    axes[0].axhline(summary["stock"]["kp"], color="k", ls="--", lw=1.2, label=f"stock PD Kp={summary['stock']['kp']:.0f}")
    axes[1].axhline(summary["stock"]["kd"], color="k", ls="--", lw=1.2, label=f"stock PD Kd={summary['stock']['kd']:.2f}")
    axes[0].set_ylabel("Kp  [N.m/rad]")
    axes[1].set_ylabel("Kd  [N.m.s/rad]")
    axes[1].set_xlabel("time  [s]")
    axes[0].set_title(f"{label}: instantaneous equivalent gains (one line per joint)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "pd_gains.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    # --- 2. distribution per joint ------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    x = np.arange(n_joints)
    axes[0].bar(x, kp_mean, yerr=kp_std, color="#3b7dd8", capsize=3)
    axes[0].axhline(summary["stock"]["kp"], color="k", ls="--", lw=1.2)
    axes[0].set_ylabel("Kp  [N.m/rad]")
    axes[1].bar(x, kd_mean, yerr=kd_std, color="#d8783b", capsize=3)
    axes[1].axhline(summary["stock"]["kd"], color="k", ls="--", lw=1.2)
    axes[1].set_ylabel("Kd  [N.m.s/rad]")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{label}: mean +- std of the equivalent gains (dashed = stock PD)")
    fig.tight_layout()
    path = os.path.join(out_dir, "pd_distribution.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    # --- 3. coupling matrix -------------------------------------------------------------
    kp_full = stacked["kp_full"].reshape(-1, n_joints, n_joints)
    coupling = np.nanmean(np.abs(kp_full), axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    image = ax.imshow(coupling, cmap="viridis")
    ax.set_xticks(np.arange(n_joints))
    ax.set_yticks(np.arange(n_joints))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("perturbed joint  j")
    ax.set_ylabel("torque joint  i")
    ax.set_title(f"{label}: |d tau_i / d q_j|  [N.m/rad]\ndiagonal/off-diagonal = "
                 f"{summary['coupling_dominance']:.1f}x")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(out_dir, "pd_coupling.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    return written
