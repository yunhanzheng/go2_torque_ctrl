"""Is the trained direct-torque policy realisable as a joint PD controller?

The question
------------
The V56 torque policy emits ``tau`` directly.  A real Go2 runs a joint PD loop.  So, with the
network's own command as the target:

    tau_j  ==  Kp_j * (q*_j - q_j) - Kd_j * qdot_j          (constant Kp, Kd, q*)

This script answers it three ways, all from one checkpoint:

1. **fit** - least squares per joint over every (q, qdot, tau) sample, five models:
   ``pd3``      ``tau ~ [q, qdot, 1]``               free Kp, Kd, q*   <- the deployable form
   ``pd2``      ``tau ~ [q*_def - q, -qdot]``        no intercept, q* = the spawn pose
   ``pd_base``  ``pd3`` + base tilt and base rate    how much needs state a joint-local PD
                                                     cannot see
   ``pd_hist``  ``pd3`` + the previous torque        how much is one-step memory
   ``pd_cross`` all 12 joints' q and qdot            joint coupling
2. **gains** - are the identified gains physically usable?  Kp > 0 and Kd > 0 for every joint,
   and how far they are from Go2's stock 25 / 0.5.
3. **replay** - the decisive test.  Close the loop with the *fitted* PD and with the *stock* PD
   instead of the network, and compare the ground reaction forces.  A PD that reproduces
   ``sum(F_z)/mg``, the load share, the CoP and the base height is an equivalent controller;
   one that drifts away or falls over is not.

Both replays drive the *same* :class:`DirectTorqueAction` as the network, so the difference
between the three rollouts is the control law and nothing else.

Identification data
-------------------
A perfectly still stand visits a tiny region of the state space, where a linear fit is
undetermined.  The identification rollout therefore adds a small open-loop sinusoid to the
*action* to excite the plant, and regresses on ``clamp(action_scale * a_network)`` - the
network's *undithered* command at the excited state.  ``--excite 0`` disables the dither and
bounds the answer to "PD along the nominal trajectory only".

Only the direct-torque tasks mean anything here (the action term must be ``DirectTorqueAction``).

Artefacts
---------
    <out>/pd_fit.json    gains, per-joint R2, every ablation, the three replays and the verdict
    <out>/pd_gains.csv   one row per joint: Kp, Kd, q*, q_ref and the R2 of each model
    <out>/pd_fit.png     torque: network vs fitted PD, environment mean
    <out>/pd_replay.png  sum(F_z)/mg and base height: network vs fitted PD vs stock PD

    python identify_pd.py --task Isaac-Go2-V56-StandTorque-v0 \
        --checkpoint logs/rsl_rl/v56_stand_torque/<run>/model_<it>.pt --output_dir pd_eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# --- same path bootstrap as train.py ---------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (_SOURCE_ROOT, os.path.join(_SOURCE_ROOT, "base_adapter"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

# Imported for its side effect only: it pre-registers the broken ``sa_16dof`` task package as an
# empty stub, without which ``import isaaclab_tasks`` dies on the training server (see the
# comment at the top of ``base_adapter/cli_args.py``).
import cli_args  # noqa: F401, isort: skip

parser = argparse.ArgumentParser(description="Fit and validate an equivalent PD for a torque policy.")
parser.add_argument("--task", type=str, default="Isaac-Go2-V56-StandTorque-v0", help="Direct-torque task id.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the ``.pt`` checkpoint.")
parser.add_argument("--output_dir", type=str, default=None, help="Artefact directory (default: next to the checkpoint).")
parser.add_argument("--num_envs", type=int, default=8, help="Robots rolled out in parallel.")
parser.add_argument("--duration", type=float, default=6.0, help="Seconds recorded per rollout.")
parser.add_argument("--warmup", type=float, default=1.0, help="Seconds dropped after the reset (landing transient).")
parser.add_argument("--seed", type=int, default=7, help="Seed for the (deterministic) evaluation.")
parser.add_argument("--excite", type=float, default=0.15, help="Action dither amplitude [action units] used to excite the fit; 0 disables it.")
parser.add_argument("--stock_kp", type=float, default=25.0, help="Kp of the reference PD (UNITREE_GO2_CFG base_legs).")
parser.add_argument("--stock_kd", type=float, default=0.5, help="Kd of the reference PD (UNITREE_GO2_CFG base_legs).")
parser.add_argument("--min_fit_r2", type=float, default=0.90, help="Pooled R2 a PD must reach to count as equivalent.")
parser.add_argument("--no_replay", action="store_true", default=False, help="Only fit, skip the two closed-loop replays.")
parser.add_argument("--no_plots", action="store_true", default=False, help="Skip the matplotlib figures.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OffPolicyRunner, OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import terrain_v56_stand_force_compare.config  # noqa: F401  (registers the four tasks)
from terrain_v56_stand_force_compare.actions import DirectTorqueAction
from terrain_v56_stand_force_compare.bc import is_bc_checkpoint, load_bc
from terrain_v56_stand_force_compare.eval_common import (
    FOOT_LABELS,
    GRAVITY,
    force_snapshot,
    play_overrides,
    summarise_force,
)
from terrain_v56_stand_force_compare.metrics import TORQUE_LIMIT_NM
from terrain_v56_stand_force_compare.pd_fit import (
    FIT_MODELS,
    fit_models,
    gains_from_fits,
    pool_rows,
    r2,
    write_gains_csv,
)

from isaaclab_tasks.utils.hydra import hydra_task_config

ACTION_TERM = "joint_pos"
"""Name of the action term in ``config.py``; kept from the stock config so that the observation
and action layouts are identical to the position-controlled baseline."""

DITHER_HZ = 0.8
"""Frequency of the identification dither.  Slow compared to the policy: the point is to move
the joints through a range of states, not to test the bandwidth."""


def _direct_torque_term(env) -> DirectTorqueAction:
    """The action term, with a hard check that the network output really is the torque."""
    term = env.action_manager.get_term(ACTION_TERM)
    if not isinstance(term, DirectTorqueAction):
        raise RuntimeError(
            f"action term '{ACTION_TERM}' is a {type(term).__name__}, not DirectTorqueAction: "
            "--task must be one of the V56 direct-torque tasks."
        )
    return term


def _stack(samples: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack(values) for key, values in samples.items()}


def _step_actions(env, actions: torch.Tensor, base_env) -> torch.Tensor:
    """One environment step; returns the fresh observation and counts resets in ``base_env``."""
    obs, _reward, dones, _extras = env.step(actions)
    if bool(dones.any()):
        reset_ids = dones.nonzero(as_tuple=False).flatten()
        base_env.reset(env_ids=reset_ids)
        obs = env.get_observations()
        base_env._v56_resets = getattr(base_env, "_v56_resets", 0) + int(reset_ids.numel())
    return obs


def _record_network(policy, env, base_env, warmup_steps: int, record_steps: int, excite: float) -> dict[str, np.ndarray]:
    """Roll out the checkpoint; record the network's own torque, not the dithered one."""
    term = _direct_torque_term(base_env)
    scale, limit = float(term.cfg.action_scale), float(term.cfg.torque_limit)
    num_joints = base_env.action_manager.total_action_dim
    phase = torch.arange(num_joints, device=base_env.device, dtype=torch.float) * 0.7
    base_env._v56_resets = 0

    env.reset()
    obs = env.get_observations()
    samples: dict[str, list[np.ndarray]] = {}
    tau_net: list[np.ndarray] = []

    with torch.inference_mode():
        for step in range(warmup_steps + record_steps):
            a_net = policy(obs)
            if step >= warmup_steps:
                # The dither only moves the plant; the fit target stays the network's command.
                dither = excite * torch.sin(2.0 * torch.pi * DITHER_HZ * step * base_env.step_dt + phase)
                tau_net.append(((a_net * scale).clamp(-limit, limit)).cpu().numpy())
                actions = a_net + dither
            else:
                actions = a_net
            obs = _step_actions(env, actions, base_env)
            if step >= warmup_steps:
                for key, value in force_snapshot(base_env).items():
                    samples.setdefault(key, []).append(value)

    rows = _stack(samples)
    tau_command = np.stack(tau_net)
    # ``samples`` holds the state *after* the step (q_1 .. q_S) while ``tau_command`` holds the
    # command of that step (tau_0 .. tau_{S-1}), and tau_k was computed from q_k.  Shifting by
    # one sample pairs every command with the state it was computed from; without it the fit
    # would see a one-step-delayed PD and reject a perfect controller.
    rows = {key: value[:-1] for key, value in rows.items()}
    rows["tau"] = tau_command[1:]  # the network's own command, not the dithered torque
    return rows


def _record_pd(
    env, base_env, kp: torch.Tensor, kd: torch.Tensor, q_des: torch.Tensor, warmup_steps: int, record_steps: int
) -> dict[str, np.ndarray]:
    """Close the loop with ``tau = Kp (q* - q) - Kd qdot`` fed through the same torque channel."""
    term = _direct_torque_term(base_env)
    scale = float(term.cfg.action_scale)
    robot = base_env.scene["robot"]
    base_env._v56_resets = 0
    saturated = 0

    env.reset()
    samples: dict[str, list[np.ndarray]] = {}
    with torch.inference_mode():
        for step in range(warmup_steps + record_steps):
            tau = kp * (q_des - robot.data.joint_pos) - kd * robot.data.joint_vel
            _step_actions(env, tau / scale, base_env)
            if step >= warmup_steps:
                saturated += int((tau.abs() > TORQUE_LIMIT_NM).sum())
                for key, value in force_snapshot(base_env).items():
                    samples.setdefault(key, []).append(value)

    rows = _stack(samples)
    rows["pd_saturated_samples"] = np.array([saturated])
    return rows


def _scalars(summary: dict) -> dict:
    """The handful of numbers a replay comparison actually reads."""
    return {
        "sumF_over_mg": summary["force"]["total_over_weight_mean"],
        "sumF_over_mg_std": summary["force"]["total_over_weight_std"],
        "load_share_pct": {key: round(value * 100.0, 2) for key, value in summary["force"]["load_share"].items()},
        "load_std_frac": summary["force"]["load_std_frac"],
        "asym_LR": summary["force"]["asym_LR"],
        "asym_FB": summary["force"]["asym_FB"],
        "cop_offset_mm": round(summary["cop"]["offset_mean_m"] * 1000.0, 2),
        "cop_speed_mmps": round(summary["cop"]["speed_mean_mps"] * 1000.0, 2),
        "base_z_m": summary["pose"]["base_z_mean_m"],
        "roll_rms_deg": round(float(np.degrees(summary["pose"]["roll_rms_rad"])), 3),
        "pitch_rms_deg": round(float(np.degrees(summary["pose"]["pitch_rms_rad"])), 3),
        "tau_rms_Nm": summary["actuation"]["tau_rms_Nm"],
        "sat_frac": summary["actuation"]["sat_frac"],
    }


def _delta(reference: dict, other: dict) -> dict:
    """``other - reference`` for the scalars where a difference means something."""
    return {
        "sumF_over_mg": round(other["sumF_over_mg"] - reference["sumF_over_mg"], 4),
        "load_share_pct_max": round(
            max(abs(other["load_share_pct"][key] - reference["load_share_pct"][key]) for key in FOOT_LABELS), 2
        ),
        "cop_offset_mm": round(other["cop_offset_mm"] - reference["cop_offset_mm"], 2),
        "base_z_mm": round((other["base_z_m"] - reference["base_z_m"]) * 1000.0, 2),
        "pitch_rms_deg": round(other["pitch_rms_deg"] - reference["pitch_rms_deg"], 3),
        "tau_rms_Nm": round(other["tau_rms_Nm"] - reference["tau_rms_Nm"], 3),
    }


def _write_plots(out_dir: str, joint_names, rows: dict[str, np.ndarray], gains, fits, replays, step_dt: float) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed, skipping the figures.", flush=True)
        return []

    written = []
    time_s = np.arange(rows["base_z"].shape[0]) * step_dt

    # --- measured vs fitted torque ------------------------------------------------------------
    show = [0, 1, 2, 4]
    fig, axes = plt.subplots(len(show), 1, figsize=(9, 1.9 * len(show)), sharex=True)
    for ax, j in zip(np.atleast_1d(axes), show):
        predicted = -gains["kp"][j] * (rows["q"][:, :, j].mean(axis=1) - gains["q_star"][j])
        predicted -= gains["kd"][j] * rows["qd"][:, :, j].mean(axis=1)
        measured = rows["tau"][:, :, j].mean(axis=1)
        ax.plot(time_s, measured, lw=1.0, label="network")
        ax.plot(time_s, predicted, lw=1.0, ls="--", label=r"fitted PD  $R^2$=%.2f" % r2(fits["pd3"]["per_joint"][j]))
        ax.set_ylabel(joint_names[j], fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("time [s]")
    np.atleast_1d(axes)[0].set_title("torque: network vs equivalent PD (environment mean)")
    fig.tight_layout()
    path = os.path.join(out_dir, "pd_fit.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # --- closed-loop replay ------------------------------------------------------------------
    fig, (ax_force, ax_height) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for label, replay in replays.items():
        # Each rollout owns its time axis: the network recordings are one sample shorter than the
        # PD replays (the command/state alignment drops one), and the fit must not stretch them.
        replay_t = np.arange(replay["rows"]["base_z"].shape[0]) * step_dt
        ax_force.plot(replay_t, replay["rows"]["f_total"].mean(axis=1) / replay["weight"], lw=1.0, label=label)
        ax_height.plot(replay_t, replay["rows"]["base_z"].mean(axis=1), lw=1.0, label=label)
    ax_force.axhline(1.0, color="C3", ls="--", lw=0.9)
    ax_force.set_ylabel(r"$\sum F_z / mg$")
    ax_force.set_title("closed loop: network vs equivalent PD")
    ax_force.grid(alpha=0.3)
    ax_force.legend(fontsize=8)
    ax_height.set_ylabel("base height [m]")
    ax_height.set_xlabel("time [s]")
    ax_height.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "pd_replay.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    out_dir = args_cli.output_dir or os.path.join(os.path.dirname(os.path.abspath(args_cli.checkpoint)), "pd_eval")
    os.makedirs(out_dir, exist_ok=True)
    play_overrides(
        env_cfg,
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        device=args_cli.device,
        warmup=args_cli.warmup,
        duration=args_cli.duration,
    )

    print(f"[INFO] task       : {args_cli.task}", flush=True)
    print(f"[INFO] checkpoint : {args_cli.checkpoint}", flush=True)
    print(
        f"[INFO] plant      : {1.0 / (env_cfg.decimation * env_cfg.sim.dt):.0f} Hz policy, "
        f"{1.0 / env_cfg.sim.dt:.0f} Hz physics, {env_cfg.scene.num_envs} envs",
        flush=True,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    step_dt = base_env.step_dt
    robot = base_env.scene["robot"]
    weight = float(robot.root_physx_view.get_masses()[0].sum()) * GRAVITY
    term = _direct_torque_term(base_env)
    joint_names = list(robot.data.joint_names)
    num_joints = base_env.action_manager.total_action_dim
    print(f"[INFO] m g        : {weight:.2f} N, {num_joints} joints, action_scale {float(term.cfg.action_scale):.2f}", flush=True)

    runner_kwargs = {"log_dir": out_dir, "device": str(env.device)}
    if is_bc_checkpoint(args_cli.checkpoint):
        # Same branch as ``evaluate_force.py``: a cloned policy carries no critic and no optimiser,
        # so the runner is skipped and the MLP is loaded straight from the checkpoint.
        policy = load_bc(args_cli.checkpoint, device=env.device)
        print("[INFO] policy     : behaviour cloning (plain MLP, no rsl_rl runner)", flush=True)
    else:
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), **runner_kwargs)
        elif agent_cfg.class_name == "OffPolicyRunner":
            runner = OffPolicyRunner(env, agent_cfg.to_dict(), **runner_kwargs)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.device)

    warmup_steps = int(round(args_cli.warmup / step_dt))
    record_steps = int(round(args_cli.duration / step_dt))
    q_ref_t = robot.data.default_joint_pos[0].clone()
    q_ref = q_ref_t.cpu().numpy()

    # --- 1. identification rollout ------------------------------------------------------------
    print(f"[INFO] recording the network ({args_cli.duration:.1f} s, dither {args_cli.excite}) ...", flush=True)
    rows_net = _record_network(policy, env, base_env, warmup_steps, record_steps, args_cli.excite)
    net_resets = int(getattr(base_env, "_v56_resets", 0))
    pool = pool_rows(rows_net)
    fits = fit_models(pool, q_ref)
    gains = gains_from_fits(fits, num_joints)
    print(f"[INFO] samples    : {pool['q'].shape[0]} (envs x steps), resets {net_resets}", flush=True)

    # --- 2. gain sanity -----------------------------------------------------------------------
    gains_ok = bool(np.all(gains["kp"] > 0.0) and np.all(gains["kd"] > 0.0))
    fit_ok = bool(fits["pd3"]["pooled"]["r2"] >= args_cli.min_fit_r2)

    # --- 3. closed-loop replays ---------------------------------------------------------------
    replays: dict[str, dict] = {}
    if not args_cli.no_replay:
        kp_fit = torch.as_tensor(gains["kp"], device=base_env.device, dtype=torch.float)
        kd_fit = torch.as_tensor(gains["kd"], device=base_env.device, dtype=torch.float)
        q_star = torch.as_tensor(np.nan_to_num(gains["q_star"], nan=0.0), device=base_env.device, dtype=torch.float)
        kp_stock = torch.full_like(kp_fit, args_cli.stock_kp)
        kd_stock = torch.full_like(kd_fit, args_cli.stock_kd)

        print("[INFO] replay: network ...", flush=True)
        rows_plain = _record_network(policy, env, base_env, warmup_steps, record_steps, 0.0)
        print("[INFO] replay: fitted PD ...", flush=True)
        rows_pd = _record_pd(env, base_env, kp_fit, kd_fit, q_star, warmup_steps, record_steps)
        print("[INFO] replay: stock PD ...", flush=True)
        rows_stock = _record_pd(env, base_env, kp_stock, kd_stock, q_ref_t, warmup_steps, record_steps)

        replays = {
            "network": {"rows": rows_plain, "weight": weight, "summary": summarise_force(rows_plain, weight, step_dt)},
            "pd_fit": {"rows": rows_pd, "weight": weight, "summary": summarise_force(rows_pd, weight, step_dt)},
            "pd_stock": {"rows": rows_stock, "weight": weight, "summary": summarise_force(rows_stock, weight, step_dt)},
        }

    # --- 4. report ----------------------------------------------------------------------------
    per_joint = []
    for j, name in enumerate(joint_names):
        fit3 = fits["pd3"]["per_joint"][j]
        per_joint.append(
            {
                "joint": name,
                "kp": round(float(gains["kp"][j]), 3),
                "kd": round(float(gains["kd"][j]), 4),
                "q_star": round(float(gains["q_star"][j]), 5),
                "q_ref": round(float(q_ref[j]), 5),
                "kp_pd2": round(float(gains["kp2"][j]), 3),
                "kd_pd2": round(float(gains["kd2"][j]), 4),
                "r2_pd3": round(1.0 - fit3["sse"] / max(fit3["sst"], 1e-12), 4),
                "rms_pd3_Nm": round(fit3["rms_Nm"], 3),
                "tau_rms_Nm": round(float(np.sqrt((pool["tau"][:, j] ** 2).mean())), 3),
            }
        )

    report = {
        "task": args_cli.task,
        "checkpoint": os.path.abspath(args_cli.checkpoint),
        "control_hz": round(1.0 / step_dt, 3),
        "num_envs": args_cli.num_envs,
        "duration_s": args_cli.duration,
        "excite_action_units": args_cli.excite,
        "dither_hz": DITHER_HZ,
        "samples": int(pool["q"].shape[0]),
        "identification_resets": net_resets,
        "weight_N": round(weight, 3),
        "state_coverage_rad": {
            name: round(float(pool["q"][:, j].std()), 5) for j, name in enumerate(joint_names)
        },
        "fit": {
            model: {
                "note": fits[model]["note"],
                "num_params_per_joint": fits[model]["num_params_per_joint"],
                "pooled_r2": round(fits[model]["pooled"]["r2"], 4),
                "pooled_rms_Nm": round(fits[model]["pooled"]["rms_Nm"], 4),
            }
            for model in FIT_MODELS
        },
        "per_joint": per_joint,
        "checks": {
            "fit_r2_threshold": args_cli.min_fit_r2,
            "fit_ok": fit_ok,
            "gains_positive": gains_ok,
            "kp_range": [round(float(gains["kp"].min()), 3), round(float(gains["kp"].max()), 3)],
            "kd_range": [round(float(gains["kd"].min()), 4), round(float(gains["kd"].max()), 4)],
            "stock": {"kp": args_cli.stock_kp, "kd": args_cli.stock_kd},
            "q_star_shift_max_rad": round(float(np.nanmax(np.abs(gains["q_star"] - q_ref))), 4),
        },
    }

    if replays:
        scalars = {label: _scalars(replay["summary"]) for label, replay in replays.items()}
        report["replay"] = {
            "network": scalars["network"],
            "pd_fit": scalars["pd_fit"],
            "pd_stock": scalars["pd_stock"],
            "delta_pd_fit": _delta(scalars["network"], scalars["pd_fit"]),
            "delta_pd_stock": _delta(scalars["network"], scalars["pd_stock"]),
            "pd_fit_saturated_samples": int(rows_pd["pd_saturated_samples"][0]),
            "pd_stock_resets": int(getattr(base_env, "_v56_resets", 0)),
        }
        delta = report["replay"]["delta_pd_fit"]
        report["checks"]["replay_ok"] = bool(
            abs(delta["sumF_over_mg"]) <= 0.05
            and abs(delta["base_z_mm"]) <= 20.0
            and abs(delta["cop_offset_mm"]) <= 20.0
        )

    report_path = os.path.join(out_dir, "pd_fit.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    write_gains_csv(os.path.join(out_dir, "pd_gains.csv"), joint_names, q_ref, gains, fits)
    if replays and not args_cli.no_plots:
        _write_plots(out_dir, joint_names, rows_net, gains, fits, replays, step_dt)

    # --- 5. console ---------------------------------------------------------------------------
    print("\n[FIT] tau = Kp (q* - q) - Kd qdot   (per joint, least squares)", flush=True)
    print(f"  {'joint':<12}{'Kp':>9}{'Kd':>9}{'q*':>10}{'q_ref':>10}{'R2':>8}{'rms':>8}", flush=True)
    for row in per_joint:
        print(
            f"  {row['joint']:<12}{row['kp']:>9.2f}{row['kd']:>9.3f}{row['q_star']:>10.4f}"
            f"{row['q_ref']:>10.4f}{row['r2_pd3']:>8.3f}{row['rms_pd3_Nm']:>8.2f}",
            flush=True,
        )
    print("\n[FIT] pooled R2 by model", flush=True)
    for model in FIT_MODELS:
        print(
            f"  {model:<9} R2={fits[model]['pooled']['r2']:.4f}  rms={fits[model]['pooled']['rms_Nm']:.3f} N.m"
            f"  ({fits[model]['note']})",
            flush=True,
        )

    if replays:
        print("\n[REPLAY] closed loop, %d x %.1f s" % (args_cli.num_envs, args_cli.duration), flush=True)
        header = f"  {'metric':<20}{'network':>10}{'PD fit':>10}{'PD stock':>10}"
        print(header, flush=True)
        for key in ("sumF_over_mg", "load_std_frac", "cop_offset_mm", "base_z_m", "pitch_rms_deg", "tau_rms_Nm"):
            print(
                f"  {key:<20}{scalars['network'][key]:>10.4f}{scalars['pd_fit'][key]:>10.4f}"
                f"{scalars['pd_stock'][key]:>10.4f}",
                flush=True,
            )
        print(
            "  load share [%]       "
            + "  ".join(
                f"{label}: " + "/".join(f"{scalars[label]['load_share_pct'][foot]:.1f}" for foot in FOOT_LABELS)
                for label in ("network", "pd_fit", "pd_stock")
            ),
            flush=True,
        )
        delta = report["replay"]["delta_pd_fit"]
        print(
            f"\n[DELTA] fitted PD - network: sumF/mg {delta['sumF_over_mg']:+.4f}   "
            f"base z {delta['base_z_mm']:+.1f} mm   CoP {delta['cop_offset_mm']:+.1f} mm   "
            f"load share max {delta['load_share_pct_max']:.1f} pt",
            flush=True,
        )

    print(
        f"\n[VERDICT] fit R2={fits['pd3']['pooled']['r2']:.3f} (>= {args_cli.min_fit_r2}: {fit_ok}), "
        f"gains positive: {gains_ok}, "
        + (f"replay consistent: {report['checks'].get('replay_ok')}" if replays else "replay skipped"),
        flush=True,
    )
    print(f"[VERDICT] artefacts: {out_dir}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    # Same teardown hang as in train.py.
    os._exit(0)


if __name__ == "__main__":
    main()
