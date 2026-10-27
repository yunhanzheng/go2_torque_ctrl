"""Roll out a trained V56 checkpoint and record what the ground feels.

One run of this script produces the evidence for one controller:

    <out>/timeseries.csv   per-environment, per-step forces / CoP / pose / torque
    <out>/summary.json     the numbers that go into the comparison table
    <out>/forces.png       four feet + total, with the body weight as a reference line
    <out>/psd.png          spectrum of the total normal force (load oscillation)
    <out>/load_share.png   mean load share per foot
    <out>/pd_gains.png     equivalent Kp / Kd of the torque policy, one sample per step
    <out>/pd_distribution.png  those gains as mean +- std against the stock 25 / 0.5
    <out>/pd_coupling.png  |d tau_i / d q_j|, the cross-joint part of the same Jacobian
    <out>/pd_gains.csv     every per-step gain, so the figures can be re-derived
    <out>/videos/          the rollout as mp4, only with --video

For a direct-torque task the equivalent gains are read off the policy Jacobian at every
control step (see ``pd_probe.py``); for the position task there is nothing to measure, so
the probe is skipped unless it is explicitly forced with ``--pd_probe on``.

``compare_force.py`` then merges any number of ``summary.json`` files into the final
position-vs-torque table.  Nothing here trains and nothing touches wandb.

Evaluation is deterministic on purpose (the training-time randomisation is switched off, see
``eval_common.play_overrides``): a force comparison between two controllers must not carry a
random friction bucket, a random base mass or a random spawn pose.

    python evaluate_force.py --checkpoint logs/rsl_rl/v56_stand_pos/<run>/model_<it>.pt \
        --task Isaac-Go2-V56-StandPos-v0 --label pos --output_dir force_eval/pos
"""
from __future__ import annotations

import argparse
import csv
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

parser = argparse.ArgumentParser(description="Record ground reaction forces of a V56 checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-Go2-V56-StandTorque-v0", help="Task id of the checkpoint.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the ``.pt`` checkpoint.")
parser.add_argument("--output_dir", type=str, default=None, help="Where to write the artefacts (default: next to the checkpoint).")
parser.add_argument("--label", type=str, default=None, help="Name used in the artefacts (default: the task name).")
parser.add_argument("--num_envs", type=int, default=4, help="Robots rolled out in parallel.")
parser.add_argument("--duration", type=float, default=6.0, help="Seconds recorded after the warm-up.")
parser.add_argument("--warmup", type=float, default=0.5, help="Seconds dropped after the reset (landing transient).")
parser.add_argument("--seed", type=int, default=7, help="Seed for the (deterministic) evaluation.")
parser.add_argument("--no_plots", action="store_true", default=False, help="Skip the matplotlib figures.")
parser.add_argument(
    "--pd_probe",
    type=str,
    default="auto",
    choices=["auto", "on", "off"],
    help="Measure the instantaneous equivalent PD gains while playing (auto: direct-torque tasks only).",
)
parser.add_argument("--stock_kp", type=float, default=25.0, help="Reference stiffness drawn on the PD figures.")
parser.add_argument("--stock_kd", type=float, default=0.5, help="Reference damping drawn on the PD figures.")
parser.add_argument("--video", action="store_true", default=False, help="Record the rollout to an mp4 (turns on --enable_cameras).")
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=[1.6, 1.6, 0.9],
    help="Viewer position in metres; the stock viewport leaves the robot a few pixels wide.",
)
parser.add_argument(
    "--camera_target",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.30],
    help="Point the viewer looks at, in metres.",
)
parser.add_argument(
    "--video_seconds",
    type=float,
    default=None,
    help="Length of the recording (default: warmup + duration, i.e. everything that is recorded).",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = parser.parse_known_args()

# The camera has to be enabled before the app starts, not when the recorder is wrapped.
if args_cli.video:
    args_cli.enable_cameras = True

# ``hydra_task_config`` parses ``sys.argv`` after this point, so drop what argparse consumed.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OffPolicyRunner, OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import terrain_v56_stand_force_compare.config  # noqa: F401  (registers the three tasks)
from terrain_v56_stand_force_compare.eval_common import (
    FOOT_LABELS,
    GRAVITY,
    force_snapshot,
    play_overrides,
    psd,
    summarise_force,
)
from terrain_v56_stand_force_compare.pd_probe import (
    PdProbe,
    stack_records,
    summarise_gains,
    torque_term,
)
from terrain_v56_stand_force_compare.pd_probe import print_table as print_pd_table
from terrain_v56_stand_force_compare.pd_probe import write_csv as write_pd_csv
from terrain_v56_stand_force_compare.pd_probe import write_plots as write_pd_plots

from isaaclab_tasks.utils.hydra import hydra_task_config

CSV_COLUMNS = (
    "env",
    "step",
    "t_s",
    *(f"f_{label}_N" for label in FOOT_LABELS),
    "f_total_N",
    "f_total_over_weight",
    "cop_x_m",
    "cop_y_m",
    "cop_offset_m",
    "base_z_m",
    "roll_rad",
    "pitch_rad",
    "tau_rms_Nm",
    "sat_count",
    *(f"contact_{label}" for label in FOOT_LABELS),
)


def _write_plots(out_dir: str, label: str, time_s: np.ndarray, rows: dict[str, np.ndarray], fs: float) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed, skipping the figures.", flush=True)
        return []

    written = []
    # Figures show the environment mean; the per-env detail is in the CSV.
    mean_feet = rows["f_feet"].mean(axis=1)  # [T, 4]
    total = rows["f_total"].mean(axis=1)  # [T]
    weight = float(rows["weight"][0, 0])

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for i, foot in enumerate(FOOT_LABELS):
        ax_top.plot(time_s, mean_feet[:, i], lw=0.9, label=foot)
    ax_top.plot(time_s, total, color="k", lw=1.2, label="total")
    ax_top.axhline(weight, color="C3", ls="--", lw=0.9, label="m g")
    ax_top.set_ylabel("normal force [N]")
    ax_top.set_title(f"{label}: ground reaction force")
    ax_top.legend(ncol=5, fontsize=8)
    ax_top.grid(alpha=0.3)

    ax_bottom.plot(time_s, total / weight, lw=1.0, color="C0")
    ax_bottom.axhline(1.0, color="C3", ls="--", lw=0.9)
    ax_bottom.set_xlabel("time [s]")
    ax_bottom.set_ylabel(r"$\sum F_z / mg$")
    ax_bottom.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "forces.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    series = [(total, "total"), *((mean_feet[:, i], foot) for i, foot in enumerate(FOOT_LABELS))]
    for values, name in series:
        freq, power = psd(values / weight, fs)
        ax.semilogy(freq[1:], power[1:], lw=0.9, label=name)
    ax.set_xlim(0, min(10.0, fs / 2.0))
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"PSD of $F_z / mg$  [1/Hz]")
    ax.set_title(f"{label}: load spectrum")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "psd.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    share = mean_feet.sum(axis=0) / mean_feet.sum()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(FOOT_LABELS, share * 100.0, yerr=mean_feet.std(axis=0) / mean_feet.sum() * 100.0, capsize=4)
    ax.axhline(25.0, color="C3", ls="--", lw=0.9)
    ax.set_ylabel("mean load share [%]")
    ax.set_title(f"{label}: load distribution")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, "load_share.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    label = args_cli.label or args_cli.task.replace("Isaac-Go2-", "").replace("-v0", "").lower()
    out_dir = args_cli.output_dir or os.path.join(os.path.dirname(os.path.abspath(args_cli.checkpoint)), f"force_eval_{label}")
    os.makedirs(out_dir, exist_ok=True)
    play_overrides(
        env_cfg,
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        device=args_cli.device,
        warmup=args_cli.warmup,
        duration=args_cli.duration,
    )

    print(f"[INFO] task      : {args_cli.task}  ({label})", flush=True)
    print(f"[INFO] checkpoint: {args_cli.checkpoint}", flush=True)
    print(f"[INFO] output    : {out_dir}", flush=True)
    print(
        f"[INFO] plant     : {1.0 / (env_cfg.decimation * env_cfg.sim.dt):.0f} Hz policy, "
        f"{1.0 / env_cfg.sim.dt:.0f} Hz physics, {env_cfg.scene.num_envs} envs",
        flush=True,
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    physics_dt = env_cfg.decimation * env_cfg.sim.dt
    video_recorder = None
    if args_cli.video:
        video_seconds = args_cli.video_seconds if args_cli.video_seconds is not None else args_cli.warmup + args_cli.duration
        video_length = int(round(video_seconds / physics_dt))
        video_folder = os.path.join(out_dir, "videos")
        # Playback has to run at the control rate or the clip is in slow motion (V55 does the same).
        env.unwrapped.metadata["render_fps"] = max(1, int(round(1.0 / physics_dt)))
        env.unwrapped.sim.set_camera_view(eye=args_cli.camera_eye, target=args_cli.camera_target)
        print(f"[INFO] video     : {video_folder} ({video_length} steps, {video_length * physics_dt:.1f} s)", flush=True)
        # Same recipe as IsaacLab's own play.py: record from step 0 for a fixed number of steps.
        video_recorder = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )
        env = video_recorder

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    step_dt = base_env.step_dt
    weight = float(base_env.scene["robot"].root_physx_view.get_masses()[0].sum()) * GRAVITY
    print(f"[INFO] m g       : {weight:.2f} N", flush=True)

    runner_kwargs = {"log_dir": out_dir, "device": str(env.device)}
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), **runner_kwargs)
    elif agent_cfg.class_name == "OffPolicyRunner":
        runner = OffPolicyRunner(env, agent_cfg.to_dict(), **runner_kwargs)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    probe = None
    if args_cli.pd_probe != "off" and torque_term(env) is not None:
        probe = PdProbe(policy, env, stock_kp=args_cli.stock_kp, stock_kd=args_cli.stock_kd)
        print(
            f"[INFO] pd probe : Kp/Kd from d(tau)/d(obs), {probe.n_joints} joints, "
            f"action_scale {probe.action_scale:.2f} N.m, limit +-{probe.torque_limit:.1f} N.m",
            flush=True,
        )
    elif args_cli.pd_probe == "on":
        print("[WARN] --pd_probe on, but this task has no direct-torque action term: skipped", flush=True)

    env.reset()
    obs = env.get_observations()

    warmup_steps = int(round(args_cli.warmup / step_dt))
    recorded_steps = int(round(args_cli.duration / step_dt))
    samples: dict[str, list[np.ndarray]] = {}
    time_s = np.arange(recorded_steps) * step_dt
    resets = 0
    # Why the episodes end is the difference between "the robot stands" and "the metrics are
    # empty", so the termination counters training logs already print are collected here too.
    terminations: dict[str, float] = {}
    pd_samples: list[dict[str, np.ndarray]] = []
    pd_summary = None
    pd_stacked = None

    with torch.inference_mode():
        for step in range(warmup_steps + recorded_steps):
            if probe is None:
                actions = policy(obs)
            else:
                # Same action the policy would have returned, plus this step's gains.  The probe
                # lifts the inference mode itself, so the probe call has to stay inside the loop
                # body rather than outside it.
                actions, gains = probe(obs)
                if step >= warmup_steps:
                    pd_samples.append(gains)
            obs, _reward, dones, _extras = env.step(actions)
            # ``RslRlVecEnvWrapper.step`` has already reset the finished environments and handed
            # back the fresh observations.  Resetting a second time here only zeroes
            # ``episode_length_buf`` again, which keeps the force metrics' warm-up mask off for
            # the rest of the run (and made every force number read 0).
            resets += int(dones.sum())
            if isinstance(_extras, dict):
                for key, value in (_extras.get("log") or {}).items():
                    if isinstance(key, str) and key.startswith("Episode_Termination"):
                        terminations[key] = terminations.get(key, 0.0) + float(np.asarray(value).mean())
            if step < warmup_steps:
                continue
            for key, value in force_snapshot(base_env).items():
                samples.setdefault(key, []).append(value)

    rows: dict[str, np.ndarray] = {key: np.stack(values) for key, values in samples.items()}
    rows["weight"] = np.full((recorded_steps, args_cli.num_envs), weight)

    # --- csv ---------------------------------------------------------------------------------
    csv_path = os.path.join(out_dir, "timeseries.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for env_id in range(args_cli.num_envs):
            for index in range(recorded_steps):
                writer.writerow(
                    [
                        env_id,
                        index,
                        f"{time_s[index]:.4f}",
                        *(f"{rows['f_feet'][index, env_id, i]:.3f}" for i in range(4)),
                        f"{rows['f_total'][index, env_id]:.3f}",
                        f"{rows['f_total'][index, env_id] / weight:.4f}",
                        f"{rows['cop_x'][index, env_id]:.4f}",
                        f"{rows['cop_y'][index, env_id]:.4f}",
                        f"{rows['cop_offset'][index, env_id]:.5f}",
                        f"{rows['base_z'][index, env_id]:.4f}",
                        f"{rows['roll'][index, env_id]:.4f}",
                        f"{rows['pitch'][index, env_id]:.4f}",
                        f"{rows['tau_rms'][index, env_id]:.3f}",
                        f"{int(rows['sat_count'][index, env_id])}",
                        *(int(rows["contact"][index, env_id, i]) for i in range(4)),
                    ]
                )

    # --- statistics over envs x steps ---------------------------------------------------------
    summary = {
        "label": label,
        "task": args_cli.task,
        "checkpoint": os.path.abspath(args_cli.checkpoint),
        "control_hz": round(1.0 / step_dt, 3),
        "physics_hz": round(1.0 / env_cfg.sim.dt, 3),
        "num_envs": args_cli.num_envs,
        "duration_s": args_cli.duration,
        "steps": recorded_steps,
        "resets": resets,
        "terminations": terminations,
        "weight_N": round(weight, 3),
        **summarise_force(rows, weight, step_dt),
    }

    if pd_samples:
        pd_stacked = stack_records(pd_samples)
        pd_summary = summarise_gains(
            pd_stacked,
            probe.joint_names,
            stock_kp=args_cli.stock_kp,
            stock_kd=args_cli.stock_kd,
        )
        summary["pd"] = pd_summary
        write_pd_csv(os.path.join(out_dir, "pd_gains.csv"), pd_stacked, probe.joint_names, step_dt)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    if not args_cli.no_plots:
        _write_plots(out_dir, label, time_s, rows, 1.0 / step_dt)
        if pd_summary is not None:
            write_pd_plots(out_dir, label, pd_stacked, pd_summary, step_dt)

    print("\n[RESULT]", flush=True)
    print(
        f"  sum(Fz)/mg = {summary['force']['total_over_weight_mean']:.3f} "
        f"+- {summary['force']['total_over_weight_std']:.3f}",
        flush=True,
    )
    print(f"  load share  = " + "  ".join(f"{k} {v * 100:.1f}%" for k, v in summary["force"]["load_share"].items()), flush=True)
    print(
        f"  asym LR/FB  = {summary['force']['asym_LR']:.3f} / {summary['force']['asym_FB']:.3f}   "
        f"CoP |offset| = {summary['cop']['offset_mean_m'] * 1000:.1f} mm   "
        f"CoP speed = {summary['cop']['speed_mean_mps'] * 1000:.1f} mm/s",
        flush=True,
    )
    print(
        f"  |tau| rms   = {summary['actuation']['tau_rms_Nm']:.2f} N.m   "
        f"sat = {summary['actuation']['sat_frac'] * 100:.2f} %   "
        f"base z = {summary['pose']['base_z_mean_m']:.3f} m   resets = {resets}",
        flush=True,
    )
    print(f"  artefacts   : {out_dir}", flush=True)

    for key, value in sorted(terminations.items()):
        print(f"  {key:52s} {value:+.3f}", flush=True)

    if pd_summary is not None:
        print_pd_table(pd_summary)

    if video_recorder is not None:
        # ``os._exit`` below skips every finaliser, so the open episode has to be flushed here;
        # V55 reaches the same point by closing the wrapper instead.
        try:
            video_recorder.stop_recording()
        except AttributeError:
            video_recorder.close()
        print(f"  video       : {os.path.join(out_dir, 'videos')}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    # Same teardown hang as in train.py.
    os._exit(0)


if __name__ == "__main__":
    main()
