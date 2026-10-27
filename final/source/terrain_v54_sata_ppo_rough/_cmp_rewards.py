"""Temporary: V54 final per-step reward breakdown vs SATA Oct14_16-33-32 (hard-coded)."""

import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

V54_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ckpt_server", "v54_run")

# SATA's final iteration (wandb output.log, iter 2999)
SATA = {
    "Episode_Reward/forward": 7.6433,
    "Episode_Reward/head_height": 2.5014,
    "Episode_Reward/moving_y": 3.0335,
    "Episode_Reward/moving_yaw": 2.4137,
    "Episode_Reward/soft_dof_pos_limits": -0.0308,
    "Episode_Reward/motor_fatigue": -3.0028,
    "Episode_Reward/dof_acc": -2.3714,
    "Episode_Reward/roll": -0.3888,
    "Episode_Reward/lin_vel_z": -0.1552,
}
SATA_EP_LEN = 1952.14
SATA_REWARD = 97.19

ea = EventAccumulator(sorted(glob.glob(os.path.join(V54_DIR, "events.out.tfevents.*")))[-1],
                      size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags()["scalars"]
print("V54 reward tags:", [t for t in tags if t.startswith("Episode_Reward")])
print("V54 episode tags:", [t for t in tags if t.startswith("Episode") and "Reward" not in t])

ev = {
    t: (ea.Scalars(t)[-1].step, ea.Scalars(t)[-1].value)
    for t in tags
    if t.startswith("Episode_Reward") or t.startswith("Episode/") or t.startswith("Episode_Termination")
}
for t in sorted(ev):
    print(f"{t:45s} step={ev[t][0]:5d} value={ev[t][1]:.6f}")

v54_ep_len = next((v[1] for k, v in ev.items() if k.startswith("Episode/episode_length")), float("nan"))
print(f"\n{'term':38s} {'SATA/step':>11s} {'V54/step':>11s} {'ratio':>8s}")
for tag, sata_val in SATA.items():
    if tag not in ev:
        print(f"{tag:38s} {sata_val / SATA_EP_LEN:11.6f} {'MISSING':>11s}")
        continue
    v54_val = ev[tag][1]
    s = sata_val / SATA_EP_LEN
    v = v54_val / v54_ep_len
    ratio = (v / s) if abs(s) > 1e-12 else float("nan")
    print(f"{tag:38s} {s:11.6f} {v:11.6f} {ratio:8.3f}")
print(f"\nepisode_length: SATA {SATA_EP_LEN:.1f} | V54 {v54_ep_len:.1f}")
print(f"total reward:   SATA {SATA_REWARD:.2f} | V54 {ev.get('Episode/reward', (0, float('nan')))[1]:.2f}")
