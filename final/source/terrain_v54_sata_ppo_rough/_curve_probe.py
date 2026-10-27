"""Temporary: dump the V54 tfevents training curves that explain the late-run degradation."""

import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ckpt_server", "v54_run")

ea = EventAccumulator(sorted(glob.glob(os.path.join(RUN, "events.out.tfevents.*")))[-1],
                      size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags()["scalars"]
print("TAGS:")
for t in tags:
    print("  ", t)

WANT = [
    "Loss/learning_rate",
    "Train/mean_episode_length",
    "Train/mean_reward",
    "Metrics/base_velocity/error_vel_xy",
    "Policy/mean_noise_std",
    "Loss/surrogate",
    "Loss/value",
    "Loss/entropy",
    "Perf/total_fps",
    "Episode/reward",
    "Episode_Termination/fell_over",
    "Episode_Termination/time_out",
    "Episode_Reward/forward",
    "Episode_Reward/roll",
    "Episode_Reward/dof_acc",
]

for t in WANT:
    if t not in tags:
        print(f"\n{t}: MISSING")
        continue
    s = ea.Scalars(t)
    print(f"\n{t}  (n={len(s)})")
    for frac in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        i = min(len(s) - 1, int(round(frac * (len(s) - 1))))
        print(f"   step {s[i].step:5d}  {s[i].value:.6f}")
