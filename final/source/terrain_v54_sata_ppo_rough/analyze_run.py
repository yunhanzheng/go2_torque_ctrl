"""Summarize a rsl_rl run directory: checkpoints + sampled scalar curves.

Usage: python analyze_run.py <run_dir>
"""
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

D = sys.argv[1]
ev = sorted(glob.glob(os.path.join(D, "events.out.tfevents.*")))[-1]
ea = EventAccumulator(ev, size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags()["scalars"]

models = [int(os.path.basename(p).split("_")[1].split(".")[0]) for p in glob.glob(os.path.join(D, "model_*.pt"))]
print("checkpoints:", len(models), "| max iter:", max(models) if models else None)


def show(tag, n=12):
    if tag not in tags:
        print(f"{tag}: <absent>")
        return
    s = ea.Scalars(tag)
    N = len(s)
    if N >= n:
        idx = sorted(set([0] + [min(N - 1, N * i // (n - 1)) for i in range(1, n)]))
    else:
        idx = list(range(N))
    print(f"{tag} (n={N}): " + " ".join(f"{s[i].step}:{s[i].value:.3f}" for i in idx))


for t in [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Curriculum/terrain_levels",
    "Episode_Termination/time_out",
    "Episode_Termination/base_contact",
    "Perf/total_fps",
    "Perf/collection_time",
    "Perf/learning_time",
    "Loss/value",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/learning_rate",
    "Policy/mean_noise_std",
]:
    show(t)

print("\n-- Episode_Reward first->last --")
for t in sorted(x for x in tags if x.startswith("Episode_Reward/")):
    s = ea.Scalars(t)
    print(f"  {t.split('/')[-1]:22s} {s[0].value:+.4f} -> {s[-1].value:+.4f}")

print("\n-- Metrics first->last --")
for t in sorted(x for x in tags if x.startswith("Metrics/")):
    s = ea.Scalars(t)
    print(f"  {t:44s} {s[0].value:+.4f} -> {s[-1].value:+.4f}")

print("\n-- all tags --")
print(", ".join(sorted(tags)))
