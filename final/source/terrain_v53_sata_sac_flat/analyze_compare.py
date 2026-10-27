"""Compare V52 (PPO) vs V53 (SAC) runs: checkpoints, curves, diagnostics."""
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

S = "/root/bayes-tmp/xiaoyulong/yunhanzheng/go2_torque_ctrl/final/source"
RUNS = {
    "V52_PPO": f"{S}/terrain_v52_sata_ppo_flat/logs/rsl_rl/terrain_v52_sata_ppo_flat/2026-10-20_12-42-39_V52_sata_ppo_flat_20261020T044229Z",
    "V53_SAC": f"{S}/terrain_v53_sata_sac_flat/logs/rsl_rl/terrain_v53_sata_sac_flat/2026-10-21_09-40-54_V53_sata_sac_flat_20261021T014043Z",
}

KEY = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Termination/time_out",
    "Episode_Termination/base_contact",
    "Episode_Termination/invalid_upright_state",
]

data = {}
for name, d in RUNS.items():
    models = sorted(glob.glob(os.path.join(d, "model_*.pt")))
    print("=" * 70)
    print(name)
    print("  checkpoints:", len(models), "| last:", os.path.basename(models[-1]) if models else None)
    ev = sorted(glob.glob(os.path.join(d, "events.out.tfevents.*")))
    ea = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    ea.Reload()
    data[name] = ea
    tags = ea.Tags()["scalars"]
    print("  scalar tags (%d):" % len(tags))
    for t in tags:
        print("   -", t)

print("\n" + "=" * 70)
print("LEARNING CURVES (sampled at ~1/8 intervals)")
for tag in KEY:
    print("\n###", tag)
    for name, ea in data.items():
        tags = ea.Tags()["scalars"]
        if tag not in tags:
            print(f"  {name}: <absent>")
            continue
        s = ea.Scalars(tag)
        n = len(s)
        idx = sorted(set([0, n // 8, n // 4, 3 * n // 8, n // 2, 5 * n // 8, 3 * n // 4, 7 * n // 8, n - 1]))
        pts = " ".join(f"{s[i].step}:{s[i].value:.3f}" for i in idx)
        print(f"  {name} (n={n}): {pts}")

print("\n" + "=" * 70)
print("EPISODE REWARD TERMS (first -> last)")
for name, ea in data.items():
    print(f"  {name}:")
    for t in sorted(t for t in ea.Tags()["scalars"] if t.startswith("Episode_Reward/")):
        s = ea.Scalars(t)
        print(f"    {t.split('/')[-1]:22s} {s[0].value:+.4f} -> {s[-1].value:+.4f}")

print("\n" + "=" * 70)
print("ALGO-SPECIFIC DIAGNOSTICS")
for name, ea in data.items():
    print(f"  {name}:")
    for t in sorted(ea.Tags()["scalars"]):
        if any(k in t for k in ("Loss", "KL", "Learning rate", "Entropy", "alpha", "Alpha", "noise", "Noise")):
            s = ea.Scalars(t)
            print(f"    {t:45s} {s[0].value:+.5f} -> {s[-1].value:+.5f}")
