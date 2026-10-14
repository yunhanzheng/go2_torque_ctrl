"""Export publication-ready V38 training curves from TensorBoard events."""
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parent
EVENTS = sorted((Path("/home/ubuntu22/logs/rsl_rl/terrain_v38_official_stairs")).glob(
    "*/events.out.tfevents.*"
))
OUT = ROOT / "figures"
TAGS = {
    "reward": "Train/mean_reward",
    "terrain_level": "Curriculum/terrain_levels",
    "episode_length": "Train/mean_episode_length",
    "invalid_upright": "Episode_Termination/invalid_upright_state",
}


def load(tag):
    # Later resume files intentionally replace overlapping points from earlier runs.
    points = {}
    for event_file in EVENTS:
        accumulator = EventAccumulator(str(event_file)).Reload()
        for event in accumulator.Scalars(tag):
            points[event.step] = event.value
    steps = np.array(sorted(points), dtype=float)
    values = np.array([points[int(step)] for step in steps], dtype=float)
    return steps, values


def ema(values, span=25):
    alpha = 2.0 / (span + 1.0)
    result = np.empty_like(values)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def main():
    OUT.mkdir(exist_ok=True)
    series = {name: load(tag) for name, tag in TAGS.items()}
    common_steps = series["reward"][0].astype(int)
    csv_path = OUT / "v38_training_curves.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["iteration", *TAGS])
        mappings = {
            name: {int(step): float(value) for step, value in zip(steps, values)}
            for name, (steps, values) in series.items()
        }
        for step in common_steps:
            writer.writerow([step, *(mappings[name].get(step, "") for name in TAGS)])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    panels = [
        ("reward", "Mean episode reward", "Reward"),
        ("terrain_level", "Terrain curriculum level", "Mean level (0–9)"),
        ("episode_length", "Mean episode length", "Steps"),
        ("invalid_upright", "Invalid-upright termination", "Reported rate"),
    ]
    for axis, (name, title, ylabel) in zip(axes.flat, panels):
        steps, values = series[name]
        axis.plot(steps, values, color="#5B8FF9", alpha=0.22, linewidth=0.8, label="raw")
        axis.plot(steps, ema(values), color="#174EA6", linewidth=2.0, label="EMA-25")
        axis.axvline(2900, color="#D97706", linestyle="--", linewidth=1.2, label="resume")
        axis.set_title(title, fontsize=12)
        axis.set_ylabel(ylabel)
        axis.margins(x=0)
    axes[1, 0].set_xlabel("Training iteration")
    axes[1, 1].set_xlabel("Training iteration")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=3, frameon=False)
    fig.suptitle("V38 multi-terrain curriculum training", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    png = OUT / "v38_training_curves.png"
    pdf = OUT / "v38_training_curves.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)
    print(csv_path)


if __name__ == "__main__":
    main()
