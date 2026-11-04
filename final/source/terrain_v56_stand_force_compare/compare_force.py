"""Merge the ``summary.json`` files of several V56 rollouts into one comparison.

This script needs no simulation, no checkpoint and no GPU - it only reads the artefacts that
``evaluate_force.py`` wrote, so it can also be run on a laptop:

    python compare_force.py --run pos=force_eval/pos --run torque=force_eval/torque \
        --out force_eval/compare

All four arms can share one table - ``--run pos=... --run torque=... --run pos200=... --run
torque200=...``.  Labels are free-form, the only convention is that the pair a conclusion is
drawn from shares the same suffix (``pos``/``torque`` at 50 Hz, ``pos200``/``torque200`` at 200 Hz).

Outputs (in ``--out``):
    compare.json        the merged numbers
    compare.md          the same, as a markdown table for the report
    compare_bars.png    the four quantities the comparison is about, side by side
    compare_psd.png     load spectra of all runs on top of each other (needs timeseries.csv)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

parser = argparse.ArgumentParser(description="Compare the ground reaction forces of several V56 runs.")
parser.add_argument(
    "--run",
    action="append",
    default=None,
    metavar="LABEL=DIR",
    help="A run: label and the directory that evaluate_force.py wrote. Repeatable.",
)
parser.add_argument("--out", type=str, default=None, help="Output directory (default: <first dir>/compare).")
parser.add_argument("--no_plots", action="store_true", default=False, help="Skip the figures.")
args_cli = parser.parse_args()

if not args_cli.run:
    parser.error("at least one --run LABEL=DIR is required")

RUNS: list[tuple[str, str]] = []
for item in args_cli.run:
    if "=" not in item:
        parser.error(f"--run expects LABEL=DIR, got '{item}'")
    label, directory = item.split("=", 1)
    RUNS.append((label, os.path.abspath(directory)))

out_dir = os.path.abspath(args_cli.out or os.path.join(RUNS[0][1], "compare"))
os.makedirs(out_dir, exist_ok=True)

SUMMARIES: dict[str, dict] = {}
for label, directory in RUNS:
    path = os.path.join(directory, "summary.json")
    if not os.path.isfile(path):
        raise SystemExit(f"[ERROR] {path} not found - run evaluate_force.py first")
    with open(path, encoding="utf-8") as handle:
        SUMMARIES[label] = json.load(handle)

LABELS = [label for label, _ in RUNS]

# (title, extractor, format, unit, lower_is_better)
METRICS: tuple[tuple[str, callable, str, str, bool], ...] = (
    ("control rate", lambda s: s["control_hz"], "{:.0f}", "Hz", False),
    (
        "sum(F_z)/mg",
        lambda s: f"{s['force']['total_over_weight_mean']:.3f} +- {s['force']['total_over_weight_std']:.3f}",
        "{}",
        "",
        False,
    ),
    ("load spread / total", lambda s: s["force"]["load_std_frac"], "{:.3f}", "", True),
    ("asymmetry L-R", lambda s: s["force"]["asym_LR"], "{:.3f}", "", True),
    ("asymmetry F-B", lambda s: s["force"]["asym_FB"], "{:.3f}", "", True),
    ("front share", lambda s: s["force"]["front_share"], "{:.3f}", "", False),
    ("all four feet loaded", lambda s: s["contact"]["duty_all4"], "{:.3f}", "", False),
    ("CoP offset", lambda s: s["cop"]["offset_mean_m"] * 1000.0, "{:.1f}", "mm", True),
    ("CoP speed", lambda s: s["cop"]["speed_mean_mps"] * 1000.0, "{:.1f}", "mm/s", True),
    ("CoP drift", lambda s: s["cop"]["drift_m"] * 1000.0, "{:.1f}", "mm", True),
    ("base height", lambda s: s["pose"]["base_z_mean_m"], "{:.3f}", "m", False),
    ("base height std", lambda s: s["pose"]["base_z_std_m"] * 1000.0, "{:.1f}", "mm", True),
    ("roll rms", lambda s: np.degrees(s["pose"]["roll_rms_rad"]), "{:.2f}", "deg", True),
    ("pitch rms", lambda s: np.degrees(s["pose"]["pitch_rms_rad"]), "{:.2f}", "deg", True),
    ("tau rms", lambda s: s["actuation"]["tau_rms_Nm"], "{:.2f}", "N.m", True),
    ("saturation", lambda s: s["actuation"]["sat_frac"] * 100.0, "{:.2f}", "%", True),
    ("resets", lambda s: s["resets"], "{:d}", "", True),
)

# Old summaries predate ``reference.py``; the rows are only added when every run has them, so a
# mixed set still renders instead of raising KeyError halfway through the table.
REFERENCE_ROWS: tuple[tuple[str, callable, str, str, bool], ...] = (
    ("Ref dq_max", lambda s: s["reference"]["dq_max_rad"], "{:.3f}", "rad", True),
    ("Ref tilt", lambda s: s["reference"]["tilt_mean_deg"], "{:.2f}", "deg", True),
    ("Ref d_ell", lambda s: s["reference"]["d_ell_mean_mm"], "{:+.1f}", "mm", False),
)
if all("reference" in summary for summary in SUMMARIES.values()):
    METRICS = METRICS + REFERENCE_ROWS

header = "| metric | unit | " + " | ".join(LABELS) + " |"
divider = "|---|---|" + "---|" * len(LABELS)
lines = [header, divider]
for title, extract, fmt, unit, _ in METRICS:
    cells = []
    for label in LABELS:
        value = extract(SUMMARIES[label])
        cells.append(value if isinstance(value, str) else fmt.format(value))
    lines.append(f"| {title} | {unit} | " + " | ".join(cells) + " |")
table = "\n".join(lines)

print()
print(table)
print()
for label, (_, directory) in zip(LABELS, RUNS):
    summary = SUMMARIES[label]
    share = "  ".join(f"{foot} {value * 100:5.1f}%" for foot, value in summary["force"]["load_share"].items())
    print(f"[{label}] load share: {share}")
    print(f"[{label}] checkpoint: {summary.get('checkpoint')}  ({directory})")

with open(os.path.join(out_dir, "compare.json"), "w", encoding="utf-8") as handle:
    json.dump({label: SUMMARIES[label] for label in LABELS}, handle, indent=2, sort_keys=True)
with open(os.path.join(out_dir, "compare.md"), "w", encoding="utf-8") as handle:
    handle.write(table + "\n")
print(f"\n[INFO] table written to {os.path.join(out_dir, 'compare.md')}")

if args_cli.no_plots:
    sys.exit(0)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[WARN] matplotlib is not installed, skipping the figures.")
    sys.exit(0)

# --- the four quantities the comparison is about ---------------------------------------------
PANELS = (
    ("load spread / total", lambda s: s["force"]["load_std_frac"]),
    ("asymmetry L-R", lambda s: s["force"]["asym_LR"]),
    ("CoP speed [mm/s]", lambda s: s["cop"]["speed_mean_mps"] * 1000.0),
    ("|tau| rms [N.m]", lambda s: s["actuation"]["tau_rms_Nm"]),
)
fig, axes = plt.subplots(1, len(PANELS), figsize=(4 * len(PANELS), 3.4))
for ax, (title, extract) in zip(np.atleast_1d(axes), PANELS):
    values = [extract(SUMMARIES[label]) for label in LABELS]
    ax.bar(LABELS, values, color=[f"C{i}" for i in range(len(LABELS))])
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}", ha="center", va="bottom", fontsize=8)
fig.suptitle("V56: position vs direct torque (flat, standing)", fontsize=10)
fig.tight_layout()
path = os.path.join(out_dir, "compare_bars.png")
fig.savefig(path, dpi=140)
plt.close(fig)
print(f"[INFO] {path}")

# --- load share -------------------------------------------------------------------------------
feet = ("FL", "FR", "RL", "RR")
fig, ax = plt.subplots(figsize=(6, 3.6))
width = 0.8 / len(LABELS)
x = np.arange(len(feet))
for i, label in enumerate(LABELS):
    share = [SUMMARIES[label]["force"]["load_share"][foot] * 100.0 for foot in feet]
    ax.bar(x + (i - (len(LABELS) - 1) / 2) * width, share, width=width, label=label)
ax.axhline(25.0, color="C3", ls="--", lw=0.9, label="even share")
ax.set_xticks(x, feet)
ax.set_ylabel("mean load share [%]")
ax.set_title("V56: static load distribution")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
path = os.path.join(out_dir, "compare_load_share.png")
fig.savefig(path, dpi=140)
plt.close(fig)
print(f"[INFO] {path}")

# --- spectra ----------------------------------------------------------------------------------
curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
for label, directory in RUNS:
    csv_path = os.path.join(directory, "timeseries.csv")
    if not os.path.isfile(csv_path):
        continue
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    # One environment is enough for a spectrum, and it keeps the time axis monotonic.
    mask = data["env"] == data["env"].min()
    total = data["f_total_N"][mask]
    sample_rate = 1.0 / float(np.median(np.diff(data["t_s"][mask])))
    signal = total / total.mean() - 1.0  # fractional fluctuation: comparable across runs
    window = np.hanning(signal.size)
    spectrum = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    power = 2.0 * np.abs(spectrum) ** 2 / (sample_rate * np.sum(window**2))
    curves[label] = (freqs, power)

if curves:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (freqs, power) in curves.items():
        ax.semilogy(freqs[1:], power[1:], lw=0.9, label=label)
    ax.set_xlim(0, 10.0)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD of relative F_z fluctuation [1/Hz]")
    ax.set_title("V56: load spectrum")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "compare_psd.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[INFO] {path}")
