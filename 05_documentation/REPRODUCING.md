# Reproducing and reviewing the project

## System requirements

- Ubuntu 22.04;
- NVIDIA GPU, compatible driver, CUDA, and Isaac Sim 5.1;
- Python 3.11 for the final Isaac Lab experiment chain;
- Git LFS for model and video files.

Isaac Sim/Isaac Gym binaries and Conda environments are not redistributed.
Install them from NVIDIA and follow the included IsaacLab-SAC README.

## Source layout

Install or expose the included IsaacLab-SAC, RSL-RL-SAC, legacy SAC workspace,
and repository root. The archived experiment scripts use the original paths
`/home/ubuntu22` and `/data/sac`; search and replace those with local absolute
paths before executing on another machine.

The original final evaluator is
`02_our_work/terrain_experiments/terrain_v50_curriculum_recovery/evaluate_final.py`.
Update its `PYTHON`, `MODEL`, `PYTHONPATH`, and subprocess working directory, and
point `MODEL` to `03_final_models/V50_curriculum_recovery/model_5099.pt`.

The evaluator imports the complete V25--V50 config chain. Do not run an isolated
copy of only the V50 folder.

## Reviewing without retraining

- compare JSON counts in the V49/V50 evaluation folders;
- watch `.webm` files in V48--V50 `video_reports/`;
- read training logs under each experiment's `results/` folder;
- validate checkpoints with `03_final_models/README.md`.

Hundreds of near-adjacent intermediate checkpoints, generated caches, and
duplicate archives were omitted. They exceed 3 GB while adding little review
value. Source, selected checkpoints, final logs, raw evaluations, and videos are
preserved.

