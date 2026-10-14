#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE=offline
export PYTHONPATH=/data/sata/SATA_flat_reproduction/legged_gym:/data/sata/SATA_flat_reproduction/rsl_rl
cd /data/sata/SATA_flat_reproduction/legged_gym

/home/ubuntu22/miniforge3/bin/conda run --no-capture-output -n sata \
  python legged_gym/scripts/train.py \
  --task=go2_torque_gait \
  --headless \
  --num_envs=4096 \
  --max_iterations=3000 \
  --seed=1 \
  --run_name=stage3_diagonal_gait_seed1

