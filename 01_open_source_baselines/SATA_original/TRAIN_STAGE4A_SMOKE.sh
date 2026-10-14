#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE=offline
export PYTHONPATH=/data/sata/SATA_flat_reproduction/legged_gym:/data/sata/SATA_flat_reproduction/rsl_rl
cd /data/sata/SATA_flat_reproduction/legged_gym

/home/ubuntu22/miniforge3/bin/conda run --no-capture-output -n sata \
  python legged_gym/scripts/train.py \
  --task=go2_torque_rough_gait \
  --resume \
  --load_run=Aug05_19-46-44_stage3_diagonal_gait_seed1_visible \
  --checkpoint=3000 \
  --max_iterations=5 \
  --run_name=stage4a_rough_smoke \
  --headless
