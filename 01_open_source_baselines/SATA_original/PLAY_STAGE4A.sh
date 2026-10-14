#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/data/sata/SATA_flat_reproduction/legged_gym:/data/sata/SATA_flat_reproduction/rsl_rl
cd /data/sata/SATA_flat_reproduction/legged_gym

/home/ubuntu22/miniforge3/bin/conda run --no-capture-output -n sata \
  python legged_gym/scripts/play_rough.py \
  --task=go2_torque_rough_gait \
  --load_run=Aug06_19-21-04_stage4a_rough_curriculum_seed1_visible \
  --checkpoint=4900
