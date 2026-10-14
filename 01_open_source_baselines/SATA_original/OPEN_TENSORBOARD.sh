#!/usr/bin/env bash
set -euo pipefail

/home/ubuntu22/miniforge3/bin/conda run --no-capture-output -n sata \
  tensorboard \
  --logdir /data/sata/SATA_flat_reproduction/legged_gym/logs \
  --port 6006 \
  --bind_all

