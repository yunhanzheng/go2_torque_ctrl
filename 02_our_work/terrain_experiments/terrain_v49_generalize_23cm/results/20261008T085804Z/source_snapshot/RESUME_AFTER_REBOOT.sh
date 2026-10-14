#!/usr/bin/env bash
set -euo pipefail

if pgrep -f 'train_eth_fixed05_v2.py' >/dev/null; then
  echo '已有 Isaac/SAC 训练进程，拒绝重复启动。'
  exit 1
fi

/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python -c 'import torch; x=torch.ones(1, device="cuda"); print("CUDA preflight:", x, torch.cuda.get_device_name(0))'
/home/ubuntu22/terrain_v49_generalize_23cm/run_visible.sh
/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python -u /home/ubuntu22/terrain_v49_generalize_23cm/evaluate_matrix.py
