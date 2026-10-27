#!/usr/bin/env bash
set -euo pipefail

/opt/conda/envs/isaac_sac/bin/python -c 'import torch; x=torch.ones(1, device="cuda"); print("CUDA preflight:", x, torch.cuda.get_device_name(0))'
/home/ubuntu22/terrain_v49_generalize_23cm/run_visible.sh
/opt/conda/envs/isaac_sac/bin/python -u /home/ubuntu22/terrain_v49_generalize_23cm/evaluate_matrix.py
