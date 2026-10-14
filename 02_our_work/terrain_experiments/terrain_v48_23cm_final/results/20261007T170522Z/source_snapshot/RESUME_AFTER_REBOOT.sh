#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22

if pgrep -f '/terrain_v48_23cm_final/train.py' >/dev/null; then
  echo 'V48 is already running; refusing to start a duplicate.' >&2
  exit 1
fi

/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is still unavailable; reboot the machine first"
probe = torch.zeros(1, device="cuda:0")
print("CUDA preflight passed:", probe)
PY

exec /home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python -u \
  /home/ubuntu22/terrain_v48_23cm_final/supervisor.py
