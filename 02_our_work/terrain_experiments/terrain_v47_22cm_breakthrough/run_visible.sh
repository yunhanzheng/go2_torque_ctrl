#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
exec /home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python -u /home/ubuntu22/terrain_v47_22cm_breakthrough/supervisor.py
