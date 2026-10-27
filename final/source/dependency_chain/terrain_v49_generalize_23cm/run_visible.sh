#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
exec /opt/conda/envs/isaac_sac/bin/python -u /home/ubuntu22/terrain_v49_generalize_23cm/supervisor.py
