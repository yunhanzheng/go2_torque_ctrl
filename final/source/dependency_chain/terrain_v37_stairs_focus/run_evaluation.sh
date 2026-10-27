#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v37_stairs_focus/evaluate_candidates.py 2>&1 | tee -a /home/ubuntu22/terrain_v37_stairs_focus/evaluation_visible.log

