#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
export V34_MEDIUM_RECOVERY_SOURCE=/home/ubuntu22/logs/rsl_rl/terrain_v34_scratch_stage2_medium/2026-09-20_15-50-00_stage2_20260920T152558Z/model_998.pt
/usr/bin/python3 -u /home/ubuntu22/terrain_v34_from_scratch_four_terrain/supervisor_curriculum.py 2>&1 | tee -a /home/ubuntu22/terrain_v34_from_scratch_four_terrain/curriculum_visible.log
echo 'V34已从自训中速 model_998 继续 0.5 m/s 及四地形流程。'
read -r -p '按回车关闭窗口' _
