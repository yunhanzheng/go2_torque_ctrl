#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
export V34_FLAT_RECOVERY_SOURCE=/home/ubuntu22/logs/rsl_rl/terrain_v34_scratch_flat_foundation/2026-09-20_16-31-55_stage3_20260920T163148Z/model_1597.pt
/usr/bin/python3 -u /home/ubuntu22/terrain_v34_from_scratch_four_terrain/supervisor_curriculum.py 2>&1 | tee -a /home/ubuntu22/terrain_v34_from_scratch_four_terrain/curriculum_visible.log
echo 'V34 0.5m/s门禁、四地形训练、评估与录像流程已结束。'
read -r -p '按回车关闭窗口' _
