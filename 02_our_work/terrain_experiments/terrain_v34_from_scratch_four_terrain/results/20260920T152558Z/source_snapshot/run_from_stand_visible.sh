#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
export V34_CURRICULUM_SOURCE=/home/ubuntu22/logs/rsl_rl/terrain_v34_scratch_stage0_stand/2026-09-20_14-37-36_stage0_20260920T143704Z/model_299.pt
/usr/bin/python3 -u /home/ubuntu22/terrain_v34_from_scratch_four_terrain/supervisor_curriculum.py 2>&1 | tee -a /home/ubuntu22/terrain_v34_from_scratch_four_terrain/curriculum_visible.log
echo 'V34从已验证的自训站立模型继续的速度课程及四地形流程已结束。'
read -r -p '按回车关闭窗口' _
