#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
export V34_SLOW_RECOVERY_SOURCE=/home/ubuntu22/logs/rsl_rl/terrain_v34_scratch_stage1_slow/2026-09-20_15-18-02_stage1_20260920T151755Z/model_350.pt
/usr/bin/python3 -u /home/ubuntu22/terrain_v34_from_scratch_four_terrain/supervisor_curriculum.py 2>&1 | tee -a /home/ubuntu22/terrain_v34_from_scratch_four_terrain/curriculum_visible.log
echo 'V34已从自训慢速 model_350 恢复，速度课程及四地形流程已结束。'
read -r -p '按回车关闭窗口' _
