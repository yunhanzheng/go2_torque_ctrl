#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v34_from_scratch_four_terrain/supervisor_curriculum.py 2>&1 | tee -a /home/ubuntu22/terrain_v34_from_scratch_four_terrain/curriculum_visible.log
echo 'V34严格从头速度课程及四地形流程已结束。'
read -r -p '按回车关闭窗口' _
