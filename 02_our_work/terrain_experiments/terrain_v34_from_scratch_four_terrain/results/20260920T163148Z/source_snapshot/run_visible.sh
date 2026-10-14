#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v34_from_scratch_four_terrain/supervisor.py 2>&1 | tee -a /home/ubuntu22/terrain_v34_from_scratch_four_terrain/visible.log
echo 'V34严格从头训练、四地形评估与WebM流程已结束。'
read -r -p '按回车关闭窗口' _

