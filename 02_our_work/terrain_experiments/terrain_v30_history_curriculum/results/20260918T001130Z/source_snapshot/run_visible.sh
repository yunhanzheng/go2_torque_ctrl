#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v30_history_curriculum/supervisor.py 2>&1 | tee -a /home/ubuntu22/terrain_v30_history_curriculum/visible.log
echo 'V30历史课程多地形Pilot、评估和WebM已完成；Firefox已打开。'
read -r -p '按回车关闭窗口' _
