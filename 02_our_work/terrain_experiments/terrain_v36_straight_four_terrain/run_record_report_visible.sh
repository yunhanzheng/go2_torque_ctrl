#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v36_straight_four_terrain/record_report.py 2>&1 | tee -a /home/ubuntu22/terrain_v36_straight_four_terrain/record_report_visible.log
echo 'V36七段WebM与Firefox报告已完成。'
read -r -p '按回车关闭窗口' _
