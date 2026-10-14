#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v37_stairs_focus/supervisor.py 2>&1 | tee -a /home/ubuntu22/terrain_v37_stairs_focus/visible.log
echo 'V37无高度观测楼梯专项已结束。'
read -r -p '按回车关闭窗口' _

