#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v35_straight_finetune/supervisor.py 2>&1 | tee -a /home/ubuntu22/terrain_v35_straight_finetune/visible.log
echo 'V35直线微调流程已结束。'
read -r -p '按回车关闭窗口' _
