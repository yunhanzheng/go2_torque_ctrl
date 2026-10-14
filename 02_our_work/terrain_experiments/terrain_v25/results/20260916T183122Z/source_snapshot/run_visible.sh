#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v25/supervisor.py 2>&1 | tee -a /home/ubuntu22/terrain_v25/visible.log
echo '训练、评估和WebM流程已结束，见 terrain_v25/latest.txt'
read -r -p '按回车关闭窗口' _
