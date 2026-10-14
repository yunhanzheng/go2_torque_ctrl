#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v40_bridge_original/supervisor.py 2>&1 | tee -a /home/ubuntu22/terrain_v40_bridge_original/visible.log
echo 'V40桥接训练已结束。'
read -r -p '按回车关闭窗口' _
