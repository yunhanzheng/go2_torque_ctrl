#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v41_height_bridge/supervisor.py 2>&1 | tee -a /home/ubuntu22/terrain_v41_height_bridge/visible.log
echo 'V41高度单变量桥接训练已结束。'
read -r -p '按回车关闭窗口' _
