#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v32_centered_start/supervisor.py 2>&1 | tee -a /home/ubuntu22/terrain_v32_centered_start/visible.log
echo 'V32平地基础步态及门禁评估已完成。'
read -r -p '按回车关闭窗口' _
