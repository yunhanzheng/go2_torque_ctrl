#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v26_official3/supervisor.py 2>&1 | tee -a /home/ubuntu22/terrain_v26_official3/visible.log
echo '训练、V24/V26评估和WebM流程已结束；Firefox将自动打开结果页。'
read -r -p '按回车关闭窗口' _
