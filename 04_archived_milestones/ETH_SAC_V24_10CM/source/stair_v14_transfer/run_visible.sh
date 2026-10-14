#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/stair_v14_transfer/supervisor.py 2>&1 | tee -a /home/ubuntu22/stair_v14_transfer/visible.log
echo '流程结束；结果见 stair_v14_transfer/latest.txt'
read -r -p '按回车关闭窗口' _
