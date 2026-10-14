#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v27_normal_gait/supervisor.py 2>&1 | tee -a /home/ubuntu22/terrain_v27_normal_gait/visible.log
echo 'V27训练、评估和V26/V27对比WebM已完成；Firefox将自动打开。'
read -r -p '按回车关闭窗口' _
