#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v38_official_stairs/supervisor_resume.py 2>&1 | tee -a /home/ubuntu22/terrain_v38_official_stairs/visible_resume.log
echo 'V38原始楼梯课程恢复训练已结束。'
read -r -p '按回车关闭窗口' _
