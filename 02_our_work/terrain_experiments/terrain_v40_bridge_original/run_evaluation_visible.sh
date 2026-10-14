#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python -u /home/ubuntu22/terrain_v40_bridge_original/evaluate_screen.py 2>&1 | tee -a /home/ubuntu22/terrain_v40_bridge_original/evaluation_visible.log
echo 'V40严格筛选评估已结束。'
read -r -p '按回车关闭窗口' _
