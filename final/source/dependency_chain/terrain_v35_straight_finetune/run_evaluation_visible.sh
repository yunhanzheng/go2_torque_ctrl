#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
/usr/bin/python3 -u /home/ubuntu22/terrain_v35_straight_finetune/evaluate_and_report.py 2>&1 | tee -a /home/ubuntu22/terrain_v35_straight_finetune/evaluation_visible.log
echo 'V35评估、WebM编码与Firefox报告已完成。'
read -r -p '按回车关闭窗口' _
