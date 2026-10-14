#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
export V34_RECOVERY_BATCH=/home/ubuntu22/terrain_v34_from_scratch_four_terrain/results/20260920T224437Z
export V34_RECOVERY_CHECKPOINT=/home/ubuntu22/logs/rsl_rl/terrain_v34_scratch_four_terrain/2026-09-20_22-45-03_mixed_curriculum_20260920T224437Z/model_2396.pt
export V34_RECOVERY_FLAT_CHECKPOINT=/home/ubuntu22/logs/rsl_rl/terrain_v34_scratch_flat_foundation/2026-09-20_16-31-55_stage3_20260920T163148Z/model_1597.pt
/usr/bin/python3 -u /home/ubuntu22/terrain_v34_from_scratch_four_terrain/resume_recordings.py 2>&1 | tee -a /home/ubuntu22/terrain_v34_from_scratch_four_terrain/recording_recovery_visible.log
echo 'V34诊断录像恢复流程已结束。'
read -r -p '按回车关闭窗口' _
