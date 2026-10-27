#!/usr/bin/env bash
# 7 x 10 s clips from the 2026-10-25_19-12-07 run, model_3000 (iter 3000 = step 72000).
# Every sub-terrain at row 8 (difficulty 0.85), except pyramid_stairs_inv which also gets row 4.
set -euo pipefail

cd /root/bayes-tmp/xiaoyulong/yunhanzheng/go2_torque_ctrl/final/source/terrain_v54_sata_ppo_rough

R=logs/rsl_rl/terrain_v54_sata_ppo_rough/2026-10-25_19-12-07_V54_sata_ppo_rough_20261025T111153Z
export CKPT=$R/model_3000.pt
export RUN=$R
export STEPS=72000      # 3000 iterations x 24 steps
export LENGTH=2000      # 2000 frames @ 200 fps = 10 s
export OUT_ROOT=$R/videos_model3000_10s
export CLIPS="pyramid_stairs:8:stairs_d8
pyramid_stairs_inv:4:stairs_inv_d4
pyramid_stairs_inv:8:stairs_inv_d8
boxes:8:boxes_d8
random_rough:8:rough_d8
hf_pyramid_slope:8:slope_d8
hf_pyramid_slope_inv:8:slope_inv_d8"

CLIPS=$(echo "$CLIPS" | tr '\n' ' ')
exec bash run_clips.sh
