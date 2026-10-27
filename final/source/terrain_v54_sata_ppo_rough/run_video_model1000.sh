#!/usr/bin/env bash
# Record a 1-minute clip of the 2026-10-23 run's model_1000 on bayesdl.
#
# Same Vulkan workaround and GPU pinning as run_video_server.sh (see its header), but:
#   * checkpoint = model_1000, i.e. the reward peak of that run (~iter 1200);
#   * --common_step_counter 24000 = 1000 iters * 24 steps/iter, so SATA's Gompertz clock
#     (torque limit / control frequency) sits where the checkpoint was trained;
#   * --fix_terrain_level 0: the run never left row 0 (terrain_levels_vel demoted every
#     env by iter 370), so row 0 is the only terrain this policy has seen;
#   * --command 0.5 0 0 0 instead of --isaaclab_play: 1.0 m/s pin would ask for a 4 Hz
#     gait clock, while this checkpoint trained around 2-2.4 Hz.  0.5 m/s is its average
#     training command (target = 0.37 + 0.13 at gs~0.13).
set -euo pipefail

FAN=/root/bayes-tmp/xiaoyulong/fanlineng
WORK=/root/bayes-tmp/xiaoyulong/yunhanzheng

export HOME="$WORK"
export TMPDIR="$WORK/tmp"
mkdir -p "$TMPDIR"
export LD_LIBRARY_PATH="$FAN/tmp/vulkan-loader/usr/lib/x86_64-linux-gnu:$FAN/tmp/vulkan-nvidia-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="$FAN/tmp/nvidia_egl_icd.json"
unset CUDA_VISIBLE_DEVICES

cd "$WORK/go2_torque_ctrl/final/source/terrain_v54_sata_ppo_rough"
RUN=logs/rsl_rl/terrain_v54_sata_ppo_rough/2026-10-23_10-46-41_V54_sata_ppo_rough_20261023T024629Z
CKPT=${CKPT:-$RUN/model_1000.pt}
OUT=${OUT:-$RUN/V54_model1000_0p5_1min}

KIT_ARGS="${KIT_ARGS:---allow-root --/renderer/multiGpu/enabled=false --/renderer/activeGpu=1 --/plugins/carb.tasking.plugin/threadCount=4}"

exec /opt/conda/envs/isaac_sac/bin/python evaluate_video.py \
    --task Isaac-Go2-V54-SataPpoRough-v0 \
    --agent rsl_rl_sac_cfg_entry_point \
    --num_envs 1 \
    --seed 42 \
    --kit_args "$KIT_ARGS" \
    --checkpoint "$CKPT" \
    --common_step_counter 24000 \
    --fix_terrain_level 0 \
    --command 0.5 0 0 0 \
    --no_time_out \
    --log_command \
    --video \
    --video_length 6000 \
    --video_fps 100 \
    --video_folder "$OUT" \
    --headless \
    "$@"
