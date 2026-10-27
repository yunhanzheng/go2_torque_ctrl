#!/usr/bin/env bash
# Render an evaluation video on bayesdl.
#
# The container ships no Vulkan runtime, so headless rendering used to hang during
# render-product creation.  Borrow the loader + NVIDIA ICD that fanlineng unpacked under
# his work directory (see his setup_isaacsim.sh): a pure-Debian libvulkan.so.1 plus the
# libEGL_nvidia/libGLX_nvidia set extracted from NVIDIA-Linux-x86_64-580.105.08.
#
# The renderer must also be pinned to GPU1: Isaac Sim defaults to activeGpu=0, which the
# running training already fills, and vkCreateRayTracingPipelinesKHR then fails with
# "Failed to create the raytracing pipeline".  fanlineng's launch_*_videos.sh scripts use
# the same flags.
#
#   bash run_video_server.sh                 # IsaacLab default play command -> ~60 s
#   bash run_video_server.sh --video_length 500
#
# The command term is replaced with IsaacLab's stock CommandsCfg, but the linear velocity and
# the heading target are both pinned: 1.0 m/s straight ahead, no turning, no standing envs.
# Both terminations that matter (fall, joint limit) stay on and the 10 s time-out is dropped,
# so the clip runs the full 6000 steps without a reset.
#
# The terrain is V54's own rough curriculum.  A tile is only 8 m wide and the spawn sits at
# its centre, so 37 s at 1.0 m/s walks across several neighbouring tiles; the borders are
# 0.25 m wide ramps, which is what keeps the crossing survivable.
set -euo pipefail

FAN=/root/bayes-tmp/xiaoyulong/fanlineng
WORK=/root/bayes-tmp/xiaoyulong/yunhanzheng

# Isaac Sim writes kit/shaderc caches into HOME; keep them out of fanlineng's tree.
export HOME="$WORK"
export TMPDIR="$WORK/tmp"
mkdir -p "$TMPDIR"
export LD_LIBRARY_PATH="$FAN/tmp/vulkan-loader/usr/lib/x86_64-linux-gnu:$FAN/tmp/vulkan-nvidia-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="$FAN/tmp/nvidia_egl_icd.json"
# Isaac Sim's Vulkan device enumeration disagrees with a restricted CUDA_VISIBLE_DEVICES list.
unset CUDA_VISIBLE_DEVICES

cd "$WORK/go2_torque_ctrl/final/source/terrain_v54_sata_ppo_rough"
RUN=logs/rsl_rl/terrain_v54_sata_ppo_rough/2026-10-22_13-52-21_V54_sata_ppo_rough_20261022T055209Z

KIT_ARGS="${KIT_ARGS:---allow-root --/renderer/multiGpu/enabled=false --/renderer/activeGpu=1 --/plugins/carb.tasking.plugin/threadCount=4}"

exec /opt/conda/envs/isaac_sac/bin/python evaluate_video.py \
    --task Isaac-Go2-V54-SataPpoRough-v0 \
    --agent rsl_rl_sac_cfg_entry_point \
    --num_envs 1 \
    --seed 42 \
    --kit_args "$KIT_ARGS" \
    --checkpoint "$RUN/model_2000.pt" \
    --common_step_counter 48000 \
    --isaaclab_play \
    --fix_terrain_level 2 \
    --no_time_out \
    --log_command \
    --video \
    --video_length 6000 \
    --video_fps 100 \
    --video_folder "$RUN/V54_seed42_row2_1min" \
    --headless \
    "$@"
