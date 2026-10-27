#!/usr/bin/env bash
# Record an N-second clip of one V55 checkpoint on bayesdl.
#
# Why the wrapper exists: the cluster driver has no Vulkan ICD, so Kit needs the bundled
# loader + NVIDIA ICD (same trick as terrain_v54_sata_ppo_rough/run_video_model1000.sh),
# and the renderer has to be pinned to a GPU explicitly.
#
# Usage:
#   CKPT=logs/rsl_rl/terrain_v55_sata_warmstart_isaac_rl/<run>/model_3000.pt \
#   SECS=10 bash run_video.sh
#
# Optional: OUT=<folder> (default <ckpt dir>/<model>_<secs>s), CUDA_VISIBLE_DEVICES (default 0),
#           TERRAIN_LEVEL=<row> to pin the clip to one difficulty, extra args are forwarded to
#           evaluate.py (e.g. --max_init_terrain_level 9).
set -euo pipefail

FAN=/root/bayes-tmp/xiaoyulong/fanlineng
WORK=/root/bayes-tmp/xiaoyulong/yunhanzheng

export HOME="$WORK"
export TMPDIR="$WORK/tmp"
mkdir -p "$TMPDIR"
export LD_LIBRARY_PATH="$FAN/tmp/vulkan-loader/usr/lib/x86_64-linux-gnu:$FAN/tmp/vulkan-nvidia-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="$FAN/tmp/nvidia_egl_icd.json"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$WORK/go2_torque_ctrl/final/source/terrain_v55_sata_warmstart_isaac_rl"

: "${CKPT:?set CKPT=/path/to/model_<iter>.pt}"
SECS="${SECS:-10}"
OUT="${OUT:-$(dirname "$CKPT")/$(basename "$CKPT" .pt)_${SECS}s}"
KIT_ARGS="${KIT_ARGS:---allow-root --/renderer/multiGpu/enabled=false --/renderer/activeGpu=0 --/plugins/carb.tasking.plugin/threadCount=4}"

echo "checkpoint : $CKPT"
echo "clip       : ${SECS}s -> $OUT"
echo "gpu        : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

exec /opt/conda/envs/isaac_sac/bin/python -u evaluate.py \
    --checkpoint "$CKPT" \
    --num_envs 1 \
    --seed 0 \
    --freeze_gs 1.0 \
    --kit_args "$KIT_ARGS" \
    --video \
    --video_seconds "$SECS" \
    --sky_plain \
    --video_folder "$OUT" \
    "$@"
