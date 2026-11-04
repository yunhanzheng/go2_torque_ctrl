#!/usr/bin/env bash
# Record a V56 rollout on bayesdl, with the equivalent PD gains measured in the same pass.
#
# Two things borrowed from V54's run_video_server.sh are load-bearing:
#   * the container ships no Vulkan runtime, so headless rendering hangs forever while the
#     render product is created unless the loader and NVIDIA ICD that fanlineng unpacked are
#     put on LD_LIBRARY_PATH / VK_ICD_FILENAMES;
#   * Isaac Sim defaults to activeGpu=0, and its Vulkan enumeration disagrees with a
#     restricted CUDA_VISIBLE_DEVICES, so the renderer is pinned explicitly and that
#     variable is dropped rather than set to a single device.
#
#   bash run_video_server.sh <pos|torque|pos200|torque200> <checkpoint.pt> [seconds]
#   RENDER_GPU=1 bash run_video_server.sh torque "$CKT" 10    # if GPU0 is busy
set -euo pipefail

FAN=/root/bayes-tmp/xiaoyulong/fanlineng
WORK=/root/bayes-tmp/xiaoyulong/yunhanzheng
REPO=$WORK/go2_torque_ctrl
PYTHON=${PYTHON:-/root/bayes-tmp/system_caches/opt_conda/envs/isaac_sac/bin/python3.11}

VARIANT=${1:?usage: run_video_server.sh <pos|torque|pos200|torque200> <checkpoint.pt> [seconds]}
CHECKPOINT=${2:?usage: run_video_server.sh <variant> <checkpoint.pt> [seconds]}
SECS=${3:-6}
shift $(( $# >= 3 ? 3 : $# ))

case "$VARIANT" in
  pos) TASK=Isaac-Go2-V56-StandPos-v0 ;;
  torque) TASK=Isaac-Go2-V56-StandTorque-v0 ;;
  pos200) TASK=Isaac-Go2-V56-StandPos200-v0 ;;
  torque200) TASK=Isaac-Go2-V56-StandTorque200-v0 ;;
  *) echo "unknown variant '$VARIANT' (pos|torque|pos200|torque200)" >&2; exit 1 ;;
esac

# Isaac Sim writes kit/shaderc caches into HOME; keep them out of fanlineng's tree.
export HOME="$WORK"
export TMPDIR="$WORK/tmp"
mkdir -p "$TMPDIR"
export LD_LIBRARY_PATH="$FAN/tmp/vulkan-loader/usr/lib/x86_64-linux-gnu:$FAN/tmp/vulkan-nvidia-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="$FAN/tmp/nvidia_egl_icd.json"
unset CUDA_VISIBLE_DEVICES

# Same rsl_rl resolution as run_trial.sh: the SAC fork, not the upstream checkout.
export PYTHONPATH="$REPO/final/source:$REPO/final/source/base_adapter:$REPO/final/source/terrain_v56_stand_force_compare:$REPO/isaac_sac_setup/rsl_rl_sac"

KIT_ARGS=${KIT_ARGS:-"--allow-root --/renderer/multiGpu/enabled=false --/renderer/activeGpu=${RENDER_GPU:-0} --/plugins/carb.tasking.plugin/threadCount=4"}

cd "$REPO/final/source/terrain_v56_stand_force_compare"
# ``_video`` suffix: a video pass runs 1 env for a different duration, and writing it into the
# same directory would overwrite the 4-env force evaluation that the comparison table is built from.
exec "$PYTHON" -u evaluate_force.py \
  --task "$TASK" \
  --checkpoint "$CHECKPOINT" \
  --label "$VARIANT" \
  --num_envs 1 \
  --duration "$SECS" \
  --output_dir "results/force_eval/${VARIANT}_video" \
  --video \
  --kit_args "$KIT_ARGS" \
  --headless "$@"
