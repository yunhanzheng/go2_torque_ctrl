#!/usr/bin/env bash
set -euo pipefail

root=/home/ubuntu22/terrain_v34_from_scratch_four_terrain
batch=/home/ubuntu22/terrain_v34_from_scratch_four_terrain/results/20260920T163148Z
stage_log="$batch/目标0.5mps_600轮.log"
run_dir=/home/ubuntu22/logs/rsl_rl/terrain_v34_scratch_flat_foundation/2026-09-20_16-31-55_stage3_20260920T163148Z
output="$batch/videos/V34_target_0.5mps"
python_bin=/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python
ffmpeg_bin=/home/ubuntu22/miniforge3/envs/env_isaaclab/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2

mkdir -p "$batch/videos"
exec > >(tee -a "$batch/videos/auto_video_after_flat.log") 2>&1

echo '[自动监管] 等待 0.5 m/s 训练完成并保存最终检查点…'
while ! rg -q '^Training time:' "$stage_log"; do
  sleep 5
done

checkpoint=$(find "$run_dir" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -1)
if [[ -z "$checkpoint" || ! -s "$checkpoint" ]]; then
  echo '[错误] 未找到最终检查点'
  exit 1
fi
echo "[自动监管] 最终检查点：$checkpoint"

# The curriculum starts its gate evaluation immediately after training. Stop
# the supervisor so recording has exclusive use of the GPU; resume later.
supervisor_pid=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["supervisor_pid"])' "$batch/status.json")
if ps -p "$supervisor_pid" -o args= | rg -q 'supervisor_curriculum.py'; then
  echo "[自动监管] 暂停后续课程，PID=$supervisor_pid"
  kill -INT "$supervisor_pid"
  while kill -0 "$supervisor_pid" 2>/dev/null; do
    sleep 2
  done
fi

export PYTHONPATH=/home/ubuntu22:/data/sac:/data/sac/scripts
export WANDB_MODE=offline
export TERRAIN_CASE=plane
export TERRAIN_DIFFICULTY=0.0
export TERRAIN_SPEED=0.5
export TERRAIN_OUTPUT="$output"

echo '[自动监管] 开始录制 0.5 m/s 最终模型…'
"$python_bin" -u "$root/evaluate.py" \
  --task Isaac-Go2-V34-ScratchFlat-v0 \
  --agent rsl_rl_sac_cfg_entry_point \
  --num_envs 1 --seed 7 --checkpoint "$checkpoint" --headless \
  --video --video_length 600

"$ffmpeg_bin" -y -v error \
  -i "${output}_video/rl-video-step-0.mp4" \
  -c:v libvpx-vp9 -crf 30 -b:v 0 -an "${output}.webm"
"$ffmpeg_bin" -v error -i "${output}.webm" -f null -

echo "[完成] WebM 已校验：${output}.webm"
firefox --new-window "file://${output}.webm"
read -r -p '按回车关闭自动监管窗口' _
