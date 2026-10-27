#!/usr/bin/env bash
set -euo pipefail

root=/home/ubuntu22/terrain_v34_from_scratch_four_terrain
batch="$root/results/20260920T152558Z"
output="$batch/videos/V34_medium_0.3mps"
checkpoint=/home/ubuntu22/logs/rsl_rl/terrain_v34_scratch_stage2_medium/2026-09-20_15-50-00_stage2_20260920T152558Z/model_998.pt
python_bin=/opt/conda/envs/isaac_sac/bin/python
ffmpeg_bin=/opt/conda/envs/isaac_sac/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2

mkdir -p "$batch/videos"
exec > >(tee -a "$batch/videos/record_medium.log") 2>&1
export PYTHONPATH=/home/ubuntu22:/data/sac:/data/sac/scripts
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY}"
export TERRAIN_CASE=plane
export TERRAIN_DIFFICULTY=0.0
export TERRAIN_SPEED=0.3
export TERRAIN_OUTPUT="$output"

"$python_bin" -u "$root/evaluate.py" \
  --task Isaac-Go2-V34-ScratchMedium-v0 \
  --agent rsl_rl_sac_cfg_entry_point \
  --num_envs 1 --seed 7 --checkpoint "$checkpoint" --headless \
  --video --video_length 600

"$ffmpeg_bin" -y -v error \
  -i "${output}_video/rl-video-step-0.mp4" \
  -c:v libvpx-vp9 -crf 30 -b:v 0 -an "${output}.webm"
"$ffmpeg_bin" -v error -i "${output}.webm" -f null -

echo "WebM ready: ${output}.webm"
firefox --new-window "file://${output}.webm"
read -r -p '按回车关闭录制窗口' _
