#!/usr/bin/env bash
set -euo pipefail
checkpoint=${1:?checkpoint path required}
label=${2:?video label required}
video_dir="/home/ubuntu22/terrain_v35_heading_corridor/videos/${label}"
mkdir -p "$video_dir"
export PYTHONPATH=/home/ubuntu22:/data/sac:/data/sac/scripts
export WANDB_MODE=offline
export V35_EVAL_HEADING=1
python_bin=/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python
evaluator=/home/ubuntu22/terrain_v35_heading_corridor/evaluate.py
ffmpeg_bin=/home/ubuntu22/miniforge3/envs/env_isaaclab/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
for terrain in stairs_up slope_up boxes random_rough; do
  export TERRAIN_CASE="$terrain"
  export TERRAIN_DIFFICULTY=0.5
  export TERRAIN_OUTPUT="$video_dir/$terrain"
  "$python_bin" -u "$evaluator" \
    --task Isaac-Go2-V35-HeadingCorridor-v0 --agent rsl_rl_sac_cfg_entry_point \
    --num_envs 1 --seed 7 --checkpoint "$checkpoint" \
    --video --video_length 600 --headless > "$video_dir/$terrain.log" 2>&1
done
"$ffmpeg_bin" -y -v error \
  -i "$video_dir/stairs_up_video/rl-video-step-0.mp4" \
  -i "$video_dir/slope_up_video/rl-video-step-0.mp4" \
  -i "$video_dir/boxes_video/rl-video-step-0.mp4" \
  -i "$video_dir/random_rough_video/rl-video-step-0.mp4" \
  -filter_complex '[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[outv]' \
  -map '[outv]' -vf fps=50 -c:v libvpx-vp9 -crf 30 -b:v 0 -an \
  "$video_dir/four_terrain_showcase.webm"
