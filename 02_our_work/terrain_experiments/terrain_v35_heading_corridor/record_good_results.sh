#!/usr/bin/env bash
set -euo pipefail
checkpoint=/home/ubuntu22/logs/rsl_rl/terrain_v35_stair_boxes/2026-09-21_08-10-08_stair_boxes_focus_200/model_744.pt
video_dir=/home/ubuntu22/terrain_v35_heading_corridor/videos/good_results_20260921
mkdir -p "$video_dir"
export PYTHONPATH=/home/ubuntu22:/data/sac:/data/sac/scripts
export WANDB_MODE=offline
export V35_EVAL_HEADING=1
python_bin=/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python
evaluator=/home/ubuntu22/terrain_v35_heading_corridor/evaluate.py
ffmpeg_bin=/home/ubuntu22/miniforge3/envs/env_isaaclab/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2

record_case() {
  local terrain=$1
  local difficulty=$2
  local view_env=$3
  export TERRAIN_CASE="$terrain"
  export TERRAIN_DIFFICULTY="$difficulty"
  export V35_VIEW_ENV="$view_env"
  export TERRAIN_OUTPUT="$video_dir/$terrain"
  "$python_bin" -u "$evaluator" \
    --task Isaac-Go2-V35-HeadingCorridor-v0 --agent rsl_rl_sac_cfg_entry_point \
    --num_envs 32 --seed 7 --checkpoint "$checkpoint" \
    --video --video_length 500 --headless > "$video_dir/$terrain.log" 2>&1
}

record_case plane 0.5 0
record_case slope_up 0.5 0
record_case slope_down 0.5 0
record_case random_rough 0.5 0
record_case boxes 0.5 13
record_case stairs_up 0.0 0
record_case stairs_down 0.2 0

"$ffmpeg_bin" -y -v error \
  -i "$video_dir/plane_video/rl-video-step-0.mp4" \
  -i "$video_dir/slope_up_video/rl-video-step-0.mp4" \
  -i "$video_dir/slope_down_video/rl-video-step-0.mp4" \
  -i "$video_dir/random_rough_video/rl-video-step-0.mp4" \
  -i "$video_dir/boxes_video/rl-video-step-0.mp4" \
  -i "$video_dir/stairs_up_video/rl-video-step-0.mp4" \
  -i "$video_dir/stairs_down_video/rl-video-step-0.mp4" \
  -filter_complex '[0:v][1:v][2:v][3:v][4:v][5:v][6:v]concat=n=7:v=1:a=0,fps=50[outv]' \
  -map '[outv]' -c:v libvpx-vp9 -crf 30 -b:v 0 -an \
  /home/ubuntu22/v35_good_results_all.webm
