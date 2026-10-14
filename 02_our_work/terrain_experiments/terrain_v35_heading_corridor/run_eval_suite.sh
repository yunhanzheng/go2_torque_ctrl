#!/usr/bin/env bash
set -euo pipefail
checkpoint=${1:?checkpoint path required}
label=${2:?result label required}
result_dir="/home/ubuntu22/terrain_v35_heading_corridor/results/${label}"
mkdir -p "$result_dir"
export PYTHONPATH=/home/ubuntu22:/data/sac:/data/sac/scripts
export WANDB_MODE=offline
export V35_EVAL_HEADING=1
python_bin=/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python
evaluator=/home/ubuntu22/terrain_v35_heading_corridor/evaluate.py
for terrain in plane stairs_up stairs_down slope_up slope_down boxes random_rough; do
  export TERRAIN_CASE="$terrain"
  export TERRAIN_DIFFICULTY=0.5
  export TERRAIN_OUTPUT="$result_dir/$terrain"
  "$python_bin" -u "$evaluator" \
    --task Isaac-Go2-V35-HeadingCorridor-v0 --agent rsl_rl_sac_cfg_entry_point \
    --num_envs 32 --seed 7 --checkpoint "$checkpoint" --headless \
    > "$result_dir/$terrain.log" 2>&1
done

