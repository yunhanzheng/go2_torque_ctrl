#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${ISAACLAB_PYTHON:-/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python}"
source_root="$repo_root/final/source"
output_path="${SAC_SMOKE_OUTPUT:-/tmp/semeterarbeit_v50_smoke}"

export PYTHONPATH="$source_root:$source_root/dependency_chain:$source_root/base_adapter:$source_root/base_adapter/initial_stair_source:${PYTHONPATH:-}"
export WANDB_MODE=offline
export STRAIGHT_OUTPUT="$output_path"
export TERRAIN_CASE=plane
export TERRAIN_DIFFICULTY=0.0

exec "$python_bin" -u \
  "$source_root/terrain_v50_curriculum_recovery/evaluate_dual.py" \
  --task Isaac-Go2-V50-CurriculumRecovery-v0 \
  --agent rsl_rl_sac_cfg_entry_point \
  --num_envs 1 \
  --seed 7 \
  --checkpoint "$repo_root/final/model/model_5099.pt" \
  --headless
