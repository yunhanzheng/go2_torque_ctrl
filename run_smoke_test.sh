#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${ISAACLAB_PYTHON:-/opt/conda/envs/isaac_sac/bin/python}"
source_root="$repo_root/final/source"
if [[ -n "${SAC_SMOKE_OUTPUT:-}" ]]; then
  output_path="$SAC_SMOKE_OUTPUT"
else
  smoke_dir="$(mktemp -d)"
  output_path="$smoke_dir/result"
fi

export PYTHONPATH="$source_root:$source_root/dependency_chain:$source_root/base_adapter:$source_root/base_adapter/initial_stair_source:${PYTHONPATH:-}"
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY}"
export STRAIGHT_OUTPUT="$output_path"
export TERRAIN_CASE=plane
export TERRAIN_DIFFICULTY=0.0

"$python_bin" -u \
  "$source_root/terrain_v50_curriculum_recovery/evaluate_dual.py" \
  --task Isaac-Go2-V50-CurriculumRecovery-v0 \
  --agent rsl_rl_sac_cfg_entry_point \
  --num_envs 1 \
  --seed 7 \
  --checkpoint "$repo_root/final/model/model_5099.pt" \
  --headless

if [[ ! -s "${output_path}.json" ]]; then
  echo "Smoke test failed: evaluator did not create ${output_path}.json" >&2
  exit 1
fi

echo "Smoke test passed: ${output_path}.json"
