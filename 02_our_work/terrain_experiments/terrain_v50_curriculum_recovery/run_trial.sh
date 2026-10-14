#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu22/terrain_v50_curriculum_recovery
PYTHON=/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python
SOURCE=/home/ubuntu22/logs/rsl_rl/terrain_v49_generalize_23cm/2026-10-08_09-01-27_mixed55upstairs_keep15fixed23_from_v48_4778_20261008T090121Z/model_4980.pt
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$ROOT/results"
LOG="$ROOT/results/V50_curriculum_recovery_${STAMP}.log"
ln -sfn "$LOG" "$ROOT/latest.log"

export PYTHONPATH=/home/ubuntu22:/data/sac:/data/sac/scripts
export WANDB_MODE=offline
export V50_SOURCE_CHECKPOINT="$SOURCE"

echo "V50 short trial: target curriculum band 6-7, source model4980"
echo "Log: $LOG"
exec "$PYTHON" -u "$ROOT/train.py" \
  --task Isaac-Go2-V50-CurriculumRecovery-v0 \
  --agent rsl_rl_sac_cfg_entry_point \
  --num_envs 4096 --max_iterations 120 --seed 1 \
  --resume --run_name "level6_trial_from_v49_4980_${STAMP}" --headless 2>&1 | tee "$LOG"
