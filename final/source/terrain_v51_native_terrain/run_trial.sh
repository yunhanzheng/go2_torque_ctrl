#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/bayes-tmp/xiaoyulong/yunhanzheng/go2_torque_ctrl/final/source/terrain_v51_native_terrain
PYTHON=/opt/conda/envs/isaac_sac/bin/python
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$ROOT/results"
LOG="$ROOT/results/V51_native_terrain_${STAMP}.log"
ln -sfn "$LOG" "$ROOT/latest.log"

SOURCE_ROOT="$ROOT/.."
export PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/dependency_chain:$SOURCE_ROOT/base_adapter:$SOURCE_ROOT/base_adapter/initial_stair_source:$ROOT:${PYTHONPATH:-}"
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY}"

echo "V51: V50 rewards with IsaacLab native rough terrain"
echo "Log: $LOG"
exec "$PYTHON" -u "$ROOT/train.py" \
  --task Isaac-Go2-V51-NativeTerrain-v0 \
  --agent rsl_rl_sac_cfg_entry_point \
  --num_envs 4096 --max_iterations 1500 --seed 1 \
  --run_name "V51_native_terrain_${STAMP}" --logger wandb --log_project_name isaac_sac_pipeline --headless 2>&1 | tee "$LOG"
