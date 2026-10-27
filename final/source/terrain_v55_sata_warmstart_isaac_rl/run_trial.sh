#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/bayes-tmp/xiaoyulong/yunhanzheng/go2_torque_ctrl/final/source/terrain_v55_sata_warmstart_isaac_rl
PYTHON=/opt/conda/envs/isaac_sac/bin/python
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$ROOT/results"
LOG="$ROOT/results/V55_warmstart_isaac_rl_${STAMP}.log"
ln -sfn "$LOG" "$ROOT/latest.log"

SOURCE_ROOT="$ROOT/.."

# SATA's finished policy (Oct14_16-33-32_, model_3000 = 3000 iters, step_count 72000, gs 0.789).
# md5 3834c010f7d8445b6769d7189f60935b, identical to the copy in the working repo.
SATA_CKPT=${SATA_CKPT:-/root/bayes-tmp/xiaoyulong/yunhanzheng/SATA/legged_gym/logs/SATA/Oct14_16-33-32_/model_3000.pt}
[[ -f "$SATA_CKPT" ]] || { echo "Missing SATA checkpoint: $SATA_CKPT" >&2; exit 1; }

# V55 reuses V54's config (and through it V52's sata_rewards and V51's sata_alignment).
export PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/dependency_chain:$SOURCE_ROOT/base_adapter:$SOURCE_ROOT/base_adapter/initial_stair_source:$SOURCE_ROOT/terrain_v51_native_terrain:$SOURCE_ROOT/terrain_v52_sata_ppo_flat:$SOURCE_ROOT/terrain_v54_sata_ppo_rough:$ROOT:${PYTHONPATH:-}"

# Reuse the shared wandb key from the V54 launcher (single source of truth, no duplicated secret).
WANDB_KEY_LINE=$(grep -m1 '^export WANDB_API_KEY=' "$SOURCE_ROOT/terrain_v54_sata_ppo_rough/run_trial.sh" || true)
[[ -n "$WANDB_KEY_LINE" ]] || { echo "Cannot find WANDB_API_KEY in V54's run_trial.sh" >&2; exit 1; }
export WANDB_API_KEY="${WANDB_KEY_LINE#export WANDB_API_KEY=}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "V55: SATA model_3000 warm start, IsaacLab Go2 rough's stock reward table, V54's terrain"
echo "     gs 0.789 -> 1.0 over 500 iters (then frozen), lr 1e-3 adaptive (IsaacLab default), 15000 iters"
echo "     checkpoint: $SATA_CKPT"
echo "Log: $LOG"

# Why 4096 envs / 15000 iters: env count and seed match V52/V54 so the runs stay comparable;
# 15000 matches V54's budget (V55 used 5000 for the first probe, 229 iters of which are in
# logs/rsl_rl/.../2026-10-24_17-58-01).  Override with --max_iterations if needed.
exec "$PYTHON" -u "$ROOT/train.py" \
  --task Isaac-Go2-V55-SataWarmstartIsaacRl-v0 \
  --agent rsl_rl_sac_cfg_entry_point \
  --num_envs 4096 --max_iterations 15000 --seed 1 \
  --sata_init "$SATA_CKPT" \
  --run_name "V55_warmstart_isaac_rl_${STAMP}" --logger wandb --log_project_name isaac_sac_pipeline --headless 2>&1 | tee "$LOG"
