#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/bayes-tmp/xiaoyulong/yunhanzheng/go2_torque_ctrl/final/source/terrain_v54_sata_ppo_rough
PYTHON=/opt/conda/envs/isaac_sac/bin/python
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
# The rev D pit-first variant is the same experiment except for the task id, so both are
# launched from here:  TASK=Isaac-Go2-V54-PitFirst-v0 LOG_TAG=V54_pit_first bash run_trial.sh
TASK=${TASK:-Isaac-Go2-V54-SataPpoRough-v0}
LOG_TAG=${LOG_TAG:-V54_sata_ppo_rough}
mkdir -p "$ROOT/results"
LOG="$ROOT/results/${LOG_TAG}_${STAMP}.log"
ln -sfn "$LOG" "$ROOT/latest.log"

SOURCE_ROOT="$ROOT/.."
# V54 reuses V52's sata_rewards and V51's sata_alignment, so both dirs are on the path.
export PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/dependency_chain:$SOURCE_ROOT/base_adapter:$SOURCE_ROOT/base_adapter/initial_stair_source:$SOURCE_ROOT/terrain_v51_native_terrain:$SOURCE_ROOT/terrain_v52_sata_ppo_flat:$ROOT:${PYTHONPATH:-}"
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY}"

echo "V54: V52 (SATA-identical: PPO + SATA's 9 rewards) on IsaacLab's official Go2 rough terrain (15000 iters)"
echo "Task: $TASK"
echo "Log: $LOG"

# Same protocol as V52's 21-13-36 run (4096 envs, seed 1, from scratch) except the budget:
# 3000 iterations was not enough -- Train/mean_episode_length peaks at 1436 near iter 900
# and regresses afterwards, while SATA sits at ~1950 from iter 500 on.  Extended to 15000,
# which is ~7.8 h at the measured 1 h 34 min per 3000 iterations.
#
# Optional warm start (off by default): continue from a finished V52 policy.  SATA's Gompertz
# clock must be restored too, otherwise general_scale restarts at ~0.13 (~9 Nm) and destroys
# the pretrained policy.  To enable, uncomment the two lines below and append
#   --init_from "$V52_CKPT" --init_from_bump 0 --growth_step_offset 72000
# to the exec line (bump 0 because 21-13-36's model_3000.pt is 1-based).
# V52_CKPT="$SOURCE_ROOT/terrain_v52_sata_ppo_flat/logs/rsl_rl/terrain_v52_sata_ppo_flat/2026-10-21_21-13-36_V52_sata_ppo_flat_20261021T131323Z/model_3000.pt"
# [[ -f "$V52_CKPT" ]] || { echo "Missing V52 checkpoint: $V52_CKPT" >&2; exit 1; }

exec "$PYTHON" -u "$ROOT/train.py" \
  --task "$TASK" \
  --agent rsl_rl_sac_cfg_entry_point \
  --num_envs 4096 --max_iterations 15000 --seed 1 \
  --run_name "${LOG_TAG}_${STAMP}" --logger wandb --log_project_name isaac_sac_pipeline --headless 2>&1 | tee "$LOG"
