#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/ubuntu22:/data/sac:/data/sac/scripts
export WANDB_MODE=offline
export V35_SOURCE_CHECKPOINT=/home/ubuntu22/logs/rsl_rl/terrain_v34_scratch_flat_foundation/2026-09-20_16-31-55_stage3_20260920T163148Z/model_1597.pt
export V35_ACTOR_ONLY=1
/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python -u /home/ubuntu22/terrain_v35_heading_corridor/train.py \
  --task Isaac-Go2-V35-HeadingFlat-v0 --agent rsl_rl_sac_cfg_entry_point \
  --num_envs 64 --max_iterations 20 --seed 1 --resume \
  --run_name smoke_heading_flat --headless \
  agent.algorithm.replay_buffer_size=8192 agent.algorithm.num_mini_batches=2 agent.algorithm.mini_batch_size=128 agent.algorithm.policy_frequency=10000 \
  2>&1 | tee /home/ubuntu22/terrain_v35_heading_corridor/flat_smoke.log
