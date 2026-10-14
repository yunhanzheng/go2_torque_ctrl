#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu22
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/ubuntu22:/data/sac:/data/sac/scripts
export WANDB_MODE=offline
export V35_SOURCE_CHECKPOINT=/home/ubuntu22/logs/rsl_rl/terrain_v35_heading_corridor/2026-09-21_01-01-18_mixed_actor_adapt_200/model_396.pt
export V35_ACTOR_LR_OVERRIDE=2e-6
unset V35_FREEZE_ACTOR_NORMALIZER
unset V35_ACTOR_ONLY
/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python -u /home/ubuntu22/terrain_v35_heading_corridor/train.py \
  --task Isaac-Go2-V35-HeadingCorridor-v0 --agent rsl_rl_sac_cfg_entry_point \
  --num_envs 4096 --max_iterations 150 --seed 1 --resume \
  --run_name mixed_actor_stage2_150 --headless \
  agent.algorithm.policy_frequency=100 \
  2>&1 | tee /home/ubuntu22/terrain_v35_heading_corridor/mixed_actor_stage2_150.log
