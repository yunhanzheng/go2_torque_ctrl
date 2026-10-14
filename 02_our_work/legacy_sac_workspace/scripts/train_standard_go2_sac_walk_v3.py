"""Warm-start v3 from standing weights with fresh optimizers and temperature."""

import math

import isaacgym  # noqa: F401

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


STAND_CHECKPOINT = "/data/sac/src/SATA_sac/legged_gym/logs/SAC_standard_go2_stand/Aug12_15-39-36_flat_zero_speed_seed1_formal_detached/model_500.pt"

args = get_args()
env, _ = task_registry.make_env(name=args.task, args=args)
runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
print("Loading standing actor/critic weights only from {}".format(STAND_CHECKPOINT))
runner.load(STAND_CHECKPOINT, load_optimizer=False)
runner.alg.log_alpha.data.fill_(math.log(train_cfg.algorithm.alpha))
runner.current_learning_iteration = 500
print("Fresh optimizers; SAC alpha reset to {:.6f}".format(runner.alg.alpha.item()))
runner.learn(train_cfg.runner.max_iterations, init_at_random_ep_len=False)
