"""Warm-start standard-Go2 SAC walking from the successful standing policy."""

import isaacgym  # noqa: F401

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


STAND_CHECKPOINT = "/data/sac/src/SATA_sac/legged_gym/logs/SAC_standard_go2_stand/Aug12_15-39-36_flat_zero_speed_seed1_formal_detached/model_500.pt"


args = get_args()
env, _ = task_registry.make_env(name=args.task, args=args)
runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
print("Warm-starting standard Go2 SAC from {}".format(STAND_CHECKPOINT))
runner.load(STAND_CHECKPOINT, load_optimizer=True)
runner.learn(
    num_learning_iterations=train_cfg.runner.max_iterations,
    init_at_random_ep_len=False,
)
