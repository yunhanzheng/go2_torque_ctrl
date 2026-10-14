"""Transfer the 48-D standing SAC into the 50-D gait-clock policy."""

import math

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


STAND_CHECKPOINT = "/data/sac/src/SATA_sac/legged_gym/logs/SAC_standard_go2_stand/Aug12_15-39-36_flat_zero_speed_seed1_formal_detached/model_500.pt"


def transfer_module(module, source):
    target = module.state_dict()
    for key, old in source.items():
        if key not in target:
            continue
        new = target[key]
        if new.shape == old.shape:
            new.copy_(old)
        elif key.endswith("obs_norm.mean") or key.endswith("obs_norm.var"):
            new[:old.shape[0]].copy_(old)
            if key.endswith("mean"):
                new[old.shape[0]:].zero_()
            else:
                new[old.shape[0]:].fill_(1.0)
        elif key == "net.0.weight":
            new[:, :old.shape[1]].copy_(old)
            new[:, old.shape[1]:].zero_()
        elif key.endswith("net.0.weight"):
            # Critic input is [observation, action]. Insert two clock columns
            # before the 12 action columns.
            old_obs = old.shape[1] - 12
            new[:, :old_obs].copy_(old[:, :old_obs])
            new[:, old_obs:old_obs + 2].zero_()
            new[:, old_obs + 2:].copy_(old[:, old_obs:])
        else:
            raise RuntimeError("Unhandled transfer shape {}: {} -> {}".format(
                key, tuple(old.shape), tuple(new.shape)
            ))
    module.load_state_dict(target)


args = get_args()
env, _ = task_registry.make_env(name=args.task, args=args)
runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
checkpoint = torch.load(STAND_CHECKPOINT, map_location=runner.device)["sac"]
transfer_module(runner.alg.actor, checkpoint["actor"])
transfer_module(runner.alg.critic, checkpoint["critic"])
runner.alg.log_alpha.data.fill_(math.log(train_cfg.algorithm.alpha))
runner.current_learning_iteration = 500
print("Transferred standing weights 48D -> gait-clock 50D; fresh optimizers")
print("SAC alpha reset to {:.6f}".format(runner.alg.alpha.item()))
runner.learn(train_cfg.runner.max_iterations, init_at_random_ep_len=False)
