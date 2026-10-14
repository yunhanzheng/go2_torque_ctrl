"""Smoke-test the timeout transition contract required by off-policy SAC."""

import isaacgym  # noqa: F401 -- must be imported before torch-backed task modules
import torch
from legged_gym.envs import *  # noqa: F401,F403 -- registers tasks
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args()
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 8
    env_cfg.env.episode_length_s = 0.02
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.loss_action_obs = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.measure_heights = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    saw_timeout = False

    for _ in range(20):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        next_obs, _, _, dones, extras = env.step(actions)

        assert "time_outs" in extras, "environment did not expose time_outs"
        assert "terminal_observation" in extras, "environment did not expose pre-reset observation"
        terminal_obs = extras["terminal_observation"]
        assert terminal_obs.shape == next_obs.shape
        assert torch.isfinite(terminal_obs).all()
        assert torch.isfinite(next_obs).all()

        timeout_mask = extras["time_outs"].bool()
        assert torch.all(~timeout_mask | dones.bool()), "timeout must also set done"
        if timeout_mask.any():
            saw_timeout = True
            changed = torch.any(terminal_obs[timeout_mask] != next_obs[timeout_mask], dim=1)
            assert changed.any(), "terminal observation unexpectedly equals every reset observation"

        obs = next_obs

    assert saw_timeout, "short smoke-test episode did not produce a timeout"
    print("PASS: done, time_outs, terminal_observation, and post-reset observation are consistent")


if __name__ == "__main__":
    main()
