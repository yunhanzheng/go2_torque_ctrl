"""Per-environment circular replay buffer with timeout-aware n-step sampling."""

from typing import Dict

import torch


class ReplayBuffer:
    def __init__(self, num_envs: int, obs_dim: int, action_dim: int, capacity: int,
                 n_steps: int, gamma: float, device: str):
        self.num_envs = num_envs
        self.capacity_per_env = max(n_steps + 1, capacity // num_envs)
        self.n_steps = n_steps
        self.gamma = gamma
        self.device = device
        shape = (num_envs, self.capacity_per_env)
        self.obs = torch.empty(*shape, obs_dim, device=device)
        self.actions = torch.empty(*shape, action_dim, device=device)
        self.rewards = torch.empty(*shape, 1, device=device)
        self.next_obs = torch.empty(*shape, obs_dim, device=device)
        self.dones = torch.empty(*shape, 1, dtype=torch.bool, device=device)
        self.timeouts = torch.empty(*shape, 1, dtype=torch.bool, device=device)
        self.position = 0
        self.size = 0

    @torch.no_grad()
    def add(self, obs, actions, rewards, next_obs, dones, timeouts) -> None:
        idx = self.position
        self.obs[:, idx].copy_(obs)
        self.actions[:, idx].copy_(actions)
        self.rewards[:, idx, 0].copy_(rewards)
        self.next_obs[:, idx].copy_(next_obs)
        self.dones[:, idx, 0].copy_(dones.bool())
        self.timeouts[:, idx, 0].copy_(timeouts.bool())
        self.position = (idx + 1) % self.capacity_per_env
        self.size = min(self.size + 1, self.capacity_per_env)

    def ready(self, batch_size: int) -> bool:
        return self.size >= self.n_steps and self.size * self.num_envs >= batch_size

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.size < self.n_steps:
            raise RuntimeError("Not enough consecutive transitions for n-step sampling")
        env_ids = torch.randint(self.num_envs, (batch_size,), device=self.device)
        # Age n is the newest valid starting point; age=size is the oldest.
        ages = torch.randint(self.n_steps, self.size + 1, (batch_size,), device=self.device)
        starts = (self.position - ages) % self.capacity_per_env
        offsets = torch.arange(self.n_steps, device=self.device).unsqueeze(0)
        indices = (starts.unsqueeze(1) + offsets) % self.capacity_per_env
        env_grid = env_ids.unsqueeze(1).expand_as(indices)

        rewards = self.rewards[env_grid, indices, 0]
        dones = self.dones[env_grid, indices, 0]
        timeouts = self.timeouts[env_grid, indices, 0]
        alive_before = torch.cat(
            [torch.ones(batch_size, 1, device=self.device), (~dones[:, :-1]).float()], dim=1
        ).cumprod(dim=1)
        discounts = self.gamma ** torch.arange(self.n_steps, device=self.device).float()
        n_reward = (rewards * alive_before * discounts.unsqueeze(0)).sum(1, keepdim=True)

        any_done = dones.any(1)
        first_done = dones.float().argmax(1)
        final_offset = torch.where(any_done, first_done, torch.full_like(first_done, self.n_steps - 1))
        final_idx = indices.gather(1, final_offset.unsqueeze(1)).squeeze(1)
        effective_steps = (final_offset + 1).unsqueeze(1)

        return {
            "obs": self.obs[env_ids, starts],
            "actions": self.actions[env_ids, starts],
            "rewards": n_reward,
            "next_obs": self.next_obs[env_ids, final_idx],
            "dones": self.dones[env_ids, final_idx].float(),
            "timeouts": self.timeouts[env_ids, final_idx].float(),
            "steps": effective_steps,
        }
