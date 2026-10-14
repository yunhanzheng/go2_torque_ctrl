"""Python-3.8-compatible SAC networks for the legacy SATA RSL-RL stack."""

import copy
import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


def _activation(name: str) -> nn.Module:
    table = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if name.lower() not in table:
        raise ValueError("Unsupported activation: {}".format(name))
    return table[name.lower()]()


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, activation: str) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for width in hidden_dims:
        layers.extend([nn.Linear(last_dim, width), _activation(activation)])
        last_dim = width
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class RunningMeanStd(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -10.0, 10.0)


class SACActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=(512, 256, 128),
                 activation="silu", init_noise_std=0.15, action_limit=4.7,
                 normalize_obs=True, log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        self.obs_norm = RunningMeanStd(obs_dim) if normalize_obs else nn.Identity()
        self.net = _mlp(obs_dim, hidden_dims, 2 * action_dim, activation)
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        limit = torch.as_tensor(action_limit, dtype=torch.float32)
        if limit.ndim == 0:
            limit = limit.repeat(action_dim)
        self.register_buffer("action_limit", limit)

        final = self.net[-1]
        nn.init.normal_(final.weight[:action_dim], mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias[:action_dim])
        nn.init.zeros_(final.weight[action_dim:])
        nn.init.constant_(final.bias[action_dim:], math.log(init_noise_std))

    def _distribution(self, obs: torch.Tensor) -> Normal:
        out = self.net(self.obs_norm(obs))
        mean, log_std = out.chunk(2, dim=-1)
        return Normal(mean, log_std.clamp(self.log_std_min, self.log_std_max).exp())

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs)
        latent = dist.rsample()
        unit_action = torch.tanh(latent)
        action = unit_action * self.action_limit
        log_prob = dist.log_prob(latent)
        log_prob -= torch.log(1.0 - unit_action.square() + 1e-6)
        log_prob -= torch.log(self.action_limit)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self._distribution(obs)
        latent = dist.mean if deterministic else dist.rsample()
        return torch.tanh(latent) * self.action_limit

    @torch.no_grad()
    def update_normalization(self, obs: torch.Tensor) -> None:
        if isinstance(self.obs_norm, RunningMeanStd):
            self.obs_norm.update(obs)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=(512, 256, 128),
                 activation="silu", normalize_obs=True):
        super().__init__()
        self.obs_norm = RunningMeanStd(obs_dim) if normalize_obs else nn.Identity()
        self.net = _mlp(obs_dim + action_dim, hidden_dims, 1, activation)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([self.obs_norm(obs), action], dim=-1))

    @torch.no_grad()
    def update_normalization(self, obs: torch.Tensor) -> None:
        if isinstance(self.obs_norm, RunningMeanStd):
            self.obs_norm.update(obs)


class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, **kwargs):
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, **kwargs)
        self.q2 = QNetwork(obs_dim, action_dim, **kwargs)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        return self.q1(obs, action), self.q2(obs, action)

    def target(self, obs: torch.Tensor, action: torch.Tensor):
        return self.target_q1(obs, action), self.target_q2(obs, action)

    @torch.no_grad()
    def soft_update(self, tau: float) -> None:
        for source, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
            for src_param, tgt_param in zip(source.parameters(), target.parameters()):
                tgt_param.mul_(1.0 - tau).add_(src_param, alpha=tau)

    @torch.no_grad()
    def update_normalization(self, obs: torch.Tensor) -> None:
        self.q1.update_normalization(obs)
        self.q2.update_normalization(obs)
        if isinstance(self.q1.obs_norm, RunningMeanStd):
            self.target_q1.obs_norm.load_state_dict(self.q1.obs_norm.state_dict())
            self.target_q2.obs_norm.load_state_dict(self.q2.obs_norm.state_dict())
