"""Behaviour-cloning policy and its checkpoint format.

The cloning policy is deliberately *not* an rsl_rl actor.  rsl_rl's 3.x actor is a stochastic
``MLPModel`` over a ``TensorDict`` with its own distribution plumbing, and wrapping behaviour
cloning into that API would add a second definition of "what the network is" that could drift
from the PPO arm's.  Instead:

* the network is built from the *same* runner config as the PPO arm
  (:func:`from_runner_cfg` reads ``policy.actor_hidden_dims`` and ``policy.activation``), so both
  arms provably have the same function class;
* ``forward`` takes whatever the evaluator has at hand - a ``TensorDict``, a ``dict`` with the
  ``policy`` group, or a plain tensor - and returns deterministic actions, which is exactly what
  ``OnPolicyRunner.get_inference_policy`` returns as well;
* the checkpoint is a plain dict with a format tag, so ``evaluate_force.py`` can tell a cloned
  policy from an rsl_rl one by looking at the file rather than at a flag.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

OBS_KEY = "policy"
"""Observation group the Go2 flat task uses for the actor."""

CHECKPOINT_FORMAT = 1
"""Bumped whenever the key layout below changes; the loader refuses anything else."""

_ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU, "gelu": nn.GELU}


def _activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(f"unknown activation '{name}', known: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


def _as_tensor(obs) -> torch.Tensor:
    """Accept a ``TensorDict``, a plain ``dict`` or a tensor."""
    if torch.is_tensor(obs):
        return obs
    return obs[OBS_KEY]


class BcPolicy(nn.Module):
    """Deterministic MLP policy: ``obs -> action``, no distribution, no state."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128, 128),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.activation = str(activation)

        layers: list[nn.Module] = []
        width = self.obs_dim
        for hidden in self.hidden_dims:
            layers += [nn.Linear(width, hidden), _activation(self.activation)]
            width = hidden
        layers.append(nn.Linear(width, self.action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs) -> torch.Tensor:
        return self.mlp(_as_tensor(obs))


def from_runner_cfg(agent_cfg, obs_dim: int, action_dim: int) -> BcPolicy:
    """Build the policy with the PPO arm's architecture, read from its runner config."""
    policy_cfg = agent_cfg.policy
    return BcPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=tuple(policy_cfg.actor_hidden_dims),
        activation=str(policy_cfg.activation),
    )


def save_bc(
    path: str,
    policy: BcPolicy,
    *,
    task: str,
    arm: str,
    rate_hz: float,
    action_scale: float,
    iteration: int,
    extra: dict | None = None,
) -> None:
    """Write a checkpoint the evaluator recognises without being told what it is."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "bc_format": CHECKPOINT_FORMAT,
            "task": task,
            "arm": arm,
            "rate_hz": float(rate_hz),
            "action_scale": float(action_scale),
            "iteration": int(iteration),
            "obs_dim": policy.obs_dim,
            "action_dim": policy.action_dim,
            "hidden_dims": list(policy.hidden_dims),
            "activation": policy.activation,
            "state_dict": policy.state_dict(),
            "extra": extra or {},
        },
        path,
    )


def read_checkpoint(path: str) -> dict | None:
    """Return the checkpoint dict if it is a BC file, ``None`` if it is something else."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - an unreadable file is simply "not ours"
        return None
    if not isinstance(payload, dict) or "bc_format" not in payload:
        return None
    if payload["bc_format"] != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path}: BC checkpoint format {payload['bc_format']}, expected {CHECKPOINT_FORMAT}"
        )
    return payload


def is_bc_checkpoint(path: str) -> bool:
    return read_checkpoint(path) is not None


def load_bc(path: str, device: str | torch.device = "cpu") -> BcPolicy:
    payload = read_checkpoint(path)
    if payload is None:
        raise ValueError(f"{path} is not a behaviour-cloning checkpoint")
    policy = BcPolicy(
        obs_dim=payload["obs_dim"],
        action_dim=payload["action_dim"],
        hidden_dims=payload["hidden_dims"],
        activation=payload["activation"],
    )
    policy.load_state_dict(payload["state_dict"])
    policy.to(device).eval()
    return policy
