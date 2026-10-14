"""SAC initialization used by V34; no checkpoint-derived parameters."""
import torch
from torch import nn
from rsl_rl.algorithms import SAC


class CenteredStartSAC(SAC):
    """Make the freshly initialized deterministic policy command the default pose."""

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        algorithm = SAC.construct_algorithm(obs, env, cfg, device)
        actor = algorithm.actor
        output = next(module for module in reversed(actor.mlp) if isinstance(module, nn.Linear))
        neutral_raw = torch.atanh(torch.clamp(-actor.action_bias / actor.action_range, -0.999, 0.999))
        with torch.no_grad():
            output.weight[: actor.output_dim].zero_()
            output.bias[: actor.output_dim].copy_(neutral_raw)
            neutral_action = actor.action_range * torch.tanh(neutral_raw) + actor.action_bias
        print("[V34_FRESH_INIT] no checkpoint loaded", flush=True)
        print("[V34_FRESH_INIT] initial deterministic action:", neutral_action.detach().cpu().tolist(), flush=True)
        return algorithm

