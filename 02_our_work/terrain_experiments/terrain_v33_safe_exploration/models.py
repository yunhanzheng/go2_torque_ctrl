"""SAC actor with current-frame bypass and a compact temporal encoder."""
import torch
from torch import nn
from rsl_rl.algorithms import SAC
from rsl_rl.models import SACActorModel


FRAME = 235
PACKED = 5 * FRAME
LATENT = 32


class HistoryActor(SACActorModel):
    """Current 235-D frame plus a 32-D encoding of four past frames."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        if self.obs_dim != PACKED:
            raise ValueError(f"Expected {PACKED} packed inputs, got {self.obs_dim}")
        self.encoder = nn.Sequential(
            nn.Linear(4 * FRAME, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, LATENT),
            nn.Tanh(),
        )
        self.mlp[0] = nn.Linear(FRAME + LATENT, self.mlp[0].out_features)

    def input_view(self, obs):
        return torch.cat([obs[key] for key in self.obs_groups], dim=-1)

    def get_latent(self, obs, masks=None, hidden_state=None):
        normalized = self.obs_normalizer(self.input_view(obs))
        return torch.cat((normalized[..., :FRAME], self.encoder(normalized[..., FRAME:])), dim=-1)

    def update_normalization(self, obs):
        if self.obs_normalization:
            self.obs_normalizer.update(self.input_view(obs))


class CenteredStartSAC(SAC):
    """Initialize the policy mean at the robot's zero/default-pose action."""

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
        print("[CENTERED_START] raw mean bias:", neutral_raw.detach().cpu().tolist(), flush=True)
        print("[CENTERED_START] initial deterministic action:", neutral_action.detach().cpu().tolist(), flush=True)
        return algorithm
