"""SAC actor with current-frame bypass and a compact temporal encoder."""
import torch
from torch import nn
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
