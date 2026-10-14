"""Transfer-safe SAC used by V41."""
from terrain_v34_from_scratch_four_terrain.models import CenteredStartSAC


class GuardedHeightTransferSAC(CenteredStartSAC):
    """Warm the critic on the shifted terrain distribution before actor updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transfer_calls = 0
        self._configured_policy_frequency = self.policy_frequency
        self._configured_auto_alpha = self.auto_alpha
        # Make the first post-warmup call eligible for an actor update.
        self.update_step = 1

    def update(self):
        warmup = self._transfer_calls < 25
        self.policy_frequency = 10**12 if warmup else self._configured_policy_frequency
        self.auto_alpha = False if warmup else self._configured_auto_alpha
        result = super().update()
        self._transfer_calls += 1
        return result
