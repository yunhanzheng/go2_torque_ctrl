"""Transfer-safe SAC for the V43 18 cm clearance curriculum."""
from terrain_v34_from_scratch_four_terrain.models import CenteredStartSAC


class V43GuardedTransferSAC(CenteredStartSAC):
    """Adapt the critic first, then permit sparse low-rate actor updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transfer_calls = 0
        self._configured_policy_frequency = self.policy_frequency
        self.update_step = 1

    def update(self):
        warmup = self._transfer_calls < 50
        self.policy_frequency = 10**12 if warmup else self._configured_policy_frequency
        result = super().update()
        self._transfer_calls += 1
        return result
