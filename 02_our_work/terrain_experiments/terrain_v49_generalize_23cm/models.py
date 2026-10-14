"""Guarded SAC transfer from the protected V48 23 cm model."""
from terrain_v34_from_scratch_four_terrain.models import CenteredStartSAC


class V49GuardedTransferSAC(CenteredStartSAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transfer_calls = 0
        self._configured_policy_frequency = self.policy_frequency
        self.update_step = 1

    def update(self):
        # Let the fresh critics see the mixed distribution before touching the
        # successful 23 cm actor.
        self.policy_frequency = 10**12 if self._transfer_calls < 30 else self._configured_policy_frequency
        result = super().update()
        self._transfer_calls += 1
        return result
