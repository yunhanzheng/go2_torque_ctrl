import torch


class SparseHistory:
    """Episode-safe history sampled at 0, 40, 80, 120 and 160 ms at 50 Hz."""

    def __init__(self, num_envs, device, width=235, offsets=(0, 2, 4, 6, 8)):
        self.offsets = torch.tensor(offsets, device=device, dtype=torch.long)
        self.length = max(offsets) + 1
        self.values = torch.zeros(num_envs, self.length, width, device=device)
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.cursor = -1
        self.last_step = None

    def reset(self, env_ids=None):
        self.valid[slice(None) if env_ids is None else env_ids] = False

    def update(self, current, step):
        if self.last_step is not None and step not in (self.last_step, self.last_step + 1):
            raise RuntimeError("History must be observed on every policy step")
        if step != self.last_step:
            self.cursor = (self.cursor + 1) % self.length
            self.last_step = step
        fresh = ~self.valid
        self.values[fresh] = current[fresh, None, :]
        self.valid.fill_(True)
        self.values[:, self.cursor] = current
        indices = (self.cursor - self.offsets) % self.length
        return self.values[:, indices].flatten(1).clone()
