import hashlib
import json
from pathlib import Path

import torch
from rsl_rl.algorithms.sac import SAC


SOURCE = "/data/sac/saved_models/ETH_SAC_V24_10CM/model_299.pt"


def install():
    original_init, original_update = SAC.__init__, SAC.update

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._transfer_calls = 0
        self._transfer_frequency = self.policy_frequency
        self._transfer_alpha = self.auto_alpha
        self.update_step = 1

    def update(self):
        warmup = self._transfer_calls < 25
        self.policy_frequency = 10**12 if warmup else self._transfer_frequency
        self.auto_alpha = False if warmup else self._transfer_alpha
        result = original_update(self)
        self._transfer_calls += 1
        return result

    SAC.__init__, SAC.update = init, update


def initialize(runner, log_dir):
    runner.load(
        SOURCE,
        load_cfg={"actor": True, "critic": True, "optimizer": True, "iteration": False, "rnd": False},
    )
    for group in runner.alg.actor_optimizer.param_groups:
        group["lr"] = 2.5e-5
    runner._initial_actor = {
        key: value.detach().cpu().clone() for key, value in runner.alg.actor.named_parameters()
    }
    metadata = {
        "source": SOURCE,
        "sha256": hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest(),
        "warmup_update_calls": 25,
        "transfer": "V24 actor, critic, optimizer and alpha; fresh replay and iteration counter",
        "actor_observation_dim": 62,
    }
    Path(log_dir, "provenance.json").write_text(json.dumps(metadata, indent=2))
    print("[V26_INIT]", metadata, flush=True)


def audit(runner, log_dir):
    state = runner.alg.actor.state_dict()
    change = max(
        float((value.detach().cpu() - runner._initial_actor[key]).abs().max())
        for key, value in runner.alg.actor.named_parameters()
    )
    result = {
        "iteration": runner.current_learning_iteration,
        "update_calls": runner.alg._transfer_calls,
        "finite": all(bool(torch.isfinite(value).all()) for value in state.values()),
        "actor_change": change,
    }
    Path(log_dir, "audit.json").write_text(json.dumps(result, indent=2))
    assert result["finite"] and (result["update_calls"] <= 25 or change > 0), result
    print("[V26_AUDIT]", result, flush=True)
