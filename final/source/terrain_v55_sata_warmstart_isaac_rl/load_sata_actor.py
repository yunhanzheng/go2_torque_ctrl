"""Copy SATA's legged_gym actor weights into the V55 policy (warm start).

Weight compatibility (checked against the two checkpoints, not assumed)
----------------------------------------------------------------------
SATA  ``logs/SATA/Oct14_16-33-32_/model_3000.pt`` (legged_gym / rsl_rl 1.x format)::

    model_state_dict: std (12,)
                      actor.0.weight (512, 60)  actor.0.bias (512,)
                      actor.2.weight (256, 512) actor.2.bias (256,)
                      actor.4.weight (128, 256) actor.4.bias (128,)
                      actor.6.weight (12, 128)  actor.6.bias (12,)
                      critic.*                  (same sizes, output 1)

this repo's rsl_rl SAC fork (``OnPolicyRunner.save``, V52/V54 checkpoints)::

    actor_state_dict : std (12,)
                       mlp.0.weight (512, 60) ... mlp.6.bias (12,)
    critic_state_dict: mlp.0 ... mlp.6 (output 1)

Same architecture (60 -> 512 -> 256 -> 128 -> 12, ELU), same per-joint ``std``, so the actor
transfers exactly and only the key prefix changes (``actor.`` -> ``mlp.``).  The critic is
**not** transferred: V55 replaces the whole reward function, so the old value function is
meaningless and starting from it would only inject stale scale.

Notes
-----
- ``std`` is **not** copied by default: V55 starts from ``init_noise_std`` (1.0, the IsaacLab
  default) so that the warm start touches only the deterministic actor.  Pass ``load_std=True``
  (CLI: ``--sata_init_std``) to also take SATA's saved std (min 1.48 / mean 2.31 / max 3.10,
  i.e. ~11 Nm of 1-sigma torque noise) as a single-variable ablation.
- Nothing here touches the optimizer: the runner starts with fresh Adam moments, which is
  what we want when the loss scale changes.
"""
from __future__ import annotations

import argparse
import torch

# SATA's rsl_rl 1.x prefix -> this repo's fork prefix.
_PREFIX_MAP = (("mlp.", "actor."), ("actor.", "mlp."))


def _candidates(key: str) -> list[str]:
    out = [key]
    for old, new in _PREFIX_MAP:
        if key.startswith(old):
            out.append(new + key[len(old):])
    return out


def load_sata_actor(module, checkpoint: str, load_std: bool = False, verbose: bool = True) -> dict:
    """Load SATA's actor network into ``module`` (rsl_rl's MLPModel).

    Returns a small report dict: ``{"copied": [...], "skipped": [...], "std": bool}``.
    Raises if a weight exists on both sides with mismatched shapes -- a silent partial load
    would produce a policy that looks trained but is not.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    src = ckpt.get("model_state_dict", ckpt)
    src = {k: v for k, v in src.items() if torch.is_tensor(v)}
    if not src:
        raise ValueError(f"{checkpoint}: no tensors found (keys={list(ckpt.keys())})")

    target = module.state_dict()
    merged, copied, skipped, shape_errors = {}, [], [], []

    for key, tensor in target.items():
        match = None
        for cand in _candidates(key):
            if cand in src:
                if tuple(src[cand].shape) != tuple(tensor.shape):
                    shape_errors.append(f"{key} {tuple(tensor.shape)} vs {cand} {tuple(src[cand].shape)}")
                else:
                    match = cand
                break
        if key == "std" and not load_std:
            merged[key] = tensor
            skipped.append(f"{key} (std disabled)")
            continue
        if match is None:
            merged[key] = tensor
            skipped.append(key)
            continue
        merged[key] = src[match].to(dtype=tensor.dtype, device=tensor.device).clone()
        copied.append(f"{key} <- {match}")

    if shape_errors:
        raise RuntimeError("Actor transfer refused, shape mismatch:\n  " + "\n  ".join(shape_errors))

    module.load_state_dict(merged, strict=True)

    report = {"copied": copied, "skipped": skipped, "std": any(c.startswith("std <-") for c in copied)}
    if verbose:
        print(f"[sata_init] {checkpoint}", flush=True)
        print(f"[sata_init] source format: {'legged_gym (model_state_dict)' if 'model_state_dict' in ckpt else 'raw'}"
              f", sata iter={ckpt.get('iter', '?')}", flush=True)
        print(f"[sata_init] copied {len(copied)} tensors:", flush=True)
        for line in copied:
            print(f"    {line}", flush=True)
        if skipped:
            print(f"[sata_init] left at init ({len(skipped)}): {skipped}", flush=True)
        std = module.state_dict()["std"]
        tag = "from checkpoint" if report["std"] else "kept at init_noise_std"
        print(f"[sata_init] std ({tag}): min={std.min():.3f} mean={std.mean():.3f} max={std.max():.3f}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a SATA checkpoint (no simulator needed).")
    parser.add_argument("--ckpt", required=True, help="Path to SATA's model_XXXX.pt")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print("top-level keys:", list(ckpt.keys()), flush=True)
    src = ckpt.get("model_state_dict", ckpt)
    for key, value in src.items():
        if torch.is_tensor(value):
            print(f"  {key:26s} {tuple(value.shape)}")
    print("iter:", ckpt.get("iter", "?"), "| infos:", list(ckpt.get("infos", {}).keys()), flush=True)


if __name__ == "__main__":
    main()
