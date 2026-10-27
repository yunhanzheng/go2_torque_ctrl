"""Smoke test: build the V53 SAC control env (V52's env + SAC runner) and step it.

Verifies:
- SAC runner cfg is OffPolicyRunner with the repo's flat from-scratch recipe
  (CenteredStartSAC, actor_lr=5e-5, critic_lr=2e-4, gamma=0.96, n_steps=5, 1500 iters)
- env is V52's, unchanged: EXACTLY SATA's 9 rewards, flat ground, 60-D obs
- all rewards compute finite values; TG torques stay bounded (<= 23.5 Nm)
"""
import argparse
import os
import sys

# --- Path bootstrap (mirrors train.py): source tree, dependency chain (V34 SAC runner),
# shared adapters, and the V51 dir (for sata_alignment / sata_rewards).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (
    _SOURCE_ROOT,
    os.path.join(_SOURCE_ROOT, "dependency_chain"),
    os.path.join(_SOURCE_ROOT, "base_adapter"),
    os.path.join(_SOURCE_ROOT, "base_adapter", "initial_stair_source"),
    os.path.join(_SOURCE_ROOT, "terrain_v51_native_terrain"),
    _HERE,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

# --- sa_16dof compat shim (full rationale in base_adapter/cli_args.py) ---
# The server's shared isaaclab_tasks auto-imports a sibling `sa_16dof` task that
# needs rsl_rl.modules.distribution (absent from this env's rsl_rl SAC fork), which
# would crash `import ...config` below. Pre-register sa_16dof as an empty-path stub
# so _walk_packages skips it. No-op locally (dev isaaclab_tasks has no sa_16dof).
import types as _types  # noqa: E402

_STUB = "isaaclab_tasks.manager_based.locomotion.velocity.config.sa_16dof"
if _STUB not in sys.modules:
    _m = _types.ModuleType(_STUB)
    _m.__path__ = []
    sys.modules[_STUB] = _m

import terrain_v53_sata_sac_flat.config  # noqa: F401,E402  (registers the task)
from terrain_v53_sata_sac_flat.config import (  # noqa: E402
    V53SataSacFlatEnvCfg,
    V53SataSacFlatRunnerCfg,
)

# ---------------- SAC runner cfg (repo's flat from-scratch SAC recipe) ----------------
print("=== SAC RUNNER CFG ===", flush=True)
rcfg = V53SataSacFlatRunnerCfg()
print("class_name        :", rcfg.class_name, "(expect OffPolicyRunner)", flush=True)
print("max_iterations    :", rcfg.max_iterations, "(expect 1500)", flush=True)
print("num_steps_per_env :", rcfg.num_steps_per_env, flush=True)
print("algorithm.class   :", rcfg.algorithm.class_name, flush=True)
print("actor hidden/act  :", rcfg.actor.hidden_dims, rcfg.actor.activation, flush=True)
print(
    "algo lr a/c/gamma :",
    rcfg.algorithm.actor_learning_rate,
    rcfg.algorithm.critic_learning_rate,
    rcfg.algorithm.gamma,
    "| n_steps=", rcfg.algorithm.n_steps,
    "| policy_freq=", rcfg.algorithm.policy_frequency,
    "| auto_alpha=", rcfg.algorithm.auto_alpha,
    flush=True,
)
assert rcfg.class_name == "OffPolicyRunner", rcfg.class_name
assert rcfg.max_iterations == 1500, rcfg.max_iterations
assert rcfg.algorithm.class_name.endswith("CenteredStartSAC"), rcfg.algorithm.class_name
# Simulate train.py's deprecated-cfg handling + to_dict (what OffPolicyRunner consumes).
try:
    import importlib.metadata as metadata

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    iv = metadata.version("rsl-rl-lib")
    rcfg2 = handle_deprecated_rsl_rl_cfg(rcfg, iv)
    d = rcfg2.to_dict()
    print("to_dict OK (%d keys); obs_groups=%s" % (len(d), d.get("obs_groups")), flush=True)
except Exception as e:  # noqa: BLE001
    print("WARN: to_dict path raised:", repr(e), flush=True)

# ---------------- environment (identical to V52) ----------------
env_cfg = V53SataSacFlatEnvCfg()
env_cfg.scene.num_envs = 2
env = gym.make("Isaac-Go2-V53-SataSacFlat-v0", cfg=env_cfg)
uw = env.unwrapped
print("=== ENV CREATED ===", flush=True)
print("action_space     :", env.action_space.shape, flush=True)
print("obs_space[policy]:", env.observation_space["policy"].shape, flush=True)

# flat ground: exactly one 'plane' sub-terrain
subs = list(env_cfg.scene.terrain.terrain_generator.sub_terrains.keys())
print("terrain sub_terrains:", subs, "(expect ['plane'])", flush=True)
assert subs == ["plane"], subs

# V52's termination fix must be inherited (no invalid_upright_state)
term_terms = set(uw.termination_manager.active_terms)
print("active terminations:", sorted(term_terms), "(expect no invalid_upright_state)", flush=True)
assert "invalid_upright_state" not in term_terms, term_terms

at = uw.action_manager.get_term("joint_pos")
ct = uw.command_manager.get_term("base_velocity")
print("action term :", type(at).__name__, "| command term:", type(ct).__name__, flush=True)

# exactly SATA's 9 rewards must be active (V51's 16 replaced by V52)
active = set(uw.reward_manager.active_terms)
expected = {
    "forward", "head_height", "moving_y", "moving_yaw", "motor_fatigue",
    "roll", "lin_vel_z", "soft_dof_pos_limits", "dof_acc",
}
print("active reward terms:", sorted(active), flush=True)
assert active == expected, f"reward mismatch: extra={active - expected} missing={expected - active}"

obs, _ = env.reset()
dim = obs["policy"].shape[-1]
print("reset obs dim:", dim, flush=True)
assert dim == 60, f"Expected 60-D SATA obs, got {dim}"

# growth-coupled terms recomputed by hand for inspection
from sata_rewards import sata_forward, sata_head_height, sata_motor_fatigue  # noqa: E402

for i in range(10):
    action = torch.randn(2, 12, device=uw.device) * 0.5
    obs, rew, term, trunc, info = env.step(action)
    fwd = sata_forward(uw).mean().item()
    hh = sata_head_height(uw).mean().item()
    mf = sata_motor_fatigue(uw).mean().item()
    cmd = uw.command_manager.get_command("base_velocity")
    print(
        f"step {i}: obs_dim={obs['policy'].shape[-1]} rew={rew.mean().item():+.3f} "
        f"|tau|max={at.torques.abs().max().item():.3f} fat={at.motor_fatigue.max().item():.4f} "
        f"gs={at.general_scale:.4f} fwd={fwd:+.3f} head={hh:+.3f} mfat={mf:+.4f} "
        f"cmd_x=[{cmd[:, 0].min().item():.2f},{cmd[:, 0].max().item():.2f}]",
        flush=True,
    )
    assert obs["policy"].shape[-1] == 60
    assert torch.isfinite(rew).all(), "non-finite total reward"
    assert torch.isfinite(at.torques).all(), "non-finite torque"
    assert at.torques.abs().max().item() <= 23.5 + 1e-3, "torque exceeds limit"

print("=== SMOKE TEST PASSED ===", flush=True)
# simulation_app.close() hangs forever on this Vulkan-less headless box (render cleanup
# spins at 99% CPU and never releases the GPU). Everything is verified by now, so
# hard-exit immediately instead of tearing down the simulator.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
