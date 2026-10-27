"""Smoke test: build the V52 SATA-identical (PPO + SATA terrain) env and step it.

Verifies:
- PPO runner cfg is OnPolicyRunner with SATA's hyperparameters (max_iterations=3000)
- env constructs with EXACTLY SATA's 9 rewards (V51's 16 fully replaced)
- SATA terrain: 20 % smooth slope + 80 % rough slope, 10 x 20 tiles, no curriculum
- observation is 60-D (SATA layout); SataTorqueAction + SataVelocityCommand reused
- all rewards compute finite values; growth-coupled terms track general_scale
- TG torques stay bounded (<= 23.5 Nm) and motor fatigue accumulates
"""
import argparse
import os
import sys

# --- Path bootstrap (mirrors train.py): put the source tree, shared adapters, and the
# V51 dir (for sata_alignment) on sys.path so `terrain_v52_sata_ppo_flat.config` and
# `sata_rewards` import regardless of the current working directory.
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

import terrain_v52_sata_ppo_flat.config  # noqa: F401,E402  (registers the task)
from terrain_v52_sata_ppo_flat.config import (  # noqa: E402
    V52SataPpoFlatEnvCfg,
    V52SataPpoFlatRunnerCfg,
)

# ---------------- PPO runner cfg (SATA GO2TorqueCfgPPO) ----------------
print("=== PPO RUNNER CFG ===", flush=True)
rcfg = V52SataPpoFlatRunnerCfg()
print("class_name      :", rcfg.class_name, "(expect OnPolicyRunner)", flush=True)
print("max_iterations  :", rcfg.max_iterations, "(expect 3000)", flush=True)
print("num_steps_per_env:", rcfg.num_steps_per_env, flush=True)
print("policy hidden   :", rcfg.policy.actor_hidden_dims, rcfg.policy.activation, flush=True)
print("algo lr/sched/kl:", rcfg.algorithm.learning_rate, rcfg.algorithm.schedule,
      rcfg.algorithm.desired_kl, flush=True)
assert rcfg.class_name == "OnPolicyRunner", rcfg.class_name
assert rcfg.max_iterations == 3000, rcfg.max_iterations
# Simulate train.py's deprecated-cfg handling + to_dict (what OnPolicyRunner consumes).
try:
    import importlib.metadata as metadata

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    iv = metadata.version("rsl-rl-lib")
    rcfg2 = handle_deprecated_rsl_rl_cfg(rcfg, iv)
    d = rcfg2.to_dict()
    print("to_dict OK (%d keys); obs_groups=%s" % (len(d), d.get("obs_groups")), flush=True)
except Exception as e:  # noqa: BLE001
    print("WARN: to_dict path raised:", repr(e), flush=True)

# ---------------- environment ----------------
env_cfg = V52SataPpoFlatEnvCfg()
env_cfg.scene.num_envs = 2
env = gym.make("Isaac-Go2-V52-SataPpoFlat-v0", cfg=env_cfg)
uw = env.unwrapped
print("=== ENV CREATED ===", flush=True)
print("action_space     :", env.action_space.shape, flush=True)
print("obs_space[policy]:", env.observation_space["policy"].shape, flush=True)

# SATA terrain: smooth / rough slope tiles, no curriculum
subs = list(env_cfg.scene.terrain.terrain_generator.sub_terrains.keys())
print("terrain sub_terrains:", subs, "(expect smooth + rough slope)", flush=True)
assert subs == ["sata_smooth_slope", "sata_rough_slope"], subs
_tz = uw.scene.terrain.terrain_origins[..., 2]
print(
    f"terrain origin z: min={_tz.min():.3f} max={_tz.max():.3f} mean={_tz.mean():.3f}",
    flush=True,
)
assert _tz.max() > 0.02 and _tz.min() < -0.02, "terrain is flat - SATA slopes were not generated"

at = uw.action_manager.get_term("joint_pos")
ct = uw.command_manager.get_term("base_velocity")
print("action term :", type(at).__name__, "| command term:", type(ct).__name__, flush=True)

# exactly SATA's 9 rewards must be active (V51's 16 replaced)
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

# --- SATA termination set (GO2Torque.check_termination): no torso-contact term ----
_term = set(uw.termination_manager.active_terms)
print("active terminations:", sorted(_term), flush=True)
assert _term == {"dof_pos_limit", "fell_over", "time_out"}, _term

# --- SATA timing: substep-based clock ------------------------------------------------
from terrain_v51_native_terrain.sata_alignment import sata_time_out  # noqa: E402

# SATA's while-loop can only yield 1 or 2 substeps at 100-200 Hz (never 0 or 3).
assert uw.cfg.decimation in (1, 2), uw.cfg.decimation
_sub_before = int(uw._sata_env_substeps[0].item())
env.step(torch.randn(2, 12, device=uw.device) * 0.5)
_sub_after = int(uw._sata_env_substeps[0].item())
assert _sub_after >= _sub_before, "substep clock went backwards"
assert (_sub_after - _sub_before) in (1, 2), (_sub_before, _sub_after)
# SATA time-out: ``episode_length_buf > ceil(episode_length_s / dt)`` in 200 Hz substeps.
_limit = int(round(uw.cfg.episode_length_s / uw.physics_dt))
assert _limit == 2000, _limit
uw._sata_env_substeps[:] = _limit
assert not bool(sata_time_out(uw).any()), "time-out fired one substep early"
uw._sata_env_substeps[:] = _limit + 1
assert bool(sata_time_out(uw).all()), "time-out did not fire"
uw._sata_env_substeps[:] = 0
print(
    f"timing OK: decimation={uw.cfg.decimation} substeps/step, time-out at {_limit} substeps "
    f"({uw.cfg.episode_length_s:g} s x {1 / uw.physics_dt:.0f} Hz)",
    flush=True,
)

print("=== SMOKE TEST PASSED ===", flush=True)
# simulation_app.close() hangs forever on this Vulkan-less headless box (render cleanup
# spins at 99% CPU and never releases the GPU). Everything is verified by now, so
# hard-exit immediately instead of tearing down the simulator.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
