"""Smoke test: build the V54 (SATA-aligned PPO + Go2 rough) env and step it.

Verifies:
- PPO runner cfg is OnPolicyRunner with SATA's hyperparameters (max_iterations=15000)
- terrain is IsaacLab's official Go2 rough: 6 sub-terrains, 10x20 tiles, official
  curriculum (generator.curriculum=True so difficulty grows with the row index,
  max_init_terrain_level=5 so the first episode starts in rows 0-5)
- terrain-level curriculum term (mdp.terrain_levels_vel) is restored (V52 had it None)
- env constructs with the 2026-10-24 rev C table: the 2026-10-22 set (SATA's 9 +
  ang_vel_xy_l2 + action_rate_l2) with ``feet_air_time`` at IsaacLab's stock 0.01 / 0.5 s;
  undesired_contacts, feet_slip, gait_contact,
  swing_clearance and excessive_foot_height stay off; ``base_contact`` (illegal_contact on
  "base", 1 N) is restored; head_height measures the base
  against the local ground line (terrain under the front vs rear feet), lin_vel_z stays
  gated on per-foot flat ground, joint limits are the stock USD range
- observation is 60-D (SATA layout); SataTorqueAction + SataVelocityCommand reused
- all rewards compute finite values on rough ground
- TG torques stay bounded (<= 23.5 Nm)
"""
import argparse
import os
import sys

# --- Path bootstrap (mirrors train.py): put the source tree, shared adapters, V51
# (sata_alignment) and V52 (sata_rewards/config) on sys.path so
# `terrain_v54_sata_ppo_rough.config` imports regardless of the current working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (
    _SOURCE_ROOT,
    os.path.join(_SOURCE_ROOT, "dependency_chain"),
    os.path.join(_SOURCE_ROOT, "base_adapter"),
    os.path.join(_SOURCE_ROOT, "base_adapter", "initial_stair_source"),
    os.path.join(_SOURCE_ROOT, "terrain_v51_native_terrain"),
    os.path.join(_SOURCE_ROOT, "terrain_v52_sata_ppo_flat"),
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

import terrain_v54_sata_ppo_rough.config  # noqa: F401,E402  (registers the task)
from terrain_v54_sata_ppo_rough.config import (  # noqa: E402
    V54SataPpoRoughEnvCfg,
    V54SataPpoRoughRunnerCfg,
)

# ---------------- PPO runner cfg (same as V52 / SATA GO2TorqueCfgPPO) ----------------
print("=== PPO RUNNER CFG ===", flush=True)
rcfg = V54SataPpoRoughRunnerCfg()
print("class_name      :", rcfg.class_name, "(expect OnPolicyRunner)", flush=True)
print("max_iterations  :", rcfg.max_iterations, "(expect 15000)", flush=True)
print("num_steps_per_env:", rcfg.num_steps_per_env, flush=True)
print("policy hidden   :", rcfg.policy.actor_hidden_dims, rcfg.policy.activation, flush=True)
print("explorer        : init_noise_std=", rcfg.policy.init_noise_std,
      "entropy_coef=", rcfg.algorithm.entropy_coef, flush=True)
assert rcfg.policy.init_noise_std == 1.0, "explorer untouched (rsl_rl default)"
assert rcfg.algorithm.entropy_coef == 0.01, "explorer untouched (rsl_rl default)"
print("algo lr/sched/kl:", rcfg.algorithm.learning_rate, rcfg.algorithm.schedule,
      rcfg.algorithm.desired_kl, flush=True)
assert rcfg.class_name == "OnPolicyRunner", rcfg.class_name
assert rcfg.max_iterations == 15000, rcfg.max_iterations
# Simulate train.py's deprecated-cfg handling + to_dict (what OnPolicyRunner consumes).
try:
    import importlib.metadata as metadata

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    iv = metadata.version("rsl-rl-lib")
    rcfg2 = handle_deprecated_rsl_rl_cfg(rcfg, iv)
    d = rcfg2.to_dict()
    print("to_dict OK (%d keys); obs_groups=%s" % (len(d), d.get("obs_groups")), flush=True)
    _og = d.get("obs_groups") or {}
    assert _og.get("actor") == ["policy"] and _og.get("critic") == ["critic"], _og
except Exception as e:  # noqa: BLE001
    print("WARN: to_dict path raised:", repr(e), flush=True)

# ---------------- environment ----------------
env_cfg = V54SataPpoRoughEnvCfg()
env_cfg.scene.num_envs = 2
env = gym.make("Isaac-Go2-V54-SataPpoRough-v0", cfg=env_cfg)
uw = env.unwrapped
print("=== ENV CREATED ===", flush=True)
print("action_space     :", env.action_space.shape, flush=True)
print("obs_space[policy]:", env.observation_space["policy"].shape, flush=True)

# official Go2 rough terrain: 6 sub-terrains, 10x20 tiles, curriculum on
gen = env_cfg.scene.terrain.terrain_generator
subs = sorted(gen.sub_terrains.keys())
expected_subs = [
    "boxes", "hf_pyramid_slope", "hf_pyramid_slope_inv",
    "pyramid_stairs", "pyramid_stairs_inv", "random_rough",
]
print("terrain sub_terrains:", subs, flush=True)
print("terrain grid        :", gen.num_rows, "x", gen.num_cols, "| curriculum:", gen.curriculum, flush=True)
print("max_init_level      :", env_cfg.scene.terrain.max_init_terrain_level, flush=True)
print("terrain curriculum  :", env_cfg.curriculum.terrain_levels, flush=True)
assert subs == expected_subs, subs
assert gen.num_rows == 10 and gen.num_cols == 20, (gen.num_rows, gen.num_cols)
assert gen.curriculum is True, "official curriculum: difficulty must follow the row index"
assert env_cfg.scene.terrain.max_init_terrain_level == 5
assert env_cfg.curriculum.terrain_levels is not None

# Difficulty ordering: ``difficulty = lower + (upper - lower) * row / num_rows``, so the
# stairs must grow monotonically and row 0 must still be climbable.
_stairs = gen.sub_terrains["pyramid_stairs"]
_step_lo, _step_hi = _stairs.step_height_range
_d_lo, _d_hi = gen.difficulty_range


def _step_at(row):
    return _step_lo + (_step_hi - _step_lo) * (_d_lo + (_d_hi - _d_lo) * row / gen.num_rows)


print(f"stairs row 0      : {100 * _step_at(0):.1f}-{100 * _step_at(1):.1f} cm ; "
      f"row 5: {100 * _step_at(5):.1f}-{100 * _step_at(6):.1f} cm ; "
      f"row 9: {100 * _step_at(9):.1f}-{100 * _step_at(10):.1f} cm", flush=True)
assert _step_at(0) < 0.07, "row 0 must stay under 7 cm so the curriculum can start"
assert _step_at(5) < 0.16, "max_init_terrain_level=5 must keep the first episode under 16 cm"
assert _step_at(9) >= 0.21, "top row must reach the 23 cm target"
# Go2-specific scaling must be applied
assert tuple(gen.sub_terrains["boxes"].grid_height_range) == (0.025, 0.1)
assert tuple(gen.sub_terrains["random_rough"].noise_range) == (0.01, 0.06)

# V51's relative_head_height reads scene.sensors['height_scanner']
assert env_cfg.scene.height_scanner is not None, "relative_head_height needs the height scanner"

at = uw.action_manager.get_term("joint_pos")
ct = uw.command_manager.get_term("base_velocity")
print("action term :", type(at).__name__, "| command term:", type(ct).__name__, flush=True)
# Command range stays at SATA's full sweep; only the reward curve and the clock moved.
print("cmd range x :", ct.cfg.ranges.lin_vel_x, flush=True)
assert tuple(ct.cfg.ranges.lin_vel_x) == (-0.5, 1.5), ct.cfg.ranges.lin_vel_x

# Fall detection stays V52's: ``projected_gravity_b[:, 2] > 0`` (tilt > 90 deg).
fo_cfg = uw.termination_manager.get_term_cfg("fell_over")
print(f"fell_over     : {fo_cfg.func.__name__}", flush=True)
assert fo_cfg.func.__name__ == "sata_body_flipped", fo_cfg.func

# ``base_contact`` is back to IsaacLab's stock definition (2026-10-24): illegal_contact
# on "base", 1 N, max over the sensor's 3-frame history.  V52 had cleared it.
bc_cfg = uw.termination_manager.get_term_cfg("base_contact")
print(f"base_contact  : {bc_cfg.func.__name__} thr={bc_cfg.params['threshold']} "
      f"bodies={bc_cfg.params['sensor_cfg'].body_names}", flush=True)
assert bc_cfg.func.__name__ == "illegal_contact", bc_cfg.func
assert bc_cfg.params["threshold"] == 1.0
assert bc_cfg.params["sensor_cfg"].body_names == ["base"], bc_cfg.params["sensor_cfg"]

# SATA's 9 terms (head_height/lin_vel_z swapped to V51's versions), two stock
# regularisers from the official Go2 rough lineup, and ``feet_air_time`` -- the
# 2026-10-22 table the module was reverted to on 2026-10-24 (rev C: back to 12 terms).
active = set(uw.reward_manager.active_terms)
expected = {
    "forward", "head_height", "moving_y", "moving_yaw", "motor_fatigue",
    "roll", "lin_vel_z_l2", "soft_dof_pos_limits", "dof_acc",
    "ang_vel_xy_l2", "action_rate_l2", "feet_air_time",
}
print("active reward terms:", sorted(active), flush=True)
assert active == expected, f"reward mismatch: extra={active - expected} missing={expected - active}"

from terrain_v54_sata_ppo_rough.config import (  # noqa: E402
    foot_ground_height,
    foot_terrain_relief,
    lin_vel_z_on_flat,
    on_flat_ground,
    relative_head_height,
)

hh_cfg = uw.reward_manager.get_term_cfg("head_height")
lz_cfg = uw.reward_manager.get_term_cfg("lin_vel_z_l2")
sp_cfg = uw.reward_manager.get_term_cfg("soft_dof_pos_limits")
da_cfg = uw.reward_manager.get_term_cfg("dof_acc")
print(f"head_height   : {hh_cfg.func.__name__} w={hh_cfg.weight} params={hh_cfg.params}", flush=True)
print(f"lin_vel_z_l2  : {lz_cfg.func.__name__} w={lz_cfg.weight} "
      f"radius={lz_cfg.params['radius']} max_relief={lz_cfg.params['max_relief']}", flush=True)
print(f"soft_dof_pos_limits: {sp_cfg.func.__name__} w={sp_cfg.weight}", flush=True)
print(f"dof_acc       : {da_cfg.func.__name__} w={da_cfg.weight}", flush=True)
assert hh_cfg.func is relative_head_height, hh_cfg.func
assert hh_cfg.weight == 5.0
# SATA's shaper verbatim with only its zero moved onto the ground line: nose-up is
# capped at ``0.2 * (1.5 - 2 gs)``, nose-down is uncapped.  No ``max_pitch_lag``.
assert set(hh_cfg.params) == {"feet_cfg", "target_height", "slope_tau"}, hh_cfg.params
assert hh_cfg.params["target_height"] == 0.3
# Ground heights are low-passed (tau = 0.3 s) so step-edge sampling cannot inject a
# 31 deg square wave into the reward.
assert hh_cfg.params["slope_tau"] == 0.3, hh_cfg.params
assert lz_cfg.func is lin_vel_z_on_flat, lz_cfg.func
assert lz_cfg.weight == -5.0
# These two are byte-equivalent to SATA's versions, so they run the stock IsaacLab
# implementations at SATA's weights.
assert sp_cfg.func.__name__ == "joint_pos_limits", sp_cfg.func
assert sp_cfg.weight == -5.0
assert da_cfg.func.__name__ == "joint_acc_l2", da_cfg.func
assert da_cfg.weight == -1e-6

# The two stock regularisers from the official Go2 rough lineup, plus ``feet_air_time``
# at IsaacLab's stock 0.01 / 0.5 s (rev B's 10.0 / 0.2 was rolled back) -- the term only
# pays while a velocity command is live, so the contact sensor must track air time.
ar_cfg = uw.reward_manager.get_term_cfg("action_rate_l2")
av_cfg = uw.reward_manager.get_term_cfg("ang_vel_xy_l2")
ft_cfg = uw.reward_manager.get_term_cfg("feet_air_time")
print(f"ang_vel_xy_l2 : {av_cfg.func.__name__} w={av_cfg.weight}", flush=True)
print(f"action_rate_l2: {ar_cfg.func.__name__} w={ar_cfg.weight}", flush=True)
print(f"feet_air_time : {ft_cfg.func.__name__} w={ft_cfg.weight} "
      f"threshold={ft_cfg.params['threshold']}", flush=True)
assert av_cfg.weight == -0.05
assert ar_cfg.weight == -0.01
assert ft_cfg.func.__name__ == "feet_air_time", ft_cfg.func
assert ft_cfg.weight == 0.01
assert "feet_contact" in uw.scene.sensors
assert uw.scene.sensors["feet_contact"].cfg.track_air_time is True, "feet_air_time needs track_air_time"
assert ft_cfg.params["command_name"] == "base_velocity"
assert ft_cfg.params["threshold"] == 0.5
assert ft_cfg.params["sensor_cfg"].body_names == ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
print(f"feet sensor bodies: {uw.scene.sensors['feet_contact'].body_names}", flush=True)

# The five 2026-10-23 terms stay off.  ``undesired_contacts`` was back in rev B and is
# off again in rev C, so the table is the 2026-10-22 one term for term.  The functions and
# the stance gate stay defined, dormant.
for _n in (
    "feet_slip",
    "gait_contact",
    "swing_clearance",
    "excessive_foot_height",
    "undesired_contacts",
):
    assert getattr(env_cfg.rewards, _n) is None, f"{_n} must be off in the restored table"
assert "undesired_contacts" not in uw.reward_manager.active_terms

# ``forward`` back at +10; the sigma 0.25 curve and SATA's lo/hi are unchanged, so the
# growth-shrunk target keeps pointing at the sampled command's centre.
fw_cfg = uw.reward_manager.get_term_cfg("forward")
print(f"forward       : w={fw_cfg.weight} sigma={fw_cfg.params['sigma']} "
      f"x=[{fw_cfg.params['lin_vel_x_lo']}, {fw_cfg.params['lin_vel_x_hi']}]", flush=True)
assert fw_cfg.weight == 10.0 and fw_cfg.params["sigma"] == 0.25
assert fw_cfg.params["lin_vel_x_lo"] == -0.5 and fw_cfg.params["lin_vel_x_hi"] == 1.5

# The stance gate, the speed-locked clock and the forward fan are dormant in this table:
# their only consumers (``gait_contact``, ``swing_clearance``, ``excessive_foot_height``)
# are switched off, so the unit probes that used to live here would only be reading the
# module-level helpers, not the training config.

# Joint limits must be the stock USD range: SATA's startup clamp (thigh front [0, 1.5],
# rear [0, 2.0]) is disabled for stairs, otherwise the default stance sits outside the
# 90 % soft bound and the calf has no travel left for a 23 cm riser.
lim = uw.scene["robot"].data.joint_pos_limits[0]
names = uw.scene["robot"].joint_names
front_thigh = [i for i, n in enumerate(names) if "thigh" in n and n.startswith(("FL", "FR"))]
rear_thigh = [i for i, n in enumerate(names) if "thigh" in n and n.startswith(("RL", "RR"))]
calf_ids = [i for i, n in enumerate(names) if "calf" in n]
print("front thigh:", lim[front_thigh].tolist(), flush=True)
print("rear  thigh:", lim[rear_thigh].tolist(), flush=True)
print("calf  limits:", lim[calf_ids].tolist(), flush=True)
# Stock Unitree Go2 USD: front [-1.5708, 3.4907], rear [-0.5236, 4.5379], calf both.
assert (lim[front_thigh, 0] < -1.5).all(), f"front thigh clamped: {lim[front_thigh]}"
assert (lim[front_thigh, 1] > 3.4).all(), f"front thigh clamped: {lim[front_thigh]}"
assert (lim[rear_thigh, 0] < -0.5).all(), f"rear thigh clamped: {lim[rear_thigh]}"
assert (lim[rear_thigh, 1] > 4.5).all(), f"rear thigh clamped: {lim[rear_thigh]}"
assert (lim[calf_ids, 0] < -2.7).all(), f"calf lower bound still clamped: {lim[calf_ids]}"

obs, _ = env.reset()
dim = obs["policy"].shape[-1]
print("reset obs dim:", dim, flush=True)
assert dim == 60, f"Expected 60-D SATA obs, got {dim}"

# Privileged critic group (rev H): clean 60-D state + 187-D height map.
print("obs groups     :", dict(uw.observation_manager.group_obs_dim), flush=True)
cdim = obs["critic"].shape[-1]
print("reset critic dim:", cdim, flush=True)
assert cdim == 60 + 187, f"Expected 247-D critic obs (60 state + 187 height), got {cdim}"
assert torch.isfinite(obs["critic"]).all(), "non-finite critic obs"
_hs = obs["critic"][:, 60:]
print(
    f"height scan    : min={_hs.min().item():+.3f} max={_hs.max().item():+.3f}"
    f" mean={_hs.mean().item():+.3f}",
    flush=True,
)

# head_height reads the local ground line: per-foot terrain heights and its slope.
_z = foot_ground_height(uw, hh_cfg.params["feet_cfg"])
_slope = torch.atan((_z[:, :2].mean(dim=1) - _z[:, 2:].mean(dim=1)) / 0.3868)
print(f"foot ground z  : {[[round(v, 3) for v in row] for row in _z.tolist()]}", flush=True)
print(f"terrain slope  : {[round(float(v) * 57.2958, 1) for v in _slope.tolist()]} deg", flush=True)

# Flat-ground gate: lin_vel_z only (head_height is terrain-relative now).
_relief = foot_terrain_relief(uw, lz_cfg.params["feet_cfg"], lz_cfg.params["radius"])
_flat = on_flat_ground(uw, lz_cfg.params["feet_cfg"], lz_cfg.params["radius"])
print(f"per-foot relief: {[[round(v, 4) for v in row] for row in _relief.tolist()]}", flush=True)
print(f"on_flat gate   : {_flat.tolist()}", flush=True)

# growth-coupled / V51 terms recomputed by hand for inspection
from sata_rewards import sata_forward, sata_motor_fatigue  # noqa: E402

for i in range(10):
    action = torch.randn(2, 12, device=uw.device) * 0.5
    obs, rew, term, trunc, info = env.step(action)
    fwd = sata_forward(uw).mean().item()
    hh = hh_cfg.func(uw, **hh_cfg.params).mean().item()
    mf = sata_motor_fatigue(uw).mean().item()
    cmd = uw.command_manager.get_command("base_velocity")
    _lpf = uw.__dict__["_reward_lowpass"]["head_height_ground_z"]
    # SATA's cap on the nose-up bonus: gs = 0 -> 0.3 (17.5 deg), gs = 0.75 -> 0.
    _cap = max(0.0, 0.2 * (1.5 - 2.0 * at.general_scale))
    print(
        f"step {i}: obs_dim={obs['policy'].shape[-1]} rew={rew.mean().item():+.3f} "
        f"|tau|max={at.torques.abs().max().item():.3f} fat={at.motor_fatigue.max().item():.4f} "
        f"gs={at.general_scale:.4f} fwd={fwd:+.3f} head={hh:+.3f} cap={_cap:.3f} mfat={mf:+.4f} "
        f"zlpf={[[round(v, 3) for v in row] for row in _lpf.tolist()]} "
        f"cmd_x=[{cmd[:, 0].min().item():.2f},{cmd[:, 0].max().item():.2f}]",
        flush=True,
    )
    assert obs["policy"].shape[-1] == 60
    assert torch.isfinite(rew).all(), "non-finite total reward"
    assert torch.isfinite(at.torques).all(), "non-finite torque"
    assert at.torques.abs().max().item() <= 23.5 + 1e-3, "torque exceeds limit"

# --- Reward audit: per-term contribution and pairwise redundancy --------------------
# Every term function is a pure read, so calling it directly reproduces exactly what
# ``RewardManager.compute`` sees (``value = raw * weight * dt``, dt = step_dt).
# The random-action probe is not a gait, so absolute magnitudes are meaningless; what
# carries over is the *share* of each term and how the terms co-vary.
_audit_names = sorted(active)
_audit_raw = {n: [] for n in _audit_names}
for _ in range(20):
    env.step(torch.randn(2, 12, device=uw.device) * 0.5)
    for _n in _audit_names:
        _c = uw.reward_manager.get_term_cfg(_n)
        # A term may return an integer count (``sum`` of a bool mask); the reward manager
        # promotes it when it multiplies by ``weight * dt``, but ``mean()`` on the raw
        # tensor refuses to guess a dtype.
        _raw = torch.as_tensor(_c.func(uw, **_c.params)).float()
        _audit_raw[_n].append(float(_raw.mean()))

_rows = []
for _n in _audit_names:
    _c = uw.reward_manager.get_term_cfg(_n)
    _mean = sum(_audit_raw[_n]) / len(_audit_raw[_n])
    _rows.append((_mean * _c.weight * uw.step_dt, _mean, _n))
_total = sum(abs(r[0]) for r in _rows) or 1e-12
print("term                 raw_mean    reward/step   share", flush=True)
for _w, _m, _n in sorted(_rows, key=lambda r: -abs(r[0])):
    print(f"{_n:<18}{_m:+11.4f}{_w:+15.6f}  {100 * abs(_w) / _total:5.1f}%", flush=True)

_mat = torch.tensor([_audit_raw[_n] for _n in _audit_names])
_mat = _mat - _mat.mean(dim=1, keepdim=True)
_unit = _mat / _mat.norm(dim=1, keepdim=True).clamp(min=1e-9)
_corr = _unit @ _unit.T
print("correlated pairs (|r| > 0.8):", flush=True)
for _i in range(len(_audit_names)):
    for _j in range(_i + 1, len(_audit_names)):
        if abs(float(_corr[_i, _j])) > 0.8:
            print(f"  {_audit_names[_i]:<18}{_audit_names[_j]:<18}r={float(_corr[_i, _j]):+.2f}", flush=True)

print("=== SMOKE TEST PASSED ===", flush=True)
# simulation_app.close() hangs forever on this Vulkan-less headless box (render cleanup
# spins at 99% CPU and never releases the GPU). Everything is verified by now, so
# hard-exit immediately instead of tearing down the simulator.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
