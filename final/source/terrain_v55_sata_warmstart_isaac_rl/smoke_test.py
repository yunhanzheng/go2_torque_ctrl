"""Smoke test for V55 (SATA warm start + IsaacLab's stock Go2 reward table).

Verifies, in this order:
- PPO runner cfg: OnPolicyRunner, 15000 iters, stock IsaacLab numbers (lr 1e-3 adaptive, KL 0.01,
  entropy 0.01, init_noise_std 1.0, [512, 256, 128] elu)
- env: 60-D SATA observation, 12 torque actions, V54's terrain + curriculum untouched
- reward table is *exactly* IsaacLab Go2 rough's eleven terms (and nothing else), with
  ``undesired_contacts`` off as decided
- terminations are still V54's four (time_out via SATA's 10 s substep clock / base_contact /
  dof_pos_limit / fell_over)
- the growth clock: gs starts at 0.7891 (torque scale 0.852 -> 20.0 Nm, 179 Hz), rises
  linearly with policy steps, reaches 1.0 at ``gs_ramp_steps`` and honours ``_v55_freeze_gs``
- the SATA actor checkpoint loads into the rsl_rl actor (shapes; ``std`` only with
  ``--sata_init_std``, otherwise it must stay at init_noise_std = 1.0)
- all rewards compute finite values and TG torques stay bounded

Run:  python smoke_test.py [--sata_init /path/model_3000.pt]
Exits with ``=== SMOKE TEST PASSED ===`` and status 0 (os._exit, render teardown hangs here).
"""
import argparse
import math
import os
import sys

# --- Path bootstrap (mirrors train.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (
    _SOURCE_ROOT,
    os.path.join(_SOURCE_ROOT, "dependency_chain"),
    os.path.join(_SOURCE_ROOT, "base_adapter"),
    os.path.join(_SOURCE_ROOT, "base_adapter", "initial_stair_source"),
    os.path.join(_SOURCE_ROOT, "terrain_v51_native_terrain"),
    os.path.join(_SOURCE_ROOT, "terrain_v52_sata_ppo_flat"),
    os.path.join(_SOURCE_ROOT, "terrain_v54_sata_ppo_rough"),
    _HERE,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

parser = argparse.ArgumentParser()
parser.add_argument("--sata_init", type=str, default=None, help="Optional SATA actor checkpoint to test-load.")
parser.add_argument(
    "--sata_init_std",
    action="store_true",
    default=False,
    help="Expect SATA's std to be transferred as well (default: std stays at 1.0).",
)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

# --- sa_16dof compat shim (full rationale in base_adapter/cli_args.py): the server's shared
# isaaclab_tasks auto-imports a sibling sa_16dof task that needs rsl_rl.modules.distribution
# (absent from this env's rsl_rl SAC fork). Pre-register it as an empty-path stub.
import types as _types  # noqa: E402

_STUB = "isaaclab_tasks.manager_based.locomotion.velocity.config.sa_16dof"
if _STUB not in sys.modules:
    _m = _types.ModuleType(_STUB)
    _m.__path__ = []
    sys.modules[_STUB] = _m

import terrain_v55_sata_warmstart_isaac_rl.config  # noqa: F401,E402
from terrain_v55_sata_warmstart_isaac_rl.config import (  # noqa: E402
    V55SataWarmstartIsaacRlEnvCfg,
    V55SataWarmstartIsaacRlRunnerCfg,
)
from terrain_v55_sata_warmstart_isaac_rl.v55_growth import (  # noqa: E402
    SATA_MODEL_3000_GS,
    v55_growth_scale,
)

TASK = "Isaac-Go2-V55-SataWarmstartIsaacRl-v0"

# ---------------------------------------------------------------------------------------
# 1. Runner cfg
# ---------------------------------------------------------------------------------------
print("=== PPO RUNNER CFG ===", flush=True)
rcfg = V55SataWarmstartIsaacRlRunnerCfg()
print("class_name       :", rcfg.class_name, flush=True)
print("max_iterations   :", rcfg.max_iterations, "| save_interval:", rcfg.save_interval, flush=True)
print("num_steps_per_env:", rcfg.num_steps_per_env, flush=True)
print("policy hidden    :", rcfg.policy.actor_hidden_dims, rcfg.policy.activation, flush=True)
print(
    "algo lr/sched/kl :",
    rcfg.algorithm.learning_rate,
    rcfg.algorithm.schedule,
    rcfg.algorithm.desired_kl,
    "| entropy",
    rcfg.algorithm.entropy_coef,
    flush=True,
)
assert rcfg.class_name == "OnPolicyRunner", rcfg.class_name
assert rcfg.max_iterations == 15000, rcfg.max_iterations
# stock IsaacLab Go2 rough numbers, deliberately not retuned
assert rcfg.algorithm.learning_rate == 1e-3, rcfg.algorithm.learning_rate
assert rcfg.algorithm.schedule == "adaptive", rcfg.algorithm.schedule
assert rcfg.algorithm.desired_kl == 0.01, rcfg.algorithm.desired_kl
assert rcfg.algorithm.entropy_coef == 0.01, rcfg.algorithm.entropy_coef
assert rcfg.policy.init_noise_std == 1.0, rcfg.policy.init_noise_std
assert rcfg.num_steps_per_env == 24, rcfg.num_steps_per_env
assert list(rcfg.policy.actor_hidden_dims) == [512, 256, 128], rcfg.policy.actor_hidden_dims

# ---------------------------------------------------------------------------------------
# 2. Environment
# ---------------------------------------------------------------------------------------
env_cfg = V55SataWarmstartIsaacRlEnvCfg()
env_cfg.scene.num_envs = 2
env = gym.make(TASK, cfg=env_cfg)
uw = env.unwrapped
print("=== ENV CREATED ===", flush=True)
print("action_space     :", env.action_space.shape, flush=True)
print("obs_space[policy]:", env.observation_space["policy"].shape, flush=True)
assert env.action_space.shape[-1] == 12, env.action_space.shape
assert env.observation_space["policy"].shape[-1] == 60, env.observation_space["policy"].shape

gen = env_cfg.scene.terrain.terrain_generator
print("terrain sub-terrains:", sorted(gen.sub_terrains.keys()), flush=True)
print("terrain grid      :", gen.size, "| curriculum:", gen.curriculum, flush=True)
assert gen.curriculum is True, "official terrain_levels_vel needs generator.curriculum=True"
curr = uw.curriculum_manager.active_terms
print("curriculum terms  :", curr, flush=True)
assert "terrain_levels" in curr, curr

# ---------------------------------------------------------------------------------------
# 3. Reward table == IsaacLab Go2 rough, exactly
# ---------------------------------------------------------------------------------------
EXPECTED_REWARDS = {
    # name: weight
    "track_lin_vel_xy_exp": 1.5,
    "track_ang_vel_z_exp": 0.75,
    "lin_vel_z_l2": -2.0,
    "ang_vel_xy_l2": -0.05,
    "dof_torques_l2": -2e-4,
    "dof_acc_l2": -2.5e-7,
    "action_rate_l2": -0.01,
    "feet_air_time": 0.01,
    "flat_orientation_l2": 0.0,
    "dof_pos_limits": 0.0,
}
active = list(uw.reward_manager.active_terms)
weights = {name: cfg.weight for name, cfg in zip(uw.reward_manager.active_terms, uw.reward_manager._term_cfgs)}
print("=== REWARD TABLE ===", flush=True)
for name in active:
    print(f"  {name:24s} {weights[name]:+.3e}", flush=True)

assert set(active) == set(EXPECTED_REWARDS), (
    f"reward table mismatch\n  extra: {sorted(set(active) - set(EXPECTED_REWARDS))}"
    f"\n  missing: {sorted(set(EXPECTED_REWARDS) - set(active))}"
)
for name, want in EXPECTED_REWARDS.items():
    got = weights[name]
    assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12), (name, got, want)
assert "undesired_contacts" not in active, "undesired_contacts must stay off in V55"
# no SATA-reward leftovers
for banned in ("forward", "head_height", "moving_y", "moving_yaw", "motor_fatigue", "roll", "sata_lin_vel_z"):
    assert banned not in active, f"leftover SATA term: {banned}"

# ---------------------------------------------------------------------------------------
# 4. Terminations == V54's four
# ---------------------------------------------------------------------------------------
terms = list(uw.termination_manager.active_terms)
print("=== TERMINATIONS ===", terms, flush=True)
assert set(terms) == {"time_out", "base_contact", "dof_pos_limit", "fell_over"}, terms

# ---------------------------------------------------------------------------------------
# 5. Growth clock
# ---------------------------------------------------------------------------------------
term = uw.action_manager.get_term("joint_pos")
print("=== GROWTH CLOCK ===", flush=True)
print("gs_start / ramp_steps:", term.cfg.gs_start, term.cfg.gs_ramp_steps, flush=True)
assert math.isclose(term.cfg.gs_start, SATA_MODEL_3000_GS, rel_tol=1e-9), term.cfg.gs_start
assert term.cfg.gs_ramp_steps == 12000, term.cfg.gs_ramp_steps

env.reset()
actions = torch.zeros((uw.num_envs, 12), device=uw.device)
env.step(actions)
gs_after_1 = term.general_scale
print(
    f"after 1 step: policy_steps={uw._v55_policy_steps} gs={gs_after_1:.4f} "
    f"torque_scale={term.current_torque_limit_scale:.4f} "
    f"({term.current_torque_limit_scale * 23.5:.2f} Nm, {100 + 100 * gs_after_1:.1f} Hz)",
    flush=True,
)
assert abs(gs_after_1 - SATA_MODEL_3000_GS) < 1e-3, gs_after_1
assert abs(term.current_torque_limit_scale * 23.5 - 20.03) < 0.05, term.current_torque_limit_scale

# ramp: jump to the middle of the schedule and to the end
uw._v55_policy_steps = 6000
env.step(actions)
gs_mid = term.general_scale
uw._v55_policy_steps = 12000
env.step(actions)
gs_end = term.general_scale
print(f"ramp: gs(6000)={gs_mid:.4f} (expect 0.8946)  gs(12000)={gs_end:.4f} (expect 1.0)", flush=True)
assert abs(gs_mid - 0.89455) < 1e-3, gs_mid
assert abs(gs_end - 1.0) < 1e-9, gs_end

# freeze hook used by evaluate/play
uw._v55_freeze_gs = 0.95
uw._v55_policy_steps = 0
env.step(actions)
print(f"freeze: gs={term.general_scale:.4f} (expect 0.95)", flush=True)
assert abs(term.general_scale - 0.95) < 1e-9, term.general_scale

# --- the other two consumers (commands + control frequency) must read the same clock ---
import terrain_v51_native_terrain.sata_alignment as _sa  # noqa: E402

assert _sa.sata_growth_scale is v55_growth_scale, "module-level clock was not replaced"

def _decimation_sequence(freeze: float, n: int = 24):
    # Long enough to see the carry pattern: at 179 Hz one control period is 5.59 ms, so the
    # loop takes 2 substeps once and 1 substep for the next ~7 steps (mean 200/179 = 1.117).
    uw._v55_freeze_gs = freeze
    seq = []
    for _ in range(n):
        env.step(actions)
        seq.append(int(uw.cfg.decimation))
    return seq

seq_full = _decimation_sequence(1.0)
seq_pre = _decimation_sequence(SATA_MODEL_3000_GS)
mean_full = sum(seq_full) / len(seq_full)
mean_pre = sum(seq_pre) / len(seq_pre)
print(f"decimation @ gs=1.0   : mean {mean_full:.3f}, set={sorted(set(seq_full))} (expect 1.000 = 200 Hz)", flush=True)
print(f"decimation @ gs=0.789 : mean {mean_pre:.3f}, set={sorted(set(seq_pre))} "
      f"(expect ~1.117 = 179 Hz, so a 2 must appear)", flush=True)
assert all(d == 1 for d in seq_full), seq_full
assert 1.0 < mean_pre < 1.3, seq_pre
assert 2 in seq_pre, seq_pre
uw._v55_freeze_gs = -1.0

# ---------------------------------------------------------------------------------------
# 6. Optional: SATA actor transfer
# ---------------------------------------------------------------------------------------
if args.sata_init:
    from terrain_v55_sata_warmstart_isaac_rl.load_sata_actor import load_sata_actor

    class _Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.std = torch.nn.Parameter(torch.ones(12) * 1.0)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(60, 512), torch.nn.ELU(),
                torch.nn.Linear(512, 256), torch.nn.ELU(),
                torch.nn.Linear(256, 128), torch.nn.ELU(),
                torch.nn.Linear(128, 12),
            )

    dummy = _Dummy()
    before = dummy.mlp[0].weight.detach().clone()
    report = load_sata_actor(dummy, args.sata_init, load_std=args.sata_init_std)
    n_expected = 9 if args.sata_init_std else 8
    assert len(report["copied"]) == n_expected, report["copied"]
    assert not torch.allclose(before, dummy.mlp[0].weight.detach()), "actor weights unchanged"
    _ck_std = torch.load(args.sata_init, map_location="cpu", weights_only=False)["model_state_dict"]["std"]
    if args.sata_init_std:
        # opt-in: std must come from the checkpoint, elementwise
        assert torch.equal(dummy.std.detach().cpu(), _ck_std), "std not transferred elementwise"
        assert not torch.allclose(dummy.std.detach().cpu(), torch.ones(12)), "std left at 1.0"
        print("=== SATA ACTOR TRANSFER OK (8 weights + std, critic untouched) ===", flush=True)
    else:
        # default: weights only, exploration stays at the IsaacLab default
        assert not report["std"], report
        assert report["skipped"] == ["std (std disabled)"], report["skipped"]
        assert torch.equal(dummy.std.detach().cpu(), torch.ones(12)), "std should stay 1.0"
        print("=== SATA ACTOR TRANSFER OK (8 weights only, std=1.0, critic untouched) ===", flush=True)
        print(f"[sata_init] SATA's std (NOT loaded) would be: "
              f"min={_ck_std.min():.3f} mean={_ck_std.mean():.3f} max={_ck_std.max():.3f}", flush=True)

# ---------------------------------------------------------------------------------------
# 7. Reward audit + torque bound on a few random-action steps
# ---------------------------------------------------------------------------------------
print("=== REWARD AUDIT (20 steps, random actions) ===", flush=True)
env.reset()
torch.manual_seed(0)
totals = {name: 0.0 for name in active}
max_tau = 0.0
steps = 20
prev = None
for _ in range(steps):
    a = 0.1 * torch.randn((uw.num_envs, 12), device=uw.device)
    env.step(a)
    max_tau = max(max_tau, float(term.torques.abs().max()))
    # ``_episode_sums`` is a running episode total: sum the increment, not the total.
    cur = {name: uw.reward_manager._episode_sums[name].clone() for name in active}
    reset_mask = uw.episode_length_buf == 0
    if prev is not None:
        for name in active:
            totals[name] += float(torch.where(reset_mask, cur[name], cur[name] - prev[name]).mean())
    prev = cur

ep_len = float(uw.episode_length_buf.float().mean())
magnitude = sum(abs(v) for v in totals.values()) / max(steps, 1)
for name in active:
    per_step = totals[name] / steps
    never = "never" if per_step == 0.0 else ""
    share = per_step / magnitude * 100 if magnitude else 0.0
    print(f"  {name:24s} {per_step:+.6f}/step {per_step / uw.step_dt:+9.4f}/s {share:+6.1f}% {never}", flush=True)
    assert math.isfinite(per_step), name
print(f"mean episode length after {steps} steps: {ep_len:.1f}", flush=True)
print(f"max |torque| over the run: {max_tau:.2f} Nm (ceiling 23.5 x 1.5 hill factor)", flush=True)
assert max_tau <= 23.5 * 1.5 + 1e-3, max_tau
assert all(math.isfinite(v) for v in weights.values())

print("=== SMOKE TEST PASSED ===", flush=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
