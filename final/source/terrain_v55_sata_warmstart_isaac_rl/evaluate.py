"""Evaluate a V55 checkpoint (gs pinned, N steps, per-term reward + termination report).

The growth clock starts at model_3000's operating point on a fresh process, so evaluation
must pin it: without ``--freeze_gs`` a 500-step episode would still be climbing the ramp and
the policy would be judged at ~0.79 instead of the 1.0 it trains into.

Examples
--------
    python evaluate.py --checkpoint logs/rsl_rl/terrain_v55_.../model_500.pt
    python evaluate.py --checkpoint .../model_500.pt --num_envs 128 --steps 1000 \
        --max_init_terrain_level 9          # 23 cm rows only
"""
import argparse
import os
import sys
from pathlib import Path

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
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--freeze_gs", type=float, default=1.0, help="gs used for the whole rollout (-1 = let it ramp).")
parser.add_argument("--max_init_terrain_level", type=int, default=None, help="Pin the spawn row (9 = 23 cm risers).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", action="store_true", default=False, help="Record an mp4 of the rollout."
                    " Needs a render surface; see run_video.sh for the Vulkan loader/ICD env vars.")
parser.add_argument("--video_seconds", type=float, default=10.0, help="Clip length in simulated"
                    " seconds. Replaces --steps and derives the playback fps from step_dt, so the"
                    " mp4 plays in real time.")
parser.add_argument("--video_fps", type=int, default=None, help="Playback fps (default: round(1/step_dt)).")
parser.add_argument("--video_folder", type=str, default=None, help="Where to write the mp4 (default:"
                    " <checkpoint dir>/<model>_seed<k>_<seconds>s).")
parser.add_argument("--sky_plain", action="store_true", default=False, help="Drop the DomeLight"
                    " HDR (it 404s on the cluster, so the terrain renders black) for a flat colour.")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
import importlib.metadata as metadata  # noqa: E402

TASK = "Isaac-Go2-V55-SataWarmstartIsaacRl-v0"
device = args.device or "cuda:0"

env_cfg = V55SataWarmstartIsaacRlEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
if args.max_init_terrain_level is not None:
    env_cfg.scene.terrain.max_init_terrain_level = args.max_init_terrain_level
if args.sky_plain:
    # The kloofendal HDR is a 404 on the cluster, so the DomeLight adds no light and the
    # terrain renders black; a flat colour keeps the clip readable.
    _sky = env_cfg.scene.sky_light.spawn
    _sky.texture_file = None
    _sky.color = (0.75, 0.75, 0.75)
    _sky.intensity = 1500.0
if args.video:
    # Third-person follow camera behind and above the robot (only env 0 is on film).
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.0, -2.5, 1.5)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
uw = env.unwrapped
uw._v55_freeze_gs = float(args.freeze_gs)

# ``step_dt`` is the growth-scaled control period (it is not 0.02 s), so a wall-clock clip
# length is only known once the env exists: 10 s = round(10 / step_dt) policy steps, played
# back at 1 / step_dt fps so the mp4 runs in real time.
steps = args.steps
video_dir = None
if args.video:
    steps = max(1, int(round(args.video_seconds / uw.step_dt)))
    fps = args.video_fps or max(1, int(round(1.0 / uw.step_dt)))
    uw.metadata["render_fps"] = fps
    video_dir = (
        Path(args.video_folder)
        if args.video_folder
        else Path(args.checkpoint).parent / f"{Path(args.checkpoint).stem}_seed{args.seed}_{args.video_seconds:g}s"
    )
    video_dir.mkdir(parents=True, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        step_trigger=lambda step: step == 0,
        video_length=steps,
        name_prefix=f"V55_{Path(args.checkpoint).stem}_gs{args.freeze_gs:g}",
        disable_logger=True,
    )
    print(f"[INFO] video: {steps} steps x {uw.step_dt:.4f} s = {steps * uw.step_dt:.2f} s at {fps} fps", flush=True)
    print(f"[INFO] video dir: {video_dir}", flush=True)

env.reset()

vec_env = RslRlVecEnvWrapper(env)
# Same conversion train.py performs: rsl_rl 4.x wants explicit ``actor``/``critic`` configs
# (with ``class_name``), while the runner cfg still carries the deprecated ``policy`` block.
rcfg = handle_deprecated_rsl_rl_cfg(V55SataWarmstartIsaacRlRunnerCfg(), metadata.version("rsl-rl-lib"))
runner = OnPolicyRunner(vec_env, rcfg.to_dict(), log_dir=None, device=device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=device)

term = uw.action_manager.get_term("joint_pos")
active = list(uw.reward_manager.active_terms)
acc = {name: 0.0 for name in active}
logs: dict[str, float] = {}
n_done = 0

obs = vec_env.get_observations()
prev = None
with torch.inference_mode():
    for _ in range(steps):
        actions = policy(obs)
        obs, _rew, dones, extras = vec_env.step(actions)
        n_done += int(dones.sum())
        # ``_episode_sums`` is a *running* episode total, so summing it every step would
        # over-count by ~steps/2.  Take the increment instead; after a reset the buffer is
        # zeroed first, so the current value already is the increment.
        cur = {name: uw.reward_manager._episode_sums[name].clone() for name in active}
        reset_mask = uw.episode_length_buf == 0
        if prev is not None:
            for name in active:
                delta = torch.where(reset_mask, cur[name], cur[name] - prev[name])
                acc[name] += float(delta.mean())
        prev = cur
        if isinstance(extras, dict):
            for key, value in (extras.get("log") or {}).items():
                logs[key] = float(value)

print("=" * 78, flush=True)
print(f"checkpoint   : {args.checkpoint}", flush=True)
print(f"envs / steps : {uw.num_envs} / {steps}", flush=True)
print(f"gs (frozen)  : {term.general_scale:.4f}  torque {term.current_torque_limit_scale * 23.5:.2f} Nm", flush=True)
print(f"episodes done: {n_done} ({n_done / (uw.num_envs * steps) * 100:.1f} % of env-steps)", flush=True)
print(f"mean ep len  : {float(uw.episode_length_buf.float().mean()):.1f} steps "
      f"({float(uw.episode_length_buf.float().mean()) * uw.step_dt:.2f} s)", flush=True)
print(f"step_dt      : {uw.step_dt:.4f} s", flush=True)
print(f"terrain level: {float(uw.scene.terrain.terrain_levels.float().mean()):.2f} "
      f"(max {int(uw.scene.terrain.terrain_levels.max())})", flush=True)
print("-" * 78, flush=True)
print(f"{'reward term':26s} {'per step':>12s}   {'per s':>10s}", flush=True)
total = 0.0
for name in active:
    per_step = acc[name] / steps
    total += per_step
    print(f"{name:26s} {per_step:+.6f}   {per_step / uw.step_dt:+10.4f}", flush=True)
print(f"{'SUM':26s} {total:+.6f}   {total / uw.step_dt:+10.4f}", flush=True)
print("-" * 78, flush=True)
for key in sorted(logs):
    if key.startswith(("Episode_Termination", "Metrics", "Curriculum", "Episode_Reward")):
        print(f"{key:52s} {logs[key]:+.4f}", flush=True)

if args.video:
    # RecordVideo only auto-closes once it has written more than ``video_length`` frames, so
    # an exact-length loop relies on close() to flush the mp4.
    env.close()
    print(f"[INFO] video saved under: {video_dir}", flush=True)

sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
