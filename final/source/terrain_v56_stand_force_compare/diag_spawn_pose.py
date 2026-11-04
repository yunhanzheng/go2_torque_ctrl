"""Is the V56 spawn pose a level stance?  Measure the four feet, do not assume it.

The whole force-comparison argument rests on one number: the height a robot would have if its
nominal joint angles were lowered until the feet touch.  ``UNITREE_GO2_CFG`` spawns the base at
z = 0.40 m with ``thigh`` 0.8 rad in front and 1.0 rad in the rear while every ``calf`` is at
-1.5 rad, so the front and rear legs are *not* the same length apart from the ground.  If the
four foot spheres are not coplanar, no single vertical translation puts them all down at once,
and a reference height derived that way would be wrong.

This script therefore measures instead of computing:

1. right after ``reset()`` (before any action is applied) dump root pose, joint positions and
   the world position of every ankle/foot body, plus the foot collision radii read from the USD
   stage, so "sole height" = foot origin z - radius can be formed per leg;
2. report the spread of the four sole heights;
3. then let the position arm (Kp 25 / Kd 0.5, zero action -> target = default angles) settle for
   a second and report the same quantities next to the measured ground reaction forces, which
   shows whether all four feet actually carry load in the pose the PD converges to.

    python diag_spawn_pose.py --task Isaac-Go2-V56-StandPos-v0 --steps 60
"""
from __future__ import annotations

import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (_SOURCE_ROOT, os.path.join(_SOURCE_ROOT, "base_adapter"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

import cli_args  # noqa: F401, isort: skip

parser = argparse.ArgumentParser(description="Measure the spawn-pose foot heights of the V56 arms.")
parser.add_argument("--task", type=str, default="Isaac-Go2-V56-StandPos-v0")
parser.add_argument("--steps", type=int, default=60, help="Policy steps to run after the dump.")
parser.add_argument("--print_every", type=int, default=20)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import terrain_v56_stand_force_compare.config  # noqa: F401  (registers the tasks)
from terrain_v56_stand_force_compare.eval_common import play_overrides
from terrain_v56_stand_force_compare.metrics import FOOT_NAMES, foot_ids

from isaaclab_tasks.utils.hydra import hydra_task_config


def _euler(quat_w: torch.Tensor) -> tuple[float, float, float]:
    """Roll, pitch, yaw from a ``wxyz`` quaternion, matching what the observations expose."""
    w, x, y, z = (float(v) for v in quat_w)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _foot_radii() -> dict[str, float]:
    """Read the sphere radius of every foot collision shape from the stage, best effort.

    This is a convenience only: the *difference* between the reset and the settled foot origin z
    already gives the gap to contact with the radius cancelling out, so an empty result here does
    not invalidate the measurement.  Every prim whose path mentions a foot is printed so the
    approximation actually shipped in ``go2.usd`` can be seen.
    """
    radii: dict[str, float] = {}
    try:
        import omni.usd
    except ImportError:
        print("[WARN] pxr/omni.usd unavailable, foot radii unknown", flush=True)
        return radii
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return radii
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        hit = next((name for name in FOOT_NAMES if name in path), None)
        if hit is None:
            continue
        radius = prim.GetAttribute("radius").Get()
        print(f"[COLLIDER] {path}  type={prim.GetTypeName()}  radius={radius}", flush=True)
        if radius is not None and hit not in radii:
            radii[hit] = float(radius)
    return radii


def _ref_line(tag: str, base_env) -> None:
    """The base-relative reference and its current error - the quantities V56 now judges on.

    Cross-checks ``reference.py`` against the world-frame numbers printed by :func:`_report`:
    ``ell_0`` must equal ``root z - mean(foot origin z)`` of the reset frame, because the spawn
    quaternion is the identity there.
    """
    ref = base_env.reference
    err = ref.tracking_error()
    print(
        f"[{tag}] REF ell_0={float(ref.ell_0[0]):.4f} m  foot_ell_0="
        + ", ".join(f"{v:.4f}" for v in ref.foot_ell_0[0].tolist())
        + f"  | dq_rms={float(err['dq_rms'][0]):.4f} dq_max={float(err['dq_max'][0]):.4f} rad"
        f"  tilt={float(err['tilt'][0]):.4f} rad"
        f"  d_ell={1e3 * float(err['d_ell'][0]):+7.2f} mm"
        f"  d_foot_ell_max={1e3 * float(err['d_foot_ell_max'][0]):6.2f} mm",
        flush=True,
    )


def _report(tag: str, robot, foot_body_ids: list[int], radii: dict[str, float], env) -> None:
    """Print base pose, foot origins, inferred soles and the sole spread for one snapshot."""
    # Without a radius the sole column collapses onto the body origin.  All four feet use the same
    # sphere in the urdf, so the *spread* between feet stays valid either way.
    fallback = 0.0 if not radii else None
    pos_w = robot.data.body_pos_w[0]
    root_pos = robot.data.root_pos_w[0]
    roll, pitch, yaw = _euler(robot.data.root_state_w[0, 3:7])
    print(f"\n[{tag}] root z={float(root_pos[2]):.4f}  roll={roll:+.4f} pitch={pitch:+.4f} yaw={yaw:+.4f}", flush=True)
    print(f"[{tag}] joint_pos={', '.join(f'{v:+.3f}' for v in robot.data.joint_pos[0].tolist())}", flush=True)
    print(
        f"[{tag}] default  ={', '.join(f'{v:+.3f}' for v in robot.data.default_joint_pos[0].tolist())}",
        flush=True,
    )
    print(
        f"[{' ' * len(tag)}]  {'foot':>8s}  {'x':>8s}  {'y':>8s}  {'origin_z':>9s}  {'radius':>7s}  {'sole_z':>8s}"
        if radii
        else f"[{' ' * len(tag)}]  {'foot':>8s}  {'x':>8s}  {'y':>8s}  {'origin_z':>9s}  (radius unknown: sole column = origin)",
        flush=True,
    )
    soles = []
    for name, body_id in zip(FOOT_NAMES, foot_body_ids):
        p = pos_w[body_id]
        r = radii.get(name, fallback)
        sole = float(p[2]) - r
        soles.append(sole)
        print(
            f"        {name:>8s}  {float(p[0]):+8.4f}  {float(p[1]):+8.4f}  {float(p[2]):8.4f}  "
            f"{r:7.4f}  {sole:8.4f}",
            flush=True,
        )
    ground = _ground_height(env)
    spread = max(soles) - min(soles)
    print(
        f"[{tag}] sole spread={spread:.4f} m  min={min(soles):.4f}  max={max(soles):.4f}  "
        f"ground={ground if ground is None else f'{ground:.4f}'}  "
        f"-> translate-down height={float(root_pos[2]) - min(soles):.4f} m "
        f"(per foot: {', '.join(f'{float(root_pos[2]) - s:.4f}' for s in soles)})",
        flush=True,
    )


def _ground_height(env) -> float | None:
    """Best-effort height of the terrain surface the feet would land on."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        return None
    data = getattr(terrain, "data", None)
    for attr in ("default_land_plane_height",):
        value = getattr(data, attr, None)
        if value is not None:
            return float(value)
    return None


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    play_overrides(env_cfg, num_envs=1, seed=args_cli.seed, device=args_cli.device, warmup=0.5, duration=2.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    print(f"[INFO] task={args_cli.task}  bodies={list(robot.data.body_names)}", flush=True)
    print(f"[INFO] joints={list(robot.data.joint_names)}", flush=True)
    ids = foot_ids(base_env)
    print(f"[INFO] foot sensor ids={list(ids[0])}  foot body ids={list(ids[1])}", flush=True)
    radii = _foot_radii()
    print(f"[INFO] foot collision radii={radii}", flush=True)

    env.reset()
    _report("RESET", robot, ids[1], radii, base_env)
    _ref_line("RESET", base_env)

    # Zero action on the position arm means "hold the default joint angles", which is exactly the
    # nominal pose; watching it settle shows what the PD does with an unbalanced spawn.
    print(f"\n[INFO] stepping {args_cli.steps} policy steps with zero actions", flush=True)
    weight_n = float(robot.root_physx_view.get_masses()[0].sum()) * 9.81
    zero = torch.zeros(base_env.num_envs, robot.num_joints, device=env.device)
    with torch.inference_mode():
        for step in range(args_cli.steps):
            _obs, _rew, _dones, _extras = env.step(zero)
            if (step + 1) % args_cli.print_every == 0 or step + 1 == args_cli.steps:
                feet_sensor = base_env.scene.sensors["feet_forces"]
                fz = feet_sensor.data.net_forces_w[0, ids[0], 2]
                loaded = " ".join(f"{name[:2]}:{float(v):7.2f}" for name, v in zip(FOOT_NAMES, fz))
                print(
                    f"[STEP {step + 1:3d}] foot Fz(N)= {loaded}  sum={float(fz.sum()):7.2f}  "
                    f"weight={weight_n:.2f}  ratio={float(fz.sum()) / weight_n:.3f}",
                    flush=True,
                )
                _report(f"STEP{step + 1}", robot, ids[1], radii, base_env)
                _ref_line(f"STEP{step + 1}", base_env)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
