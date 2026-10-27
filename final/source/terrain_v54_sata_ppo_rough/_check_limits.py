"""Temporary check: joint limits / gains of the Go2 USD vs SATA's URDF values."""

import argparse
import os
import sys

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

import cli_args  # noqa: E402,F401  (registers the sa_16dof stub before the app loads)
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import terrain_v52_sata_ppo_flat.config as v52cfg  # noqa: E402,F401
from terrain_v51_native_terrain.sata_alignment import SATA_JOINT_NAMES  # noqa: E402

torch.set_printoptions(precision=4, sci_mode=False, linewidth=160)

cfg = v52cfg.V52SataPpoFlatEnvCfg()
cfg.scene.num_envs = 1
env = gym.make("Isaac-Go2-V52-SataPpoFlat-v0", cfg=cfg)
env.reset()
robot = env.unwrapped.scene["robot"]
ids, names = robot.find_joints(list(SATA_JOINT_NAMES), preserve_order=True)
d = robot.data


def show(label, tensor):
    print(f"CHECK {label}: {tensor.tolist()}", flush=True)


print(f"CHECK joint_names: {names}", flush=True)
show("joint_pos_limits", d.joint_pos_limits[0, ids].cpu())
show("default_joint_pos", d.default_joint_pos[0, ids].cpu())
show("joint_vel_limits", d.joint_vel_limits[0, ids].cpu())
show("joint_effort_limits", d.joint_effort_limits[0, ids].cpu())
for attr in ("joint_armature", "joint_damping", "joint_friction", "joint_stiffness"):
    if hasattr(d, attr):
        show(attr, getattr(d, attr)[0, ids].cpu())

# actuation model actually configured on the asset
actuators = getattr(robot, "actuators", {})
for name, act in actuators.items():
    print(f"CHECK actuator {name}: {type(act).__name__} stiffness={getattr(act.cfg, 'stiffness', None)} "
          f"damping={getattr(act.cfg, 'damping', None)} effort_limit={getattr(act.cfg, 'effort_limit', None)} "
          f"velocity_limit={getattr(act.cfg, 'velocity_limit', None)} "
          f"armature={getattr(act.cfg, 'armature', None)} friction={getattr(act.cfg, 'friction', None)}",
          flush=True)

env.close()
sys.stdout.flush()
os._exit(0)
