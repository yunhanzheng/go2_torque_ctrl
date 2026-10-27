"""Dump the current V54 env cfg to yaml, in the same format train.py writes to params/.

Usage (on the server):
    CUDA_VISIBLE_DEVICES=1 OUT_YAML=/tmp/v54_now_env.yaml python -u _dump_cfg.py

The result is comparable with a past run's ``logs/.../params/env.yaml``, so a plain
``diff`` lists every config difference between the current code and that run.

``train.py`` dumps *after* ``gym.make``, and constructing the env mutates the cfg
(``TerrainGenerator._add_sub_terrain`` copies the generator size / slope threshold down
into every sub-terrain, the scene clears ``env_spacing`` once the terrain owns the origins).
This script therefore walks the same path: build the cfg, make the env, then dump.
"""
import argparse
import os
import sys
import types

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

_STUB = "isaaclab_tasks.manager_based.locomotion.velocity.config.sa_16dof"
if _STUB not in sys.modules:
    _m = types.ModuleType(_STUB)
    _m.__path__ = []
    sys.modules[_STUB] = _m

import gymnasium as gym  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
import terrain_v54_sata_ppo_rough.config as C  # noqa: E402,F401

_out = os.environ.get("OUT_YAML", "/tmp/v54_now_env.yaml")

env_cfg = C.V54SataPpoRoughEnvCfg()
env_cfg.seed = 1  # the 2026-10-22 run was launched with --seed 1
env_cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", "2"))
# Constructing the env is what applies the generator -> sub-terrain size/slope propagation.
gym.make("Isaac-Go2-V54-SataPpoRough-v0", cfg=env_cfg)

dump_yaml(_out, env_cfg)
print(f"[DUMP] env cfg written to {_out}", flush=True)

# Isaac Sim hangs in close() on a headless box without a working Vulkan runtime.
os._exit(0)