"""Smoke test: build the V51 SATA-aligned env and step it a few times.

Verifies:
- env constructs (all custom cfgs/terms instantiate)
- observation is 60-D (SATA layout)
- SataTorqueAction produces bounded torques + accumulates fatigue
- growth scale / command term work
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import terrain_v51_native_terrain.config  # noqa: F401,E402  (registers the task)
from terrain_v51_native_terrain.config import V51NativeTerrainEnvCfg  # noqa: E402

env_cfg = V51NativeTerrainEnvCfg()
env_cfg.scene.num_envs = 2
# Use a single flat plane to skip the slow native rough-terrain mesh generation;
# this smoke test validates the action/command/obs components, not the terrain.
import isaaclab.terrains as terrain_gen  # noqa: E402
env_cfg.scene.terrain.terrain_generator.sub_terrains = {
    "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
}
env_cfg.scene.terrain.terrain_generator.num_rows = 1
env_cfg.scene.terrain.terrain_generator.num_cols = 1
env_cfg.scene.terrain.max_init_terrain_level = None
env_cfg.curriculum.terrain_levels = None
env = gym.make("Isaac-Go2-V51-NativeTerrain-v0", cfg=env_cfg)
uw = env.unwrapped
print("=== ENV CREATED ===", flush=True)
print("action_space:", env.action_space.shape, flush=True)
print("obs_space[policy]:", env.observation_space["policy"].shape, flush=True)

at = uw.action_manager.get_term("joint_pos")
print("action term:", type(at).__name__, flush=True)
ct = uw.command_manager.get_term("base_velocity")
print("command term:", type(ct).__name__, flush=True)

# Diagnostic: compare the USD-reported joint velocity limits (which the hill model
# no longer uses) against the hardcoded SATA URDF values now baked into the action.
robot = uw.scene["robot"]
from sata_alignment import SATA_JOINT_NAMES, SATA_DOF_VEL_LIMITS  # noqa: E402
jids, _ = robot.find_joints(list(SATA_JOINT_NAMES), preserve_order=True)
usd_vlim = robot.data.joint_vel_limits[:, jids][0].tolist()
print("USD  joint_vel_limits:", [round(v, 2) for v in usd_vlim], flush=True)
print("SATA dof_vel_limits  :", list(SATA_DOF_VEL_LIMITS), flush=True)
print("SATA torque_limit    :", at.cfg.torque_limit, "| hill uses hardcoded SATA vel limits", flush=True)

obs, _ = env.reset()
dim = obs["policy"].shape[-1]
print("reset obs dim:", dim, flush=True)
assert dim == 60, f"Expected 60-D SATA obs, got {dim}"

for i in range(10):
    action = torch.randn(2, 12, device=uw.device) * 0.5
    obs, rew, term, trunc, info = env.step(action)
    cmd = uw.command_manager.get_command("base_velocity")
    print(
        f"step {i}: obs_dim={obs['policy'].shape[-1]} rew={rew.mean().item():+.3f} "
        f"|tau|max={at.torques.abs().max().item():.3f} fat_max={at.motor_fatigue.max().item():.4f} "
        f"gs={at.general_scale:.4f} cmd_x=[{cmd[:,0].min().item():.2f},{cmd[:,0].max().item():.2f}]",
        flush=True,
    )
    assert obs["policy"].shape[-1] == 60
    assert torch.isfinite(at.torques).all(), "non-finite torque"
    assert at.torques.abs().max().item() <= 23.5 + 1e-3, "torque exceeds limit"

print("=== SMOKE TEST PASSED ===", flush=True)
sim_app.close()
