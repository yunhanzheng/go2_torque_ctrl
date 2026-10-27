"""Dump the effective V52 env config (events/terminations/rewards/obs) without starting sim."""
import sys

sys.path.insert(0, "/home/ubuntu22/data/go2_torque_ctrl/final/source")
sys.path.insert(0, "/home/ubuntu22/data/go2_torque_ctrl/final/source/terrain_v51_native_terrain")

from terrain_v52_sata_ppo_flat.config import V52SataPpoFlatEnvCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg

cfg = V52SataPpoFlatEnvCfg()
print("decimation", cfg.decimation, "| episode", cfg.episode_length_s, "| dt", cfg.sim.dt)
print("robot init pos", cfg.scene.robot.init_state.pos)
print("robot init joint_pos", cfg.scene.robot.init_state.joint_pos)

for n, v in vars(cfg.events).items():
    if isinstance(v, EventTermCfg):
        print("EVENT ", v.mode, n, "->", v.func.__name__, v.params)
for n, v in vars(cfg.terminations).items():
    if isinstance(v, TerminationTermCfg):
        print("TERM  ", n, "->", v.func.__name__, v.params)
print("REWARDS", [(n, v.weight) for n, v in vars(cfg.rewards).items() if isinstance(v, RewardTermCfg)])
for n, v in vars(cfg.observations.policy).items():
    print("OBS   ", n, type(v).__name__)
