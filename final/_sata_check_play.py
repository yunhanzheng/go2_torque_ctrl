"""Reproduce SATA play.py (model_3000, test.vel=+1.0) headless and print what the policy actually does.

Run with the `sata` conda env (Isaac Gym):
  /home/ubuntu22/.holosoma_deps/miniconda3/envs/sata/bin/python final/_sata_check_play.py
"""
import sys

sys.argv = [
    sys.argv[0],
    "--task=go2_torque",
    "--headless",
    "--num_envs=1",
    "--sim_device=cuda:0",
    "--rl_device=cuda:0",
]
sys.path.insert(0, "/home/ubuntu22/data/SATA/legged_gym")

import isaacgym  # noqa: F401,E402  must be imported before torch
import torch  # noqa: E402
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402

args = get_args()
env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

# --- same overrides as legged_gym/scripts/play.py ---
env_cfg.env.num_envs = 1
env_cfg.env.episode_length_s = 10.0
env_cfg.control.control_type = "T"
env_cfg.test.use_test = True
env_cfg.test.checkpoint = 3000
env_cfg.test.vel = torch.tensor([1.0, 0.0, 0.0, 0.0])
env_cfg.control.activation_process = True
env_cfg.control.hill_model = True
env_cfg.control.motor_fatigue = True
env_cfg.commands.heading_command = True
env_cfg.terrain.mesh_type = "plane"
env_cfg.terrain.num_rows = 5
env_cfg.terrain.num_cols = 5
env_cfg.terrain.curriculum = False
env_cfg.noise.add_noise = False
env_cfg.domain_rand.randomize_friction = False
env_cfg.domain_rand.push_robots = False
env_cfg.domain_rand.randomize_base_mass = False
env_cfg.terrain.terrain_proportions = [0, 1, 0, 0, 0]

env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
train_cfg.runner.resume = True
ppo_runner, _ = task_registry.make_alg_runner(
    env=env, name=args.task, args=args, train_cfg=train_cfg
)
policy = ppo_runner.get_inference_policy(device=env.device)
obs = env.get_observations()

print(f"[cfg] env.test.vel = {env.cfg.test.vel.tolist()}")
print(f"[cfg] env.cfg is env_cfg: {env.cfg is env_cfg}")
print(f"[cfg] gs(step_count={env.step_count}) = {env.general_scale:.3f}")
print("[spawn] dof_pos      :", [round(v, 3) for v in env.dof_pos[0].tolist()])
print("[spawn] default      :", [round(v, 3) for v in env.default_dof_pos[0].tolist()])
print("[spawn] root_z       :", round(env.root_states[0, 2].item(), 3))
env.gym.refresh_rigid_body_state_tensor(env.sim)
print("[spawn] body_z       :", [round(v, 3) for v in env.rigid_body_states[0, :, 2].tolist()])

for i in range(2000):  # 200 Hz x 10 s = one full episode
    actions = policy(obs.detach())
    out = env.step(actions)  # go2_torque may return more than 4 values
    obs = out[0]
    extras = out[3] if len(out) > 3 else (out[-1] if isinstance(out[-1], dict) else {})
    if isinstance(extras, dict) and extras.get("episode"):
        ep = extras["episode"]
        print("EPISODE LOG at step", i, "->")
        for k, v in ep.items():
            print(f"   {k} = {v}")
        break
    if i % 500 == 0 or i == 1999:
        q = env.root_states[0, 3:7]
        _fi = env.reward_names.index("forward")
        _r_step = env.reward_functions[_fi]()[0].item() * env.reward_scales["forward"]
        _acc = env.episode_sums["forward"][0].item()
        print(
            f"step {i:3d}: cmd_x={env.commands[0, 0].item():+.3f} "
            f"vx_body={env.base_lin_vel[0, 0].item():+.3f} "
            f"root_x={env.root_states[0, 0].item():+.3f} "
            f"gs={env.general_scale:.3f} "
            f"quat=({q[0]:+.2f},{q[1]:+.2f},{q[2]:+.2f},{q[3]:+.2f})"
            f" | forward_step={_r_step:+.4f} (max 10) sum={_acc:.2f}",
            flush=True,
        )
