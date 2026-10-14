"""Measure frozen Stage-3 PPO actions without writing into /data/sata."""

import json
import os

import isaacgym  # must precede torch
import torch

from legged_gym.envs import *  # noqa: F401,F403 -- register tasks
from legged_gym.utils import class_to_dict, get_args, task_registry
from rsl_rl.runners import OnPolicyRunner


CHECKPOINT = (
    "/data/sata/SATA_flat_reproduction/legged_gym/logs/"
    "SATA_flat_gait_coordination/"
    "Aug05_19-46-44_stage3_diagonal_gait_seed1_visible/model_3000.pt"
)


def tensor_summary(values):
    probs = torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    quantiles = torch.quantile(values.float(), probs, dim=0)
    return {
        "mean": values.float().mean(0).tolist(),
        "std": values.float().std(0, unbiased=False).tolist(),
        "min": quantiles[0].tolist(),
        "p01": quantiles[1].tolist(),
        "p05": quantiles[2].tolist(),
        "median": quantiles[3].tolist(),
        "p95": quantiles[4].tolist(),
        "p99": quantiles[5].tolist(),
        "max": quantiles[6].tolist(),
        "global_abs_max": values.abs().max().item(),
    }


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs("go2_torque_gait")
    env_cfg.env.num_envs = 256
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.measure_heights = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.loss_action_obs = False
    # Freeze the SATA growth schedule at checkpoint 3000 while retaining random
    # command sampling over the final configured 0.0--0.8 m/s range.
    env_cfg.control.control_type = "T"
    env_cfg.test.checkpoint = 3000
    env_cfg.test.use_test = False

    env, _ = task_registry.make_env("go2_torque_gait", args=args, env_cfg=env_cfg)
    runner = OnPolicyRunner(env, class_to_dict(train_cfg), log_dir=None, device=args.rl_device)
    runner.load(CHECKPOINT, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    obs = env.get_observations()
    command_targets = torch.linspace(0.0, 0.8, env.num_envs, device=env.device)

    action_rows = []
    command_rows = []
    requested_torque_rows = []
    actual_torque_rows = []
    saturation_rows = []
    for _ in range(200):
        # Cover the complete final Stage-3 command interval in every vector
        # step. Recompute observations so the policy sees the assigned command.
        env.commands[:, 0] = command_targets
        env.commands[:, 1:3] = 0.0
        env.compute_observations()
        obs = env.get_observations()
        with torch.no_grad():
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
        requested = actions[:, :12] * env.cfg.control.action_scale
        limits = env.current_torque_limit_scale * env.torque_limits
        rear_scale = torch.ones_like(limits)
        rear_scale[6:] = env.r_leg_scaled
        limits = limits * rear_scale
        saturation = requested.abs() >= limits.unsqueeze(0)
        action_rows.append(actions.detach().cpu())
        command_rows.append(env.commands[:, 0].detach().cpu())
        requested_torque_rows.append(requested.detach().cpu())
        actual_torque_rows.append(env.torques.detach().cpu())
        saturation_rows.append(saturation.detach().cpu())

    actions = torch.cat(action_rows)
    commands = torch.cat(command_rows)
    requested_torques = torch.cat(requested_torque_rows)
    actual_torques = torch.cat(actual_torque_rows)
    saturation = torch.cat(saturation_rows)
    result = {
        "checkpoint": CHECKPOINT,
        "num_envs": env.num_envs,
        "policy_steps": 200,
        "samples": actions.shape[0],
        "joint_names": list(env.dof_names),
        "configured_command_range_mps": [0.0, 0.8],
        "observed_command_mps": tensor_summary(commands.unsqueeze(1)),
        "policy_actions": tensor_summary(actions),
        "requested_torque_nm": tensor_summary(requested_torques),
        "actual_torque_nm": tensor_summary(actual_torques),
        "requested_over_dynamic_limit_fraction_per_joint": saturation.float().mean(0).tolist(),
        "requested_over_dynamic_limit_fraction_global": saturation.float().mean().item(),
        "provisional_sac_action_limit": 4.7,
    }

    output = "/data/sac/results/ppo_stage3_action_distribution.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("Wrote", output)
    print("Observed command range:", commands.min().item(), commands.max().item())
    print("PPO global |action| max:", actions.abs().max().item())
    print("PPO global action p99:", torch.quantile(actions.abs(), 0.99).item())
    print("Requested-over-limit fraction:", saturation.float().mean().item())


if __name__ == "__main__":
    main()
