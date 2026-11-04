"""DAgger behaviour cloning for one V56 standing arm.

Why cloning and not a reward
----------------------------
The comparison is "position control vs direct torque", so the training signal must not be a
hand-written reward that happens to favour one interface.  Here the label comes from the stock
actuator law evaluated on the standing reference of ``reference.py``,

    tau* = 25 (q_0 - q) - 0.5 qdot,     clipped to +-23.5 N.m,

and each arm is asked for it in its own action units (``teacher_action_position`` /
``teacher_action_torque``).  The reward table of the environment is untouched and is only logged.

Why DAgger and not "record the teacher once"
------------------------------------------
Because on the position arm the label is (almost) the constant zero: ``q_0`` equals the default
joint angles, so an action of 0 makes the built-in PD produce exactly ``tau*``.  A network trained
on teacher rollouts alone learns that constant and nothing else, and so does - wrongly - the
torque arm, because the states it is shown never contain the deviation the feedback term is
supposed to correct.  Rolling out *each arm's own policy* and labelling the states it actually
reaches is what gives the torque arm a chance to learn the feedback term, and it is the same
procedure for both arms.

The teacher mix is annealed: for the first iterations the environments are driven by the teacher
(so the buffer starts with standable states), and the fraction of teacher-driven environments
falls linearly to zero.

Budget
------
``--steps`` defaults to ``agent_cfg.num_steps_per_env``, which the 200 Hz runners set to 96: 24
policy steps at 50 Hz is 0.48 s of physical time, so the same 0.48 s at 200 Hz is 96 steps.  Both
arms of a pair therefore see the same amount of simulated time per iteration, with the same
network, the same optimiser, the same seed and the same number of gradient steps.

    python bc_train.py --task Isaac-Go2-V56-StandPos200-v0
    python bc_train.py --task Isaac-Go2-V56-StandTorque200-v0 --iterations 5 --num_envs 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (_SOURCE_ROOT, os.path.join(_SOURCE_ROOT, "base_adapter"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

import cli_args  # noqa: F401, isort: skip

parser = argparse.ArgumentParser(description="V56 behaviour cloning (DAgger), one arm per process.")
parser.add_argument("--task", type=str, default="Isaac-Go2-V56-StandPos200-v0", help="V56 task id.")
parser.add_argument("--num_envs", type=int, default=1024, help="Environments used for the rollouts.")
parser.add_argument("--iterations", type=int, default=None, help="Rollout/train rounds; default: agent config.")
parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="Policy steps per iteration; default: agent_cfg.num_steps_per_env (24 at 50 Hz, 96 at 200 Hz).",
)
parser.add_argument("--updates", type=int, default=32, help="Gradient steps per iteration (identical for both arms).")
parser.add_argument("--batch_size", type=int, default=512, help="Minibatch size for the cloning loss.")
parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
parser.add_argument("--keep_iters", type=int, default=8, help="Iterations kept in the DAgger buffer.")
parser.add_argument(
    "--teacher_frac_start", type=float, default=1.0, help="Fraction of environments driven by the teacher at it 0."
)
parser.add_argument("--teacher_frac_end", type=float, default=0.0, help="...at the last iteration (linear).")
parser.add_argument("--seed", type=int, default=1, help="Environment and network seed (same for both arms).")
parser.add_argument("--output_dir", type=str, default=None, help="Default: results/bc/<task>.")
parser.add_argument("--save_interval", type=int, default=50, help="Iterations between checkpoints.")
# ``--device`` comes from ``AppLauncher``; declaring it here makes the kit abort with a duplicate-
# argument error before the sim is even created.
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn.functional as F

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import terrain_v56_stand_force_compare.config  # noqa: F401  (registers the four tasks)
from terrain_v56_stand_force_compare.actions import DirectTorqueAction
from terrain_v56_stand_force_compare.bc import OBS_KEY, from_runner_cfg, save_bc
from terrain_v56_stand_force_compare.reference import teacher_action_position, teacher_action_torque

LOG_KEYS: tuple[str, ...] = (
    "Force/total_over_weight",
    "Contact/duty_all4",
    "Ref/dq_max_rad",
    "Ref/tilt_rad",
    "Ref/d_ell_mm",
)


def _mean_log(log: dict) -> dict[str, float]:
    """Average the per-environment metric tensors of one step into plain floats."""
    out = {}
    for key in LOG_KEYS:
        if key in log:
            out[key] = float(torch.as_tensor(log[key]).float().mean())
    return out


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    # Observation noise stays ON: it is part of the task the PPO arms train under, so switching
    # it off only here would hand the cloned policies an easier observation than the baseline.
    iterations = args_cli.iterations if args_cli.iterations is not None else int(agent_cfg.max_iterations)
    steps = args_cli.steps if args_cli.steps is not None else int(agent_cfg.num_steps_per_env)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    term = base_env.action_manager.get_term("joint_pos")
    ref = base_env.reference
    is_torque = isinstance(term, DirectTorqueAction)
    arm = "torque" if is_torque else "position"
    action_scale = float(term.cfg.action_scale) if is_torque else float(term.cfg.scale)
    rate_hz = 1.0 / base_env.step_dt

    obs, _ = env.reset()
    obs_dim = int(obs[OBS_KEY].shape[-1])
    action_dim = int(base_env.action_manager.total_action_dim)

    torch.manual_seed(args_cli.seed)
    policy = from_runner_cfg(agent_cfg, obs_dim, action_dim).to(base_env.device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=args_cli.lr)

    # Under the package's own ``results/`` - the same tree ``run_trial.sh`` writes its logs and its
    # force evaluations to.  It used to point at ``<repo>/results/bc``, one level up and outside the
    # directory the launcher and the docs talk about, so a finished run was hard to find.
    out_dir = args_cli.output_dir or os.path.join(
        _HERE, "results", "bc", f"{arm}_{int(round(rate_hz))}hz"
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    progress_path = os.path.join(out_dir, "progress.jsonl")

    print(
        f"\n=== BC {arm} arm | {args_cli.task} | {rate_hz:.0f} Hz policy | {args_cli.num_envs} envs",
        flush=True,
    )
    print(
        f"  net {policy.hidden_dims} {policy.activation} | lr {args_cli.lr} | "
        f"{iterations} it x {steps} steps x {args_cli.updates} updates | action_scale {action_scale}",
        flush=True,
    )
    print(f"  out {out_dir}", flush=True)

    def teacher_label() -> torch.Tensor:
        if is_torque:
            return teacher_action_torque(robot, ref, action_scale)
        return teacher_action_position(robot, ref, action_scale)

    def checkpoint(path: str, iteration: int, loss: float) -> None:
        save_bc(
            path,
            policy,
            task=args_cli.task,
            arm=arm,
            rate_hz=rate_hz,
            action_scale=action_scale,
            iteration=iteration,
            extra={"steps": steps, "num_envs": args_cli.num_envs, "seed": args_cli.seed, "loss": loss},
        )

    obs_buffer: list[torch.Tensor] = []
    label_buffer: list[torch.Tensor] = []
    total_samples = 0
    for iteration in range(iterations):
        frac = iteration / max(1, iterations - 1)
        teacher_frac = args_cli.teacher_frac_start + frac * (args_cli.teacher_frac_end - args_cli.teacher_frac_start)

        step_obs: list[torch.Tensor] = []
        step_label: list[torch.Tensor] = []
        log_sums: dict[str, float] = {}
        log_n = 0
        resets = 0
        for _ in range(steps):
            current = obs[OBS_KEY]
            label = teacher_label()  # the state the observation belongs to, before stepping
            with torch.no_grad():
                action = policy(current)
            if teacher_frac > 0.0:
                # Drive a fraction of the environments with the teacher.  ``beta`` decays to 0, at
                # which point every state in the buffer was produced by the arm's own policy.
                mask = torch.rand(base_env.num_envs, device=base_env.device) < teacher_frac
                action = torch.where(mask[:, None], label, action)
            if agent_cfg.clip_actions is not None:
                action = action.clamp(-agent_cfg.clip_actions, agent_cfg.clip_actions)

            step_obs.append(current)
            step_label.append(label)
            obs, _reward, terminated, truncated, extras = env.step(action)
            resets += int(terminated.sum().item()) + int(truncated.sum().item())
            for key, value in _mean_log(extras.get("log") or {}).items():
                log_sums[key] = log_sums.get(key, 0.0) + value
            log_n += 1

        obs_buffer.append(torch.cat(step_obs))
        label_buffer.append(torch.cat(step_label))
        while len(obs_buffer) > max(1, args_cli.keep_iters):
            obs_buffer.pop(0)
            label_buffer.pop(0)
        states = torch.cat(obs_buffer)
        labels = torch.cat(label_buffer)
        total_samples += states.shape[0]

        # --- the cloning update: identical shape, optimiser and count on both arms -------------
        policy.train()
        losses = []
        for _ in range(args_cli.updates):
            index = torch.randint(0, states.shape[0], (args_cli.batch_size,), device=base_env.device)
            loss = F.mse_loss(policy(states[index]), labels[index])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(float(loss))
        policy.eval()

        # Report the loss after the updates, plus the mean over them, on the same buffer prefix.
        with torch.no_grad():
            head = min(4096, states.shape[0])
            final_loss = float(F.mse_loss(policy(states[:head]), labels[:head]))
        means = {key: value / max(1, log_n) for key, value in log_sums.items()}
        record = {
            "iteration": iteration,
            "loss": final_loss,
            "loss_mean": sum(losses) / max(1, len(losses)),
            "teacher_frac": teacher_frac,
            "samples": int(states.shape[0]),
            "total_samples": int(total_samples),
            "resets": resets,
            **means,
        }
        with open(progress_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"[it {iteration:3d}/{iterations}] mse {final_loss:.4f} | teacher {teacher_frac * 100:5.1f} %"
            f" | dq_max {means.get('Ref/dq_max_rad', float('nan')):.3f} rad"
            f" | tilt {means.get('Ref/tilt_rad', float('nan')):.3f}"
            f" | d_ell {means.get('Ref/d_ell_mm', float('nan')):6.1f} mm"
            f" | F/mg {means.get('Force/total_over_weight', float('nan')):.3f}"
            f" | duty4 {means.get('Contact/duty_all4', float('nan')):.2f}"
            f" | buf {states.shape[0]}",
            flush=True,
        )

        if (iteration + 1) % args_cli.save_interval == 0 or iteration == iterations - 1:
            path = os.path.join(out_dir, f"model_{iteration}.pt")
            checkpoint(path, iteration, final_loss)
            print(f"  -> {path}", flush=True)
            final_path = os.path.join(out_dir, "final.pt")
            if path != final_path:
                checkpoint(final_path, iteration, final_loss)

    env.close()
    print(f"\nBC finished: {out_dir}/final.pt", flush=True)
    sys.stdout.flush()
    # Same teardown hang as in train.py: the render context does not close on this host.
    os._exit(0)


if __name__ == "__main__":
    main()
