"""Can a *torque* command hold the stance, and at which control rate?

Why this exists: the imitation teacher in ``reference.py`` is the stock Go2 joint PD
(``tau = kp (q_0 - q) - kd qdot``).  On the position arm that law is evaluated by the actuator
every physics step (200 Hz) no matter how often the policy runs.  On the torque arm the same law
is only evaluated at the *policy* rate, and the value is then held for ``decimation`` substeps.
So before any behaviour-cloning run is started, the teacher itself has to be proven to stand -
otherwise a "learned" torque policy would be measured against a teacher that cannot execute.

The script steps the env with a hand-computed torque PD (exactly what ``teacher_torque`` does)
and prints, every few steps, the quantities that distinguish the three failure modes:

* ``qdot`` spike + ``clip`` on one joint   -> the DCMotor envelope removed the authority;
* roll/pitch growing while torques stay small -> the body loop diverged, not the joint loop;
* ``d_ell`` growing with everything small  -> the leg yields, i.e. pure compliance, no instability.

    python diag_torque_hold.py --task Isaac-Go2-V56-StandTorque-v0 --kp 25 --kd 0.5

The sweep that matters is over ``--kd``: at 50 Hz the stock value stands, while a larger one
(doubling the damping to compensate the slower loop - the intuitive thing to do) injects energy
through the zero-order hold and collapses the robot.  ``--task`` accepts any registered V56 task,
so the same law can be driven through the position arm's envelope for comparison.

One process per configuration: the simulation context is a singleton.
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

parser = argparse.ArgumentParser(description="V56 torque-hold diagnostic.")
parser.add_argument("--task", type=str, default="Isaac-Go2-V56-StandTorque-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--seconds", type=float, default=1.5)
parser.add_argument("--every", type=int, default=5, help="Print every N policy steps.")
parser.add_argument("--kp", type=float, default=25.0)
parser.add_argument("--kd", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import terrain_v56_stand_force_compare.config  # noqa: F401
from terrain_v56_stand_force_compare.metrics import TORQUE_LIMIT_NM

JOINT_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 7
    env_cfg.observations.policy.enable_corruption = False
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    term = unwrapped.action_manager.get_term("joint_pos")
    scale = float(getattr(term.cfg, "action_scale", 1.0))
    legs = robot.actuators["base_legs"]

    rate = 1.0 / unwrapped.step_dt
    print(
        f"\n=== {args_cli.task}: {rate:.0f} Hz policy, sim {1.0 / unwrapped.physics_dt:.0f} Hz, "
        f"teacher kp={args_cli.kp} kd={args_cli.kd}",
        flush=True,
    )
    print(f"  actuator {type(legs).__name__} stiffness={float(torch.as_tensor(legs.stiffness).max())}", flush=True)

    env.reset()
    n_steps = int(round(args_cli.seconds * rate))
    worst_clip, worst_qdot, worst_qdot_joint = 0.0, 0.0, "-"
    for step in range(n_steps):
        torque = args_cli.kp * (robot.data.default_joint_pos - robot.data.joint_pos) - args_cli.kd * (
            robot.data.joint_vel
        )
        action = torque / scale
        obs, reward, terminated, truncated, extras = env.step(action)

        log = extras.get("log") or {}
        clip = (torque.abs() > TORQUE_LIMIT_NM).float().mean().item()
        qdot_max = float(robot.data.joint_vel.abs().max())
        if qdot_max > worst_qdot:
            worst_qdot = qdot_max
            flat = robot.data.joint_vel.abs().argmax().item()
            worst_qdot_joint = f"{JOINT_NAMES[flat % 12]}({flat // 12})"
        worst_clip = max(worst_clip, clip)

        if step % args_cli.every == 0 or step == n_steps - 1:
            g = robot.data.projected_gravity_b[0]
            roll = math.atan2(float(g[1]), -float(g[2]))
            pitch = -math.asin(max(-1.0, min(1.0, float(g[0]))))
            print(
                f"  step {step:4d}  z={float(robot.data.root_pos_w[0, 2]):.3f}"
                f"  roll={roll: .3f} pitch={pitch: .3f}"
                f"  |tau|max={float(torque.abs().max()):5.1f}"
                f"  clip={clip * 100:4.1f}%"
                f"  |qdot|max={qdot_max:5.2f}"
                f"  F/mg={float(log.get('Force/total_over_weight', float('nan'))):.3f}"
                f"  duty4={float(log.get('Contact/duty_all4', float('nan'))):.2f}"
                f"  d_ell={float(log.get('Ref/d_ell_mm', float('nan'))):6.1f}mm"
                f"  dq_max={float(log.get('Ref/dq_max_rad', float('nan'))):.3f}",
                flush=True,
            )

    resets = int(terminated.sum().item()) + int(truncated.sum().item())
    print(
        f"  -> after {args_cli.seconds:.1f} s: worst clip {worst_clip * 100:.1f}% of joints,"
        f" worst |qdot| {worst_qdot:.2f} rad/s on {worst_qdot_joint}, resets this step {resets}",
        flush=True,
    )
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
