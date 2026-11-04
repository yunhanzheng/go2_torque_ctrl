"""Smoke test for the V56 standing comparison.  No training, no checkpoint, a few seconds.

Four things are checked per task:

1. wiring - 12 leg actions, 48-D policy observation, the ``feet_forces`` sensor exists and
   ``metrics.py`` can address all four feet by name;
2. the torque path - ``processed_actions == clamp(action_scale * raw_action, +-23.5)`` and the
   articulation's own ``applied_torque`` is the same tensor, i.e. no hidden PD, no filter, no
   saturation beyond the clamp;
3. the plant - the ground carries the robot (``sum(GRF) / m g`` near 1) and nothing is NaN;
4. the logging channel - ``extras["log"]`` carries every force/CoP/actuation key that wandb
   will receive.

Standing with a random torque policy is impossible, so (3) for the torque variant drives the
torque path with the stock Go2 PD torque computed *inside the test* (see
``_stance_pd_action``).  That keeps the experiment free of any PD while still proving that the
feet can carry the robot through a torque command.

The same driver is what ``diag_torque_hold.py`` sweeps: it is the execution check for the
imitation teacher of ``reference.py``.  Running that sweep is also how the one asymmetry the
config cannot remove was measured - the position arm's PD is re-evaluated by the actuator every
physics step (200 Hz), the torque arm's controller only once per 20 ms policy step (see
``config.py`` item 6).

    python smoke_test.py                          # all four tasks, headless by default
    python smoke_test.py --task Isaac-Go2-V56-StandTorque-v0

Exit code 0 means every assertion passed.
"""
from __future__ import annotations

import argparse
import os
import sys

# --- same path bootstrap as train.py ---------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (_SOURCE_ROOT, os.path.join(_SOURCE_ROOT, "base_adapter"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

# Imported for its side effect only: it pre-registers the broken ``sa_16dof`` task package as an
# empty stub, without which ``import isaaclab_tasks`` dies on the training server (see the
# comment at the top of ``base_adapter/cli_args.py``).
import cli_args  # noqa: F401, isort: skip

TASK_POS = "Isaac-Go2-V56-StandPos-v0"
TASK_TORQUE = "Isaac-Go2-V56-StandTorque-v0"
TASK_POS200 = "Isaac-Go2-V56-StandPos200-v0"
TASK_TORQUE200 = "Isaac-Go2-V56-StandTorque200-v0"

OBS_DIM = 48
"""Stock Go2 flat observation: 3 + 3 + 3 + 3 base/command terms and 12 + 12 + 12 joint/action."""
NUM_LEG_JOINTS = 12

STOCK_EFFORT_LIMIT_NM = 23.5
STOCK_SATURATION_EFFORT_NM = 23.5
STOCK_VELOCITY_LIMIT_RAD_S = 30.0
STOCK_STIFFNESS = 25.0
STOCK_DAMPING = 0.5
"""``UNITREE_GO2_CFG`` leg ``DCMotorCfg`` values, checked against the live actuator of every
variant: the envelope is shared, only the gains are the experimental variable."""

parser = argparse.ArgumentParser(description="V56 smoke test (no training, no checkpoint).")
parser.add_argument("--task", type=str, default=None, help="Task id; default: all four tasks, one process each.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments used for the checks.")
parser.add_argument("--settle_s", type=float, default=1.5, help="Physical seconds before averaging forces.")
parser.add_argument("--check_s", type=float, default=0.4, help="Physical seconds averaged for the force check.")
parser.add_argument(
    "--ok_file",
    type=str,
    default=None,
    help="Write this file once every check passed; the multi-task parent reads the verdict from it.",
)
AppLauncher.add_app_launcher_args(parser)
# The test is headless by nature; an explicit ``--headless`` on the command line still wins.
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

TASKS = (TASK_POS, TASK_TORQUE, TASK_POS200, TASK_TORQUE200)

if args_cli.task is None:
    # One interpreter per task: SimulationContext is a singleton, so a second ``gym.make`` in
    # the same process would silently reuse the first simulation.
    import subprocess

    failed = []
    for task_id in TASKS:
        # The return code alone cannot be trusted: Isaac Sim's teardown runs from an atexit hook
        # and exits with 0 even when the checks below raised, which made this loop report every
        # variant as passed while all three were crashing on an attribute name.  A child that
        # reached the end of its checks writes ``ok_file``, and that file is the verdict.
        ok_file = f"/tmp/v56_smoke_ok_{task_id.split('-')[-2]}_{os.getpid()}.marker"
        if os.path.exists(ok_file):
            os.remove(ok_file)
        child = [
            sys.executable,
            os.path.abspath(__file__),
            "--task",
            task_id,
            "--headless",
            "--num_envs",
            str(args_cli.num_envs),
            "--settle_s",
            str(args_cli.settle_s),
            "--check_s",
            str(args_cli.check_s),
            "--ok_file",
            ok_file,
        ]
        if args_cli.device is not None:
            child += ["--device", args_cli.device]
        returncode = subprocess.run(child).returncode
        if returncode != 0 or not os.path.exists(ok_file):
            print(f"[FAILED] {task_id}: returncode={returncode} marker={os.path.exists(ok_file)}", flush=True)
            failed.append(task_id)
        else:
            os.remove(ok_file)
    if failed:
        raise SystemExit(f"V56 smoke test FAILED for: {', '.join(failed)}")
    print("\nAll V56 smoke checks passed.")
    raise SystemExit(0)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import terrain_v56_stand_force_compare.config  # noqa: F401  (registers the four tasks)
from terrain_v56_stand_force_compare.actions import DirectTorqueAction
from terrain_v56_stand_force_compare.metrics import FOOT_NAMES, TORQUE_LIMIT_NM

REQUIRED_LOG_KEYS: tuple[str, ...] = (
    "Force/total_N",
    "Force/total_over_weight",
    *(f"Force/{foot[:2]}_N" for foot in FOOT_NAMES),
    "Force/load_std_frac",
    "Force/asym_LR",
    "Force/asym_FB",
    "Force/front_share",
    "Contact/duty_foot",
    "Contact/duty_all4",
    "CoP/offset_m",
    "CoP/speed_mps",
    "Ref/dq_rms_rad",
    "Ref/dq_max_rad",
    "Ref/tilt_rad",
    "Ref/d_ell_mm",
    "Ref/d_foot_ell_max_mm",
    "Ref/ell_0_m",
    "Act/torque_rms",
    "Act/torque_max",
    "Act/sat_frac",
)


def _check(condition: bool, message: str) -> None:
    """Fail loudly with the measured value instead of silently printing a warning."""
    if not condition:
        raise AssertionError(message)


def _worst(tensor: torch.Tensor) -> float:
    return float(tensor.abs().max())


def _actuator_value(legs, name: str) -> torch.Tensor:
    """Read one actuator limit, public or private.

    ``ActuatorBase`` exposes ``effort_limit`` and ``velocity_limit`` but keeps the DCMotor's
    saturation torque as ``_saturation_effort``; a test that wants to pin the envelope has to
    reach around that, rather than read the configuration the object was built from.
    """
    value = getattr(legs, name, None)
    if value is None:
        value = getattr(legs, f"_{name}", None)
    if value is None:
        raise AssertionError(f"actuator exposes neither '{name}' nor '_{name}'")
    return torch.as_tensor(value, dtype=torch.float)


def _dcmotor_band(joint_vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The four-quadrant torque-speed band of the stock ``DCMotor``, as :class:`ActuatorBase` applies it.

    Mirrors ``DCMotor._clip_effort`` (``actuator_pd.py:295``): the positive and negative limits are
    *not* symmetric, they depend on the sign and magnitude of the joint velocity, and the band
    collapses to zero driving torque at ``velocity_limit``.  The smoke test needs this because the
    torque variant shares the envelope with the baseline: demanding ``applied_torque ==
    commanded_torque`` everywhere would only be true for a motor that does not exist.
    """
    vel_at_effort_lim = STOCK_VELOCITY_LIMIT_RAD_S * (1.0 + STOCK_EFFORT_LIMIT_NM / STOCK_SATURATION_EFFORT_NM)
    speed = joint_vel.clamp(min=-vel_at_effort_lim, max=vel_at_effort_lim)
    top = (STOCK_SATURATION_EFFORT_NM * (1.0 - speed / STOCK_VELOCITY_LIMIT_RAD_S)).clamp(max=STOCK_EFFORT_LIMIT_NM)
    bottom = (-STOCK_SATURATION_EFFORT_NM * (1.0 + speed / STOCK_VELOCITY_LIMIT_RAD_S)).clamp(
        min=-STOCK_EFFORT_LIMIT_NM
    )
    # Above ``velocity_limit`` the curve crosses zero and the two edges swap; torch.clip would then
    # force the torque to ``top``.  Ordering them keeps the assertion about membership honest.
    low, high = torch.minimum(bottom, top), torch.maximum(bottom, top)
    return low, high


def _check_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, what: str) -> None:
    error = float((actual - expected).abs().max())
    _check(error <= atol, f"{what}: max |error| = {error:.5f} > {atol:.5f}")


def _build_env(task_id: str):
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 7
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    # The check is about physics, not about the noise the policy input carries.
    env_cfg.observations.policy.enable_corruption = False
    return gym.make(task_id, cfg=env_cfg)


def _stance_pd_action(robot, term) -> torch.Tensor:
    """The stock Go2 PD torque, expressed as a *network output* for the torque path.

    Test-only, and deliberately the *official* gains (kp 25, kd 0.5) at both control rates.
    The intuition that a velocity feedback has to be strengthened when the loop closes less
    often is wrong here, and ``diag_torque_hold.py`` measured how wrong: at 50 Hz the same law
    stands with kd 0.5 (duty 1.00, |qdot| < 5 rad/s, no clipping) and collapses with kd 1.0
    (roll 0.10 rad within 0.4 s, three joints pinned at the 30 rad/s velocity limit, 40 % of
    the commands clipped) and even faster with kd 2.0.  A zero-order hold on a stale velocity
    injects energy instead of dissipating it, because the torque keeps braking a motion that
    has already reversed.  Nothing of this is part of the experiment: the experiment has no PD.
    """
    kp, kd = STOCK_STIFFNESS, STOCK_DAMPING
    torque = kp * (robot.data.default_joint_pos - robot.data.joint_pos) - kd * robot.data.joint_vel
    return torque / term.cfg.action_scale


def run_task(task_id: str) -> None:
    env = _build_env(task_id)
    unwrapped = env.unwrapped
    term = unwrapped.action_manager.get_term("joint_pos")
    robot = unwrapped.scene["robot"]
    is_torque = isinstance(term, DirectTorqueAction)
    label = "direct torque" if is_torque else "position"
    print(f"\n=== {task_id}  [{label}, {1.0 / unwrapped.step_dt:.0f} Hz] ===", flush=True)

    # --- 1. wiring ---------------------------------------------------------------------
    _check(
        unwrapped.action_manager.total_action_dim == NUM_LEG_JOINTS,
        f"expected {NUM_LEG_JOINTS} leg actions, got {unwrapped.action_manager.total_action_dim}",
    )
    _check("feet_forces" in unwrapped.scene.sensors, "'feet_forces' sensor is missing (see config.py)")
    missing_feet = [name for name in FOOT_NAMES if name not in unwrapped.scene.sensors["feet_forces"].body_names]
    _check(not missing_feet, f"'feet_forces' does not track {missing_feet}")

    legs = robot.actuators["base_legs"]
    # ``UNITREE_GO2_CFG`` ships the legs as a DCMotor with these numbers.  The envelope is part of
    # the motor, not of the controller, so both variants must run against exactly the same one - a
    # wider envelope would hand the torque variant a stronger actuator than the baseline has.
    effort = _actuator_value(legs, "effort_limit")
    saturation = _actuator_value(legs, "saturation_effort")
    velocity = _actuator_value(legs, "velocity_limit")
    _check(
        float(effort.min()) == STOCK_EFFORT_LIMIT_NM and float(saturation.min()) == STOCK_EFFORT_LIMIT_NM,
        f"effort envelope is not the stock {STOCK_EFFORT_LIMIT_NM} N.m: "
        f"{float(effort.min())} / {float(saturation.min())}",
    )
    _check(
        float(velocity.min()) == STOCK_VELOCITY_LIMIT_RAD_S,
        f"velocity_limit is not the stock {STOCK_VELOCITY_LIMIT_RAD_S} rad/s: {float(velocity.min())}",
    )
    if is_torque:
        # The whole point of the variant: the actuator must not add a control law of its own.
        _check(_worst(torch.as_tensor(legs.stiffness)) == 0.0, "actuator stiffness is not 0")
        _check(_worst(torch.as_tensor(legs.damping)) == 0.0, "actuator damping is not 0")
        # A DelayedPDActuator / ActuatorNetMLP would hide dynamics behind the same zero gains.
        _check(
            type(legs).__name__ == "DCMotor",
            f"torque variant runs on {type(legs).__name__}, expected the stock DCMotor",
        )
    else:
        _check(isinstance(term, JointPositionAction), f"position task uses {type(term).__name__}")
        _check(
            float(torch.as_tensor(legs.stiffness).min()) == STOCK_STIFFNESS
            and float(torch.as_tensor(legs.damping).min()) == STOCK_DAMPING,
            "the baseline arm lost its stock gains: "
            f"{float(torch.as_tensor(legs.stiffness).min())} / {float(torch.as_tensor(legs.damping).min())}",
        )

    # --- 2./3. run ---------------------------------------------------------------------
    settle_steps = max(1, round(args_cli.settle_s / unwrapped.step_dt))
    check_steps = max(1, round(args_cli.check_s / unwrapped.step_dt))
    obs, _ = env.reset()
    _check(
        tuple(obs["policy"].shape) == (args_cli.num_envs, OBS_DIM),
        f"policy observation is {tuple(obs['policy'].shape)}, expected {(args_cli.num_envs, OBS_DIM)}",
    )

    ratio_sum, ratio_n, resets = 0.0, 0, 0
    last_log: dict[str, torch.Tensor] = {}
    envelope_bit, envelope_n, worst_gap = 0, 0, 0.0
    total_steps = settle_steps + check_steps
    for step in range(total_steps):
        if is_torque:
            action = _stance_pd_action(robot, term)
        else:
            action = torch.zeros((unwrapped.num_envs, NUM_LEG_JOINTS), device=unwrapped.device)
        obs, reward, terminated, truncated, extras = env.step(action)

        if is_torque:
            # ``JointAction._reset_idx`` zeroes ``_raw_actions`` for the environments it resets
            # but leaves ``_processed_actions`` at the torque commanded on the previous step, so
            # on those rows the identity below cannot hold by construction.  ``step()`` ends with
            # ``episode_length_buf[reset] = 0``, which is exactly the set to leave out: it says
            # nothing about the action layer, only about the reset bookkeeping.
            alive = unwrapped.episode_length_buf > 0
            if bool(alive.any()):
                expected = (term.raw_actions * term.cfg.action_scale).clamp(-TORQUE_LIMIT_NM, TORQUE_LIMIT_NM)
                _check_close(
                    term.processed_actions[alive], expected[alive], 1e-5,
                    f"step {step}: tau != clamp(scale * a) on {int(alive.sum())} non-reset envs",
                )
                _check(
                    _worst(term.processed_actions[alive]) <= TORQUE_LIMIT_NM + 1e-5,
                    f"step {step}: |tau| = {_worst(term.processed_actions[alive]):.2f} N.m exceeds the"
                    f" {TORQUE_LIMIT_NM} clamp",
                )
                # Not "applied == commanded": the stock envelope is allowed to take authority away,
                # exactly as it does on the baseline arm.  What must hold is that the applied torque
                # sits inside the band that envelope predicts, which no PD, filter or delay can do.
                # ``DCMotor.compute`` stores the velocity it used in ``_joint_vel``; reading it avoids
                # the one-substep lag that ``robot.data.joint_vel`` (post-step) would introduce.  The
                # remaining tolerance covers the solver drifting a few rad/s within one substep.
                low, high = _dcmotor_band(getattr(legs, "_joint_vel", robot.data.joint_vel))
                tolerance = 3.0
                # How far the *command* sits outside the band: that is authority the envelope takes
                # away, which the baseline loses too, so it is reported instead of failed.
                gap = torch.maximum(
                    term.processed_actions[alive] - high[alive], low[alive] - term.processed_actions[alive]
                ).clamp(min=0.0)
                # How far the *applied* torque sits outside it: this may never happen.  A PD, a filter
                # or a delay would show up here as a nonzero excess even with the gains set to zero.
                applied = robot.data.applied_torque[alive]
                excess = torch.maximum(applied - high[alive], low[alive] - applied).clamp(min=0.0)
                _check(
                    _worst(excess) <= tolerance,
                    f"step {step}: applied torque leaves the DCMotor band by {_worst(excess):.2f} N.m "
                    f"(tolerance {tolerance:.1f}) - hidden dynamics?",
                )
                # How often the envelope actually bound: reported, because it is a result of the
                # fairness decision, not a failure - and it bounds what the torque arm can do.
                envelope_bit += int((gap > 0.05).any(dim=1).sum() > 0)
                worst_gap = max(worst_gap, float(gap.max()))
                envelope_n += 1

        _check(
            bool(torch.isfinite(obs["policy"]).all()),
            f"step {step}: non-finite observation",
        )
        _check(bool(torch.isfinite(reward).all()), f"step {step}: non-finite reward")

        resets += int(terminated.sum().item()) + int(truncated.sum().item())
        # ``extras["log"]`` also carries the Episode_* statistics right after a reset, so only
        # the second half of the run (well past the warm-up gate) must have the force keys.
        log = extras.get("log") or {}
        if step < settle_steps:
            continue
        missing_keys = [key for key in REQUIRED_LOG_KEYS if key not in log]
        _check(not missing_keys, f"step {step}: extras['log'] misses {missing_keys}")
        _check(
            all(bool(torch.isfinite(torch.as_tensor(value)).all()) for value in log.values()),
            f"step {step}: non-finite force metric",
        )
        last_log = log
        ratio_sum += float(log["Force/total_over_weight"])
        ratio_n += 1

    # --- 4. did the ground actually carry the robot? ------------------------------------
    _check(ratio_n > 0, "no metric was ever logged - the warm-up gate never opened")
    ratio = ratio_sum / ratio_n
    feet = " ".join(f"{foot[:2]}={float(last_log[f'Force/{foot[:2]}_N']):.1f}N" for foot in FOOT_NAMES)
    print(f"  feet: {feet}", flush=True)
    print(
        f"  sum(GRF)/mg = {ratio:.3f} | CoP offset = {float(last_log['CoP/offset_m']) * 1000:.1f} mm"
        f" | duty(all4) = {float(last_log['Contact/duty_all4']):.2f}"
        f" | |tau|_rms = {float(last_log['Act/torque_rms']):.2f} N.m"
        f" | sat = {float(last_log['Act/sat_frac']) * 100:.1f} %",
        flush=True,
    )
    # The base-relative answer to "is it still standing?" - no ground, no absolute height, no
    # reward term: drift from this episode's own first frame, tilt from projected gravity, and
    # how much the base-to-foot extension changed.
    print(
        f"  Ref: dq_max = {float(last_log['Ref/dq_max_rad']):.3f} rad"
        f" | tilt = {float(last_log['Ref/tilt_rad']):.3f} rad"
        f" | d_ell = {float(last_log['Ref/d_ell_mm']):.1f} mm"
        f" | ell_0 = {float(last_log['Ref/ell_0_m']):.3f} m",
        flush=True,
    )
    print(f"  resets during the run: {resets}", flush=True)
    if is_torque:
        print(
            f"  stock DCMotor envelope bound the command on {envelope_bit}/{envelope_n} steps "
            f"(worst shortfall {worst_gap:.2f} N.m)",
            flush=True,
        )
    _check(
        0.80 <= ratio <= 1.20,
        f"the feet carry {ratio:.2f} x body weight, expected ~1.0 - is the robot standing?",
    )


def main() -> None:
    run_task(args_cli.task)
    print("\nAll V56 smoke checks passed.")
    if args_cli.ok_file is not None:
        # Written before the teardown, which is exactly the point: whatever Isaac Sim's atexit
        # hook does to the exit code, the file only appears when every check above returned.
        with open(args_cli.ok_file, "w") as handle:
            handle.write(args_cli.task)
    sys.stdout.flush()
    # Same teardown hang as in train.py: the render context does not close on this host.
    os._exit(0)


if __name__ == "__main__":
    main()
