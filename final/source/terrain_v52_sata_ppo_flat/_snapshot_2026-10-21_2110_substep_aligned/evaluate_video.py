"""V52 evaluator for video recording (PPO + flat ground + SATA's 9 rewards).

Mirrors terrain_v51_native_terrain/evaluate_video.py but for V52:
- V52 is PPO (OnPolicyRunner), not SAC (OffPolicyRunner): the policy is obtained via
  runner.get_inference_policy() and called as policy(obs) (MLPModel.forward with
  stochastic_output=False -> deterministic actions).
- No SAC replay_buffer_size override.
- env.close() flushes the RecordVideo mp4, then os._exit(0) skips the simulator
  teardown that hangs on Vulkan-less boxes.

Run (locally -- the local GPU supports Vulkan; the server's driver does not):
  python evaluate_video.py --task Isaac-Go2-V52-SataPpoFlat-v0 \
      --agent rsl_rl_sac_cfg_entry_point --num_envs 1 --seed 42 \
      --checkpoint <path>/model_2999.pt --video --video_length 1000 --headless \
      --cmd_x 1.0

``--cmd_x/--cmd_y/--cmd_yaw`` are written into ``env_cfg.commands.base_velocity.test_vel``
with ``use_test=True`` -- exactly SATA play.py's ``test.use_test`` / ``test.vel`` pair
(go2_torque_config.py:168-171).  The command term then overwrites the resampled command
every step (go2_torque.py:320-323), so the clip shows a single direction.  Without it the
growth-scaled ``SataVelocityCommand`` resamples x in ``[0.5 - 2*gs, 0.5 + 2*gs]`` every
5 s and the robot reverses direction mid-clip.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch

# --- Path bootstrap (mirrors train.py): source tree, shared adapters, and the V51 dir
# (for sata_alignment) so `import cli_args` and `terrain_v52_sata_ppo_flat.config`
# resolve from any CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (
    _SOURCE_ROOT,
    os.path.join(_SOURCE_ROOT, "dependency_chain"),
    os.path.join(_SOURCE_ROOT, "base_adapter"),
    os.path.join(_SOURCE_ROOT, "base_adapter", "initial_stair_source"),
    os.path.join(_SOURCE_ROOT, "terrain_v51_native_terrain"),
    _HERE,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

# local imports (cli_args carries the sa_16dof import shim for the shared server env)
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate V52 (PPO + flat) with video recording.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_folder", type=str, default=None, help="Folder to save recorded videos.")
parser.add_argument("--cmd_x", type=float, default=1.0, help="Pinned forward velocity command in m/s (SATA test.vel).")
parser.add_argument("--cmd_y", type=float, default=0.0, help="Pinned lateral velocity command in m/s.")
parser.add_argument("--cmd_yaw", type=float, default=0.0, help="Pinned yaw-rate command in rad/s.")
parser.add_argument("--dump_cfg", action="store_true", help="Print the effective event/termination/reward config and exit details.")
parser.add_argument("--train_mode", action="store_true",
                    help="Evaluate with training-time settings (DR on, growth restarts from 0). "
                         "Default is SATA play.py parity: freeze growth, no noise/friction/mass DR, 20 s episodes.")
parser.add_argument("--freeze_checkpoint", type=int, default=3000,
                    help="SATA cfg.test.checkpoint used to pin the growth counter in play mode.")
parser.add_argument("--fixed_cmd", action="store_true",
                    help="Hold --cmd_x constant instead of SATA play.py's CHANGE_VEL ramp "
                         "(1.0 -> 1.2 -> 1.4 -> 1.6 m/s, +0.2 every 100 steps, then held).")
parser.add_argument("--spawn_z", type=float, default=None,
                    help="Override the spawn base height. SATA uses 0.10, which puts the foot "
                         "spheres ~5 cm below the floor for the first frames; 0.19 rests them on it.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_sac_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments (includes --checkpoint, --resume, --run_name, --logger, ...)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import logging

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

# Import V52 config to register the task (also registers V51 via its import chain)
import terrain_v52_sata_ppo_flat.config  # noqa: F401
import isaaclab_tasks  # noqa: F401
import isaaclab.terrains as terrain_gen  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

# Check RSL-RL version
installed_version = metadata.version("rsl-rl-lib")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """Evaluate V52 (PPO) with video recording."""
    # override configurations with CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # SATA play.py (line 45): ``env_cfg.env.num_envs = min(num_envs, 1)`` - a single robot.
    play_mode = not args_cli.train_mode
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    elif play_mode:
        env_cfg.scene.num_envs = 1

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed

    # flat ground -> terrain seed is irrelevant, but keep for parity with V51
    if hasattr(env_cfg.scene.terrain.terrain_generator, "seed"):
        env_cfg.scene.terrain.terrain_generator.seed = args_cli.seed if args_cli.seed else 42

    # SATA play.py MOVE_CAMERA (lines 109-112): camera at ``robot_pos + (1, 1, 1)``
    # looking at ``robot_pos``.  IsaacLab's viewer expresses both as offsets from the
    # asset root, so the equivalent is eye=(1,1,1), lookat=(0,0,0).
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (1.0, 1.0, 1.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # Set up video recording directory
    checkpoint_path = Path(args_cli.checkpoint)
    if args_cli.video_folder:
        video_dir = Path(args_cli.video_folder)
    else:
        video_dir = checkpoint_path.parent / f"V52_seed{args_cli.seed or 42}_video"
    video_dir.mkdir(parents=True, exist_ok=True)

    # SATA play.py: test.use_test = True + test.vel = [1.0, 0.0, 0.0, 0.0].  The command
    # term resamples as usual and then overwrites every env with test_vel before the
    # reward/observation are computed (go2_torque.py:320-323).
    env_cfg.commands.base_velocity.use_test = True
    env_cfg.commands.base_velocity.test_vel = (
        args_cli.cmd_x, args_cli.cmd_y, args_cli.cmd_yaw, 0.0,
    )

    # === Faithful SATA play.py (`legged_gym/scripts/play.py:46-65`) ===
    #   episode_length_s      = 20                     (play overrides 10 -> 20)
    #   noise.add_noise       = False
    #   domain_rand.randomize_friction   = False
    #   domain_rand.push_robots          = False
    #   domain_rand.randomize_base_mass  = False
    #   terrain.mesh_type     = 'plane'                (already flat in V52)
    # Everything else (spawn joints/xy, joint limits, TG model) stays as in training.
    freeze_steps = -1
    if play_mode:
        env_cfg.episode_length_s = 20.0
        env_cfg.events.physics_material = None
        env_cfg.events.push_robot = None
        env_cfg.events.push_robots = None
        env_cfg.events.add_base_mass = None
        env_cfg.observations.policy.base_lin_vel.params["add_noise"] = False
        # play.py line 57: ``env_cfg.terrain.mesh_type = 'plane'`` (training terrain is trimesh).
        _gen = env_cfg.scene.terrain.terrain_generator
        _gen.sub_terrains = {"plane": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
        _gen.num_rows = 1
        _gen.num_cols = 1
        env_cfg.scene.terrain.max_init_terrain_level = None
        # play.py line 55: ``env_cfg.commands.heading_command = True``.  The heading servo
        # runs after the test-vel overwrite inside SataVelocityCommand.compute.
        env_cfg.commands.base_velocity.heading_command = True
        freeze_steps = 24 * int(args_cli.freeze_checkpoint)

    if args_cli.spawn_z is not None:
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, float(args_cli.spawn_z))
        print(f"[INFO] spawn base height overridden to {args_cli.spawn_z:.3f} m (SATA default 0.10)", flush=True)

    if args_cli.dump_cfg:
        from isaaclab.managers import (
            EventTermCfg,
            ObservationTermCfg,
            RewardTermCfg,
            TerminationTermCfg,
        )

        print(f"[CFG] decimation={env_cfg.decimation} episode={env_cfg.episode_length_s} dt={env_cfg.sim.dt}", flush=True)
        print(f"[CFG] play_mode={play_mode} freeze_step_count={freeze_steps}", flush=True)
        for n, v in vars(env_cfg.events).items():
            if isinstance(v, EventTermCfg):
                print(f"[CFG] event {v.mode:8s} {n} -> {v.func.__name__}", flush=True)
        for n, v in vars(env_cfg.terminations).items():
            if isinstance(v, TerminationTermCfg):
                print(f"[CFG] term  {n} -> {v.func.__name__}", flush=True)
        print("[CFG] rewards", [(n, v.weight) for n, v in vars(env_cfg.rewards).items() if isinstance(v, RewardTermCfg)], flush=True)
        print(
            "[CFG] obs",
            [n for n, v in vars(env_cfg.observations.policy).items() if isinstance(v, ObservationTermCfg)],
            flush=True,
        )
        _tgen = env_cfg.scene.terrain.terrain_generator
        print(
            f"[CFG] terrain rows={_tgen.num_rows} cols={_tgen.num_cols} size={_tgen.size} "
            f"sub={[(n, round(c.proportion, 3)) for n, c in _tgen.sub_terrains.items()]} "
            f"border={_tgen.border_width} curriculum={_tgen.curriculum} "
            f"h/v_scale={_tgen.horizontal_scale}/{_tgen.vertical_scale} slope_thr={_tgen.slope_threshold}",
            flush=True,
        )

    # create environment -- only pay for the render pipeline when a video is requested
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if args_cli.dump_cfg:
        _terrain = env.unwrapped.scene.terrain
        if _terrain.terrain_origins is not None:
            _z = _terrain.terrain_origins[..., 2]
            print(
                f"[INFO] terrain origin z: min={_z.min():.3f} max={_z.max():.3f} "
                f"mean={_z.mean():.3f} nonzero={(  _z != 0).float().mean():.2f}",
                flush=True,
            )
        print(
            f"[INFO] env origin z (first 8): {[round(v, 3) for v in _terrain.env_origins[:8, 2].tolist()]}",
            flush=True,
        )

    # Pin the Gompertz counter *before* the first step: SATA play sets
    # step_count = num_steps_per_env * test.checkpoint so general_scale is constant 0.789
    # (current_freq 178.9 Hz, torque-limit scale 0.852).  Without this the policy would
    # be evaluated at gs~0 (30 % torque, 100 Hz), which no trained checkpoint can survive.
    if freeze_steps >= 0:
        env.unwrapped._sata_freeze_step_count = freeze_steps
        term = env.unwrapped.action_manager.get_term("joint_pos")
        term._update_growth()
        gs = term.general_scale
        print(
            f"[INFO] SATA play growth frozen at step_count={freeze_steps}: "
            f"gs={gs:.4f} torque_scale={term.current_torque_limit_scale:.4f} "
            f"freq={100.0 + 100.0 * gs:.1f} Hz",
            flush=True,
        )

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording (skipped for plain numeric checks)
    if args_cli.video:
        video_kwargs = {
            "video_folder": str(video_dir),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print(f"[INFO] Video folder: {video_dir}")
        print(f"[INFO] Video length: {args_cli.video_length} steps")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---- PPO runner (V52 is OnPolicyRunner, unlike V51's SAC OffPolicyRunner) ----
    from rsl_rl.runners import OnPolicyRunner

    log_dir = str(checkpoint_path.parent)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    # Load checkpoint
    print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)

    # deterministic inference policy: MLPModel.forward(obs, stochastic_output=False)
    policy = runner.get_inference_policy(device=env.device)

    # Run evaluation
    print(f"[INFO] Running evaluation for {args_cli.video_length} steps...")
    # SATA calls reset_idx() inside __init__ (legged_robot.py:132); IsaacLab does not, and
    # without this the clip replays the frozen default pose instead of SATA's randomised spawn.
    env.reset()
    obs = env.get_observations()  # TensorDict

    # ---- Diagnostic handle on the command term (the pinning happens inside it) ----
    base_env = env.unwrapped
    cmd_term = base_env.command_manager.get_term("base_velocity")
    print(
        f"[INFO] test.use_test=True test.vel=({args_cli.cmd_x:+.2f}, "
        f"{args_cli.cmd_y:+.2f}, {args_cli.cmd_yaw:+.2f}, 0.00)",
        flush=True,
    )

    start_time = time.time()
    robot = base_env.scene["robot"]
    print("[INFO] joint_names:", robot.data.joint_names, flush=True)
    print("[INFO] body_names:", robot.data.body_names, flush=True)
    print("[INFO] body_z:", [round(v, 3) for v in robot.data.body_pos_w[0, :, 2].tolist()], flush=True)
    print("[INFO] default_joint_pos:", [round(v, 3) for v in robot.data.default_joint_pos[0].tolist()], flush=True)
    print("[INFO] joint_pos(reset):", [round(v, 3) for v in robot.data.joint_pos[0].tolist()], flush=True)
    # --- contact-sensor diagnostic (which body registers the illegal contact?) ---
    cs = base_env.scene.sensors.get("contact_forces")
    if cs is not None:
        print("[INFO] contact bodies:", cs.body_names, flush=True)
        # ``base_contact`` is disabled since V52 aligns its termination set with SATA
        # (dof-limit / inverted / time-out only), so the term cfg may be absent.
        if "base_contact" in base_env.termination_manager.active_terms:
            _hc = base_env.termination_manager.get_term_cfg("base_contact")
            _sc = _hc.params.get("sensor_cfg")
            print("[INFO] base_contact sensor_cfg body_ids:", getattr(_sc, "body_ids", "n/a"),
                  "body_names:", getattr(_sc, "body_names", "n/a"), flush=True)
        nf = cs.data.net_forces_w_history[:, 0].norm(dim=-1)
        print(
            "[INFO] contact forces |",
            {n: round(f, 1) for n, f in zip(cs.body_names, nf[0].tolist()) if f > 0.5},
            flush=True,
        )
    resets = 0
    vx_acc = 0.0
    vx_n = 0
    term_hits: dict[str, int] = {}
    # SATA play.py CHANGE_VEL (lines 87-102): vel_x starts at 1.0, +0.2 every 100 steps,
    # and is held once it exceeds 1.5 (so the real sequence is 1.0 / 1.2 / 1.4 / 1.6).
    # The sign of the ramp follows --cmd_x so a negative command ramps downwards in the
    # same way (-1.0 / -1.2 / -1.4 / -1.6) instead of ramping back through zero.
    cmd_ramp_x = float(args_cli.cmd_x)
    cmd_ramp_step = 0.2 if args_cli.cmd_x >= 0.0 else -0.2
    for step in range(args_cli.video_length):
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, infos = env.step(actions)
        robot = base_env.scene["robot"]
        resets += int(dones.sum().item())
        vx_acc += robot.data.root_lin_vel_b[0, 0].item()
        vx_n += 1
        # per-term termination accounting (which SATA condition actually fires?)
        tm = base_env.termination_manager
        for name in tm.active_terms:
            hit = int(tm.get_term(name).sum().item())
            if hit:
                term_hits[name] = term_hits.get(name, 0) + hit
        if play_mode and not args_cli.fixed_cmd:
            if step % 100 == 0:
                if abs(cmd_ramp_x) > 1.5:
                    cmd_ramp_step = 0.0
                cmd_ramp_x += cmd_ramp_step
                # CommandManager deep-copies the term cfg, so the *term's* copy is what
                # ``SataVelocityCommand.compute`` reads every step.
                cmd_term.cfg.test_vel = (
                    cmd_ramp_x, args_cli.cmd_y, args_cli.cmd_yaw, 0.0,
                )
        if step < 5 and cs is not None:
            _bf = cs.data.net_forces_w_history[:, :, [0], :].norm(dim=-1)
            _foot_ids = [i for i, n in enumerate(robot.data.body_names) if n.endswith("_foot")]
            _fz = robot.data.body_pos_w[0, _foot_ids, 2]
            print(
                f"    [diag step {step}] base_force={_bf.max().item():.2f}N "
                f"foot_z={[round(v, 4) for v in _fz.tolist()]} "
                f"min={_fz.min().item():+.4f}",
                flush=True,
            )

        if step % 100 == 0 or step == args_cli.video_length - 1:
            body_z = robot.data.body_pos_w[0, :, 2]
            print(
                f"  Step {step}/{args_cli.video_length}"
                f" | cmd_x={cmd_term.vel_command_b[0, 0].item():+.3f}"
                f" cmd_yaw={cmd_term.vel_command_b[0, 2].item():+.3f}"
                f" vx_body={robot.data.root_lin_vel_b[0, 0].item():+.3f}"
                f" vx_mean={vx_acc / max(vx_n, 1):+.3f}"
                f" root_x={robot.data.root_pos_w[0, 0].item():+.3f}"
                f" root_z={robot.data.root_pos_w[0, 2].item():+.3f}"
                f" body_z_min={body_z.min().item():+.3f}"
                f" resets={resets} terms={term_hits}",
                flush=True,
            )
            vx_acc, vx_n = 0.0, 0

    elapsed = time.time() - start_time
    print(f"[INFO] Evaluation completed in {elapsed:.1f}s")

    # close the environment -- this flushes the RecordVideo mp4 to disk.
    env.close()
    print(f"[INFO] Video saved under: {video_dir}", flush=True)
    # simulation_app.close() hangs forever on Vulkan-less boxes (render cleanup spins at
    # 99% CPU). The mp4 is already written by env.close(), so hard-exit to guarantee the
    # process terminates and frees the GPU.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
