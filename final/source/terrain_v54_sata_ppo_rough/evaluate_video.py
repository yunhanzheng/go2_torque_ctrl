"""V54 evaluator for video recording (PPO + IsaacLab Go2 rough terrain + SATA's 9 rewards).

Mirrors terrain_v52_sata_ppo_flat/evaluate_video.py but for V54:
- registers the rough-terrain task (6 sub-terrains, 10x20 tiles, terrain curriculum).
- --common_step_counter pins SATA's Gompertz clock to the checkpoint's training point
  (``_sata_freeze_step_count``), so the torque limit and control frequency during
  evaluation match training.  Without it a fresh env starts at counter=0 (general_scale
  ~0.13 -> ~9 Nm, 113 Hz); model_3000 of the 23-09-03 run was trained at counter
  = 3000 * 24 = 72000 (general_scale 0.789 -> 20 Nm, 178.9 Hz).
- --terrain_level overrides max_init_terrain_level (default 0 = easiest row).
- --fix_terrain_level pins the row (difficulty = row / num_rows) and --terrain_type pins the
  sub-terrain (one whole column block per type: pyramid_stairs 0-3, pyramid_stairs_inv 4-7,
  boxes 8-11, random_rough 12-15, hf_pyramid_slope 16-17, hf_pyramid_slope_inv 18-19), so a
  clip can target one exact terrain at one exact difficulty.
- prints deterministic-policy statistics (resets, per-term termination counts, vx tracking,
  distance from spawn) so a checkpoint can be judged without PPO's exploration noise.
- env.close() flushes the RecordVideo mp4, then os._exit(0) skips the simulator teardown
  that hangs on Vulkan-less boxes.

Run (locally -- the local GPU supports Vulkan; the server's driver does not):
  python evaluate_video.py --task Isaac-Go2-V54-SataPpoRough-v0 \
      --agent rsl_rl_sac_cfg_entry_point --num_envs 1 --seed 42 \
      --checkpoint <path>/model_3000.pt --common_step_counter 72000 \
      --video --video_length 1000 --headless
Statistics only (no render surface, ~10x faster, --num_envs can be raised freely):
  python evaluate_video.py --task Isaac-Go2-V54-SataPpoRough-v0 \
      --agent rsl_rl_sac_cfg_entry_point --num_envs 32 --seed 42 \
      --checkpoint <path>/model_3000.pt --common_step_counter 72000 \
      --video_length 2200 --headless
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch

# --- Path bootstrap (mirrors train.py): source tree, shared adapters, V51 (sata_alignment),
# V52 (sata_rewards) and this V54 directory so imports resolve from any CWD.
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

# local imports (cli_args carries the sa_16dof import shim for the shared server env)
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate V54 (PPO + rough terrain) with video recording.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=1000, help="Number of policy steps to evaluate (= video length when --video).")
parser.add_argument("--video_folder", type=str, default=None, help="Folder to save recorded videos.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_sac_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--common_step_counter",
    type=int,
    default=None,
    help="Pin SATA's Gompertz clock to this step count (env._sata_freeze_step_count) so the "
    "torque limit and control frequency match the checkpoint's training point, e.g. 12000 "
    "for model_500 or 72000 for model_3000 (= num_steps_per_env * iteration).",
)
parser.add_argument(
    "--log_falls",
    action="store_true",
    default=False,
    help="Record roll/pitch/velocity at the instant each fell_over fires (the state is still "
    "pre-reset inside TerminationManager.reset) and print a fall-mode breakdown.",
)
parser.add_argument(
    "--obs_noise",
    action="store_true",
    default=False,
    help="Keep observation noise + 10%% observation loss during evaluation.  Off by default: "
    "SATA's play.py sets cfg.noise.add_noise=False which also zeroes loss_rate "
    "(go2_torque.py:59-60), so a 'deterministic' evaluation must not inject either.",
)
parser.add_argument(
    "--sata_actor",
    type=str,
    default=None,
    help="Evaluate a SATA legged_gym checkpoint (rsl_rl 1.x ActorCritic, 60->512->256->128->12 "
    "ELU) instead of a V54 checkpoint.  Answers whether SATA's own policy survives here.",
)
parser.add_argument(
    "--terrain_level",
    type=int,
    default=None,
    help="Override max_init_terrain_level (0 = easiest row, higher = harder tiles).",
)
parser.add_argument(
    "--fix_terrain_level",
    type=int,
    default=None,
    help="Force every env onto this terrain row and re-spawn.  ``max_init_terrain_level`` is "
    "only an upper bound, so this is the way to pin a single difficulty for a clip.",
)
parser.add_argument(
    "--terrain_type",
    type=str,
    default=None,
    help="Pin every env to a column of this sub-terrain (e.g. boxes, pyramid_stairs_inv).  "
    "The generator hands out whole columns per type in sub_terrains order by proportion "
    "(4/4/4/4/2/2 of the 20 columns), and num_envs=1 always lands on column 0, so without "
    "this flag every clip is shot on pyramid_stairs.  Combine with --fix_terrain_level for "
    "the row and --freeze_terrain so a reset cannot re-row it.",
)
parser.add_argument(
    "--terrain_seed",
    type=int,
    default=42,
    help="Seed for the rough-terrain mesh.  Kept separate from --seed so changing the spawn "
    "randomness does not rebuild every tile (config.py pins it to 42 for training).",
)
parser.add_argument(
    "--freeze_terrain",
    action="store_true",
    default=False,
    help="Disable the terrain-level curriculum but keep the rough grid, so --terrain_level / "
    "--fix_terrain_level survive episode resets.  Without it every reset re-runs the "
    "promotion term and can re-row the env mid-clip (V54 rev D: out of the pit after 4 m).",
)
parser.add_argument(
    "--plane",
    action="store_true",
    default=False,
    help="Swap the rough curriculum for a single infinite plane, like SATA's play.py "
    "(terrain.mesh_type = 'plane').  The 8 m tiles cannot hold a long run at walking speed.",
)
parser.add_argument(
    "--isaaclab_play",
    action="store_true",
    default=False,
    help="Use IsaacLab's stock CommandsCfg for the whole episode instead of SATA's growth-scaled "
    "sampler: 10 s resampling, vx/vy/wz in +-1.0, heading in +-pi with the 0.5 heading servo, "
    "2%% standing envs.  Official play.py keeps the training config, so this is that config.",
)
parser.add_argument(
    "--fixed_speed",
    type=float,
    default=1.0,
    help="Constant forward speed pinned by --isaaclab_play.  Heading still resamples, so the "
    "robot keeps turning towards a new target every 10 s.",
)
parser.add_argument(
    "--no_time_out",
    action="store_true",
    default=False,
    help="Drop the 10 s episode time-out so a long clip runs without a mid-shot reset.  "
    "Fall and joint-limit terminations stay active, so a real failure still shows up.",
)
parser.add_argument(
    "--command",
    type=float,
    nargs=4,
    default=None,
    metavar=("VX", "VY", "WZ", "HEADING"),
    help="Pin SATA's test velocity like play.py: sets use_test/test_vel and turns the heading "
    "servo on, so the policy gets a constant command instead of the 5 s random resample. "
    "E.g. --command 0.4 0 0 0 walks straight along the spawn heading.",
)
parser.add_argument(
    "--sky_plain",
    action="store_true",
    default=False,
    help="Drop the DomeLight HDR and use a flat colour instead.  The kloofendal sky texture in "
    "velocity_env_cfg.py returns 404 from the asset server, so on the cluster the dome adds "
    "no light and the terrain renders black.",
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=None,
    help="Playback rate for the recorded mp4.  IsaacLab stamps 200 fps while the policy runs at "
    "~161.5 Hz, so the default already plays 1.24x fast; lower it to stretch the same clip.",
)
parser.add_argument(
    "--sky_intensity",
    type=float,
    default=1500.0,
    help="DomeLight intensity used together with --sky_plain (SATA/IsaacLab default is 750).",
)
parser.add_argument(
    "--reward_probe",
    action="store_true",
    default=False,
    help="Accumulate each reward term's per-step value (weight * raw term, independent of dt) "
    "and compare it against SATA's final iteration.  A per-term gap then points at the "
    "mechanism instead of the total.",
)
parser.add_argument(
    "--log_command",
    action="store_true",
    default=False,
    help="Print the velocity command every time it is resampled (5 s) together with the base "
    "heading.  The command lives in the base frame, so a nonzero wz is the robot turning, "
    "not the command pointing away from the spawn heading.",
)
parser.add_argument(
    "--joint_probe",
    action="store_true",
    default=False,
    help="Report per-joint position range and mean soft-limit excursion.  Locates which joint "
    "sits on its 90%% soft limit (a stretched hip abduction joint explains lateral toppling).",
)
parser.add_argument(
    "--no_push",
    action="store_true",
    default=False,
    help="Disable the push_robot event.  A/B against the default push interval of 4 s isolates "
    "'cannot recover from perturbations' from 'the gait itself topples'.",
)
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

# Import V54 config to register the task (its import chain also registers V51/V52)
import terrain_v54_sata_ppo_rough.config  # noqa: F401
import isaaclab_tasks  # noqa: F401
import isaaclab.terrains as terrain_gen  # noqa: E402
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config


class FixedSpeedHeadingCommand(UniformVelocityCommand):
    """IsaacLab's stock command with the linear velocity pinned.

    ``_update_command`` only rewrites ``wz`` (heading servo) and zeroes the standing envs, so
    overriding ``_resample_command`` keeps vx/vy constant for the whole episode while the
    rest of the behaviour stays official.
    """

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.vel_command_b[env_ids, 0] = self.cfg.fixed_vx
        self.vel_command_b[env_ids, 1] = self.cfg.fixed_vy
        if self.cfg.heading_command:
            # Pin the target so the servo holds the spawn heading instead of turning.
            self.heading_target[env_ids] = self.cfg.fixed_heading


@configclass
class FixedSpeedHeadingCommandCfg(UniformVelocityCommandCfg):
    """Official command cfg plus the constant linear velocity."""

    class_type: type = FixedSpeedHeadingCommand
    fixed_vx: float = 1.0
    fixed_vy: float = 0.0
    fixed_heading: float = 0.0

logger = logging.getLogger(__name__)

# Check RSL-RL version
installed_version = metadata.version("rsl-rl-lib")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _load_sata_actor(path: str, device) -> callable:
    """Load SATA's legged_gym ``ActorCritic`` policy as a plain inference callable.

    SATA saves ``{'model_state_dict': {'actor.0.weight': [512, 60], ...}}`` (rsl_rl 1.x,
    ELU, three hidden layers, mean action only).  Feeding it this env's 60-D observation
    separates "the environment is wrong" from "the training did not converge": if SATA's
    own policy falls here too, the bug is in the env/control/observation port.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    act = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
    dims = [
        act["0.weight"].shape[1],
        act["0.weight"].shape[0],
        act["2.weight"].shape[0],
        act["4.weight"].shape[0],
        act["6.weight"].shape[0],
    ]
    net = torch.nn.Sequential(
        torch.nn.Linear(dims[0], dims[1]),
        torch.nn.ELU(),
        torch.nn.Linear(dims[1], dims[2]),
        torch.nn.ELU(),
        torch.nn.Linear(dims[2], dims[3]),
        torch.nn.ELU(),
        torch.nn.Linear(dims[3], dims[4]),
    )
    net.load_state_dict(act)
    net.to(device).eval()
    print(
        f"[INFO] Loaded SATA actor: obs {dims[0]} -> actions {dims[4]} "
        f"hidden {dims[1]}/{dims[2]}/{dims[3]} (ELU)",
        flush=True,
    )

    def _policy(obs):
        x = obs["policy"] if hasattr(obs, "keys") else obs
        return net(x)

    return _policy


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """Evaluate V54 (PPO + rough) with video recording."""
    # override configurations with CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # ``--device`` moves PhysX off cuda:0.  On the cluster GPU0 runs the training job and is
    # full, while the renderer is pinned to GPU1 (see run_video_server.sh), so a clip has to
    # put *both* halves on GPU1 or it competes with the training for the last few hundred MB.
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device
        print(f"[INFO] sim.device := {args_cli.device}", flush=True)

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed

    # rough terrain: keep the training mesh reproducible and independent of --seed, so a new
    # seed only rerolls the spawn/command randomness instead of regenerating every tile.
    if hasattr(env_cfg.scene.terrain.terrain_generator, "seed"):
        env_cfg.scene.terrain.terrain_generator.seed = args_cli.terrain_seed
        print(f"[INFO] terrain_generator.seed := {args_cli.terrain_seed} (training value 42)", flush=True)

    if args_cli.terrain_level is not None:
        env_cfg.scene.terrain.max_init_terrain_level = args_cli.terrain_level
        print(f"[INFO] max_init_terrain_level := {args_cli.terrain_level}", flush=True)

    if args_cli.freeze_terrain:
        # Only the curriculum is switched off; the grid (and --fix_terrain_level) stays.
        env_cfg.curriculum.terrain_levels = None
        print("[INFO] terrain curriculum := off (row pinned for the whole clip)", flush=True)

    if args_cli.plane:
        gen = env_cfg.scene.terrain.terrain_generator
        gen.sub_terrains = {"plane": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
        gen.num_rows = 1
        gen.num_cols = 1
        env_cfg.scene.terrain.max_init_terrain_level = None
        env_cfg.curriculum.terrain_levels = None
        print("[INFO] terrain := single plane (SATA play.py mesh_type='plane')", flush=True)

    if args_cli.no_time_out:
        env_cfg.terminations.time_out = None
        print("[INFO] terminations.time_out := None (no 10 s episode reset)", flush=True)

    if args_cli.isaaclab_play:
        # Official play.py does not overwrite the command term, so it runs exactly this cfg,
        # except that the linear velocity is pinned to a constant.
        from isaaclab.envs.mdp import UniformVelocityCommandCfg

        env_cfg.commands.base_velocity = FixedSpeedHeadingCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.0,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=False,
            fixed_vx=args_cli.fixed_speed,
            fixed_vy=0.0,
            fixed_heading=0.0,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
        )
        print(
            f"[INFO] commands := IsaacLab default play, vx pinned at {args_cli.fixed_speed:.2f} m/s, "
            "heading pinned at 0 (no turning, no standing envs)",
            flush=True,
        )

    if args_cli.command is not None and not args_cli.isaaclab_play:
        vx, vy, wz, heading = (float(v) for v in args_cli.command)
        env_cfg.commands.base_velocity.use_test = True
        env_cfg.commands.base_velocity.test_vel = (vx, vy, wz, heading)
        env_cfg.commands.base_velocity.heading_command = True
        print(
            f"[INFO] command pinned: vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} "
            f"heading={heading:+.2f} rad (heading servo on)",
            flush=True,
        )

    if args_cli.sky_plain:
        sky = env_cfg.scene.sky_light.spawn
        sky.texture_file = None
        sky.color = (0.75, 0.75, 0.75)
        sky.intensity = args_cli.sky_intensity
        print(
            f"[INFO] sky light: flat colour {sky.color} at {sky.intensity} nits (HDR dropped)",
            flush=True,
        )

    if not args_cli.obs_noise:
        # loss_rate in that case.  Without this a "deterministic" evaluation still feeds the
        # policy noisy observations and replays a stale frame 10% of the time.
        for _term_cfg in vars(env_cfg.observations.policy).values():
            if hasattr(_term_cfg, "params") and "add_noise" in _term_cfg.params:
                _term_cfg.params["add_noise"] = False
                _term_cfg.params["loss_rate"] = 0.0
                _term_cfg.params["noise_level"] = 0.0
        print("[INFO] observation noise + 10% obs loss disabled (SATA play parity)", flush=True)

    if args_cli.no_push:
        # the EventManager lists this term as ``push_robots`` (plural)
        env_cfg.events.push_robots = None
        print("[INFO] push_robots event disabled", flush=True)

    # Configure viewer for follow-camera (third-person view)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.0, -2.5, 1.5)  # behind and above the robot
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)  # look at robot center

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # Set up video recording directory (only when a clip is requested)
    checkpoint_path = Path(args_cli.checkpoint or args_cli.sata_actor)
    if args_cli.video_folder:
        video_dir = Path(args_cli.video_folder)
    else:
        video_dir = checkpoint_path.parent / f"V54_seed{args_cli.seed or 42}_video"
    if args_cli.video:
        video_dir.mkdir(parents=True, exist_ok=True)

    # create environment (only ask for a render surface when the clip is actually recorded --
    # rendering dominates the runtime of a statistics-only sweep)
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # ---- terrain difficulty actually sampled for this episode --------------------------
    # ``terrain_levels`` is the row index; the generator maps it linearly onto
    # ``difficulty_range``, and each sub-terrain turns that into its own step heights.
    _terrain = env.unwrapped.scene.terrain
    _gen = env.unwrapped.cfg.scene.terrain.terrain_generator

    # ---- pin row / sub-terrain type for the clip --------------------------------------
    def _column_names(gen) -> list[str]:
        """Sub-terrain name of every column, in ``TerrainGenerator``'s assignment order.

        ``_generate_curriculum_terrains`` walks the columns left to right and picks the first
        type whose cumulative proportion exceeds ``col / num_cols`` (plus its 1e-3 rounding
        guard), so with 20 columns and 0.2/0.2/0.2/0.2/0.1/0.1 the blocks are 0-3, 4-7, 8-11,
        12-15, 16-17, 18-19.
        """
        names = list(gen.sub_terrains.keys())
        props = [float(c.proportion) for c in gen.sub_terrains.values()]
        total = sum(props)
        cums, acc = [], 0.0
        for p in props:
            acc += p / total
            cums.append(acc)
        out = []
        for col in range(gen.num_cols):
            frac = col / gen.num_cols + 0.001
            out.append(next((n for n, c in zip(names, cums) if frac < c), names[-1]))
        return out

    _col_names = _column_names(_gen)
    _pin_row = args_cli.fix_terrain_level
    _pin_type = args_cli.terrain_type
    if _pin_row is not None or _pin_type is not None:
        _col = None
        if _pin_type is not None:
            if _pin_type not in _col_names:
                raise SystemExit(
                    f"unknown --terrain_type {_pin_type!r}; this grid holds {sorted(set(_col_names))}"
                )
            _col = _col_names.index(_pin_type)
            _terrain.terrain_types[:] = _col
        if _pin_row is not None:
            _terrain.terrain_levels[:] = int(_pin_row)
        _terrain.env_origins[:] = _terrain.terrain_origins[_terrain.terrain_levels, _terrain.terrain_types]
        env.unwrapped.reset()
        # ``_reset_idx`` runs ``curriculum_manager.compute`` first, so ``terrain_levels_vel``
        # moves the row (a fresh spawn counts as "moved up").  Write the pin back afterwards:
        # the robot already sits on the requested tile, only the bookkeeping was overwritten.
        if _col is not None:
            _terrain.terrain_types[:] = _col
        if _pin_row is not None:
            _terrain.terrain_levels[:] = int(_pin_row)
        _terrain.env_origins[:] = _terrain.terrain_origins[_terrain.terrain_levels, _terrain.terrain_types]
        print(
            f"[INFO] terrain pinned: row={int(_terrain.terrain_levels[0])} "
            f"column={int(_terrain.terrain_types[0])} "
            f"({_col_names[int(_terrain.terrain_types[0])]}), robots re-spawned",
            flush=True,
        )

    _row = int(_terrain.terrain_levels[0])
    _col = int(_terrain.terrain_types[0])
    _lo, _hi = _gen.difficulty_range
    _difficulty = (_row + 0.5) / _gen.num_rows * (_hi - _lo) + _lo
    print(
        f"[INFO] terrain: row={_row} (of {_gen.num_rows - 1}) column={_col} "
        f"({_col_names[_col]}) difficulty~{_difficulty:.2f} "
        f"origin=({_terrain.env_origins[0, 0]:.1f}, {_terrain.env_origins[0, 1]:.1f}) "
        f"max_init_level={env.unwrapped.cfg.scene.terrain.max_init_terrain_level}",
        flush=True,
    )

    # Pin SATA's Gompertz clock to the checkpoint's training point *before* the first step.
    # ``sata_growth_scale`` reads ``_sata_freeze_step_count`` first and only falls back to
    # ``common_step_counter`` / ``_sata_physics_steps`` when it is unset, so writing
    # common_step_counter (as this script used to) had no effect: the policy ran at
    # gs~0.13 (~9 Nm, 113 Hz) instead of the value it was trained with.
    if args_cli.common_step_counter is not None:
        env.unwrapped._sata_freeze_step_count = int(args_cli.common_step_counter)
        term = env.unwrapped.action_manager.get_term("joint_pos")
        term._update_growth()
        print(
            f"[INFO] SATA growth frozen at step_count={args_cli.common_step_counter}: "
            f"gs={term.general_scale:.4f} torque_scale={term.current_torque_limit_scale:.4f} "
            f"freq={100.0 + 100.0 * term.general_scale:.1f} Hz",
            flush=True,
        )

    # ---- deterministic-evaluation statistics ------------------------------------------
    # ``TerminationManager.reset()`` is the only place where the per-term flags still line up
    # with the env that just finished (``step()`` respawns terminated envs immediately), so
    # wrap it once and accumulate exact per-term firing counts.  One reset can fire several
    # terms (e.g. fell_over + dof_pos_limit), so the counts may exceed the reset total.
    eval_env = env.unwrapped
    term_manager = eval_env.termination_manager
    term_names = list(term_manager._term_names)
    term_counts = {name: 0 for name in term_names}
    resets_per_env = torch.zeros(eval_env.num_envs, dtype=torch.long, device=eval_env.device)
    total_resets = [0]
    eval_robot = eval_env.scene["robot"]
    fell_idx = term_names.index("fell_over") if "fell_over" in term_names else None
    # (step, env_id, episode_step, roll_deg, pitch_deg, vx, vy, wz, height_above_origin, cmd_x)
    fall_log: list[tuple] = []
    current_step = [0]

    def _record_falls(local_rows, ids):
        gids = local_rows if isinstance(ids, slice) else torch.as_tensor(ids)[local_rows]
        quat = eval_robot.data.root_quat_w[gids]  # w, x, y, z
        w, x, y, z = quat.unbind(dim=1)
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        vel = eval_robot.data.root_lin_vel_b[gids]
        ang = eval_robot.data.root_ang_vel_b[gids]
        height = eval_robot.data.root_pos_w[gids, 2] - eval_env.scene.env_origins[gids, 2]
        cmd = eval_env.command_manager.get_command("base_velocity")[gids, 0]
        ep_step = eval_env.episode_length_buf[gids]
        for k in range(gids.numel()):
            fall_log.append(
                (
                    current_step[0],
                    int(gids[k].item()),
                    int(ep_step[k].item()),
                    math.degrees(float(roll[k])),
                    math.degrees(float(pitch[k])),
                    float(vel[k, 0]),
                    float(vel[k, 1]),
                    float(ang[k, 2]),
                    float(height[k]),
                    float(cmd[k]),
                )
            )

    _orig_term_reset = term_manager.reset
    _orig_term_compute = term_manager.compute

    def _recording_term_compute():
        # ``_reset_idx`` calls ``scene.reset`` *before* ``termination_manager.reset``, so by
        # the time the reset hook runs the robot is already back at the spawn pose.  This
        # hook runs right after the termination check and still sees the falling state.
        out = _orig_term_compute()
        if args_cli.log_falls and fell_idx is not None:
            fell_rows = term_manager._term_dones[:, fell_idx].nonzero(as_tuple=True)[0]
            if fell_rows.numel() > 0:
                _record_falls(fell_rows, slice(None))
        return out

    def _counting_term_reset(env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        rows = term_manager._last_episode_dones[ids]
        for i, name in enumerate(term_names):
            term_counts[name] += int(rows[:, i].sum().item())
        resets_per_env[ids] += 1
        total_resets[0] += int(rows.shape[0])
        return _orig_term_reset(env_ids)

    term_manager.compute = _recording_term_compute
    term_manager.reset = _counting_term_reset

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording (optional: statistics-only runs skip RecordVideo + rendering)
    if args_cli.video:
        if args_cli.video_fps:
            env.unwrapped.metadata["render_fps"] = args_cli.video_fps
            print(f"[INFO] Video playback fps := {args_cli.video_fps}", flush=True)
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

    # ---- policy ---------------------------------------------------------------------
    if args_cli.sata_actor:
        # SATA's own actor, no rsl_rl runner needed.  clip_actions must stay None: SATA
        # trains with clip_actions = 100 (legged_robot_config.py:168), i.e. no clipping.
        print(f"[INFO] Evaluating SATA checkpoint: {args_cli.sata_actor}")
        policy = _load_sata_actor(args_cli.sata_actor, env.device)
    else:
        # ---- PPO runner (V54 uses OnPolicyRunner) ----
        from rsl_rl.runners import OnPolicyRunner

        log_dir = str(checkpoint_path.parent)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)

        # Load checkpoint
        print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
        # Inference needs the actor only, and a checkpoint from before rev H carries a 60-D
        # critic while the rev H critic takes 247-D (60 state + 187 height map).  Loading with
        # ``strict=True`` would die on that size mismatch, so the critic / optimizer / iteration
        # entries are skipped: nothing downstream of ``get_inference_policy`` reads them.
        runner.load(
            args_cli.checkpoint,
            load_cfg={"actor": True, "critic": False, "optimizer": False,
                      "iteration": False, "rnd": False},
        )

        # deterministic inference policy: MLPModel.forward(obs, stochastic_output=False)
        policy = runner.get_inference_policy(device=env.device)

    # Run evaluation
    print(f"[INFO] Running evaluation for {args_cli.video_length} steps...")
    # IsaacLab does not reset in __init__; without this the whole clip replays the frozen
    # default pose instead of SATA's randomised spawn (legged_robot.py:132 + 398-442).
    env.reset()
    obs = env.get_observations()  # TensorDict

    # the spawn reset is not part of the statistics
    for _name in term_names:
        term_counts[_name] = 0
    resets_per_env.zero_()
    total_resets[0] = 0

    robot = eval_env.scene["robot"]

    def _zeros():
        return torch.zeros(eval_env.num_envs, device=eval_env.device)

    stat_steps, vx_sum, err_sum, cmd_sum, height_sum, max_dist = (
        _zeros(),
        _zeros(),
        _zeros(),
        _zeros(),
        _zeros(),
        _zeros(),
    )
    # ``RewardManager.compute`` runs once per physics substep (1-2 per control period) and
    # overwrites ``_step_reward`` each time, so a hook is required to see every substep.
    # ``_step_reward`` holds ``raw_term * weight`` (the manager divides the dt scaling back
    # out); multiplying by the dt passed in reconstructs the reward increment that is summed
    # into the episode return.  Dividing the total by the simulated seconds then matches
    # SATA's ``rew_*`` (``episodic_sum / max_episode_length_s``) exactly.
    reward_mgr = eval_env.reward_manager
    reward_term_names = list(reward_mgr._term_names)
    reward_probe_sum = torch.zeros(len(reward_term_names), device=eval_env.device)
    reward_probe_time = 0.0
    if args_cli.reward_probe:
        _orig_reward_compute = reward_mgr.compute

        def _probe_reward_compute(dt, **kwargs):
            value = _orig_reward_compute(dt=dt, **kwargs)
            reward_probe_sum.add_(reward_mgr._step_reward.sum(dim=0) * dt)
            return value

        reward_mgr.compute = _probe_reward_compute

    joint_names = list(robot.joint_names)
    n_joints = len(joint_names)
    jpos_min = torch.full((n_joints,), float("inf"), device=eval_env.device)
    jpos_max = torch.full((n_joints,), -float("inf"), device=eval_env.device)
    jpos_abs_sum = torch.zeros(n_joints, device=eval_env.device)
    jexc_sum = torch.zeros(n_joints, device=eval_env.device)
    joint_probe_rows = 0

    start_time = time.time()
    # ``step_dt`` is fixed at construction, but SataControlFreqEnv rewrites
    # ``cfg.decimation`` every step (1-2 substeps, growing policy frequency), so accumulate
    # the real simulated time instead of multiplying by the stale attribute.
    sim_dt_sum = 0.0
    for step in range(args_cli.video_length):
        current_step[0] = step
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, infos = env.step(actions)
        if args_cli.log_command and step % 100 == 0:
            _cmd = eval_env.command_manager.get_command("base_velocity")[0]
            _heading = math.degrees(float(robot.data.heading_w[0]))
            print(
                f"  [CMD] step {step:4d} t={sim_dt_sum:5.2f}s "
                f"vx={_cmd[0]:+.2f} vy={_cmd[1]:+.2f} wz={_cmd[2]:+.2f} "
                f"heading={_heading:+.0f} deg",
                flush=True,
            )
        sim_dt_sum += eval_env.physics_dt * eval_env.cfg.decimation
        if args_cli.reward_probe:
            reward_probe_time += (
                eval_env.physics_dt * eval_env.cfg.decimation * eval_env.num_envs
            )
        if args_cli.joint_probe:
            jp = robot.data.joint_pos
            jpos_min = torch.minimum(jpos_min, jp.min(dim=0).values)
            jpos_max = torch.maximum(jpos_max, jp.max(dim=0).values)
            jpos_abs_sum += jp.abs().sum(dim=0)
            _lim = robot.data.joint_pos_limits
            _mid = 0.5 * (_lim[..., 0] + _lim[..., 1])
            _half = 0.5 * (_lim[..., 1] - _lim[..., 0]) * 0.9
            _exc = -(jp - (_mid - _half)).clip(max=0.0) + (jp - (_mid + _half)).clip(min=0.0)
            jexc_sum += _exc.sum(dim=0)
            joint_probe_rows += eval_env.num_envs

        # accumulate only for envs that survived this step (terminated ones already respawned)
        done_mask = dones.bool()
        keep = (~done_mask).float()
        vx = robot.data.root_lin_vel_b[:, 0]
        cmd_x = eval_env.command_manager.get_command("base_velocity")[:, 0]
        dist = torch.norm(
            robot.data.root_pos_w[:, :2] - eval_env.scene.env_origins[:, :2], dim=1
        )
        max_dist[done_mask] = 0.0  # a fresh episode starts over
        max_dist = torch.maximum(max_dist, dist * keep)
        stat_steps += keep
        vx_sum += vx * keep
        cmd_sum += cmd_x * keep
        err_sum += (vx - cmd_x).abs() * keep
        height_sum += (robot.data.root_pos_w[:, 2] - eval_env.scene.env_origins[:, 2]) * keep

        if step % 100 == 0:
            print(f"  Step {step}/{args_cli.video_length}", flush=True)

    elapsed = time.time() - start_time
    print(f"[INFO] Evaluation completed in {elapsed:.1f}s")

    # ---- deterministic summary --------------------------------------------------------
    n_per_env = stat_steps.clamp(min=1.0)
    env_steps = args_cli.video_length * eval_env.num_envs
    print("\n[EVAL] ===== deterministic policy (no exploration noise) =====")
    n_steps = max(1, args_cli.video_length)
    print(
        f"[EVAL] {args_cli.video_length} steps x {eval_env.num_envs} envs "
        f"({sim_dt_sum:.1f} s simulated, "
        f"{1000.0 * sim_dt_sum / n_steps:.2f} ms per policy step -> "
        f"{n_steps / max(sim_dt_sum, 1e-9):.1f} Hz)"
    )
    print(
        f"[EVAL] resets: {total_resets[0]} total | "
        f"{total_resets[0] / eval_env.num_envs:.2f} per env | "
        f"{100.0 * total_resets[0] / env_steps:.2f} % of env-steps"
    )
    print(
        "[EVAL] terminations (may overlap): "
        + " | ".join(
            f"{name} {count} ({100.0 * count / env_steps:.2f} %)"
            for name, count in term_counts.items()
        )
    )
    if args_cli.log_falls:
        if not fall_log:
            print("[EVAL] fall log: no fell_over events")
        else:
            n = len(fall_log)
            rolls = [abs(r[3]) for r in fall_log]
            pitches = [abs(r[4]) for r in fall_log]
            side = sum(1 for r in fall_log if abs(r[3]) > abs(r[4]))
            ep_steps = [r[2] for r in fall_log]
            avg_hz = max(1.0, n_steps / max(sim_dt_sum, 1e-9))
            ep_secs = sorted(s / avg_hz for s in ep_steps)
            early = sum(1 for s in ep_secs if s < 1.0)
            print(
                f"[EVAL] fall log: {n} events | side roll {side} ({100.0 * side / n:.0f} %) | "
                f"pitch {n - side} ({100.0 * (n - side) / n:.0f} %)"
            )
            print(
                f"[EVAL] fall |roll| mean {sum(rolls) / n:.0f} max {max(rolls):.0f} deg | "
                f"|pitch| mean {sum(pitches) / n:.0f} max {max(pitches):.0f} deg | "
                f"vx mean {sum(r[5] for r in fall_log) / n:+.2f} m/s | "
                f"height mean {sum(r[8] for r in fall_log) / n:.3f} m"
            )
            print(
                f"[EVAL] fall within episode: <1 s {early} ({100.0 * early / n:.0f} %) | "
                f"median {ep_secs[n // 2]:.1f} s | max {ep_secs[-1]:.1f} s"
            )
            worst = sorted(fall_log, key=lambda r: max(abs(r[3]), abs(r[4])), reverse=True)[:8]
            print("[EVAL] worst falls (step env ep_step roll pitch vx vy wz h cmd):")
            for r in worst:
                print(
                    f"        {r[0]:5d} env{r[1]:3d} ep{r[2]:4d} roll {r[3]:+7.1f} "
                    f"pitch {r[4]:+7.1f} vx {r[5]:+5.2f} vy {r[6]:+5.2f} wz {r[7]:+6.2f} "
                    f"h {r[8]:.3f} cmd {r[9]:.2f}"
                )
    print(
        f"[EVAL] vx: mean {vx_sum.div(n_per_env).mean():.2f} m/s | "
        f"mean |vx-cmd_x| {err_sum.div(n_per_env).mean():.2f} m/s | "
        f"mean cmd_x {cmd_sum.div(n_per_env).mean():.2f} m/s"
    )
    print(
        f"[EVAL] max distance from tile origin: mean {max_dist.mean():.2f} m | "
        f"max {max_dist.max():.2f} m"
    )
    print(
        f"[EVAL] base height above tile origin: mean {height_sum.div(n_per_env).mean():.3f} m"
    )
    if args_cli.reward_probe and reward_probe_time > 0.0:
        # SATA's final iteration (wandb output.log, iter 2999): ``rew_x`` is already
        # ``sum(raw*weight*dt) / max_episode_length_s``, i.e. reward per simulated second,
        # so it can be compared against ``sum(reward_buf) / simulated_seconds`` directly --
        # no episode-length or dt rescaling involved.
        _SATA_REW = {
            "forward": 7.6433,
            "head_height": 2.5014,
            "moving_y": 3.0335,
            "moving_yaw": 2.4137,
            "soft_dof_pos_limits": -0.0308,
            "motor_fatigue": -3.0028,
            "dof_acc": -2.3714,
            "roll": -0.3888,
            "lin_vel_z": -0.1552,
        }
        mean_terms = (reward_probe_sum / reward_probe_time).cpu()
        print("\n[EVAL] ===== reward per simulated second vs SATA iter 2999 =====")
        print(f"[EVAL] {'term':24s} {'ours':>10s} {'SATA':>10s} {'ours/SATA':>10s}")
        ours_total = 0.0
        sata_total = 0.0
        for i, name in enumerate(reward_term_names):
            ours = float(mean_terms[i])
            ours_total += ours
            if name not in _SATA_REW:
                print(f"[EVAL] {name:24s} {ours:10.4f} {'-':>10s} {'-':>10s}")
                continue
            ref = _SATA_REW[name]
            sata_total += ref
            ratio = ours / ref if abs(ref) > 1e-9 else float("nan")
            print(f"[EVAL] {name:24s} {ours:10.4f} {ref:10.4f} {ratio:10.3f}")
        print(
            f"[EVAL] {'TOTAL (9 SATA terms)':24s} {ours_total:10.4f} {sata_total:10.4f} "
            f"{ours_total / sata_total if abs(sata_total) > 1e-9 else float('nan'):10.3f}"
        )
    if args_cli.joint_probe and joint_probe_rows:
        print("\n[EVAL] ===== per-joint position range (rad) and soft-limit excursion =====")
        print(
            f"[EVAL] {'joint':22s} {'mean|q|':>9s} {'min':>9s} {'max':>9s} "
            f"{'mean exc':>10s} {'exc %':>7s}"
        )
        for i, name in enumerate(joint_names):
            lo, hi = robot.data.joint_pos_limits[0, i, 0].item(), robot.data.joint_pos_limits[0, i, 1].item()
            _mid = 0.5 * (lo + hi)
            _half = 0.5 * (hi - lo)
            print(
                f"[EVAL] {name:22s} {float(jpos_abs_sum[i]) / joint_probe_rows:9.3f} "
                f"{float(jpos_min[i]):9.3f} {float(jpos_max[i]):9.3f} "
                f"{float(jexc_sum[i]) / joint_probe_rows:10.4f} "
                f"{100.0 * float(jexc_sum[i]) / joint_probe_rows / max(2.0 * _half, 1e-9):6.1f}%"
            )
    if total_resets[0]:
        print(f"[EVAL] resets per env: {resets_per_env.tolist()}")

    # Flush the mp4 explicitly.  RecordVideo only auto-closes once recorded_frames exceeds
    # video_length, so a loop of exactly video_length steps relies on close(); env.close()
    # can hang on Vulkan-less boxes and be killed by the os._exit() below, truncating the
    # clip (observed: 480 of 1000 frames written).
    if getattr(env, "video_recorder", None) is not None:
        env.video_recorder.close_video()
    # close the environment -- this flushes the RecordVideo mp4 to disk.
    env.close()
    if args_cli.video:
        print(f"[INFO] Video saved under: {video_dir}", flush=True)
    # simulation_app.close() hangs forever on Vulkan-less boxes (render cleanup spins at
    # 99% CPU). The mp4 is already written by env.close(), so hard-exit to guarantee the
    # process terminates and frees the GPU.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
