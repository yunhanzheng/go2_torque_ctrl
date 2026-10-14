"""Deterministic straight-line evaluation for V34/V35 checkpoints."""
from pathlib import Path


base = Path("/home/ubuntu22/eval_eth_sata_v17_stairs.py")
source = base.read_text()


def replace(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)


replace(
    "import eth_sata_reward_v17_stairs",
    "import eth_sata_reward_v17_stairs\n"
    "import terrain_v35_straight_finetune.config\n"
    "from terrain_v34_from_scratch_four_terrain.config import terrain_catalog",
)
start = source.index("    # Fixed geometry for both checkpoints")
end = source.index("    env_cfg.curriculum.terrain_levels = None", start)
source = source[:start] + """    output = os.environ['STRAIGHT_OUTPUT']
    case = os.environ.get('TERRAIN_CASE', 'plane')
    difficulty = float(os.environ.get('TERRAIN_DIFFICULTY', '0.0'))
    generator = env_cfg.scene.terrain.terrain_generator
    selected = terrain_catalog()[case]
    selected.proportion = 1.0
    generator.sub_terrains = {case: selected}
    generator.num_rows = 1
    generator.num_cols = 8
    generator.curriculum = False
    generator.seed = 7
    generator.difficulty_range = (difficulty, difficulty)
""" + source[end:]
replace(
    "    env_cfg.episode_length_s = 20.0",
    """    env_cfg.episode_length_s = 20.0
    command = env_cfg.commands.base_velocity
    command.heading_command = True
    command.rel_heading_envs = 1.0
    command.rel_standing_envs = 0.0
    command.heading_control_stiffness = 0.5
    command.ranges.lin_vel_x = (0.5, 0.5)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (-0.5, 0.5)
    command.ranges.heading = (0.0, 0.0)""",
)
replace("    # load previously trained model", "    agent_cfg.algorithm.replay_buffer_size = 4096\n    # load previously trained model")
replace("env_cfg.viewer.eye = (2.2, -2.2, 1.35)", "env_cfg.viewer.eye = (0.8, -3.2, 1.5)")
replace("        env = gym.wrappers.RecordVideo(env, **video_kwargs)", "        env = gym.wrappers.RecordVideo(env, fps=50, **video_kwargs)")
replace(
    "    max_x = torch.zeros(n, device=raw.device)",
    """    max_x = torch.zeros(n, device=raw.device)
    max_abs_y = torch.zeros(n, device=raw.device)
    max_abs_heading = torch.zeros(n, device=raw.device)
    final_y = torch.zeros(n, device=raw.device)
    final_heading = torch.zeros(n, device=raw.device)""",
)
replace(
    "            valid = active & ~dones.bool()",
    """            heading = torch.atan2(torch.sin(robot.data.heading_w), torch.cos(robot.data.heading_w))
            valid = active & ~dones.bool()
            max_abs_y[valid] = torch.maximum(max_abs_y[valid], local[valid, 1].abs())
            max_abs_heading[valid] = torch.maximum(max_abs_heading[valid], heading[valid].abs())
            final_y[valid] = local[valid, 1]
            final_heading[valid] = heading[valid]""",
)
replace(
    "            cleared = (local[:, 0] > 3.5) & (feet_x.amin(dim=1) > 3.05) & (robot.data.projected_gravity_b[:, 2] < -0.85)",
    """            cleared = ((local[:, 0] > 3.5) & (feet_x.amin(dim=1) > 3.05)
                       & (robot.data.projected_gravity_b[:, 2] < -0.85)
                       & (local[:, 1].abs() < 0.25)
                       & (heading.abs() < 0.0872665))""",
)
replace(
    "trace.append({'step': timestep, 'x': float(local[0,0]), 'z': float(local[0,2]), 'done': bool(dones[0]), 'active': bool(active[0])})",
    "trace.append({'step': timestep, 'x': float(local[0,0]), 'y': float(local[0,1]), 'z': float(local[0,2]), 'heading_deg': float(torch.rad2deg(heading[0])), 'done': bool(dones[0]), 'active': bool(active[0])})",
)
old_result = """result = {'checkpoint': resume_path, 'direction': direction, 'height_m': height, 'num_envs': n,
              'steps': timestep, 'dt': dt, 'success': int(success.sum()), 'terminated': int(failed.sum()),
              'other_no_success': int((~success & ~failed).sum()), 'max_x_m': max_x.tolist(),
              'trace_env0': trace, 'criterion': 'first episode; base x > 3.5m, all feet x > 3.05m, gravity_z < -0.85 for 0.5s; no reset'}"""
new_result = """result = {'checkpoint': resume_path, 'num_envs': n, 'steps': timestep, 'dt': dt,
              'success': int(success.sum()), 'terminated': int(failed.sum()),
              'other_no_success': int((~success & ~failed).sum()), 'max_x_m': max_x.tolist(),
              'max_abs_y_m': max_abs_y.tolist(), 'final_y_m': final_y.tolist(),
              'max_abs_heading_deg': torch.rad2deg(max_abs_heading).tolist(),
              'final_heading_deg': torch.rad2deg(final_heading).tolist(),
              'mean_max_abs_y_m': float(max_abs_y.mean()),
              'mean_final_abs_y_m': float(final_y.abs().mean()),
              'mean_max_abs_heading_deg': float(torch.rad2deg(max_abs_heading).mean()),
              'trace_env0': trace, 'terrain': case, 'difficulty': difficulty,
              'criterion': 'first episode; x>3.5m; abs(y)<0.25m; abs(heading)<5deg; upright for 0.5s; no reset'}"""
replace(old_result, new_result)
replace("'[STAIR_RESULT]'", "'[STRAIGHT_RESULT]'")
exec(compile(source, str(base), "exec"), {"__name__": "__main__", "__file__": str(base)})
