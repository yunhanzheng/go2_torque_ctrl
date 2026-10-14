"""Matched deterministic evaluator for V35 with explicit corridor checking."""
from pathlib import Path


base = Path("/home/ubuntu22/eval_eth_sata_v17_stairs.py")
source = base.read_text()


def replace(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)


replace(
    "import eth_sata_reward_v17_stairs",
    "import eth_sata_reward_v17_stairs\nimport terrain_v35_heading_corridor.config\n"
    "from terrain_v34_from_scratch_four_terrain.config import terrain_catalog",
)
start = source.index("    # Fixed geometry for both checkpoints")
end = source.index("    env_cfg.curriculum.terrain_levels = None", start)
source = source[:start] + """    case = os.environ['TERRAIN_CASE']
    difficulty = float(os.environ['TERRAIN_DIFFICULTY'])
    output = os.environ['TERRAIN_OUTPUT']
    generator = env_cfg.scene.terrain.terrain_generator
    sub_terrain = terrain_catalog()[case]
    sub_terrain.proportion = 1.0
    generator.sub_terrains = {case: sub_terrain}
    generator.num_rows = 1
    generator.num_cols = 8
    generator.curriculum = False
    terrain_seed = int(os.environ.get('TERRAIN_SEED', '7'))
    generator.seed = terrain_seed
    generator.difficulty_range = (difficulty, difficulty)
""" + source[end:]
replace("    env_cfg.episode_length_s = 20.0", """    env_cfg.episode_length_s = 20.0
    command = env_cfg.commands.base_velocity
    heading_feedback = os.environ.get('V35_EVAL_HEADING', '0') == '1'
    command.heading_command = heading_feedback
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 1.0 if heading_feedback else 0.0
    command.heading_control_stiffness = 0.3
    command.ranges.lin_vel_x = (0.5, 0.5)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (-0.3, 0.3) if heading_feedback else (0.0, 0.0)
    command.ranges.heading = (0.0, 0.0)""")
replace("    # load previously trained model", "    agent_cfg.algorithm.replay_buffer_size = 4096\n    # load previously trained model")
replace(
    "    env_cfg.viewer.env_index = 0",
    "    env_cfg.viewer.env_index = int(os.environ.get('V35_VIEW_ENV', '0'))",
)
replace("            valid = active & ~dones.bool()", """            outside = local[:, 1].abs() > 0.75
            failed |= active & outside
            valid = active & ~dones.bool() & ~outside""")
replace("            active &= ~reached & ~dones.bool()", "            active &= ~reached & ~dones.bool() & ~outside")
replace("'direction': direction, 'height_m': height", "'terrain': case, 'difficulty': difficulty, 'terrain_seed': terrain_seed")
replace(
    "first episode; base x > 3.5m, all feet x > 3.05m, gravity_z < -0.85 for 0.5s; no reset",
    "first episode; fixed 0.5 m/s; base x > 3.5m, all feet x > 3.05m, gravity_z < -0.85 for 0.5s; abs(y)<=0.75m; no reset",
)
exec(compile(source, str(base), "exec"), {"__name__": "__main__", "__file__": str(base)})
