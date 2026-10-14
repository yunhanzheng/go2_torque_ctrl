"""Matched forward traversal evaluation for the V28 official-observation policy."""
from pathlib import Path


base = Path("/home/ubuntu22/eval_eth_sata_v17_stairs.py")
source = base.read_text()


def replace(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)


replace(
    "import eth_sata_reward_v17_stairs",
    "import eth_sata_reward_v17_stairs\nimport terrain_v28_scratch_pilot.config\n"
    "from terrain_v28_scratch_pilot.config import balanced_catalog",
)
start = source.index("    # Fixed geometry for both checkpoints")
end = source.index("    env_cfg.curriculum.terrain_levels = None", start)
source = source[:start] + """    case = os.environ['TERRAIN_CASE']
    difficulty = float(os.environ['TERRAIN_DIFFICULTY'])
    output = os.environ['TERRAIN_OUTPUT']
    generator = env_cfg.scene.terrain.terrain_generator
    catalog = balanced_catalog(generator.sub_terrains)
    sub_terrain = catalog[case]
    sub_terrain.proportion = 1.0
    generator.sub_terrains = {case: sub_terrain}
    generator.num_rows = 1
    generator.num_cols = 8
    generator.curriculum = False
    generator.seed = 7
    generator.difficulty_range = (difficulty, difficulty)
""" + source[end:]
replace(
    "    env_cfg.episode_length_s = 20.0",
    """    env_cfg.episode_length_s = 20.0
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)""",
)
replace("    # load previously trained model", "    agent_cfg.algorithm.replay_buffer_size = 4096\n    # load previously trained model")
replace("env_cfg.viewer.eye = (2.2, -2.2, 1.35)", "env_cfg.viewer.eye = (0.7, -2.7, 1.05)")
replace(
    "            valid = active & ~dones.bool()",
    """            outside = local[:, 1].abs() > 0.75
            failed |= active & outside
            valid = active & ~dones.bool() & ~outside""",
)
replace("            active &= ~reached & ~dones.bool()", "            active &= ~reached & ~dones.bool() & ~outside")
replace("'direction': direction, 'height_m': height", "'terrain': case, 'difficulty': difficulty, 'terrain_seed': 7")
replace(
    "first episode; base x > 3.5m, all feet x > 3.05m, gravity_z < -0.85 for 0.5s; no reset",
    "first episode; fixed 0.5 m/s forward command; base x > 3.5m, all feet x > 3.05m, gravity_z < -0.85 for 0.5s; abs(y)<=0.75m; no reset",
)
exec(compile(source, str(base), "exec"), {"__name__": "__main__", "__file__": str(base)})
