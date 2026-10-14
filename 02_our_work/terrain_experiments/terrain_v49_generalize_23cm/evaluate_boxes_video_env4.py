"""V49 dual evaluator with configurable terrain seed, columns, and camera env."""
from pathlib import Path


path = Path("/home/ubuntu22/terrain_v45_23cm_from_18cm/evaluate_dual.py")
source = path.read_text()
old = "terrain_v45_23cm_from_18cm.config"
new = "terrain_v49_generalize_23cm.config"
assert source.count(old) == 1
source = source.replace(old, new)

# Allow the recorder to reproduce either the 32-environment audit or a truly
# isolated one-environment terrain instance.
needle = 'assert "physical_success |= physical_reached" in source\nexec(compile(source, str(base), "exec"), {"__name__": "__main__", "__file__": str(base)})'
replacement = (
    'assert "physical_success |= physical_reached" in source\n'
    'source = source.replace("generator.num_cols = 8", "generator.num_cols = int(os.environ.get(\\\"TERRAIN_COLS\\\", \\\"8\\\"))")\n'
    'source = source.replace("generator.seed = 7", "generator.seed = int(os.environ.get(\\\"TERRAIN_SEED\\\", \\\"7\\\"))")\n'
    'source = source.replace("env_cfg.viewer.env_index = 0", "env_cfg.viewer.env_index = int(os.environ.get(\\\"VIDEO_ENV_INDEX\\\", \\\"0\\\"))")\n'
    'gym_line = \'    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)\'\n'
    'terrain_retarget = \'    env_cfg.commands.base_velocity.debug_vis = False\\n\' + gym_line + \'\\n    base_env = env.unwrapped\\n    terrain_type_index = int(os.environ.get("TERRAIN_TYPE_INDEX", "-1"))\\n    if terrain_type_index >= 0:\\n        target_origin = base_env.scene.terrain.terrain_origins[0, terrain_type_index].clone()\\n        base_env.scene.env_origins[0] = target_origin\\n        base_env.scene.terrain.env_origins[0] = target_origin\\n        base_env.scene.terrain.terrain_types[0] = terrain_type_index\\n        base_env.reset()\'\n'
    'terrain_retarget += \'\\n    visible_env_index = int(os.environ.get("VISIBLE_ENV_INDEX", "-1"))\\n    if visible_env_index >= 0:\\n        hidden_env_ids = [index for index in range(base_env.num_envs) if index != visible_env_index]\\n        base_env.scene["robot"].set_visibility(False, env_ids=hidden_env_ids)\\n        base_env.scene["robot"].set_visibility(True, env_ids=[visible_env_index])\'\n'
    'assert source.count(gym_line) == 1\n'
    'source = source.replace(gym_line, terrain_retarget)\n'
    'assert "TERRAIN_SEED" in source and "VIDEO_ENV_INDEX" in source and "VISIBLE_ENV_INDEX" in source\n'
    'exec(compile(source, str(base), "exec"), {"__name__": "__main__", "__file__": str(base)})'
)
assert source.count(needle) == 1
source = source.replace(needle, replacement)
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
