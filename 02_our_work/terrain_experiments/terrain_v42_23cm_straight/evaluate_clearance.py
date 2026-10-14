"""Generate the V42 evaluator, then inject measured foot clearance."""
from pathlib import Path


path = Path("/home/ubuntu22/terrain_v35_straight_finetune/evaluate_straight.py")
wrapper = path.read_text()
wrapper = wrapper.replace(
    '"import eth_sata_reward_v17_stairs\\n"\n'
    '    "import terrain_v35_straight_finetune.config\\n"\n'
    '    "from terrain_v34_from_scratch_four_terrain.config import terrain_catalog",',
    '"import eth_sata_reward_v17_stairs\\n"\n'
    '    "import terrain_v42_23cm_straight.config\\n"\n'
    '    "from copy import deepcopy",',
)
wrapper = wrapper.replace(
    "    selected = terrain_catalog()[case]",
    "    selected = deepcopy(generator.sub_terrains[case])",
)

final_exec = 'exec(compile(source, str(base), "exec"), {"__name__": "__main__", "__file__": str(base)})'
injection = r'''
source = source.replace(
    "    max_x = torch.zeros(n, device=raw.device)\n    max_abs_y",
    "    max_x = torch.zeros(n, device=raw.device)\n"
    "    max_foot_clearance = torch.zeros((n, 4), device=raw.device)\n"
    "    max_abs_y",
)
source = source.replace(
    "            feet_x = robot.data.body_pos_w[:, feet_ids, 0] - raw.scene.env_origins[:, 0:1]",
    """            feet_pos = robot.data.body_pos_w[:, feet_ids]
            feet_x = feet_pos[:, :, 0] - raw.scene.env_origins[:, 0:1]
            ray_hits = raw.scene.sensors['height_scanner'].data.ray_hits_w
            ray_valid = torch.isfinite(ray_hits).all(dim=-1)
            safe_ray_xy = torch.nan_to_num(ray_hits[:, :, :2], nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)
            distance = torch.cdist(feet_pos[:, :, :2], safe_ray_xy)
            distance = distance.masked_fill(~ray_valid[:, None, :], float('inf'))
            nearest = distance.argmin(dim=-1)
            ground_z = torch.gather(ray_hits[:, :, 2], 1, nearest)
            clearance = torch.nan_to_num(feet_pos[:, :, 2] - ground_z, nan=0.0, posinf=0.0, neginf=0.0)
            max_foot_clearance[valid] = torch.maximum(max_foot_clearance[valid], clearance[valid])""",
)
source = source.replace(
    "              'max_abs_y_m': max_abs_y.tolist(), 'final_y_m': final_y.tolist(),",
    """              'max_abs_y_m': max_abs_y.tolist(), 'final_y_m': final_y.tolist(),
              'max_foot_clearance_m': max_foot_clearance.tolist(),
              'mean_episode_max_foot_clearance_m': float(max_foot_clearance.amax(dim=1).mean()),
              'median_episode_max_foot_clearance_m': float(max_foot_clearance.amax(dim=1).median()),
              'global_max_foot_clearance_m': float(max_foot_clearance.max()),""",
)
assert "max_foot_clearance[valid]" in source
exec(compile(source, str(base), "exec"), {"__name__": "__main__", "__file__": str(base)})
'''
assert wrapper.count(final_exec) == 1
wrapper = wrapper.replace(final_exec, injection)
assert "terrain_catalog()[case]" not in wrapper
exec(compile(wrapper, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
