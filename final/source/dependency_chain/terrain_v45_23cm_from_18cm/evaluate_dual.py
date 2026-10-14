"""V45 evaluator reporting physical traversal and strict-straight success."""
from pathlib import Path


path = Path("/home/ubuntu22/terrain_v35_straight_finetune/evaluate_straight.py")
wrapper = path.read_text()
wrapper = wrapper.replace(
    '"import eth_sata_reward_v17_stairs\\n"\n'
    '    "import terrain_v35_straight_finetune.config\\n"\n'
    '    "from terrain_v34_from_scratch_four_terrain.config import terrain_catalog",',
    '"import eth_sata_reward_v17_stairs\\n"\n'
    '    "import terrain_v45_23cm_from_18cm.config\\n"\n'
    '    "from copy import deepcopy",',
)
wrapper = wrapper.replace(
    "    selected = terrain_catalog()[case]",
    "    selected = deepcopy(generator.sub_terrains[case])",
)

final_exec = 'exec(compile(source, str(base), "exec"), {"__name__": "__main__", "__file__": str(base)})'
injection = r'''
source = source.replace(
    "    success = torch.zeros_like(active)\n    failed = torch.zeros_like(active)",
    "    success = torch.zeros_like(active)\n"
    "    physical_success = torch.zeros_like(active)\n"
    "    failed = torch.zeros_like(active)",
)
source = source.replace(
    "    stable_steps = torch.zeros(n, dtype=torch.long, device=raw.device)",
    "    stable_steps = torch.zeros(n, dtype=torch.long, device=raw.device)\n"
    "    physical_stable_steps = torch.zeros(n, dtype=torch.long, device=raw.device)",
)
strict_block = """            cleared = ((local[:, 0] > 3.5) & (feet_x.amin(dim=1) > 3.05)
                       & (robot.data.projected_gravity_b[:, 2] < -0.85)
                       & (local[:, 1].abs() < 0.25)
                       & (heading.abs() < 0.0872665))"""
dual_block = """            physical_cleared = ((local[:, 0] > 3.5) & (feet_x.amin(dim=1) > 3.05)
                                & (robot.data.projected_gravity_b[:, 2] < -0.85))
            cleared = (physical_cleared & (local[:, 1].abs() < 0.25)
                       & (heading.abs() < 0.0872665))"""
assert source.count(strict_block) == 1
source = source.replace(strict_block, dual_block)
source = source.replace(
    "            reached = valid & (stable_steps >= round(0.5 / dt))\n            success |= reached",
    """            reached = valid & (stable_steps >= round(0.5 / dt))
            physical_stable_steps = torch.where(
                valid & physical_cleared, physical_stable_steps + 1, torch.zeros_like(physical_stable_steps)
            )
            physical_reached = valid & (physical_stable_steps >= round(0.5 / dt))
            physical_success |= physical_reached
            success |= reached""",
)
source = source.replace(
    "              'success': int(success.sum()), 'terminated': int(failed.sum()),",
    "              'success': int(success.sum()), 'physical_success': int(physical_success.sum()),\n"
    "              'terminated': int(failed.sum()),",
)
assert "physical_success |= physical_reached" in source
exec(compile(source, str(base), "exec"), {"__name__": "__main__", "__file__": str(base)})
'''
assert wrapper.count(final_exec) == 1
wrapper = wrapper.replace(final_exec, injection)
assert "terrain_catalog()[case]" not in wrapper
exec(compile(wrapper, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
