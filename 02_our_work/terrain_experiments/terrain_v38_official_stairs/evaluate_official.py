"""Use the strict straight-line evaluator with V38's original stair geometry."""
from pathlib import Path


path = Path("/home/ubuntu22/terrain_v35_straight_finetune/evaluate_straight.py")
source = path.read_text()
source = source.replace(
    '"import eth_sata_reward_v17_stairs\\n"\n'
    '    "import terrain_v35_straight_finetune.config\\n"\n'
    '    "from terrain_v34_from_scratch_four_terrain.config import terrain_catalog",',
    '"import eth_sata_reward_v17_stairs\\n"\n'
    '    "import terrain_v38_official_stairs.config\\n"\n'
    '    "from copy import deepcopy",',
)
source = source.replace(
    "    selected = terrain_catalog()[case]",
    "    selected = deepcopy(generator.sub_terrains[case])",
)
assert "terrain_catalog()[case]" not in source
assert "import terrain_v38_official_stairs.config" in source
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
