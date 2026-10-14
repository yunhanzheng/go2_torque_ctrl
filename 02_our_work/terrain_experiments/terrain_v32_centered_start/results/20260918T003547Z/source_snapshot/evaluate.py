"""Reuse the matched evaluator with the V32 history tasks registered."""
from pathlib import Path


path = Path("/home/ubuntu22/terrain_v28_scratch_pilot/evaluate.py")
source = path.read_text().replace("terrain_v28_scratch_pilot.config", "terrain_v32_centered_start.config")
source = source.replace("from terrain_v32_centered_start.config import balanced_catalog", "from terrain_v28_scratch_pilot.config import balanced_catalog")
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
