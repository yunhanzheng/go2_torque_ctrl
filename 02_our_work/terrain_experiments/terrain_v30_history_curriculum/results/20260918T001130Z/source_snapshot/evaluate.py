"""Reuse the matched V28 evaluator with the V30 history task registered."""
from pathlib import Path


path = Path("/home/ubuntu22/terrain_v28_scratch_pilot/evaluate.py")
source = path.read_text().replace("terrain_v28_scratch_pilot.config", "terrain_v30_history_curriculum.config")
source = source.replace("from terrain_v30_history_curriculum.config import balanced_catalog", "from terrain_v28_scratch_pilot.config import balanced_catalog")
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
