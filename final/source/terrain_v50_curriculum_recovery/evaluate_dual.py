"""Run the established physical/strict evaluator with the V50 task registered."""
from pathlib import Path


path = Path("/home/ubuntu22/terrain_v49_generalize_23cm/evaluate_dual.py")
source = path.read_text()
old = 'new = "import terrain_v49_generalize_23cm.config"'
new = 'new = "import terrain_v50_curriculum_recovery.config"'
assert source.count(old) == 1
source = source.replace(old, new)
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
