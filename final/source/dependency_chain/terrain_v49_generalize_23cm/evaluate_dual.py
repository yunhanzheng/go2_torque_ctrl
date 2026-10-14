"""Run the physical/strict evaluator with the V49 task registered."""
from pathlib import Path


path = Path("/home/ubuntu22/terrain_v48_23cm_final/evaluate_dual.py")
source = path.read_text()
old = 'new = "import terrain_v48_23cm_final.config"'
new = 'new = "import terrain_v49_generalize_23cm.config"'
assert source.count(old) == 1
source = source.replace(old, new)
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
