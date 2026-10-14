"""Run the dual physical/strict evaluator with the V46 task registered."""
from pathlib import Path


path = Path("/home/ubuntu22/terrain_v45_23cm_from_18cm/evaluate_dual.py")
source = path.read_text()
old = "import terrain_v45_23cm_from_18cm.config"
new = "import terrain_v46_23cm_from_20cm.config"
assert source.count(old) == 1
source = source.replace(old, new)
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
