"""Run the physical/strict evaluator with the V48 task registered."""
from pathlib import Path


path = Path("/home/ubuntu22/terrain_v47_22cm_breakthrough/evaluate_dual.py")
source = path.read_text()
old = 'new = "import terrain_v47_22cm_breakthrough.config"'
new = 'new = "import terrain_v48_23cm_final.config"'
assert source.count(old) == 1
source = source.replace(old, new)
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
