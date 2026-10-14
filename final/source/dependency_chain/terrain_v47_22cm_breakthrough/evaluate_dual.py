"""Run the physical/strict evaluator with the V47 task registered."""
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "terrain_v45_23cm_from_18cm" / "evaluate_dual.py"
source = path.read_text()
old = "import terrain_v45_23cm_from_18cm.config"
new = "import terrain_v47_22cm_breakthrough.config"
assert source.count(old) == 1
source = source.replace(old, new)
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
