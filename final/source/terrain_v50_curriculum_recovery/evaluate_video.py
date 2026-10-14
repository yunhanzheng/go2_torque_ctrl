"""V50 evaluator with selectable camera and visible environment."""
from pathlib import Path


path = (
    Path(__file__).resolve().parent.parent
    / "dependency_chain"
    / "terrain_v49_generalize_23cm"
    / "evaluate_boxes_video_env4.py"
)
source = path.read_text()
old = "terrain_v49_generalize_23cm.config"
new = "terrain_v50_curriculum_recovery.config"
assert source.count(old) == 1
source = source.replace(old, new)
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
