"""Register V30 and launch the established SAC trainer from random initialization."""
from pathlib import Path


path = Path("/data/sac/scripts/train_eth_fixed05_v2.py")
source = path.read_text()
old = "import eth_fixed05_v2  # noqa: F401"
assert source.count(old) == 1
source = source.replace(old, old + "\nimport terrain_v30_history_curriculum.config")
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
