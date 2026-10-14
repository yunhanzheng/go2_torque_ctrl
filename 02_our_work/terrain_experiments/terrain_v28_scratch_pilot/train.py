"""Use the established SAC launcher, but register V28 and do not load any checkpoint."""
from pathlib import Path


path = Path("/data/sac/scripts/train_eth_fixed05_v2.py")
source = path.read_text()
old = "import eth_fixed05_v2  # noqa: F401"
assert source.count(old) == 1
source = source.replace(old, old + "\nimport terrain_v28_scratch_pilot.config")
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
