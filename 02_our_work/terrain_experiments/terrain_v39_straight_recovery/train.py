"""Launch V39 from the protected V38 model_3798 checkpoint."""
from pathlib import Path


path = Path("/data/sac/scripts/train_eth_fixed05_v2.py")
source = path.read_text()
old = "import eth_fixed05_v2  # noqa: F401"
assert source.count(old) == 1
source = source.replace(old, old + "\nimport terrain_v39_straight_recovery.config")
needle = "resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)"
assert source.count(needle) == 1
source = source.replace(needle, 'resume_path = os.environ["V39_SOURCE_CHECKPOINT"]')
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
