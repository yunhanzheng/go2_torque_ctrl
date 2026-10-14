"""Launch V41 with an actor-preserving, fresh-optimizer transfer."""
from pathlib import Path


path = Path("/data/sac/scripts/train_eth_fixed05_v2.py")
source = path.read_text()
old = "import eth_fixed05_v2  # noqa: F401"
assert source.count(old) == 1
source = source.replace(old, old + "\nimport terrain_v41_height_bridge.config")

resume_needle = "resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)"
assert source.count(resume_needle) == 1
source = source.replace(resume_needle, 'resume_path = os.environ["V41_SOURCE_CHECKPOINT"]')

load_needle = "        runner.load(resume_path)"
assert source.count(load_needle) == 1
source = source.replace(
    load_needle,
    "        loaded = torch.load(resume_path, weights_only=False, map_location=runner.device)\n"
    "        runner.load(resume_path, load_cfg={\"actor\": True, \"critic\": True, "
    "\"optimizer\": False, \"iteration\": True, \"rnd\": False})\n"
    "        if loaded.get(\"log_alpha\") is not None:\n"
    "            runner.alg.log_alpha.data.copy_(loaded[\"log_alpha\"].to(runner.device))\n"
    "            runner.alg.alpha = runner.alg.log_alpha.exp().item()\n"
    "        print(\"[V41_TRANSFER] actor+critic+iteration loaded; fresh optimizers; alpha restored; "
    "25-call actor warmup\", flush=True)",
)

exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
