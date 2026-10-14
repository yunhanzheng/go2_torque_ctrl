"""Load protected V46 model 4180 with fresh optimizers for V47."""
from pathlib import Path


path = Path("/data/sac/scripts/train_eth_fixed05_v2.py")
source = path.read_text()
old = "import eth_fixed05_v2  # noqa: F401"
assert source.count(old) == 1
source = source.replace(old, old + "\nimport terrain_v47_22cm_breakthrough.config")
needle = "resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)"
assert source.count(needle) == 1
source = source.replace(needle, 'resume_path = os.environ["V47_SOURCE_CHECKPOINT"]')
needle = "        runner.load(resume_path)"
assert source.count(needle) == 1
source = source.replace(
    needle,
    "        loaded = torch.load(resume_path, weights_only=False, map_location=runner.device)\n"
    "        runner.load(resume_path, load_cfg={\"actor\": True, \"critic\": True, "
    "\"optimizer\": False, \"iteration\": True, \"rnd\": False})\n"
    "        source_alpha = loaded.get(\"alpha\")\n"
    "        if source_alpha is None and loaded.get(\"log_alpha\") is not None:\n"
    "            source_alpha = loaded[\"log_alpha\"].exp().item()\n"
    "        if source_alpha is not None:\n"
    "            runner.alg.alpha = float(source_alpha)\n"
    "            runner.alg.log_alpha.data.copy_(torch.log(torch.tensor(float(source_alpha), device=runner.device)))\n"
    "        print(\"[V47_TRANSFER] protected V46 model4180 actor+critic loaded; fresh optimizers; "
    "fixed alpha; 30-call actor warmup\", flush=True)",
)
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
