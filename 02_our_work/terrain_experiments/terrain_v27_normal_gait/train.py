from pathlib import Path


path = Path("/data/sac/scripts/train_eth_fixed05_v2.py")
source = path.read_text()


def replace(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)


replace(
    "import eth_fixed05_v2  # noqa: F401",
    "import eth_fixed05_v2\nimport terrain_v27_normal_gait.config\n"
    "from terrain_v27_normal_gait.training import install, initialize, audit, SOURCE\ninstall()",
)
replace("resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)", "resume_path = SOURCE")
replace("        runner.load(resume_path)", "        initialize(runner, log_dir)")
needle = "    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)"
replace(needle, needle + "\n    audit(runner, log_dir)")
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
