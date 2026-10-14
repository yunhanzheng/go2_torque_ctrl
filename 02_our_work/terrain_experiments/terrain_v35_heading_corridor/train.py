"""Register V35 and launch SAC from the protected V34 flat checkpoint."""
from pathlib import Path


path = Path("/data/sac/scripts/train_eth_fixed05_v2.py")
source = path.read_text()
old = "import eth_fixed05_v2  # noqa: F401"
assert source.count(old) == 1
source = source.replace(old, old + "\nimport terrain_v35_heading_corridor.config")
needle = "resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)"
assert source.count(needle) == 1
source = source.replace(
    needle,
    'resume_path = os.environ.get("V35_SOURCE_CHECKPOINT") or get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)',
)
load_call = "        runner.load(resume_path)"
assert source.count(load_call) == 1
source = source.replace(
    load_call,
    """        if os.environ.get("V35_ACTOR_ONLY") == "1":
            runner.load(resume_path, load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False})
            print("[V35_TRANSFER] loaded actor only; fresh critic and optimizers", flush=True)
        else:
            runner.load(resume_path)
        actor_lr_override = os.environ.get("V35_ACTOR_LR_OVERRIDE")
        if actor_lr_override is not None:
            actor_lr = float(actor_lr_override)
            runner.alg.actor_learning_rate = actor_lr
            for param_group in runner.alg.actor_optimizer.param_groups:
                param_group["lr"] = actor_lr
            print(f"[V35_TRAIN] actor learning rate overridden after load: {actor_lr:g}", flush=True)
        if os.environ.get("V35_FREEZE_ACTOR_NORMALIZER") == "1":
            runner.alg.actor.update_normalization = lambda obs: None
            print("[V35_TRAIN] actor observation normalizer frozen", flush=True)""",
)
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
