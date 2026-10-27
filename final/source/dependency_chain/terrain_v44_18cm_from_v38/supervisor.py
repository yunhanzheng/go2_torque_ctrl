"""Supervise V44's 10--18 cm continuation from the proven V38 lineage."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parent
PYTHON = "/opt/conda/envs/isaac_sac/bin/python"
SOURCE = Path(
    "/home/ubuntu22/logs/rsl_rl/terrain_v38_official_stairs/"
    "2026-09-26_11-55-01_official_5to23cm_resume2900_to3799_20260926T115454Z/model_3798.pt"
)
EXPERIMENT = "terrain_v44_18cm_from_v38"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
BATCH = ROOT / "results" / STAMP
RUN_NAME = f"height_10to18cm_width30_from_v38_3798_{STAMP}"
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts",
           WANDB_API_KEY=os.environ["WANDB_API_KEY"], V44_SOURCE_CHECKPOINT=str(SOURCE))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status(stage, **details):
    temporary = BATCH / "status.tmp"
    temporary.write_text(json.dumps(
        {"stage": stage, "updated_at": time.time(), "supervisor_pid": os.getpid(), **details},
        ensure_ascii=False, indent=2,
    ))
    temporary.replace(BATCH / "status.json")


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if subprocess.run(["pgrep", "-f", "train_eth_fixed05_v2.py"], capture_output=True).returncode == 0:
        raise RuntimeError("Another Isaac/SAC training process is active")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if gpu.returncode:
        raise RuntimeError("NVIDIA preflight failed: " + gpu.stderr.strip())

    BATCH.mkdir(parents=True)
    (ROOT / "latest.txt").write_text(str(BATCH) + "\n")
    snapshot = BATCH / "source_snapshot"
    snapshot.mkdir()
    for path in ROOT.iterdir():
        if path.suffix in (".py", ".sh", ".md"):
            shutil.copy2(path, snapshot / path.name)

    source_hash = sha256(SOURCE)
    (BATCH / "provenance.json").write_text(json.dumps({
        "source_checkpoint": str(SOURCE), "source_sha256": source_hash,
        "source_physical_success": {"12cm": "31/32", "14cm": "32/32", "16cm": "25/32", "18cm": "2/32"},
        "actor_observation_dim": 62, "height_scan_in_actor": False,
        "stair_height_m": [0.10, 0.18], "stair_step_width_m": 0.30,
        "swing_target_m": {"flat": 0.09, "at_18cm_relief": 0.18},
        "excessive_foot_height_threshold_m": 0.24,
        "actor_warmup_update_calls": 50, "optimizer_transfer": False,
        "fixed_alpha_from_source": True, "actor_learning_rate": 5.0e-7,
        "critic_learning_rate": 5.0e-5, "policy_frequency": 5,
        "iterations": 250, "save_interval": 10, "num_envs": 4096,
    }, ensure_ascii=False, indent=2))

    log_path = BATCH / "V44_从V38向18cm推进_250轮.log"
    command = [PYTHON, "-u", str(ROOT / "train.py"),
               "--task", "Isaac-Go2-V44-18cmFromV38-v0", "--agent", "rsl_rl_sac_cfg_entry_point",
               "--num_envs", "4096", "--max_iterations", "250", "--seed", "1", "--resume",
               "--run_name", RUN_NAME, "--logger", "wandb", "--log_project_name", "isaac_sac_pipeline", "--headless"]
    print("[V44] V38 model3798 -> 10-18 cm；30 cm踏面；先保住12-16，再突破18", flush=True)
    print("[GPU]", gpu.stdout.strip(), flush=True)
    print("[命令]", " ".join(command), flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT.parent, env=ENV, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        try:
            with log_path.open() as reader:
                while True:
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end="", flush=True)
                    elapsed = round(time.monotonic() - started)
                    status("training", child_pid=process.pid, elapsed_seconds=elapsed, log=str(log_path))
                    if process.poll() is not None:
                        print(reader.read(), end="", flush=True)
                        if process.returncode:
                            raise RuntimeError(f"training exited with {process.returncode}")
                        break
                    if elapsed > 10800:
                        raise TimeoutError("V44 training exceeded three hours")
                    time.sleep(2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)

    runs = list((Path("/home/ubuntu22/logs/rsl_rl") / EXPERIMENT).glob(f"*_{RUN_NAME}"))
    if len(runs) != 1:
        raise RuntimeError(f"Expected one run directory, got: {runs}")
    models = list(runs[0].glob("model_*.pt"))
    checkpoint = max(models, key=lambda item: int(item.stem.split("_")[-1]))
    if sha256(SOURCE) != source_hash:
        raise RuntimeError("Protected V38 source checkpoint changed")
    status("complete", checkpoint=str(checkpoint), source_checkpoint=str(SOURCE), log=str(log_path))
    print("[完成] V44 checkpoint:", checkpoint, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if BATCH.exists():
            status("failed", error=repr(error))
        raise
