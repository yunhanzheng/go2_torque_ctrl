"""Supervise the V49 multi-terrain generalization run."""
import datetime
import fcntl
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
    "/home/ubuntu22/saved_models/SATA_STAIRS_23CM_BEST_2026-10-07/"
    "V48_23CM_21OF32_model_4778.pt"
)
SOURCE_SHA256 = "b5355ed3eaefd17b7b606044594b720f9bdf291f9ae6c0f4b07462880d5dcfda"
EXPERIMENT = "terrain_v49_generalize_23cm"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
BATCH = ROOT / "results" / STAMP
RUN_NAME = f"mixed55upstairs_keep15fixed23_from_v48_4778_{STAMP}"
ENV = dict(
    os.environ,
    PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts",
    WANDB_API_KEY=os.environ["WANDB_API_KEY"],
    V49_SOURCE_CHECKPOINT=str(SOURCE),
)
LOCK_PATH = ROOT / ".training.lock"


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
    # A process-name check is insufficient because train.py dynamically
    # executes the shared IsaacLab entry point.  Hold an OS-level lock for the
    # entire train/evaluate pipeline so a second supervisor cannot start.
    lock = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another V49 supervisor already holds the training lock") from error
    lock.write(f"{os.getpid()}\n")
    lock.flush()

    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source_hash = sha256(SOURCE)
    if source_hash != SOURCE_SHA256:
        raise RuntimeError(f"Protected V48 source hash mismatch: {source_hash}")
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
    for item in ROOT.iterdir():
        if item.suffix in (".py", ".sh", ".md"):
            shutil.copy2(item, snapshot / item.name)
    (BATCH / "provenance.json").write_text(json.dumps({
        "source_checkpoint": str(SOURCE),
        "source_sha256": source_hash,
        "source_evaluation": {"23cm_physical": "21/32", "22cm_physical": "24/32"},
        "terrain_proportions": {
            "all_upstairs": 0.55,
            "variable_upstairs_21.5_to_23cm": 0.25,
            "fixed_upstairs_23cm": 0.15,
            "fixed_upstairs_22cm": 0.10,
            "fixed_upstairs_21cm": 0.05,
            "plane": 0.10,
            "stairs_down_5_to_23cm": 0.10,
            "slope_up": 0.075,
            "slope_down": 0.075,
            "boxes": 0.05,
            "random_rough": 0.05,
        },
        "actor_warmup_update_calls": 30,
        "optimizer_transfer": False,
        "fixed_alpha_from_source": True,
        "actor_learning_rate": 3.0e-7,
        "critic_learning_rate": 5.0e-5,
        "policy_frequency": 5,
        "iterations": 400,
        "save_interval": 10,
        "num_envs": 4096,
    }, ensure_ascii=False, indent=2))

    log_path = BATCH / "V49_保23cm能力的多地形泛化_400轮.log"
    command = [
        PYTHON, "-u", str(ROOT / "train.py"),
        "--task", "Isaac-Go2-V49-Generalize23cm-v0",
        "--agent", "rsl_rl_sac_cfg_entry_point",
        "--num_envs", "4096", "--max_iterations", "400", "--seed", "1",
        "--resume", "--run_name", RUN_NAME, "--logger", "wandb", "--log_project_name", "isaac_sac_pipeline", "--headless",
    ]
    print("[V49] protected model4778 -> 55% upstairs + 45% generalization terrains", flush=True)
    print("[V49] exact 23 cm anchor remains 15%", flush=True)
    print("[GPU]", gpu.stdout.strip(), flush=True)
    print("[命令]", " ".join(command), flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT.parent, env=ENV, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
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
                    if elapsed > 18000:
                        raise TimeoutError("V49 training exceeded five hours")
                    time.sleep(2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)

    runs = list((Path("/home/ubuntu22/logs/rsl_rl") / EXPERIMENT).glob(f"*_{RUN_NAME}"))
    if len(runs) != 1:
        raise RuntimeError(f"Expected one run directory, got: {runs}")
    checkpoints = list(runs[0].glob("model_*.pt"))
    checkpoint = max(checkpoints, key=lambda item: int(item.stem.split("_")[-1]))
    if sha256(SOURCE) != source_hash:
        raise RuntimeError("Protected V48 source checkpoint changed")
    status("complete", checkpoint=str(checkpoint), source_checkpoint=str(SOURCE), log=str(log_path))
    print("[完成] V49 checkpoint:", checkpoint, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if BATCH.exists():
            status("failed", error=repr(error))
        raise
