"""Guarded V41 height-only bridge from the protected V37 checkpoint."""
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
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
SOURCE = Path("/home/ubuntu22/saved_models/SATA_STAIRS_PAPER_2026-09-26/V37_PAPER_RECOMMENDED_model_2799.pt")
EXPERIMENT = "terrain_v41_height_bridge"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
BATCH = ROOT / "results" / STAMP
RUN_NAME = f"height_5to12cm_width40_from_v37_2799_{STAMP}"
ENV = dict(
    os.environ,
    PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts",
    WANDB_MODE="offline",
    V41_SOURCE_CHECKPOINT=str(SOURCE),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status(stage, **details):
    temporary = BATCH / "status.tmp"
    temporary.write_text(json.dumps({
        "stage": stage,
        "updated_at": time.time(),
        "supervisor_pid": os.getpid(),
        **details,
    }, ensure_ascii=False, indent=2))
    temporary.replace(BATCH / "status.json")


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if subprocess.run(["pgrep", "-f", "train_eth_fixed05_v2.py"], capture_output=True).returncode == 0:
        raise RuntimeError("Another Isaac/SAC training process is active")
    gpu = subprocess.run([
        "nvidia-smi",
        "--query-gpu=name,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader",
    ], capture_output=True, text=True)
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
        "source_checkpoint": str(SOURCE),
        "source_sha256": source_hash,
        "actor_observation_dim": 62,
        "height_scan_in_actor": False,
        "changed_geometry": "height only",
        "stair_height_m": [0.05, 0.12],
        "stair_step_width_m": 0.40,
        "actor_warmup_update_calls": 25,
        "optimizer_transfer": False,
        "alpha_transfer": True,
        "actor_learning_rate": 2.0e-7,
        "critic_learning_rate": 5.0e-5,
        "iterations": 100,
        "save_interval": 5,
        "num_envs": 4096,
    }, ensure_ascii=False, indent=2))

    log_path = BATCH / "V41高度单变量桥接_100轮.log"
    command = [
        PYTHON, "-u", str(ROOT / "train.py"),
        "--task", "Isaac-Go2-V41-HeightBridge-v0",
        "--agent", "rsl_rl_sac_cfg_entry_point",
        "--num_envs", "4096",
        "--max_iterations", "100",
        "--seed", "1",
        "--resume",
        "--run_name", RUN_NAME,
        "--headless",
    ]
    print("[V41] 只扩展阶高至12 cm；保持40 cm踏步；前25轮冻结Actor；新optimizer", flush=True)
    print("[GPU]", gpu.stdout.strip(), flush=True)
    print("[命令]", " ".join(command), flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT.parent,
            env=ENV,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
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
                    if elapsed > 7200:
                        raise TimeoutError("V41 training exceeded two hours")
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
        raise RuntimeError("Protected V37 source checkpoint changed")
    status("complete", checkpoint=str(checkpoint), source_checkpoint=str(SOURCE), log=str(log_path))
    print("[完成] V41 checkpoint:", checkpoint, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if BATCH.exists():
            status("failed", error=repr(error))
        raise
