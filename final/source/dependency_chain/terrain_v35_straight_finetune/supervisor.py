"""Visible and guarded V35 straight-line fine-tuning run."""
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
SOURCE = Path("/home/ubuntu22/logs/rsl_rl/terrain_v34_scratch_four_terrain/2026-09-20_22-45-03_mixed_curriculum_20260920T224437Z/model_2396.pt")
EXPERIMENT = "terrain_v35_straight_finetune"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
BATCH = ROOT / "results" / STAMP
RUN_NAME = f"straight_from_v34_{STAMP}"
ENV = dict(
    os.environ,
    PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts",
    WANDB_API_KEY=os.environ["WANDB_API_KEY"],
    V35_SOURCE_CHECKPOINT=str(SOURCE),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_status(stage, **details):
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
        raise RuntimeError("Another Isaac/SAC training process is already active")
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
        "source_checkpoint": str(SOURCE),
        "source_sha256": source_hash,
        "observation_dim": 62,
        "network_changed": False,
        "heading_command": True,
        "heading_target_rad": 0.0,
        "heading_control_stiffness": 0.5,
        "yaw_rate_limit_rad_s": 0.5,
        "actor_learning_rate": 1.0e-5,
        "iterations": 300,
        "num_envs": 4096,
    }, ensure_ascii=False, indent=2))

    log_path = BATCH / "V35直线微调_300轮.log"
    command = [
        PYTHON, "-u", str(ROOT / "train.py"),
        "--task", "Isaac-Go2-V35-StraightFourTerrain-v0",
        "--agent", "rsl_rl_sac_cfg_entry_point",
        "--num_envs", "4096", "--max_iterations", "300",
        "--seed", "1", "--resume", "--run_name", RUN_NAME, "--logger", "wandb", "--log_project_name", "isaac_sac_pipeline", "--headless",
    ]
    print("[V35] 保持62维输入和原网络，从V34 model_2396.pt开始直线微调", flush=True)
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
                    write_status("training", child_pid=process.pid, elapsed_seconds=elapsed, log=str(log_path))
                    if process.poll() is not None:
                        print(reader.read(), end="", flush=True)
                        if process.returncode:
                            raise RuntimeError(f"training exited with {process.returncode}")
                        break
                    if elapsed > 10800:
                        raise TimeoutError("V35 training exceeded three hours")
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
        raise RuntimeError("Protected V34 source checkpoint changed during training")
    write_status("complete", checkpoint=str(checkpoint), source_checkpoint=str(SOURCE), log=str(log_path))
    print("[完成] V35 checkpoint:", checkpoint, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if BATCH.exists():
            write_status("failed", error=repr(error))
        raise

