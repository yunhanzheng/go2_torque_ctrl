"""Resume the interrupted V38 run from model_2900 through model_3799."""
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
    "2026-09-26_00-50-15_official_5to23cm_from_v37_2799_20260926T005009Z/"
    "model_2900.pt"
)
EXPERIMENT = "terrain_v38_official_stairs"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
BATCH = ROOT / "results" / STAMP
RUN_NAME = f"official_5to23cm_resume2900_to3799_{STAMP}"
ENV = dict(
    os.environ,
    PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts",
    WANDB_API_KEY=os.environ["WANDB_API_KEY"],
    V38_SOURCE_CHECKPOINT=str(SOURCE),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status(stage, **details):
    temporary = BATCH / "status.tmp"
    temporary.write_text(
        json.dumps(
            {"stage": stage, "updated_at": time.time(), "supervisor_pid": os.getpid(), **details},
            ensure_ascii=False,
            indent=2,
        )
    )
    temporary.replace(BATCH / "status.json")


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if subprocess.run(["pgrep", "-f", "train_eth_fixed05_v2.py"], capture_output=True).returncode == 0:
        raise RuntimeError("Another Isaac/SAC training process is active")
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
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
    (BATCH / "provenance.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(SOURCE),
                "source_sha256": source_hash,
                "resume_iteration": 2900,
                "target_iteration": 3799,
                "actor_observation_dim": 62,
                "height_scan_in_actor": False,
                "network_changed": False,
                "stair_height_m": [0.05, 0.23],
                "stair_step_width_m": 0.30,
                "curriculum_max_initial_level": 1,
                "terrain_proportions": {
                    "plane": 0.15,
                    "stairs_up": 0.50,
                    "stairs_down": 0.15,
                    "slope_up": 0.025,
                    "slope_down": 0.025,
                    "boxes": 0.075,
                    "random_rough": 0.075,
                },
                "actor_learning_rate": 2.0e-6,
                "additional_iterations": 899,
                "num_envs": 4096,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    log_path = BATCH / "V38原始楼梯课程_从2900恢复到3799.log"
    command = [
        PYTHON,
        "-u",
        str(ROOT / "train.py"),
        "--task",
        "Isaac-Go2-V38-OfficialStairs-v0",
        "--agent",
        "rsl_rl_sac_cfg_entry_point",
        "--num_envs",
        "4096",
        "--max_iterations",
        "899",
        "--seed",
        "1",
        "--resume",
        "--run_name",
        RUN_NAME,
        "--logger", "wandb", "--log_project_name", "isaac_sac_pipeline", "--headless",
    ]
    print("[V38 resume] 原始5-23 cm楼梯课程；从model_2900恢复至model_3799", flush=True)
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
                    if elapsed > 18000:
                        raise TimeoutError("V38 resumed training exceeded five hours")
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
        raise RuntimeError("Protected V38 model_2900 source checkpoint changed")
    status("complete", checkpoint=str(checkpoint), source_checkpoint=str(SOURCE), log=str(log_path))
    print("[完成] V38 checkpoint:", checkpoint, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if BATCH.exists():
            status("failed", error=repr(error))
        raise
