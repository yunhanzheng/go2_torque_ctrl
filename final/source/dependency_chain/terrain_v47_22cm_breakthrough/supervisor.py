"""Supervise the focused 22 cm breakthrough run."""
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
SOURCE = Path("/home/ubuntu22/saved_models/SATA_STAIRS_22CM_FRONTIER_2026-10-07/V46_22CM_4OF32_model_4180.pt")
EXPERIMENT = "terrain_v47_22cm_breakthrough"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
BATCH = ROOT / "results" / STAMP
RUN_NAME = f"focused_20to22cm_from_v46_4180_{STAMP}"
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts",
           WANDB_API_KEY=os.environ["WANDB_API_KEY"], V47_SOURCE_CHECKPOINT=str(SOURCE))


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
    for item in ROOT.iterdir():
        if item.suffix in (".py", ".sh", ".md"):
            shutil.copy2(item, snapshot / item.name)

    source_hash = sha256(SOURCE)
    expected_hash = "304badf0f2dd551d0344c851b5f70da184c89ff17e951d04ffbd4f17896fa18b"
    if source_hash != expected_hash:
        raise RuntimeError(f"Protected source hash mismatch: {source_hash}")
    (BATCH / "provenance.json").write_text(json.dumps({
        "source_checkpoint": str(SOURCE), "source_sha256": source_hash,
        "source_evaluation": {"20cm_physical": "24/32", "21cm_physical": "16/32",
                              "22cm_physical": "4/32", "23cm_physical": "0/32"},
        "main_stair_height_m": [0.20, 0.22], "stair_step_width_m": 0.30,
        "anchors": {"20cm": 0.10, "21cm": 0.20, "22cm": 0.15},
        "main_upstairs_proportion": 0.45, "initial_max_level": 4,
        "swing_target_at_22cm_m": 0.233, "swing_clearance_weight": 6.0,
        "actor_warmup_update_calls": 30, "optimizer_transfer": False,
        "fixed_alpha_from_source": True, "actor_learning_rate": 3.0e-7,
        "critic_learning_rate": 5.0e-5, "policy_frequency": 5,
        "iterations": 300, "save_interval": 5, "num_envs": 4096,
    }, ensure_ascii=False, indent=2))

    log_path = BATCH / "V47_突破22cm_300轮.log"
    command = [PYTHON, "-u", str(ROOT / "train.py"),
               "--task", "Isaac-Go2-V47-22cmBreakthrough-v0", "--agent", "rsl_rl_sac_cfg_entry_point",
               "--num_envs", "4096", "--max_iterations", "300", "--seed", "1", "--resume",
               "--run_name", RUN_NAME, "--logger", "wandb", "--log_project_name", "isaac_sac_pipeline", "--headless"]
    print("[V47] protected model4180 -> focused 20-22 cm；固定20/21/22 cm锚点", flush=True)
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
                    if elapsed > 14400:
                        raise TimeoutError("V47 training exceeded four hours")
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
        raise RuntimeError("Protected V46 source checkpoint changed")
    status("complete", checkpoint=str(checkpoint), source_checkpoint=str(SOURCE), log=str(log_path))
    print("[完成] V47 checkpoint:", checkpoint, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if BATCH.exists():
            status("failed", error=repr(error))
        raise
