"""Guarded V31 workflow: learn a flat gait first, then unlock mixed terrain."""
import datetime
import html
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from catalog import CASES


ROOT = Path(__file__).resolve().parent
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
FFMPEG = "/home/ubuntu22/miniforge3/envs/env_isaaclab/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
BATCH = ROOT / "results" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
FLAT_TASK = ["--task", "Isaac-Go2-Terrain-V31-HistoryFlat-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]
MIXED_TASK = ["--task", "Isaac-Go2-Terrain-V31-HistoryMixed-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]


def status(stage, **details):
    temporary = BATCH / "status.tmp"
    temporary.write_text(json.dumps(dict(stage=stage, updated_at=time.time(), supervisor_pid=os.getpid(), **details), ensure_ascii=False, indent=2))
    temporary.replace(BATCH / "status.json")


def run(command, stage, timeout=3600, env=ENV):
    print(f"\n[监管] {stage}", flush=True)
    log_path = BATCH / f"{stage}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT.parent, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        try:
            with log_path.open() as reader:
                while True:
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end="", flush=True)
                    elapsed = round(time.monotonic() - started)
                    status(stage, child_pid=process.pid, elapsed_seconds=elapsed)
                    if process.poll() is not None:
                        print(reader.read(), end="", flush=True)
                        if process.returncode:
                            raise RuntimeError(f"{stage}: exit {process.returncode}")
                        return
                    if elapsed > timeout:
                        raise TimeoutError(stage)
                    time.sleep(2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def checkpoint(experiment, run_name):
    runs = list(Path("/home/ubuntu22/logs/rsl_rl", experiment).glob(f"*_{run_name}"))
    assert len(runs) == 1, runs
    models = list(runs[0].glob("model_*.pt"))
    assert models, runs[0]
    return max(models, key=lambda path: int(path.stem.split("_")[-1]))


def evaluate(task, trained, case, difficulty, prefix, results):
    output = BATCH / prefix
    env = dict(ENV, TERRAIN_CASE=case, TERRAIN_DIFFICULTY=str(difficulty), TERRAIN_OUTPUT=str(output))
    run([PYTHON, "-u", str(ROOT / "evaluate.py"), *task, "--num_envs", "32", "--seed", "7", "--checkpoint", str(trained), "--headless"], f"评估_{prefix}", 1200, env)
    data = json.loads(Path(str(output) + ".json").read_text())
    assert data["num_envs"] == 32 and sum(data[key] for key in ("success", "terminated", "other_no_success")) == 32
    results[case] = data


def record(task, trained, case, difficulty, prefix):
    videos = BATCH / "videos"
    videos.mkdir(exist_ok=True)
    output = videos / prefix
    env = dict(ENV, TERRAIN_CASE=case, TERRAIN_DIFFICULTY=str(difficulty), TERRAIN_OUTPUT=str(output))
    run([PYTHON, "-u", str(ROOT / "evaluate.py"), *task, "--num_envs", "1", "--seed", "7", "--checkpoint", str(trained), "--headless", "--video", "--video_length", "600"], f"录制_{prefix}", 1200, env)
    webm = Path(str(output) + ".webm")
    run([FFMPEG, "-y", "-v", "error", "-i", str(output) + "_video/rl-video-step-0.mp4", "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-an", str(webm)], f"编码_{prefix}", 900)
    run([FFMPEG, "-v", "error", "-i", str(webm), "-f", "null", "-"], f"校验_{prefix}", 120)


def publish(results, flat_checkpoint, mixed_checkpoint=None):
    rows = ["| 测试条件 | V31 |", "|---|---:|"]
    for case, (label, _) in CASES.items():
        data = results.get(case)
        rows.append(f'| {label} | {data["success"] if data else "待评估"}/32 |')
    videos = BATCH / "videos"
    cards = []
    if videos.exists():
        for video in sorted(videos.glob("*.webm")):
            cards.append(f'<article><h2>{html.escape(video.stem)}</h2><video controls loop muted preload="metadata" src="videos/{video.name}"></video></article>')
    page = BATCH / "V31平地门禁与多地形.html"
    page.write_text(
        '<!doctype html><meta charset="utf-8"><title>V31平地门禁与多地形</title>'
        '<style>body{font-family:sans-serif;margin:24px;background:#eee}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article{background:white;padding:16px}video{width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>V31：先学正常步态，再解锁多地形</h1>'
        '<p>随机初始化；5帧/160 ms历史；固定0.5 m/s前进；平地32环境通过率至少24/32才允许进入多地形阶段。</p>'
        '<pre>' + html.escape("\n".join(rows)) + '</pre><main>' + ''.join(cards) + '</main>'
    )
    (BATCH / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (BATCH / "结果.md").write_text("# V31 分阶段从头训练\n\n" + "\n".join(rows) + f"\n\n平地模型：{flat_checkpoint}\n多地形模型：{mixed_checkpoint}\n")
    return page


def main():
    BATCH.mkdir(parents=True)
    (ROOT / "latest.txt").write_text(str(BATCH) + "\n")
    snapshot = BATCH / "source_snapshot"
    snapshot.mkdir()
    for path in ROOT.iterdir():
        if path.suffix in (".py", ".sh", ".md"):
            shutil.copy2(path, snapshot / path.name)
    results = {}
    try:
        run([PYTHON, "-u", str(ROOT / "train.py"), *FLAT_TASK, "--num_envs", "64", "--max_iterations", "20", "--seed", "1", "--run_name", f"smoke_{STAMP}", "--headless", "agent.algorithm.replay_buffer_size=4096", "agent.algorithm.num_mini_batches=2", "agent.algorithm.mini_batch_size=128"], "V31平地短检查", 600)
        flat_name = f"flat_{STAMP}"
        run([PYTHON, "-u", str(ROOT / "train.py"), *FLAT_TASK, "--num_envs", "4096", "--max_iterations", "300", "--seed", "1", "--run_name", flat_name, "--headless"], "V31平地基础_300轮", 7200)
        flat_model = checkpoint("terrain_v31_history_flat_foundation", flat_name)
        evaluate(FLAT_TASK, flat_model, "plane", 0.0, "V31_flat_gate", results)
        record(FLAT_TASK, flat_model, "plane", 0.0, "V31_flat_gate")
        page = publish(results, flat_model)
        subprocess.Popen(["firefox", "--new-window", page.as_uri()], env=ENV)
        flat_success = results["plane"]["success"]
        if flat_success < 24:
            status("flat_gate_failed", checkpoint=str(flat_model), success=flat_success, page=str(page))
            print(f"[门禁停止] 平地仅 {flat_success}/32，不进入多地形训练。", flush=True)
            return

        mixed_name = f"mixed_{STAMP}"
        resume_env = dict(ENV, V31_SOURCE=str(flat_model))
        run([PYTHON, "-u", str(ROOT / "train.py"), *MIXED_TASK, "--num_envs", "4096", "--max_iterations", "400", "--seed", "1", "--resume", "--run_name", mixed_name, "--headless"], "V31低难度多地形_400轮", 9000, resume_env)
        mixed_model = checkpoint("terrain_v31_history_multiterrain", mixed_name)
        results = {}
        for case, (_, difficulty) in CASES.items():
            evaluate(MIXED_TASK, mixed_model, case, difficulty, f"V31_{case}", results)
            publish(results, flat_model, mixed_model)
        for case, (_, difficulty) in CASES.items():
            record(MIXED_TASK, mixed_model, case, difficulty, f"V31_{case}")
        page = publish(results, flat_model, mixed_model)
        subprocess.Popen(["firefox", "--new-window", page.as_uri()], env=ENV)
        status("complete", checkpoint=str(mixed_model), flat_checkpoint=str(flat_model), page=str(page))
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
