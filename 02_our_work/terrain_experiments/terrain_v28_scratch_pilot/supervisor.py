"""Visible from-scratch pilot, evaluation, WebM generation and Firefox report."""
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
TASK = ["--task", "Isaac-Go2-Terrain-V28-ScratchMulti-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]


def status(stage, **details):
    temporary = BATCH / "status.tmp"
    temporary.write_text(json.dumps(dict(stage=stage, updated_at=time.time(), supervisor_pid=os.getpid(), **details), ensure_ascii=False, indent=2))
    temporary.replace(BATCH / "status.json")


def run(command, stage, timeout=1800, env=ENV):
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
                        break
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


def publish(results, checkpoint=None):
    (BATCH / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    rows = ["| 测试条件 | V28从头训练Pilot |", "|---|---:|"]
    for case, (label, _) in CASES.items():
        data = results.get(case)
        rows.append(f'| {label} | {data["success"] if data else "待评估"}/32 |')
    (BATCH / "结果.md").write_text(
        "# V28 从头多地形训练Pilot\n\n" + "\n".join(rows)
        + "\n\n随机初始化；官方高度扫描观测；固定seed=7和0.5 m/s前进指令进行首回合路线穿越测试。\n\n模型："
        + str(checkpoint) + "\n"
    )
    cards = []
    for case, (label, _) in CASES.items():
        video = BATCH / "videos" / f"V28_{case}.webm"
        if video.exists():
            data = results.get(case, {})
            cards.append(f'<article><h2>{html.escape(label)}（{data.get("success", "?")}/32）</h2><video controls loop muted preload="metadata" src="videos/{video.name}"></video></article>')
    page = BATCH / "V28从头多地形Pilot.html"
    page.write_text(
        '<!doctype html><meta charset="utf-8"><title>V28从头多地形Pilot</title>'
        '<style>body{font-family:sans-serif;margin:24px;background:#eee}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article{background:white;padding:16px}video{width:100%}pre{white-space:pre-wrap}</style>'
        "<h1>V28从随机初始化进行多地形统一训练</h1>"
        "<p>平地、规则楼梯、斜坡、随机方块、随机粗糙面均在同一次训练中出现；Policy使用官方高度扫描观测。</p>"
        "<pre>" + html.escape("\n".join(rows)) + "</pre><main>" + "".join(cards) + "</main>"
    )
    return page


def main():
    BATCH.mkdir(parents=True)
    (ROOT / "latest.txt").write_text(str(BATCH) + "\n")
    snapshot = BATCH / "source_snapshot"
    snapshot.mkdir()
    for path in ROOT.iterdir():
        if path.suffix in (".py", ".sh", ".md"):
            shutil.copy2(path, snapshot / path.name)
    name = f"pilot_{STAMP}"
    results = {}
    try:
        run(
            [PYTHON, "-u", str(ROOT / "train.py"), *TASK, "--num_envs", "64", "--max_iterations", "20", "--seed", "1", "--run_name", f"smoke_{STAMP}", "--headless", "agent.algorithm.replay_buffer_size=8192", "agent.algorithm.num_mini_batches=2", "agent.algorithm.mini_batch_size=128"],
            "从头训练短检查", 600,
        )
        run(
            [PYTHON, "-u", str(ROOT / "train.py"), *TASK, "--num_envs", "4096", "--max_iterations", "300", "--seed", "1", "--run_name", name, "--headless"],
            "从头多地形Pilot_300轮", 7200,
        )
        paths = list(Path("/home/ubuntu22/logs/rsl_rl/terrain_v28_scratch_multiterrain").glob(f"*_{name}/model_299.pt"))
        assert len(paths) == 1, paths
        trained = paths[0]
        videos = BATCH / "videos"
        videos.mkdir()
        for case, (_, difficulty) in CASES.items():
            output = BATCH / f"V28_{case}"
            env = dict(ENV, TERRAIN_CASE=case, TERRAIN_DIFFICULTY=str(difficulty), TERRAIN_OUTPUT=str(output))
            run([PYTHON, "-u", str(ROOT / "evaluate.py"), *TASK, "--num_envs", "32", "--seed", "7", "--checkpoint", str(trained), "--headless"], f"评估_V28_{case}", 1200, env)
            data = json.loads(Path(str(output) + ".json").read_text())
            assert data["num_envs"] == 32 and sum(data[key] for key in ("success", "terminated", "other_no_success")) == 32
            results[case] = data
            publish(results, trained)
        for case, (_, difficulty) in CASES.items():
            output = videos / f"V28_{case}"
            env = dict(ENV, TERRAIN_CASE=case, TERRAIN_DIFFICULTY=str(difficulty), TERRAIN_OUTPUT=str(output))
            run([PYTHON, "-u", str(ROOT / "evaluate.py"), *TASK, "--num_envs", "1", "--seed", "7", "--checkpoint", str(trained), "--headless", "--video", "--video_length", "600"], f"录制_V28_{case}", 1200, env)
            webm = Path(str(output) + ".webm")
            run([FFMPEG, "-y", "-v", "error", "-i", str(output) + "_video/rl-video-step-0.mp4", "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-an", str(webm)], f"编码_V28_{case}", 900)
            run([FFMPEG, "-v", "error", "-i", str(webm), "-f", "null", "-"], f"校验_V28_{case}", 120)
            page = publish(results, trained)
        subprocess.Popen(["firefox", "--new-window", page.as_uri()], env=ENV)
        status("complete", checkpoint=str(trained), page=str(page), ready_for_overnight_review=True)
        print("[Pilot完成] 请先检查Firefox；确认后再安排夜间续训：", page, flush=True)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
