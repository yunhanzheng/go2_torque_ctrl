"""Visible V27 fine-tuning followed by matched V26/V27 evaluation and videos."""
import datetime
import hashlib
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
SOURCE = Path("/home/ubuntu22/logs/rsl_rl/terrain_v26_official3_from_v24/2026-09-17_21-18-06_v26_20260917T211724Z/model_599.pt")
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
BATCH = ROOT / "results" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
TASK = ["--task", "Isaac-Go2-Terrain-V27-NormalGait-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]


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
    rows = ["| 测试条件 | V26 | V27正常步态 |", "|---|---:|---:|"]
    for case, (label, _) in CASES.items():
        before = results.get("V26", {}).get(case)
        after = results.get("V27", {}).get(case)
        rows.append(f'| {label} | {before["success"] if before else "待评估"}/32 | {after["success"] if after else "待评估"}/32 |')
    (BATCH / "结果.md").write_text(
        "# V26 → V27 正常步态微调\n\n" + "\n".join(rows)
        + "\n\nV27将最大摆腿目标由12 cm恢复为10 cm，过高抬脚约束由14 cm提前到11.5 cm并增强。固定seed=7、首回合、3.5 m路线穿越率。\n\n模型："
        + str(checkpoint) + "\n"
    )
    cards = []
    for case, (label, _) in CASES.items():
        pair = []
        for model_label in ("V26", "V27"):
            video = BATCH / "videos" / f"{model_label}_{case}.webm"
            if video.exists():
                pair.append(f'<div><h3>{model_label}</h3><video controls loop muted preload="metadata" src="videos/{video.name}"></video></div>')
        if pair:
            cards.append(f"<article><h2>{html.escape(label)}</h2><section>{''.join(pair)}</section></article>")
    page = BATCH / "V26_V27步态对比.html"
    page.write_text(
        '<!doctype html><meta charset="utf-8"><title>V26与V27正常步态对比</title>'
        '<style>body{font-family:sans-serif;margin:24px;background:#eee}article{background:white;padding:16px;margin:18px 0}section{display:flex;gap:16px}section div{width:50%}video{width:100%}pre{white-space:pre-wrap}</style>'
        "<h1>V26与V27正常步态微调对比</h1><p>左V26，右V27；相同地图、seed和摄像机，视频保留失败与重置。</p>"
        '<button onclick="document.querySelectorAll(\'video\').forEach(v=>{v.currentTime=0;v.play()})">全部从头播放</button>'
        "<pre>" + html.escape("\n".join(rows)) + "</pre>" + "".join(cards)
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
    original_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    reference_dir = BATCH / "v26_reference"
    reference_dir.mkdir()
    reference = reference_dir / "model_599.pt"
    shutil.copy2(SOURCE, reference)
    name = f"v27_{STAMP}"
    results = {"V26": {}, "V27": {}}
    try:
        run(
            [PYTHON, "-u", str(ROOT / "train.py"), *TASK, "--num_envs", "64", "--max_iterations", "29", "--seed", "1", "--resume", "--run_name", f"smoke_{STAMP}", "--headless", "agent.algorithm.replay_buffer_size=8192", "agent.algorithm.num_mini_batches=2", "agent.algorithm.mini_batch_size=128"],
            "奖励与更新检查", 600,
        )
        run(
            [PYTHON, "-u", str(ROOT / "train.py"), *TASK, "--num_envs", "4096", "--max_iterations", "300", "--seed", "1", "--resume", "--run_name", name, "--headless"],
            "正常步态微调300轮", 7200,
        )
        paths = list(Path("/home/ubuntu22/logs/rsl_rl/terrain_v27_normal_gait_from_v26").glob(f"*_{name}/model_299.pt"))
        assert len(paths) == 1, paths
        trained = paths[0]
        audit = json.loads((trained.parent / "audit.json").read_text())
        assert audit["finite"] and audit["iteration"] == 299 and audit["actor_change"] > 0, audit
        for model_label, checkpoint in (("V26", reference), ("V27", trained)):
            for case, (_, difficulty) in CASES.items():
                output = BATCH / f"{model_label}_{case}"
                env = dict(ENV, TERRAIN_CASE=case, TERRAIN_DIFFICULTY=str(difficulty), TERRAIN_OUTPUT=str(output))
                run([PYTHON, "-u", str(ROOT / "evaluate.py"), *TASK, "--num_envs", "32", "--seed", "7", "--checkpoint", str(checkpoint), "--headless"], f"评估_{model_label}_{case}", 1200, env)
                data = json.loads(Path(str(output) + ".json").read_text())
                assert data["num_envs"] == 32 and sum(data[key] for key in ("success", "terminated", "other_no_success")) == 32
                results[model_label][case] = data
                publish(results, trained)
        videos = BATCH / "videos"
        videos.mkdir()
        for case, (_, difficulty) in CASES.items():
            for model_label, checkpoint in (("V26", reference), ("V27", trained)):
                output = videos / f"{model_label}_{case}"
                env = dict(ENV, TERRAIN_CASE=case, TERRAIN_DIFFICULTY=str(difficulty), TERRAIN_OUTPUT=str(output))
                run([PYTHON, "-u", str(ROOT / "evaluate.py"), *TASK, "--num_envs", "1", "--seed", "7", "--checkpoint", str(checkpoint), "--headless", "--video", "--video_length", "2400"], f"录制_{model_label}_{case}", 1800, env)
                webm = Path(str(output) + ".webm")
                run([FFMPEG, "-y", "-v", "error", "-i", str(output) + "_video/rl-video-step-0.mp4", "-vf", "fps=50", "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-an", str(webm)], f"编码_{model_label}_{case}", 900)
                run([FFMPEG, "-v", "error", "-i", str(webm), "-f", "null", "-"], f"校验_{model_label}_{case}", 120)
                page = publish(results, trained)
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == original_hash
        subprocess.Popen(["firefox", "--new-window", page.as_uri()], env=ENV)
        status("complete", checkpoint=str(trained), page=str(page))
        print("[全部完成] Firefox已打开：", page, flush=True)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
