"""Visible V24-to-V26 training, matched evaluation, WebM recording and Firefox report."""
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

from catalog import DIFFICULTIES, NAMES


ROOT = Path(__file__).resolve().parent
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
FFMPEG = "/home/ubuntu22/miniforge3/envs/env_isaaclab/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
SOURCE = Path("/data/sac/saved_models/ETH_SAC_V24_10CM/model_299.pt")
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
BATCH = ROOT / "results" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
TASK = ["--task", "Isaac-Go2-Terrain-V26-Official3-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]


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
    rows = ["| 地形 | 难度 | V24原模型 | V26三类地形微调 |", "|---|---|---:|---:|"]
    for case, label in NAMES.items():
        for level in DIFFICULTIES:
            before = results.get("V24", {}).get(case, {}).get(level)
            after = results.get("V26", {}).get(case, {}).get(level)
            before_text = f'{before["success"]}/32' if before else "待评估"
            after_text = f'{after["success"]}/32' if after else "待评估"
            rows.append(f"| {label} | {level} | {before_text} | {after_text} |")
    (BATCH / "结果.md").write_text(
        "# V24 → V26：官方 Rough 非楼梯三类地形\n\n"
        + "\n".join(rows)
        + "\n\n三类为随机粗糙地面、随机方块和斜坡；上/下坡是斜坡的一类两个方向。"
        + "固定 seed=7、首回合、3.5 m 路线穿越率；每条件32环境。\n\n模型："
        + str(checkpoint)
        + "\n"
    )
    cards = []
    for case, label in NAMES.items():
        video = BATCH / "videos" / f"V26_{case}_hard.webm"
        if video.exists():
            data = results.get("V26", {}).get(case, {}).get("hard", {})
            cards.append(
                f"<article><h2>{html.escape(label)}（hard：{data.get('success', '?')}/32）</h2>"
                f'<video controls loop muted preload="metadata" src="videos/{video.name}"></video></article>'
            )
    page = BATCH / "视频演示.html"
    page.write_text(
        '<!doctype html><meta charset="utf-8"><title>V26 官方 Rough 三类地形</title>'
        '<style>body{font-family:sans-serif;margin:24px;background:#eee}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}video{width:100%}article{background:white;padding:16px}pre{white-space:pre-wrap}</style>'
        "<h1>V24 初始化 → V26 官方 Rough 非楼梯三类地形</h1>"
        "<p>随机粗糙地面、随机方块／不规则台阶、斜坡（含上坡和下坡）。视频为 hard 条件，保留失败和重置。</p>"
        "<details open><summary>V24 与 V26 批量结果</summary><pre>"
        + html.escape("\n".join(rows))
        + "</pre></details><main>"
        + "".join(cards)
        + "</main>"
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
    reference_dir = BATCH / "v24_reference"
    reference_dir.mkdir()
    reference = reference_dir / "model_299.pt"
    shutil.copy2(SOURCE, reference)
    name = f"v26_{STAMP}"
    results = {"V24": {}, "V26": {}}
    try:
        run([PYTHON, "-u", str(ROOT / "preflight.py"), str(BATCH / "catalog.json")], "三类地形生成检查", 600)
        catalog = json.loads((BATCH / "catalog.json").read_text())
        assert len(catalog) == 8, catalog
        run(
            [PYTHON, "-u", str(ROOT / "train.py"), *TASK, "--num_envs", "64", "--max_iterations", "29", "--seed", "1", "--resume", "--run_name", f"smoke_{STAMP}", "--headless", "agent.algorithm.replay_buffer_size=8192", "agent.algorithm.num_mini_batches=2", "agent.algorithm.mini_batch_size=128"],
            "奖励与更新检查",
            600,
        )
        run(
            [PYTHON, "-u", str(ROOT / "train.py"), *TASK, "--num_envs", "4096", "--max_iterations", "600", "--seed", "1", "--resume", "--run_name", name, "--headless"],
            "正式训练600轮",
            10800,
        )
        paths = list(Path("/home/ubuntu22/logs/rsl_rl/terrain_v26_official3_from_v24").glob(f"*_{name}/model_599.pt"))
        assert len(paths) == 1, paths
        trained = paths[0]
        audit = json.loads((trained.parent / "audit.json").read_text())
        assert audit["finite"] and audit["iteration"] == 599 and audit["actor_change"] > 0, audit
        for model_label, checkpoint in (("V24", reference), ("V26", trained)):
            for case in NAMES:
                results[model_label][case] = {}
                for level, difficulty in DIFFICULTIES.items():
                    stem = f"{model_label}_{case}_{level}"
                    output = BATCH / stem
                    env = dict(ENV, TERRAIN_CASE=case, TERRAIN_DIFFICULTY=str(difficulty), TERRAIN_OUTPUT=str(output))
                    run([PYTHON, "-u", str(ROOT / "evaluate.py"), *TASK, "--num_envs", "32", "--seed", "7", "--checkpoint", str(checkpoint), "--headless"], f"评估_{stem}", 1200, env)
                    data = json.loads(Path(str(output) + ".json").read_text())
                    assert data["num_envs"] == 32 and sum(data[key] for key in ("success", "terminated", "other_no_success")) == 32
                    results[model_label][case][level] = data
                    publish(results, trained)
        videos = BATCH / "videos"
        videos.mkdir()
        for case in NAMES:
            stem = f"V26_{case}_hard"
            output = videos / stem
            env = dict(ENV, TERRAIN_CASE=case, TERRAIN_DIFFICULTY=str(DIFFICULTIES["hard"]), TERRAIN_OUTPUT=str(output))
            run([PYTHON, "-u", str(ROOT / "evaluate.py"), *TASK, "--num_envs", "1", "--seed", "7", "--checkpoint", str(trained), "--headless", "--video", "--video_length", "2400"], f"录制_{stem}", 1800, env)
            webm = Path(str(output) + ".webm")
            run([FFMPEG, "-y", "-v", "error", "-i", str(output) + "_video/rl-video-step-0.mp4", "-vf", "fps=50", "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-an", str(webm)], f"编码_{stem}", 900)
            run([FFMPEG, "-v", "error", "-i", str(webm), "-f", "null", "-"], f"校验_{stem}", 120)
            page = publish(results, trained)
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == original_hash
        subprocess.Popen(["firefox", "--new-window", page.as_uri()], env=ENV)
        status("complete", checkpoint=str(trained), page=str(page))
        print("[全部完成] Firefox 已打开：", page, flush=True)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
