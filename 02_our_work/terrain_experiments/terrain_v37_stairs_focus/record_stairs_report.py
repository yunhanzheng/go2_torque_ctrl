"""Record V37's selected checkpoint on 8 cm and 10 cm upstairs."""
import datetime
import html
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
EVALUATOR = Path("/home/ubuntu22/terrain_v35_straight_finetune/evaluate_straight.py")
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
FFMPEG = "/home/ubuntu22/miniforge3/envs/env_isaaclab/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
SUMMARY = Path((ROOT / "latest_evaluation.txt").read_text().strip()) / "summary.json"
RESULTS = json.loads(SUMMARY.read_text())
MODEL = Path(RESULTS["best_checkpoint"])
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "video_reports" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
TASK = ["--task", "Isaac-Go2-V35-StraightFourTerrain-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]


def run(command, label, env):
    print(f"\n[阶段] {label}", flush=True)
    with (OUT / f"{label}.log").open("w") as stream:
        process = subprocess.Popen(command, cwd="/home/ubuntu22", env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"{label} exited with {code}")


def main():
    OUT.mkdir(parents=True)
    cards = []
    for height in (0.08, 0.10):
        cm = round(height * 100)
        stem = f"V37_upstairs_{cm}cm"
        output = OUT / stem
        difficulty = (height - 0.05) / (0.12 - 0.05)
        env = dict(ENV, STRAIGHT_OUTPUT=str(output), TERRAIN_CASE="stairs_up",
                   TERRAIN_DIFFICULTY=str(difficulty))
        run([PYTHON, "-u", str(EVALUATOR), *TASK, "--num_envs", "1", "--seed", "7",
             "--checkpoint", str(MODEL), "--headless", "--video", "--video_length", "500"],
            f"录制_{cm}cm上楼梯", env)
        mp4 = Path(str(output) + "_video") / "rl-video-step-0.mp4"
        webm = OUT / f"{stem}.webm"
        run([FFMPEG, "-nostdin", "-y", "-v", "error", "-i", str(mp4),
             "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-an", str(webm)],
            f"编码_{cm}cm", env)
        run([FFMPEG, "-nostdin", "-v", "error", "-i", str(webm), "-f", "null", "-"],
            f"校验_{cm}cm", env)
        score = RESULTS["height_sweep"][f"{height:.2f}"]["success"]
        cards.append(f"<article><h2>{cm} cm 上楼梯（严格评估 {score}/32）</h2>"
                     f"<video controls loop muted autoplay src='{webm.name}'></video></article>")

    page = OUT / "V37楼梯视频报告.html"
    page.write_text("<!doctype html><meta charset='utf-8'><title>V37 楼梯视频</title>"
        "<style>body{font-family:sans-serif;margin:24px;background:#eee}main{display:grid;grid-template-columns:1fr 1fr;gap:18px}article{background:white;padding:14px}video{width:100%}</style>"
        f"<h1>V37 无高度观测 SAC：楼梯视频</h1><p>Checkpoint：{html.escape(str(MODEL))}</p>"
        "<p>固定 seed=7，左侧 8 cm，右侧 10 cm；策略输入不含高度扫描。</p><main>" + "".join(cards) + "</main>")
    (ROOT / "latest_video_report.txt").write_text(str(OUT) + "\n")
    print(f"\n[完成] {page}", flush=True)
    subprocess.Popen(["firefox", "--new-window", page.as_uri()], env=ENV)


if __name__ == "__main__":
    main()
