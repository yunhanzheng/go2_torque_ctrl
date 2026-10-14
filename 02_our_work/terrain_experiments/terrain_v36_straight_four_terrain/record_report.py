"""Record the selected V36 checkpoint on plane and six terrain variants."""
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
MODEL = Path("/home/ubuntu22/logs/rsl_rl/terrain_v36_straight_four_terrain/2026-09-24_00-42-28_four_terrain_from_v35_2525_20260924T004222Z/model_2550.pt")
EVAL = Path((ROOT / "latest_evaluation.txt").read_text().strip())
MATRIX = json.loads((EVAL / "summary.json").read_text())["evaluated_matrix"]["model_2550"]
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "reports" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
TASK = ["--task", "Isaac-Go2-V35-StraightFourTerrain-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]
CASES = {
    "plane": ("平地直线", 0.0), "random_rough": ("随机粗糙面", 0.75),
    "boxes": ("随机方块", 0.75), "slope_up": ("上坡", 0.75),
    "slope_down": ("下坡", 0.75), "stairs_up": ("上楼梯", 0.7142857142857143),
    "stairs_down": ("下楼梯", 0.7142857142857143),
}


def run(command, label, env=ENV):
    print(f"\n[阶段] {label}", flush=True)
    log = OUT / f"{label}.log"
    with log.open("w") as stream:
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
    videos = []
    for case, (label, difficulty) in CASES.items():
        output = OUT / case
        env = dict(ENV, STRAIGHT_OUTPUT=str(output), TERRAIN_CASE=case,
                   TERRAIN_DIFFICULTY=str(difficulty))
        run([PYTHON, "-u", str(EVALUATOR), *TASK, "--num_envs", "1", "--seed", "7",
             "--checkpoint", str(MODEL), "--headless", "--video", "--video_length", "450"],
            f"录制_{label}", env)
        mp4 = Path(str(output) + "_video") / "rl-video-step-0.mp4"
        webm = OUT / f"{case}.webm"
        run([FFMPEG, "-nostdin", "-y", "-v", "error", "-i", str(mp4),
             "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-an", str(webm)], f"编码_{label}")
        run([FFMPEG, "-nostdin", "-v", "error", "-i", str(webm), "-f", "null", "-"], f"校验_{label}")
        videos.append((case, label, webm))
    rows = []
    cards = []
    for case, label, webm in videos:
        data = MATRIX[case]
        rows.append(f"<tr><td>{label}</td><td>{data['success']}/32</td><td>{data['mean_final_abs_y_m']:.3f}</td></tr>")
        cards.append(f"<article><h2>{label}（{data['success']}/32）</h2><video controls loop muted preload='metadata' src='{webm.name}'></video></article>")
    page = OUT / "V36直线四地形视频报告.html"
    page.write_text("<!doctype html><meta charset='utf-8'><title>V36直线四地形视频报告</title>"
        "<style>body{font-family:sans-serif;margin:24px;background:#eee}table{border-collapse:collapse;background:white}td,th{border:1px solid #999;padding:8px}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:20px}article{background:white;padding:14px}video{width:100%}</style>"
        f"<h1>V36 model_2550：直线与四地形</h1><p>Checkpoint：{html.escape(str(MODEL))}</p>"
        "<p>固定 seed=7；32 环境；严格要求 |y|&lt;0.25 m、航向误差&lt;5°。</p>"
        "<table><tr><th>条件</th><th>成功</th><th>平均最终|y| (m)</th></tr>" + "".join(rows) + "</table>"
        "<main>" + "".join(cards) + "</main>")
    (ROOT / "latest_report.txt").write_text(str(OUT) + "\n")
    print("[完成]", page, flush=True)
    subprocess.Popen(["firefox", "--new-window", page.as_uri()], env=ENV)


if __name__ == "__main__":
    main()

