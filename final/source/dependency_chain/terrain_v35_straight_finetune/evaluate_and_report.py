"""Screen V35 checkpoints, record the best one, and open a WebM report."""
import datetime
import html
import json
import os
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parent
PYTHON = "/opt/conda/envs/isaac_sac/bin/python"
FFMPEG = "/opt/conda/envs/isaac_sac/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
RUN = Path("/home/ubuntu22/logs/rsl_rl/terrain_v35_straight_finetune/2026-09-23_23-49-41_straight_from_v34_20260923T234933Z")
V34 = Path("/home/ubuntu22/logs/rsl_rl/terrain_v34_scratch_four_terrain/2026-09-20_22-45-03_mixed_curriculum_20260920T224437Z/model_2396.pt")
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "evaluations" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_API_KEY=os.environ["WANDB_API_KEY"])
TASK = ["--task", "Isaac-Go2-V35-StraightFourTerrain-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]


def run(command, label, env):
    print(f"\n[阶段] {label}\n[命令] {' '.join(command)}", flush=True)
    log = OUT / f"{label}.log"
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd="/home/ubuntu22", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"{label} exited with {code}")


def evaluate(model, name, num_envs=32, video=False):
    output = OUT / name
    env = dict(ENV, STRAIGHT_OUTPUT=str(output))
    command = [PYTHON, "-u", str(ROOT / "evaluate_straight.py"), *TASK,
               "--num_envs", str(num_envs), "--seed", "7", "--checkpoint", str(model), "--headless"]
    if video:
        command += ["--video", "--video_length", "450"]
    run(command, ("录制_" if video else "评估_") + name, env)
    return json.loads(Path(str(output) + ".json").read_text())


def main():
    OUT.mkdir(parents=True)
    candidates = [V34] + [RUN / f"model_{iteration}.pt" for iteration in (2475, 2525, 2575, 2625, 2675, 2695)]
    results = {}
    for model in candidates:
        if model.is_file():
            results[model.stem if model != V34 else "V34_2396"] = evaluate(model, model.stem if model != V34 else "V34_2396")
    v35 = {key: value for key, value in results.items() if key.startswith("model_")}
    best_name, best = min(
        v35.items(),
        key=lambda item: (-item[1]["success"], item[1]["mean_final_abs_y_m"], item[1]["mean_max_abs_heading_deg"]),
    )
    best_model = RUN / f"{best_name}.pt"
    video_result = evaluate(best_model, f"{best_name}_video", num_envs=1, video=True)
    mp4 = OUT / f"{best_name}_video_video" / "rl-video-step-0.mp4"
    webm = OUT / f"{best_name}_straight.webm"
    run([FFMPEG, "-nostdin", "-y", "-v", "error", "-i", str(mp4),
         "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-an", str(webm)], "编码_WebM", ENV)
    run([FFMPEG, "-nostdin", "-v", "error", "-i", str(webm), "-f", "null", "-"], "校验_WebM", ENV)
    rows = []
    for name, data in results.items():
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{data['success']}/32</td>"
            f"<td>{data['mean_final_abs_y_m']:.3f}</td><td>{data['mean_max_abs_heading_deg']:.2f}</td></tr>"
        )
    page = OUT / "V35直线微调评估报告.html"
    page.write_text(
        "<!doctype html><meta charset='utf-8'><title>V35直线微调评估</title>"
        "<style>body{font-family:sans-serif;margin:28px;background:#eee}section{background:white;padding:20px;max-width:1100px}table{border-collapse:collapse}td,th{border:1px solid #999;padding:8px}video{width:100%;max-width:1000px}</style>"
        f"<section><h1>V35 直线闭环微调</h1><p>最佳 checkpoint：{html.escape(str(best_model))}</p>"
        "<p>门禁：前进超过3.5 m、横向偏移小于0.25 m、航向误差小于5°并稳定0.5 s。</p>"
        "<table><tr><th>模型</th><th>通过</th><th>平均最终|y| (m)</th><th>平均最大航向误差 (°)</th></tr>"
        + "".join(rows) + "</table>"
        f"<h2>{html.escape(best_name)} WebM</h2><video controls autoplay loop muted src='{webm.name}'></video></section>"
    )
    summary = {"best_checkpoint": str(best_model), "best_name": best_name, "results": results,
               "video_result": video_result, "webm": str(webm), "report": str(page)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (ROOT / "latest_evaluation.txt").write_text(str(OUT) + "\n")
    print("[完成]", json.dumps({"best": str(best_model), "webm": str(webm), "report": str(page)}, ensure_ascii=False), flush=True)
    subprocess.Popen(["firefox", "--new-window", page.as_uri()], env=ENV)


if __name__ == "__main__":
    main()

