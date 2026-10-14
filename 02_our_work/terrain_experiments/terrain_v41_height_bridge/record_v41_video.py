"""Record V41's strongest retained checkpoint on 10 cm and 12 cm stairs."""
import datetime
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
EVALUATOR = ROOT / "evaluate_height.py"
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
FFMPEG = "/home/ubuntu22/miniforge3/envs/env_isaaclab/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
LATEST = Path((ROOT / "latest.txt").read_text().strip())
RUN = Path(__import__("json").loads((LATEST / "status.json").read_text())["checkpoint"]).parent
MODEL = RUN / "model_2820.pt"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "video_reports" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
TASK = ["--task", "Isaac-Go2-V41-HeightBridge-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]


def run(command, label, env):
    print(f"\n[阶段] {label}", flush=True)
    with (OUT / f"{label}.log").open("w") as stream:
        process = subprocess.Popen(
            command, cwd="/home/ubuntu22", env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"{label} exited with {code}")


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    OUT.mkdir(parents=True)
    for height in (0.10, 0.12):
        cm = round(height * 100)
        stem = f"V41_model2820_upstairs_{cm}cm_40cmstep"
        output = OUT / stem
        difficulty = (height - 0.05) / (0.12 - 0.05)
        env = dict(
            ENV,
            STRAIGHT_OUTPUT=str(output),
            TERRAIN_CASE="stairs_up",
            TERRAIN_DIFFICULTY=str(difficulty),
        )
        run(
            [PYTHON, "-u", str(EVALUATOR), *TASK, "--num_envs", "1", "--seed", "7",
             "--checkpoint", str(MODEL), "--headless", "--video", "--video_length", "500"],
            f"录制_{cm}cm上楼梯",
            env,
        )
        mp4 = Path(str(output) + "_video") / "rl-video-step-0.mp4"
        webm = OUT / f"{stem}.webm"
        run(
            [FFMPEG, "-nostdin", "-y", "-v", "error", "-i", str(mp4),
             "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-an", str(webm)],
            f"编码_{cm}cm",
            env,
        )
        run(
            [FFMPEG, "-nostdin", "-v", "error", "-i", str(webm), "-f", "null", "-"],
            f"校验_{cm}cm",
            env,
        )
        print("[WEBM]", webm, flush=True)
    (ROOT / "latest_video_report.txt").write_text(str(OUT) + "\n")


if __name__ == "__main__":
    main()
