"""Record V47 model 4479 traversing 22 cm stairs."""
import datetime
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
FFMPEG = "/home/ubuntu22/miniforge3/envs/env_isaaclab/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
MODEL = Path(
    "/home/ubuntu22/logs/rsl_rl/terrain_v47_22cm_breakthrough/"
    "2026-10-07_12-58-38_focused_20to22cm_from_v46_4180_20261007T125832Z/"
    "model_4479.pt"
)
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "video_reports" / STAMP
STEM = "V47_model4479_upstairs_22cm_30cmstep"


def main():
    OUT.mkdir(parents=True)
    output = OUT / STEM
    env = dict(
        os.environ,
        PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts",
        WANDB_MODE="offline",
        STRAIGHT_OUTPUT=str(output),
        TERRAIN_CASE="stairs_up",
        TERRAIN_DIFFICULTY="1.0",
    )
    command = [
        PYTHON, "-u", str(ROOT / "evaluate_dual.py"),
        "--task", "Isaac-Go2-V47-22cmBreakthrough-v0", "--agent", "rsl_rl_sac_cfg_entry_point",
        "--num_envs", "1", "--seed", "7", "--checkpoint", str(MODEL), "--headless",
        "--video", "--video_length", "500",
    ]
    with (OUT / "record.log").open("w") as stream:
        process = subprocess.run(command, cwd="/home/ubuntu22", env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise RuntimeError(process.returncode)
    mp4 = Path(str(output) + "_video") / "rl-video-step-0.mp4"
    webm = OUT / f"{STEM}.webm"
    subprocess.run([
        FFMPEG, "-nostdin", "-y", "-v", "error", "-i", str(mp4),
        "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-an", str(webm),
    ], check=True)
    (ROOT / "latest_video_report.txt").write_text(str(OUT) + "\n")
    print(webm, flush=True)


if __name__ == "__main__":
    main()
