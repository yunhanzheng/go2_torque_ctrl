"""Record V49 model 4980 on its successful hard-boxes seed-7 environment."""
import datetime
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PYTHON = "/opt/conda/envs/isaac_sac/bin/python"
FFMPEG = (
    "/opt/conda/envs/isaac_sac/lib/python3.11/"
    "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
MODEL = Path(
    "/home/ubuntu22/logs/rsl_rl/terrain_v49_generalize_23cm/"
    "2026-10-08_09-01-27_mixed55upstairs_keep15fixed23_from_v48_4778_20261008T090121Z/"
    "model_4980.pt"
)
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "video_reports" / STAMP
STEM = "V49_model4980_boxes_hard_seed7_only_env4_visible"


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    OUT.mkdir(parents=True)
    output = OUT / STEM
    env = dict(
        os.environ,
        PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts",
        WANDB_API_KEY=os.environ["WANDB_API_KEY"],
        STRAIGHT_OUTPUT=str(output),
        TERRAIN_CASE="boxes",
        TERRAIN_DIFFICULTY="0.75",
        TERRAIN_SEED="7",
        TERRAIN_COLS="8",
        TERRAIN_TYPE_INDEX="-1",
        VIDEO_ENV_INDEX="4",
        VISIBLE_ENV_INDEX="4",
    )
    command = [
        PYTHON, "-u", str(ROOT / "evaluate_boxes_video_env4.py"),
        "--task", "Isaac-Go2-V49-Generalize23cm-v0",
        "--agent", "rsl_rl_sac_cfg_entry_point",
        "--num_envs", "32", "--seed", "7", "--checkpoint", str(MODEL),
        "--headless", "--video", "--video_length", "500",
    ]
    log = OUT / "record.log"
    with log.open("w") as stream:
        process = subprocess.run(
            command, cwd="/home/ubuntu22", env=env,
            stdout=stream, stderr=subprocess.STDOUT, text=True,
        )
    if process.returncode:
        raise RuntimeError(f"recording exited with {process.returncode}; see {log}")
    mp4 = Path(str(output) + "_video") / "rl-video-step-0.mp4"
    if not mp4.is_file():
        raise FileNotFoundError(mp4)
    webm = OUT / f"{STEM}.webm"
    subprocess.run([
        FFMPEG, "-nostdin", "-y", "-v", "error", "-i", str(mp4),
        "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-an", str(webm),
    ], check=True)
    (ROOT / "latest_video_report.txt").write_text(str(OUT) + "\n")
    print(webm, flush=True)


if __name__ == "__main__":
    main()
