"""Record successful V50 runs on a slope, boxes, and fixed 23 cm stairs."""
import datetime
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
FFMPEG = (
    "/home/ubuntu22/miniforge3/envs/env_isaaclab/lib/python3.11/"
    "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
MODEL = Path(
    "/home/ubuntu22/logs/rsl_rl/terrain_v50_curriculum_recovery/"
    "2026-10-08_14-10-57_level6_trial_from_v49_4980_20261008T141051Z/model_5099.pt"
)
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "video_reports" / STAMP

# Successful environment indices identified from the original seed-7 audit.
CASES = (
    ("slope_up", "slope_up", 0.75, 0),
    ("boxes", "boxes", 0.75, 5),
    ("upstairs_23cm", "stairs_up_23_anchor", 1.0, 0),
)


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    OUT.mkdir(parents=True)

    for label, terrain, difficulty, env_index in CASES:
        stem = f"V50_model5099_{label}_seed7_env{env_index}"
        output = OUT / stem
        env = dict(
            os.environ,
            PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts",
            WANDB_MODE="offline",
            STRAIGHT_OUTPUT=str(output),
            TERRAIN_CASE=terrain,
            TERRAIN_DIFFICULTY=str(difficulty),
            TERRAIN_SEED="7",
            TERRAIN_COLS="8",
            TERRAIN_TYPE_INDEX="-1",
            VIDEO_ENV_INDEX=str(env_index),
            VISIBLE_ENV_INDEX=str(env_index),
        )
        command = [
            PYTHON,
            "-u",
            str(ROOT / "evaluate_video.py"),
            "--task",
            "Isaac-Go2-V50-CurriculumRecovery-v0",
            "--agent",
            "rsl_rl_sac_cfg_entry_point",
            "--num_envs",
            "32",
            "--seed",
            "7",
            "--checkpoint",
            str(MODEL),
            "--headless",
            "--video",
            "--video_length",
            "500",
        ]
        log = OUT / f"{stem}.log"
        print(f"[record] {label}", flush=True)
        with log.open("w") as stream:
            process = subprocess.run(
                command,
                cwd="/home/ubuntu22",
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if process.returncode:
            raise RuntimeError(f"{label} exited with {process.returncode}; see {log}")

        mp4 = Path(str(output) + "_video") / "rl-video-step-0.mp4"
        if not mp4.is_file():
            raise FileNotFoundError(mp4)
        webm = OUT / f"{stem}.webm"
        subprocess.run(
            [
                FFMPEG,
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-i",
                str(mp4),
                "-c:v",
                "libvpx-vp9",
                "-crf",
                "30",
                "-b:v",
                "0",
                "-an",
                str(webm),
            ],
            check=True,
        )
        print(webm, flush=True)

    (ROOT / "latest_video_report.txt").write_text(str(OUT) + "\n")


if __name__ == "__main__":
    main()
