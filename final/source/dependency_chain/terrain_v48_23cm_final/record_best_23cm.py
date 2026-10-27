"""Record V48 model 4778 traversing fixed 23 cm stairs."""
import datetime
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PYTHON = "/opt/conda/envs/isaac_sac/bin/python"
FFMPEG = "/opt/conda/envs/isaac_sac/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
MODEL = Path(
    "/home/ubuntu22/logs/rsl_rl/terrain_v48_23cm_final/"
    "2026-10-07_17-05-30_focused_215to23cm_from_v47_4479_20261007T170522Z/"
    "model_4778.pt"
)
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "video_reports" / STAMP
STEM = "V48_model4778_upstairs_23cm_30cmstep"


def main():
    OUT.mkdir(parents=True)
    output = OUT / STEM
    env = dict(
        os.environ,
        PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts",
        WANDB_API_KEY=os.environ["WANDB_API_KEY"],
        STRAIGHT_OUTPUT=str(output),
        TERRAIN_CASE="stairs_up_23_anchor",
        TERRAIN_DIFFICULTY="0",
    )
    command = [
        PYTHON, "-u", str(ROOT / "evaluate_dual.py"),
        "--task", "Isaac-Go2-V48-23cmFinal-v0", "--agent", "rsl_rl_sac_cfg_entry_point",
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
