"""Record V51 videos with 5 random seeds, 2000 frames each."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import random
import sys


# Parse command line arguments
parser = argparse.ArgumentParser(description="Record V51 videos")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
args = parser.parse_args()


ROOT = Path(__file__).resolve().parent
# Auto-detect Python environment (works on both local and server)
if os.path.exists("/opt/conda/envs/isaac_sac/bin/python"):
    # Server environment
    PYTHON = "/opt/conda/envs/isaac_sac/bin/python"
elif os.path.exists("/home/ubuntu22/.holosoma_deps/miniconda3/envs/isaac_sac/bin/python"):
    # Local environment
    PYTHON = "/home/ubuntu22/.holosoma_deps/miniconda3/envs/isaac_sac/bin/python"
else:
    # Fallback to current Python
    PYTHON = sys.executable
# Find ffmpeg in the current environment
FFMPEG_CANDIDATES = [
    Path(PYTHON).parent.parent / "lib" / "python3.11" / "site-packages" / "imageio_ffmpeg" / "binaries" / "ffmpeg-linux-x86_64-v7.0.2",
    Path(PYTHON).parent.parent / "lib" / "python3.11" / "site-packages" / "imageio_ffmpeg" / "binaries" / "ffmpeg",
]
FFMPEG = None
for candidate in FFMPEG_CANDIDATES:
    if candidate.exists():
        FFMPEG = str(candidate)
        break
if FFMPEG is None:
    # Fallback to system ffmpeg
    FFMPEG = "ffmpeg"

# Use checkpoint from command line (convert to absolute path)
MODEL = Path(args.checkpoint).resolve()

STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "video_reports" / STAMP

# 5 fixed seeds for reproducibility
SEEDS = [42, 123, 456, 789, 1024]


def main():
    print(f"[record] Using model: {MODEL}")
    print(f"[record] Seeds: {SEEDS}")
    print(f"[record] Total: {len(SEEDS)} videos, 500 frames each")
    print("=" * 60)
    OUT.mkdir(parents=True)

    for idx, seed in enumerate(SEEDS, 1):
        print(f"\n[{idx}/{len(SEEDS)}] Recording seed={seed}...", flush=True)
        stem = f"V51_seed{seed}_500frames"
        output = OUT / stem
        # Build PYTHONPATH dynamically based on current environment
        source_root = ROOT.parent
        pythonpath_parts = [
            str(source_root),
            str(source_root / "dependency_chain"),
            str(source_root / "base_adapter"),
            str(source_root / "base_adapter" / "initial_stair_source"),
            str(ROOT),
        ]
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        pythonpath = ":".join(pythonpath_parts)
        
        env = dict(
            os.environ,
            PYTHONPATH=pythonpath,
            WANDB_API_KEY=os.environ["WANDB_API_KEY"],
        )
        command = [
            PYTHON,
            "-u",
            str(ROOT / "evaluate_video.py"),
            "--task",
            "Isaac-Go2-V51-NativeTerrain-v0",
            "--agent",
            "rsl_rl_sac_cfg_entry_point",
            "--num_envs",
            "1",  # Single environment for video recording
            "--seed",
            str(seed),
            "--checkpoint",
            str(MODEL),
            "--logger",
            "wandb",
            "--log_project_name",
            "isaac_sac_pipeline",
            "--headless",
            "--video",
            "--video_length",
            "500",
            "--video_folder",
            str(output) + "_video",
        ]
        log = OUT / f"{stem}.log"
        print(f"[{idx}/{len(SEEDS)}] Command: {' '.join(command)}", flush=True)
        print(f"[{idx}/{len(SEEDS)}] Log file: {log}", flush=True)
        start_time = datetime.datetime.now()
        
        # Run with real-time output display
        with log.open("w") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            # Read output in real-time and write to both terminal and log file
            for line in process.stdout:
                print(f"  [{seed}] {line}", end="", flush=True)
                log_file.write(line)
            process.wait()
        
        elapsed = datetime.datetime.now() - start_time
        if process.returncode:
            print(f"[{idx}/{len(SEEDS)}] FAILED after {elapsed.seconds}s", flush=True)
            raise RuntimeError(f"seed={seed} exited with {process.returncode}; see {log}")
        print(f"[{idx}/{len(SEEDS)}] Completed in {elapsed.seconds}s", flush=True)

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
    print("=" * 60)
    print(f"[record] All done! {len(SEEDS)} videos saved to: {OUT}")


if __name__ == "__main__":
    main()
