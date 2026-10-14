"""Find one reproducible hard-box terrain seed traversed by V49 model 4980."""
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
MODEL = Path(
    "/home/ubuntu22/logs/rsl_rl/terrain_v49_generalize_23cm/"
    "2026-10-08_09-01-27_mixed55upstairs_keep15fixed23_from_v48_4778_20261008T090121Z/"
    "model_4980.pt"
)
OUT = ROOT / "single_boxes_seed_scan"


def main():
    OUT.mkdir(exist_ok=True)
    for terrain_seed in range(32):
        output = OUT / f"seed_{terrain_seed:02d}"
        env = dict(
            os.environ,
            PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts",
            WANDB_MODE="offline",
            STRAIGHT_OUTPUT=str(output),
            TERRAIN_CASE="boxes",
            TERRAIN_DIFFICULTY="0.75",
            TERRAIN_SEED=str(terrain_seed),
            TERRAIN_COLS="1",
            VIDEO_ENV_INDEX="0",
        )
        command = [
            PYTHON, "-u", str(ROOT / "evaluate_boxes_video_env4.py"),
            "--task", "Isaac-Go2-V49-Generalize23cm-v0",
            "--agent", "rsl_rl_sac_cfg_entry_point",
            "--num_envs", "1", "--seed", "7", "--checkpoint", str(MODEL), "--headless",
        ]
        with (OUT / f"seed_{terrain_seed:02d}.log").open("w") as stream:
            result = subprocess.run(
                command, cwd="/home/ubuntu22", env=env,
                stdout=stream, stderr=subprocess.STDOUT, text=True,
            )
        if result.returncode:
            raise RuntimeError(f"seed {terrain_seed} evaluator exited with {result.returncode}")
        data = json.loads(Path(str(output) + ".json").read_text())
        print(
            f"seed={terrain_seed} physical={data['physical_success']} "
            f"max_x={data['max_x_m'][0]:.3f}",
            flush=True,
        )
        if data["physical_success"] == 1:
            (ROOT / "single_boxes_seed.txt").write_text(str(terrain_seed) + "\n")
            return
    raise RuntimeError("No successful single-environment boxes seed found in 0..31")


if __name__ == "__main__":
    main()
