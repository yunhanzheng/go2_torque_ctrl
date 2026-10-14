"""Measure actual final-policy foot clearance at representative stair heights."""
import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
EVALUATOR = ROOT / "evaluate_clearance.py"
TASK = ["--task", "Isaac-Go2-V42-23cmStraight-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "clearance_measurements" / STAMP


def main():
    status = json.loads((Path((ROOT / "latest.txt").read_text().strip()) / "status.json").read_text())
    model = Path(status["checkpoint"])
    OUT.mkdir(parents=True)
    summary = {}
    for height in (0.10, 0.12, 0.23):
        cm = round(height * 100)
        output = OUT / f"model_3019_{cm}cm"
        env = dict(ENV, STRAIGHT_OUTPUT=str(output), TERRAIN_CASE="stairs_up",
                   TERRAIN_DIFFICULTY=str((height - 0.05) / 0.18))
        command = [PYTHON, "-u", str(EVALUATOR), *TASK, "--num_envs", "32", "--seed", "7",
                   "--checkpoint", str(model), "--headless"]
        print(f"[测量真实抬脚高度] {cm}cm", flush=True)
        with (OUT / f"model_3019_{cm}cm.log").open("w") as stream:
            process = subprocess.Popen(command, cwd="/home/ubuntu22", env=env,
                                       stdout=stream, stderr=subprocess.STDOUT, text=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"{cm}cm exited with {code}")
        data = json.loads(Path(str(output) + ".json").read_text())
        summary[f"{cm}cm"] = data
        print(f"[结果] mean episode max={data['mean_episode_max_foot_clearance_m']:.3f}m, "
              f"median={data['median_episode_max_foot_clearance_m']:.3f}m, "
              f"global max={data['global_max_foot_clearance_m']:.3f}m", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (ROOT / "latest_clearance_measurement.txt").write_text(str(OUT) + "\n")


if __name__ == "__main__":
    main()
