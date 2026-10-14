"""Strict fixed-height audit of the final V42 checkpoint."""
import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
EVALUATOR = ROOT / "evaluate_height.py"
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "evaluations" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
TASK = ["--task", "Isaac-Go2-V42-23cmStraight-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]


def difficulty_for_height(height_m):
    return (height_m - 0.05) / (0.23 - 0.05)


def evaluate(model, case, difficulty, prefix):
    output = OUT / prefix
    env = dict(ENV, STRAIGHT_OUTPUT=str(output), TERRAIN_CASE=case, TERRAIN_DIFFICULTY=str(difficulty))
    command = [
        PYTHON, "-u", str(EVALUATOR), *TASK, "--num_envs", "32", "--seed", "7",
        "--checkpoint", str(model), "--headless",
    ]
    print(f"\n[V42严格评估] {model.name} / {case} / difficulty={difficulty:.6f}", flush=True)
    log = OUT / f"{prefix}.log"
    with log.open("w") as stream:
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
        raise RuntimeError(f"{prefix} exited with {code}")
    return json.loads(Path(str(output) + ".json").read_text())


def main():
    latest = Path((ROOT / "latest.txt").read_text().strip())
    status = json.loads((latest / "status.json").read_text())
    model = Path(status["checkpoint"])
    OUT.mkdir(parents=True)
    results = {}
    for height in (0.10, 0.12, 0.15, 0.18, 0.20, 0.23):
        cm = round(height * 100)
        results[f"{cm}cm"] = evaluate(
            model, "stairs_up", difficulty_for_height(height), f"model_3019_stairs_up_{cm}cm"
        )
    results["plane"] = evaluate(model, "plane", 0.0, "model_3019_plane")
    compact = {
        name: {
            "success": data["success"],
            "terminated": data["terminated"],
            "mean_max_abs_y_m": data["mean_max_abs_y_m"],
            "mean_max_abs_heading_deg": data["mean_max_abs_heading_deg"],
        }
        for name, data in results.items()
    }
    summary = {
        "checkpoint": str(model),
        "compact": compact,
        "results": results,
        "criterion": "32 environments; x>3.5m; |y|<0.25m; |heading|<5deg; upright 0.5s; first episode; no reset",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (ROOT / "latest_evaluation.txt").write_text(str(OUT) + "\n")
    print("\n[V42_FINAL_RESULT]", json.dumps(compact, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
