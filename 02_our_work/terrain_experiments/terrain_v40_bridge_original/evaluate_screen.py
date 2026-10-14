"""Strictly screen V37 and representative V40 checkpoints."""
import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
EVALUATOR = ROOT / "evaluate_bridge.py"
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "evaluations" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
TASK = ["--task", "Isaac-Go2-V40-BridgeOriginal-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]


def difficulty_for_height(height_m):
    return (height_m - 0.05) / (0.12 - 0.05)


def evaluate(model, case, difficulty, prefix):
    output = OUT / prefix
    env = dict(ENV, STRAIGHT_OUTPUT=str(output), TERRAIN_CASE=case,
               TERRAIN_DIFFICULTY=str(difficulty))
    command = [PYTHON, "-u", str(EVALUATOR), *TASK, "--num_envs", "32", "--seed", "7",
               "--checkpoint", str(model), "--headless"]
    print(f"\n[严格评估] {model.name} / {case} / difficulty={difficulty:.6f}", flush=True)
    log = OUT / f"{prefix}.log"
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd="/home/ubuntu22", env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
    if status["stage"] != "complete":
        raise RuntimeError(f"Training is not complete: {status}")
    run = Path(status["checkpoint"]).parent
    source = Path(status["source_checkpoint"])
    candidates = [source, run / "model_2900.pt", run / "model_3000.pt", run / "model_3098.pt"]
    candidates = [path for path in candidates if path.is_file()]
    OUT.mkdir(parents=True)

    results = {}
    for model in candidates:
        label = "V37_source" if model == source else model.stem
        results[label] = {
            "checkpoint": str(model),
            "stairs_up_10cm": evaluate(model, "stairs_up", difficulty_for_height(0.10),
                                         f"{label}_stairs_up_10cm"),
            "stairs_up_12cm": evaluate(model, "stairs_up", difficulty_for_height(0.12),
                                         f"{label}_stairs_up_12cm"),
            "plane": evaluate(model, "plane", 0.0, f"{label}_plane"),
        }

    def rank(item):
        data = item[1]
        ten = data["stairs_up_10cm"]["success"]
        twelve = data["stairs_up_12cm"]["success"]
        plane = data["plane"]["success"]
        passes_retention = ten >= 26 and plane >= 26
        return (passes_retention, twelve, ten, plane)

    best_label, best = max(results.items(), key=rank)
    compact = {
        label: {
            "10cm": data["stairs_up_10cm"]["success"],
            "12cm": data["stairs_up_12cm"]["success"],
            "plane": data["plane"]["success"],
        }
        for label, data in results.items()
    }
    summary = {
        "best_label": best_label,
        "best_checkpoint": best["checkpoint"],
        "results": results,
        "compact": compact,
        "selection_gate": "10cm >= 26/32 and plane >= 26/32, then maximize 12cm",
        "criterion": "32 environments; strict straight gate; first episode; no reset",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (ROOT / "latest_evaluation.txt").write_text(str(OUT) + "\n")
    print("\n[V40_SCREEN_RESULT]", json.dumps({
        "best": best_label,
        "checkpoint": best["checkpoint"],
        "compact": compact,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
