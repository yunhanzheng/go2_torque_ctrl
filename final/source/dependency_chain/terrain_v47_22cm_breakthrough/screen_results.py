"""Screen V47 checkpoints at 22/21 cm and audit 20 cm retention."""
import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
EVALUATOR = ROOT / "evaluate_dual.py"
TASK = ["--task", "Isaac-Go2-V47-22cmBreakthrough-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "evaluations" / STAMP
SOURCE = Path(
    "/home/ubuntu22/saved_models/SATA_STAIRS_22CM_FRONTIER_2026-10-07/"
    "V46_22CM_4OF32_model_4180.pt"
)


def evaluate(model, height):
    cm = round(height * 100)
    label = "V46_model4180_source" if model == SOURCE else model.stem
    stem = f"{label}_{cm}cm"
    output = OUT / stem
    env = dict(ENV, STRAIGHT_OUTPUT=str(output), TERRAIN_CASE="stairs_up",
               TERRAIN_DIFFICULTY=str((height - 0.20) / 0.02))
    command = [PYTHON, "-u", str(EVALUATOR), *TASK, "--num_envs", "32", "--seed", "7",
               "--checkpoint", str(model), "--headless"]
    print(f"[V47评估] {label} / {cm}cm", flush=True)
    with (OUT / f"{stem}.log").open("w") as stream:
        process = subprocess.run(command, cwd="/home/ubuntu22", env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise RuntimeError(f"{stem} exited with {process.returncode}")
    data = json.loads(Path(str(output) + ".json").read_text())
    print(f"[结果] physical={data['physical_success']}/32 strict={data['success']}/32 "
          f"fall={data['terminated']}", flush=True)
    return data


def score(values):
    return tuple(
        item for case in ("22cm", "21cm")
        for item in (values[case]["physical_success"], values[case]["success"],
                     -values[case]["terminated"])
    )


def main():
    latest = Path((ROOT / "latest.txt").read_text().strip())
    status = json.loads((latest / "status.json").read_text())
    if status["stage"] != "complete":
        raise RuntimeError(status)
    run = Path(status["checkpoint"]).parent
    models = [SOURCE] + [run / f"model_{index}.pt" for index in (4240, 4300, 4360, 4420, 4479)]
    missing = [str(model) for model in models if not model.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True)
    (ROOT / "latest_evaluation.txt").write_text(str(OUT) + "\n")
    results = {}
    for model in models:
        label = "V46_model4180_source" if model == SOURCE else model.stem
        results[label] = {"checkpoint": str(model)}
        for height in (0.22, 0.21):
            results[label][f"{round(height * 100)}cm"] = evaluate(model, height)
    ranking = sorted(results, key=lambda label: score(results[label]), reverse=True)
    for label in ranking[:3]:
        results[label]["20cm"] = evaluate(Path(results[label]["checkpoint"]), 0.20)
    compact = {
        label: {
            case: {"physical": data["physical_success"], "strict": data["success"],
                   "fall": data["terminated"]}
            for case, data in values.items() if isinstance(data, dict)
        }
        for label, values in results.items()
    }
    summary = {
        "ranking": ranking, "compact": compact, "results": results,
        "criterion": "physical: whole robot clears and upright 0.5s; strict additionally |y|<0.25m and |heading|<5deg",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[V47最终筛选]", json.dumps(compact, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
