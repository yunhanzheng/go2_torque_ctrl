"""Screen V44 checkpoints at 18 cm, then audit the best across lower heights."""
import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PYTHON = "/opt/conda/envs/isaac_sac/bin/python"
EVALUATOR = ROOT / "evaluate_dual.py"
TASK = ["--task", "Isaac-Go2-V44-18cmFromV38-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_API_KEY=os.environ["WANDB_API_KEY"])
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "evaluations" / STAMP
SOURCE = Path(
    "/home/ubuntu22/logs/rsl_rl/terrain_v38_official_stairs/"
    "2026-09-26_11-55-01_official_5to23cm_resume2900_to3799_20260926T115454Z/model_3798.pt"
)


def evaluate(model, height):
    cm = round(height * 100)
    label = "V38_source" if model == SOURCE else model.stem
    stem = f"{label}_{cm}cm"
    output = OUT / stem
    env = dict(ENV, STRAIGHT_OUTPUT=str(output), TERRAIN_CASE="stairs_up",
               TERRAIN_DIFFICULTY=str((height - 0.10) / 0.08))
    command = [PYTHON, "-u", str(EVALUATOR), *TASK, "--num_envs", "32", "--seed", "7",
               "--checkpoint", str(model), "--headless"]
    print(f"[V44评估] {label} / {cm}cm", flush=True)
    with (OUT / f"{stem}.log").open("w") as stream:
        process = subprocess.Popen(command, cwd="/home/ubuntu22", env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, text=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"{stem} exited with {code}")
    data = json.loads(Path(str(output) + ".json").read_text())
    print(f"[结果] physical={data['physical_success']}/32 strict={data['success']}/32 "
          f"fall={data['terminated']}", flush=True)
    return data


def main():
    latest = Path((ROOT / "latest.txt").read_text().strip())
    status = json.loads((latest / "status.json").read_text())
    if status["stage"] != "complete":
        raise RuntimeError(status)
    run = Path(status["checkpoint"]).parent
    models = [SOURCE] + [run / f"model_{i}.pt" for i in (3850, 3890, 3930, 3970, 4010, 4047)]
    OUT.mkdir(parents=True)
    results = {}
    for model in models:
        label = "V38_source" if model == SOURCE else model.stem
        results[label] = {"checkpoint": str(model), "18cm": evaluate(model, 0.18)}

    ranking = sorted(results, key=lambda k: (
        results[k]["18cm"]["physical_success"], results[k]["18cm"]["success"],
        -results[k]["18cm"]["terminated"]), reverse=True)
    for label in ranking[:3]:
        model = Path(results[label]["checkpoint"])
        for height in (0.12, 0.14, 0.16):
            results[label][f"{round(height * 100)}cm"] = evaluate(model, height)

    compact = {
        label: {
            case: {"physical": data["physical_success"], "strict": data["success"], "fall": data["terminated"]}
            for case, data in values.items() if isinstance(data, dict)
        }
        for label, values in results.items()
    }
    summary = {"ranking": ranking, "compact": compact, "results": results,
               "criterion": "physical: whole robot clears and upright 0.5s; strict additionally |y|<0.25m and |heading|<5deg"}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (ROOT / "latest_evaluation.txt").write_text(str(OUT) + "\n")
    print("[V44最终筛选]", json.dumps(compact, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
