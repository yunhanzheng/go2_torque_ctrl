"""Screen V45 checkpoints at 23/20 cm, then audit the best for retention."""
import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PYTHON = "/opt/conda/envs/isaac_sac/bin/python"
EVALUATOR = ROOT / "evaluate_dual.py"
TASK = ["--task", "Isaac-Go2-V45-23cmFrom18cm-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_API_KEY=os.environ["WANDB_API_KEY"])
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "evaluations" / STAMP
SOURCE = Path(
    "/home/ubuntu22/saved_models/SATA_STAIRS_18CM_2026-10-06/"
    "V44_18CM_BEST_model_3930.pt"
)


def evaluate(model, height):
    cm = round(height * 100)
    label = "V44_18cm_source" if model == SOURCE else model.stem
    stem = f"{label}_{cm}cm"
    output = OUT / stem
    env = dict(ENV, STRAIGHT_OUTPUT=str(output), TERRAIN_CASE="stairs_up",
               TERRAIN_DIFFICULTY=str((height - 0.12) / 0.11))
    command = [PYTHON, "-u", str(EVALUATOR), *TASK, "--num_envs", "32", "--seed", "7",
               "--checkpoint", str(model), "--headless"]
    print(f"[V45评估] {label} / {cm}cm", flush=True)
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


def score(values):
    return tuple(
        item
        for case in ("23cm", "20cm")
        for item in (values[case]["physical_success"], values[case]["success"],
                     -values[case]["terminated"])
    )


def main():
    latest = Path((ROOT / "latest.txt").read_text().strip())
    status = json.loads((latest / "status.json").read_text())
    if status["stage"] != "complete":
        raise RuntimeError(status)
    run = Path(status["checkpoint"]).parent
    indices = (3960, 3980, 4000, 4020, 4040, 4060, 4080, 4100, 4120, 4129)
    models = [SOURCE] + [run / f"model_{i}.pt" for i in indices]
    OUT.mkdir(parents=True)
    results = {}
    for model in models:
        label = "V44_18cm_source" if model == SOURCE else model.stem
        results[label] = {"checkpoint": str(model)}
        for height in (0.23, 0.20):
            results[label][f"{round(height * 100)}cm"] = evaluate(model, height)

    ranking = sorted(results, key=lambda label: score(results[label]), reverse=True)
    for label in ranking[:3]:
        model = Path(results[label]["checkpoint"])
        for height in (0.18, 0.16, 0.14, 0.12):
            results[label][f"{round(height * 100)}cm"] = evaluate(model, height)

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
    (ROOT / "latest_evaluation.txt").write_text(str(OUT) + "\n")
    print("[V45最终筛选]", json.dumps(compact, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
