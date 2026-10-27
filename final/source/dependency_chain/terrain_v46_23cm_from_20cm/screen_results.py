"""Screen V46 checkpoints at 23/20 cm, then audit the best at 22/21/18 cm."""
import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PYTHON = "/opt/conda/envs/isaac_sac/bin/python"
EVALUATOR = ROOT / "evaluate_dual.py"
TASK = ["--task", "Isaac-Go2-V46-23cmFrom20cm-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_API_KEY=os.environ["WANDB_API_KEY"])
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "evaluations" / STAMP
SOURCE = Path(
    "/home/ubuntu22/saved_models/SATA_STAIRS_20CM_2026-10-07/"
    "V45_20CM_BEST_model_4060.pt"
)


def evaluate(model, height):
    cm = round(height * 100)
    label = "V45_model4060_source" if model == SOURCE else model.stem
    stem = f"{label}_{cm}cm"
    output = OUT / stem
    difficulty = (height - 0.18) / 0.05
    env = dict(ENV, STRAIGHT_OUTPUT=str(output), TERRAIN_CASE="stairs_up",
               TERRAIN_DIFFICULTY=str(difficulty))
    command = [PYTHON, "-u", str(EVALUATOR), *TASK, "--num_envs", "32", "--seed", "7",
               "--checkpoint", str(model), "--headless"]
    print(f"[V46评估] {label} / {cm}cm", flush=True)
    with (OUT / f"{stem}.log").open("w") as stream:
        process = subprocess.run(command, cwd="/home/ubuntu22", env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise RuntimeError(f"{stem} exited with {process.returncode}")
    data = json.loads(Path(str(output) + ".json").read_text())
    print(f"[结果] physical={data['physical_success']}/32 strict={data['success']}/32 "
          f"fall={data['terminated']}", flush=True)
    return data


def primary_score(values):
    return (
        values["23cm"]["physical_success"], values["23cm"]["success"],
        -values["23cm"]["terminated"], values["20cm"]["physical_success"],
        values["20cm"]["success"], -values["20cm"]["terminated"],
    )


def final_score(values):
    return tuple(
        item
        for case in ("23cm", "22cm", "21cm", "20cm", "18cm")
        if case in values
        for item in (values[case]["physical_success"], values[case]["success"],
                     -values[case]["terminated"])
    )


def main():
    latest = Path((ROOT / "latest.txt").read_text().strip())
    status = json.loads((latest / "status.json").read_text())
    if status["stage"] != "complete":
        raise RuntimeError(status)
    run = Path(status["checkpoint"]).parent
    indices = (4120, 4180, 4240, 4300, 4359)
    models = [SOURCE] + [run / f"model_{index}.pt" for index in indices]
    missing = [str(model) for model in models if not model.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True)
    (ROOT / "latest_evaluation.txt").write_text(str(OUT) + "\n")

    results = {}
    for model in models:
        label = "V45_model4060_source" if model == SOURCE else model.stem
        results[label] = {"checkpoint": str(model)}
        for height in (0.23, 0.20):
            results[label][f"{round(height * 100)}cm"] = evaluate(model, height)

    preliminary = sorted(results, key=lambda label: primary_score(results[label]), reverse=True)
    for label in preliminary[:3]:
        model = Path(results[label]["checkpoint"])
        for height in (0.22, 0.21, 0.18):
            results[label][f"{round(height * 100)}cm"] = evaluate(model, height)

    ranking = sorted(results, key=lambda label: final_score(results[label]), reverse=True)
    compact = {
        label: {
            case: {"physical": data["physical_success"], "strict": data["success"],
                   "fall": data["terminated"]}
            for case, data in values.items() if isinstance(data, dict)
        }
        for label, values in results.items()
    }
    summary = {
        "ranking": ranking, "preliminary_ranking": preliminary,
        "compact": compact, "results": results,
        "criterion": "physical: whole robot clears and upright 0.5s; strict additionally |y|<0.25m and |heading|<5deg",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[V46最终筛选]", json.dumps(compact, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
