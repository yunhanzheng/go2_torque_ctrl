"""Audit 23 cm retention and six generalization terrains at training quartiles."""
import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PYTHON = "/opt/conda/envs/isaac_sac/bin/python"
SOURCE = Path(
    "/home/ubuntu22/saved_models/SATA_STAIRS_23CM_BEST_2026-10-07/"
    "V48_23CM_21OF32_model_4778.pt"
)
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "evaluations" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_API_KEY=os.environ["WANDB_API_KEY"])
TASK = ["--task", "Isaac-Go2-V49-Generalize23cm-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]
CASES = {
    "fixed_upstairs_23cm": ("stairs_up_23_anchor", 1.0),
    "plane": ("plane", 0.0),
    "stairs_down_18cm": ("stairs_down", (0.18 - 0.05) / (0.23 - 0.05)),
    "slope_up": ("slope_up", 0.75),
    "slope_down": ("slope_down", 0.75),
    "boxes": ("boxes", 0.75),
    "random_rough": ("random_rough", 0.75),
}


def iteration(path):
    return int(path.stem.split("_")[-1])


def candidates(run):
    checkpoints = sorted(run.glob("model_*.pt"), key=iteration)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints in {run}")
    positions = (0.25, 0.50, 0.75, 1.0)
    selected = [checkpoints[round((len(checkpoints) - 1) * position)] for position in positions]
    return [SOURCE, *dict.fromkeys(selected)]


def evaluate(model, label, case, difficulty):
    stem = f"{model.stem}_{label}"
    output = OUT / stem
    env = dict(
        ENV,
        STRAIGHT_OUTPUT=str(output),
        TERRAIN_CASE=case,
        TERRAIN_DIFFICULTY=str(difficulty),
    )
    command = [
        PYTHON, "-u", str(ROOT / "evaluate_dual.py"), *TASK,
        "--num_envs", "32", "--seed", "7", "--checkpoint", str(model), "--headless",
    ]
    print(f"\n[评估] {model.name} / {label}", flush=True)
    log = OUT / f"{stem}.log"
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
        raise RuntimeError(f"{stem} exited with {code}")
    return json.loads(Path(str(output) + ".json").read_text())


def main():
    latest = Path((ROOT / "latest.txt").read_text().strip())
    status = json.loads((latest / "status.json").read_text())
    if status["stage"] != "complete":
        raise RuntimeError(f"Training is not complete: {status}")
    run = Path(status["checkpoint"]).parent
    OUT.mkdir(parents=True)

    results = {}
    for model in candidates(run):
        model_results = {}
        for label, (case, difficulty) in CASES.items():
            model_results[label] = evaluate(model, label, case, difficulty)
        results[str(model)] = model_results

    ranked = []
    for model, model_results in results.items():
        anchor = model_results["fixed_upstairs_23cm"]["physical_success"]
        general = [
            data["physical_success"]
            for label, data in model_results.items()
            if label != "fixed_upstairs_23cm"
        ]
        ranked.append({
            "checkpoint": model,
            "fixed_23cm_physical": anchor,
            "generalization_physical_mean": sum(general) / len(general),
            "generalization_physical_min": min(general),
            "eligible_retains_23cm": anchor >= 16,
            "by_terrain": {
                label: {
                    "physical": data["physical_success"],
                    "strict": data["success"],
                    "terminated": data["terminated"],
                }
                for label, data in model_results.items()
            },
        })
    ranked.sort(
        key=lambda item: (
            item["eligible_retains_23cm"],
            item["generalization_physical_mean"],
            item["generalization_physical_min"],
            item["fixed_23cm_physical"],
        ),
        reverse=True,
    )
    summary = {
        "criterion": (
            "Prefer checkpoints retaining at least 16/32 physical success on fixed 23 cm; "
            "then maximize mean/min physical traversal across six generalization terrains."
        ),
        "recommended": ranked[0]["checkpoint"],
        "ranking": ranked,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (ROOT / "latest_evaluation.txt").write_text(str(OUT) + "\n")
    print("\n[V49_RECOMMENDED]", json.dumps(ranked[0], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
