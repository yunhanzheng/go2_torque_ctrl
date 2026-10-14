"""Select a V37 checkpoint by 10 cm stairs, then audit height and terrain retention."""
import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
EVALUATOR = Path("/home/ubuntu22/terrain_v35_straight_finetune/evaluate_straight.py")
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "evaluations" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
TASK = ["--task", "Isaac-Go2-V35-StraightFourTerrain-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]


def difficulty_for_height(height_m):
    return (height_m - 0.05) / (0.12 - 0.05)


def evaluate(model, case, difficulty, prefix):
    output = OUT / prefix
    env = dict(ENV, STRAIGHT_OUTPUT=str(output), TERRAIN_CASE=case,
               TERRAIN_DIFFICULTY=str(difficulty))
    command = [PYTHON, "-u", str(EVALUATOR), *TASK, "--num_envs", "32", "--seed", "7",
               "--checkpoint", str(model), "--headless"]
    print(f"\n[评估] {model.name} / {case} / difficulty={difficulty:.6f}", flush=True)
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
    OUT.mkdir(parents=True)

    wanted = (2550, 2600, 2650, 2700, 2750, 2799)
    candidates = [run / f"model_{i}.pt" for i in wanted if (run / f"model_{i}.pt").is_file()]
    ten_cm = {
        model.stem: evaluate(model, "stairs_up", difficulty_for_height(0.10),
                             f"{model.stem}_stairs_up_10cm")
        for model in candidates
    }
    top = sorted(candidates, key=lambda model: (-ten_cm[model.stem]["success"],
        ten_cm[model.stem]["terminated"]))[:3]
    plane = {model.stem: evaluate(model, "plane", 0.0, f"{model.stem}_plane") for model in top}
    retained = [model for model in top if plane[model.stem]["success"] >= 28]
    pool = retained or top
    best = min(pool, key=lambda model: (-ten_cm[model.stem]["success"],
        -plane[model.stem]["success"], plane[model.stem]["mean_final_abs_y_m"]))

    height_sweep = {"0.10": ten_cm[best.stem]}
    for height in (0.05, 0.06, 0.07, 0.08, 0.09):
        height_sweep[f"{height:.2f}"] = evaluate(best, "stairs_up", difficulty_for_height(height),
            f"{best.stem}_stairs_up_{round(height * 100):02d}cm")

    retention = {"plane": plane[best.stem]}
    cases = {
        "random_rough": 0.75, "boxes": 0.75, "slope_up": 0.75,
        "slope_down": 0.75, "stairs_down": difficulty_for_height(0.10),
    }
    for case, difficulty in cases.items():
        retention[case] = evaluate(best, case, difficulty, f"{best.stem}_{case}")

    summary = {
        "best_checkpoint": str(best), "ten_cm_screen": ten_cm,
        "plane_screen": plane, "height_sweep": height_sweep, "retention": retention,
        "criterion": "32 environments; strict straight gate; first episode; no reset",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (ROOT / "latest_evaluation.txt").write_text(str(OUT) + "\n")
    print("\n[V37_RESULT]", json.dumps({
        "best_checkpoint": str(best),
        "upstairs_by_height": {h: d["success"] for h, d in height_sweep.items()},
        "retention": {case: data["success"] for case, data in retention.items()},
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

