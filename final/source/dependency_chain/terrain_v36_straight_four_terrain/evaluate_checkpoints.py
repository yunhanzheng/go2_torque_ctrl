"""Straight-gate V36 candidates, then evaluate survivors on all terrain cases."""
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
EVALUATOR = Path("/home/ubuntu22/terrain_v35_straight_finetune/evaluate_straight.py")
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
RUN = Path("/home/ubuntu22/logs/rsl_rl/terrain_v36_straight_four_terrain/2026-09-24_00-42-28_four_terrain_from_v35_2525_20260924T004222Z")
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "evaluations" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
TASK = ["--task", "Isaac-Go2-V35-StraightFourTerrain-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]
CASES = {
    "plane": 0.0, "random_rough": 0.75, "boxes": 0.75,
    "slope_up": 0.75, "slope_down": 0.75,
    "stairs_up": 0.7142857142857143, "stairs_down": 0.7142857142857143,
}


def evaluate(model, case, prefix):
    output = OUT / prefix
    env = dict(ENV, STRAIGHT_OUTPUT=str(output), TERRAIN_CASE=case,
               TERRAIN_DIFFICULTY=str(CASES[case]))
    command = [PYTHON, "-u", str(EVALUATOR), *TASK, "--num_envs", "32", "--seed", "7",
               "--checkpoint", str(model), "--headless"]
    print(f"\n[评估] {model.name} / {case}", flush=True)
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
    OUT.mkdir(parents=True)
    candidates = [RUN / f"model_{iteration}.pt" for iteration in (2525, 2550, 2600, 2650, 2700, 2724)]
    plane_results = {model.stem: evaluate(model, "plane", f"{model.stem}_plane") for model in candidates}
    survivors = sorted(
        (item for item in plane_results.items() if item[1]["success"] >= 30),
        key=lambda item: (-item[1]["success"], item[1]["mean_final_abs_y_m"], item[1]["mean_max_abs_heading_deg"]),
    )[:3]
    if not survivors:
        survivors = sorted(plane_results.items(), key=lambda item: -item[1]["success"])[:1]
    matrix = {}
    for name, plane in survivors:
        model = RUN / f"{name}.pt"
        matrix[name] = {"plane": plane}
        for case in CASES:
            if case != "plane":
                matrix[name][case] = evaluate(model, case, f"{name}_{case}")
    def rank(item):
        _, results = item
        terrain_sum = sum(results[case]["success"] for case in CASES if case != "plane")
        return (-terrain_sum, -results["plane"]["success"], results["plane"]["mean_final_abs_y_m"])
    best_name, best_results = min(matrix.items(), key=rank)
    best_model = RUN / f"{best_name}.pt"
    summary = {"best_checkpoint": str(best_model), "plane_screen": plane_results,
               "evaluated_matrix": matrix, "criteria": CASES}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (ROOT / "latest_evaluation.txt").write_text(str(OUT) + "\n")
    print("\n[V36_RESULT]", json.dumps({
        "best_checkpoint": str(best_model),
        "scores": {case: data["success"] for case, data in best_results.items()},
        "plane_final_abs_y_m": best_results["plane"]["mean_final_abs_y_m"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

