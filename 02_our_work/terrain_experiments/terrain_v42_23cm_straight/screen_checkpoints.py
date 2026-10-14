"""Find whether V42 had a useful intermediate checkpoint before final regression."""
import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
EVALUATOR = ROOT / "evaluate_height.py"
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "checkpoint_screens" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
TASK = ["--task", "Isaac-Go2-V42-23cmStraight-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]


def difficulty(height):
    return (height - 0.05) / 0.18


def evaluate(model, height):
    cm = round(height * 100)
    stem = f"{model.stem}_{cm}cm"
    output = OUT / stem
    env = dict(ENV, STRAIGHT_OUTPUT=str(output), TERRAIN_CASE="stairs_up", TERRAIN_DIFFICULTY=str(difficulty(height)))
    command = [PYTHON, "-u", str(EVALUATOR), *TASK, "--num_envs", "32", "--seed", "7",
               "--checkpoint", str(model), "--headless"]
    print(f"\n[V42检查点筛选] {model.name} / {cm}cm", flush=True)
    with (OUT / f"{stem}.log").open("w") as stream:
        process = subprocess.Popen(command, cwd="/home/ubuntu22", env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"{stem} exited with {code}")
    data = json.loads(Path(str(output) + ".json").read_text())
    print(f"[结果] {stem}: {data['success']}/32, fall={data['terminated']}, "
          f"mean|y|max={data['mean_max_abs_y_m']:.3f}m", flush=True)
    return data


def main():
    status = json.loads((Path((ROOT / "latest.txt").read_text().strip()) / "status.json").read_text())
    run = Path(status["checkpoint"]).parent
    iterations = (2820, 2840, 2860, 2880, 2900, 2920, 2940, 2960, 2980, 3000, 3019)
    models = [run / f"model_{iteration}.pt" for iteration in iterations]
    OUT.mkdir(parents=True)
    results = {model.stem: {"checkpoint": str(model), "12cm": evaluate(model, 0.12)} for model in models}
    ranking = sorted(results, key=lambda key: (
        results[key]["12cm"]["success"], -results[key]["12cm"]["terminated"]), reverse=True)
    # Test only the strongest two 12 cm candidates at retained and harder heights.
    for key in ranking[:2]:
        model = Path(results[key]["checkpoint"])
        for height in (0.10, 0.15, 0.23):
            results[key][f"{round(height * 100)}cm"] = evaluate(model, height)
    compact = {
        key: {case: value["success"] for case, value in data.items() if isinstance(value, dict)}
        for key, data in results.items()
    }
    summary = {"ranking": ranking, "compact": compact, "results": results}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (ROOT / "latest_checkpoint_screen.txt").write_text(str(OUT) + "\n")
    print("\n[V42_CHECKPOINT_SCREEN]", json.dumps(compact, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
