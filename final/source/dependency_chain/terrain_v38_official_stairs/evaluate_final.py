"""Strict fixed-height audit of the completed V38 checkpoint."""
import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
EVALUATOR = ROOT / "evaluate_official.py"
PYTHON = "/opt/conda/envs/isaac_sac/bin/python"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "evaluations" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_API_KEY=os.environ["WANDB_API_KEY"])
TASK = ["--task", "Isaac-Go2-V38-OfficialStairs-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]


def difficulty_for_height(height_m):
    return (height_m - 0.05) / (0.23 - 0.05)


def evaluate(model, case, difficulty, prefix):
    output = OUT / prefix
    env = dict(
        ENV,
        STRAIGHT_OUTPUT=str(output),
        TERRAIN_CASE=case,
        TERRAIN_DIFFICULTY=str(difficulty),
    )
    command = [
        PYTHON,
        "-u",
        str(EVALUATOR),
        *TASK,
        "--num_envs",
        "32",
        "--seed",
        "7",
        "--checkpoint",
        str(model),
        "--headless",
    ]
    print(f"\n[评估] {model.name} / {case} / difficulty={difficulty:.6f}", flush=True)
    log = OUT / f"{prefix}.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd="/home/ubuntu22",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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
    if status["stage"] != "complete":
        raise RuntimeError(f"Training is not complete: {status}")
    model = Path(status["checkpoint"])
    OUT.mkdir(parents=True)

    heights = (0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.23)
    height_sweep = {}
    for height in heights:
        key = f"{height:.2f}"
        height_sweep[key] = evaluate(
            model,
            "stairs_up",
            difficulty_for_height(height),
            f"model_3798_stairs_up_{round(height * 100):02d}cm",
        )

    retention = {}
    for case, difficulty in {
        "plane": 0.0,
        "random_rough": 0.75,
        "boxes": 0.75,
        "slope_up": 0.75,
        "slope_down": 0.75,
        "stairs_down": difficulty_for_height(0.18),
    }.items():
        retention[case] = evaluate(model, case, difficulty, f"model_3798_{case}")

    summary = {
        "checkpoint": str(model),
        "height_sweep": height_sweep,
        "retention": retention,
        "criterion": "32 environments; strict straight gate; first episode; no reset",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (ROOT / "latest_evaluation.txt").write_text(str(OUT) + "\n")
    print(
        "\n[V38_RESULT]",
        json.dumps(
            {
                "checkpoint": str(model),
                "upstairs_by_height": {h: data["success"] for h, data in height_sweep.items()},
                "retention": {case: data["success"] for case, data in retention.items()},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
