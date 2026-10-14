"""Screen the early V40 checkpoints that were omitted from the first audit."""
import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
EVALUATOR = ROOT / "evaluate_bridge.py"
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "evaluations" / f"{STAMP}_early"
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
TASK = ["--task", "Isaac-Go2-V40-BridgeOriginal-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]


def difficulty_for_height(height_m):
    return (height_m - 0.05) / (0.12 - 0.05)


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
    print(f"\n[早期检查点评估] {model.name} / {case} / difficulty={difficulty:.6f}", flush=True)
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
    run = Path(status["checkpoint"]).parent
    candidates = [run / f"model_{iteration}.pt" for iteration in (2800, 2825, 2850, 2875)]
    missing = [str(path) for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True)

    results = {}
    for model in candidates:
        label = model.stem
        ten = evaluate(model, "stairs_up", difficulty_for_height(0.10), f"{label}_stairs_up_10cm")
        plane = evaluate(model, "plane", 0.0, f"{label}_plane")
        results[label] = {
            "checkpoint": str(model),
            "stairs_up_10cm": ten,
            "plane": plane,
        }
        # Spend the extra evaluation only on checkpoints that preserve the
        # two mandatory abilities. The original screen used the same gate.
        if ten["success"] >= 26 and plane["success"] >= 26:
            results[label]["stairs_up_12cm"] = evaluate(
                model,
                "stairs_up",
                difficulty_for_height(0.12),
                f"{label}_stairs_up_12cm",
            )

    compact = {
        label: {
            "10cm": data["stairs_up_10cm"]["success"],
            "12cm": data.get("stairs_up_12cm", {}).get("success", "not tested; retention gate failed"),
            "plane": data["plane"]["success"],
        }
        for label, data in results.items()
    }
    summary = {
        "results": results,
        "compact": compact,
        "selection_gate": "10cm >= 26/32 and plane >= 26/32, then test 12cm",
        "criterion": "32 environments; strict straight gate; first episode; no reset",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (ROOT / "latest_early_evaluation.txt").write_text(str(OUT) + "\n")
    print("\n[V40_EARLY_SCREEN_RESULT]", json.dumps(compact, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
