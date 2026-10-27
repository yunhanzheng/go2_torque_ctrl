"""Evaluate final V50 against the same seven-case V49 matrix."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SOURCE = ROOT.parent
PYTHON = os.environ.get("ISAACLAB_PYTHON", sys.executable)
MODEL = REPO / "final" / "model" / "model_5099.pt"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = REPO / "final" / "results" / "reproduced" / STAMP
PYTHON_PATHS = (
    SOURCE,
    SOURCE / "dependency_chain",
    SOURCE / "base_adapter",
    SOURCE / "base_adapter" / "initial_stair_source",
)
ENV = dict(
    os.environ,
    PYTHONPATH=os.pathsep.join(map(str, PYTHON_PATHS))
    + os.pathsep
    + os.environ.get("PYTHONPATH", ""),
    WANDB_API_KEY=os.environ["WANDB_API_KEY"],
)
TASK = ["--task", "Isaac-Go2-V50-CurriculumRecovery-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]
CASES = {
    "fixed_upstairs_23cm": ("stairs_up_23_anchor", 1.0),
    "plane": ("plane", 0.0),
    "stairs_down_18cm": ("stairs_down", (0.18 - 0.05) / (0.23 - 0.05)),
    "slope_up": ("slope_up", 0.75),
    "slope_down": ("slope_down", 0.75),
    "boxes": ("boxes", 0.75),
    "random_rough": ("random_rough", 0.75),
}


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    OUT.mkdir(parents=True)
    (ROOT / "latest_evaluation.txt").write_text(str(OUT) + "\n")
    results = {}
    for label, (case, difficulty) in CASES.items():
        output = OUT / f"model_5099_{label}"
        env = dict(ENV, STRAIGHT_OUTPUT=str(output), TERRAIN_CASE=case, TERRAIN_DIFFICULTY=str(difficulty))
        command = [
            PYTHON, "-u", str(ROOT / "evaluate_dual.py"), *TASK,
            "--num_envs", "32", "--seed", "7", "--checkpoint", str(MODEL), "--headless",
        ]
        print(f"\n[评估] model_5099 / {label}", flush=True)
        log = OUT / f"model_5099_{label}.log"
        with log.open("w") as stream:
            process = subprocess.Popen(
                command, cwd=REPO, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
            code = process.wait()
        if code:
            raise RuntimeError(f"{label} exited with {code}")
        results[label] = json.loads(Path(str(output) + ".json").read_text())

    general = [data["physical_success"] for label, data in results.items() if label != "fixed_upstairs_23cm"]
    summary = {
        "checkpoint": str(MODEL),
        "curriculum_level_at_training_end": 6.3861,
        "fixed_23cm_physical": results["fixed_upstairs_23cm"]["physical_success"],
        "generalization_physical_mean": sum(general) / len(general),
        "generalization_physical_min": min(general),
        "by_terrain": {
            label: {
                "physical": data["physical_success"],
                "strict": data["success"],
                "terminated": data["terminated"],
            }
            for label, data in results.items()
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n[V50_RESULT]", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
