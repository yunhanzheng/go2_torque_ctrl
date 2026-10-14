"""Strict from-scratch speed curriculum followed by the four-terrain stage."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import supervisor as common
from catalog import CASES


ROOT = common.ROOT
PYTHON = common.PYTHON
ENV = common.ENV
BATCH = common.BATCH
STAMP = common.STAMP
AGENT = ["--agent", "rsl_rl_sac_cfg_entry_point"]
STAGES = [
    ("站立稳定_300轮", "Isaac-Go2-V34-ScratchStand-v0", "terrain_v34_scratch_stage0_stand", 300),
    ("慢速0.1mps_300轮", "Isaac-Go2-V34-ScratchSlow-v0", "terrain_v34_scratch_stage1_slow", 300),
    ("中速0.3mps_400轮", "Isaac-Go2-V34-ScratchMedium-v0", "terrain_v34_scratch_stage2_medium", 400),
    ("目标0.5mps_600轮", "Isaac-Go2-V34-ScratchFlat-v0", "terrain_v34_scratch_flat_foundation", 600),
]


def main():
    BATCH.mkdir(parents=True)
    (ROOT / "latest.txt").write_text(str(BATCH) + "\n")
    snapshot = BATCH / "source_snapshot"
    snapshot.mkdir()
    for path in ROOT.iterdir():
        if path.suffix in (".py", ".sh", ".md"):
            shutil.copy2(path, snapshot / path.name)
    hashes = {"V24": common.sha256(common.V24), "V26": common.sha256(common.V26)}
    provenance = {
        "strict_from_scratch": True,
        "external_checkpoint_loaded": False,
        "curriculum": ["stand", "0.1mps", "0.3mps", "0.5mps", "four_terrain"],
        "training_seed": 1,
        "reference_hashes": hashes,
    }
    (BATCH / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    results = {}
    try:
        # Smoke only the fresh first stage.  No resume variable is present.
        common.run(
            [PYTHON, "-u", str(ROOT / "train.py"), "--task", STAGES[0][1], *AGENT,
             "--num_envs", "64", "--max_iterations", "29", "--seed", "1",
             "--run_name", f"smoke_curriculum_{STAMP}", "--headless",
             "agent.algorithm.replay_buffer_size=8192", "agent.algorithm.num_mini_batches=2",
             "agent.algorithm.mini_batch_size=128"],
            "速度课程严格从头检查", 900,
        )

        source = None
        stage_models = {}
        for index, (label, task, experiment, iterations) in enumerate(STAGES):
            run_name = f"stage{index}_{STAMP}"
            command = [PYTHON, "-u", str(ROOT / "train.py"), "--task", task, *AGENT,
                       "--num_envs", "4096", "--max_iterations", str(iterations),
                       "--seed", "1", "--run_name", run_name, "--headless"]
            env = ENV
            if source is not None:
                command.insert(-1, "--resume")
                env = dict(ENV, V34_OWN_FLAT_CHECKPOINT=str(source))
            common.run(command, label, 14400, env)
            source = common.checkpoint(experiment, run_name)
            stage_models[label] = str(source)
            common.status(f"完成_{label}", checkpoint=str(source), stages=stage_models)

        flat_model = source
        common.evaluate(flat_model, "plane", 0.0, "V34_flat_gate", results)
        common.publish(results, flat_model)
        if results["plane"]["success"] < 24:
            page = common.publish(results, flat_model)
            common.status("flat_gate_failed", checkpoint=str(flat_model), success=results["plane"]["success"], page=str(page), stages=stage_models)
            subprocess.Popen(["firefox", "--new-window", page.as_uri()], env=ENV)
            print(f'[门禁停止] 平地仅 {results["plane"]["success"]}/32。', flush=True)
            return

        mixed_name = f"mixed_curriculum_{STAMP}"
        mixed_env = dict(ENV, V34_OWN_FLAT_CHECKPOINT=str(flat_model))
        common.run(
            [PYTHON, "-u", str(ROOT / "train.py"), *common.MIXED_TASK, "--num_envs", "4096",
             "--max_iterations", "800", "--seed", "1", "--resume",
             "--run_name", mixed_name, "--headless"],
            "四地形课程_800轮", 14400, mixed_env,
        )
        mixed_model = common.checkpoint("terrain_v34_scratch_four_terrain", mixed_name)
        results = {}
        for case, (_, difficulty) in CASES.items():
            common.evaluate(mixed_model, case, difficulty, f"V34_{case}", results)
            common.publish(results, flat_model, mixed_model)
        for case, (_, difficulty) in CASES.items():
            common.record(mixed_model, case, difficulty, f"V34_{case}")
            common.publish(results, flat_model, mixed_model)
        assert common.sha256(common.V24) == hashes["V24"]
        assert common.sha256(common.V26) == hashes["V26"]
        page = common.publish(results, flat_model, mixed_model)
        subprocess.Popen(["firefox", "--new-window", page.as_uri()], env=ENV)
        common.status("complete", checkpoint=str(mixed_model), flat_checkpoint=str(flat_model), page=str(page), stages=stage_models)
    except BaseException as error:
        common.status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()

