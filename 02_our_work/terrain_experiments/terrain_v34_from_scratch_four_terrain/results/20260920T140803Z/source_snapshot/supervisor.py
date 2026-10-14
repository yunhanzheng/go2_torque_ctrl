"""Visible, guarded V34 workflow from fresh weights through four-terrain WebMs."""
import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from catalog import CASES


ROOT = Path(__file__).resolve().parent
PYTHON = "/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python"
FFMPEG = "/home/ubuntu22/miniforge3/envs/env_isaaclab/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
BATCH = ROOT / "results" / STAMP
ENV = dict(os.environ, PYTHONPATH="/home/ubuntu22:/data/sac:/data/sac/scripts", WANDB_MODE="offline")
FLAT_TASK = ["--task", "Isaac-Go2-V34-ScratchFlat-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]
MIXED_TASK = ["--task", "Isaac-Go2-V34-ScratchFourTerrain-v0", "--agent", "rsl_rl_sac_cfg_entry_point"]
V24 = Path("/data/sac/saved_models/ETH_SAC_V24_10CM/model_299.pt")
V26 = Path("/home/ubuntu22/logs/rsl_rl/terrain_v26_official3_from_v24/2026-09-17_21-18-06_v26_20260917T211724Z/model_599.pt")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status(stage, **details):
    temporary = BATCH / "status.tmp"
    temporary.write_text(json.dumps(dict(stage=stage, updated_at=time.time(), supervisor_pid=os.getpid(), **details), ensure_ascii=False, indent=2))
    temporary.replace(BATCH / "status.json")


def run(command, stage, timeout=3600, env=ENV):
    print(f"\n[监管] {stage}", flush=True)
    print("[命令]", " ".join(command), flush=True)
    log_path = BATCH / f"{stage}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT.parent, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        try:
            with log_path.open() as reader:
                while True:
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end="", flush=True)
                    elapsed = round(time.monotonic() - started)
                    status(stage, child_pid=process.pid, elapsed_seconds=elapsed, log=str(log_path))
                    if process.poll() is not None:
                        print(reader.read(), end="", flush=True)
                        if process.returncode:
                            raise RuntimeError(f"{stage}: exit {process.returncode}")
                        return
                    if elapsed > timeout:
                        raise TimeoutError(stage)
                    time.sleep(2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def checkpoint(experiment, run_name):
    runs = list(Path("/home/ubuntu22/logs/rsl_rl", experiment).glob(f"*_{run_name}"))
    assert len(runs) == 1, runs
    models = list(runs[0].glob("model_*.pt"))
    assert models, runs[0]
    return max(models, key=lambda path: int(path.stem.split("_")[-1]))


def evaluate(trained, case, difficulty, prefix, results):
    output = BATCH / prefix
    env = dict(ENV, TERRAIN_CASE=case, TERRAIN_DIFFICULTY=str(difficulty), TERRAIN_OUTPUT=str(output))
    run([PYTHON, "-u", str(ROOT / "evaluate.py"), *MIXED_TASK, "--num_envs", "32", "--seed", "7", "--checkpoint", str(trained), "--headless"], f"评估_{prefix}", 1800, env)
    data = json.loads(Path(str(output) + ".json").read_text())
    assert data["num_envs"] == 32
    assert sum(data[key] for key in ("success", "terminated", "other_no_success")) == 32
    results[case] = data


def record(trained, case, difficulty, prefix):
    videos = BATCH / "videos"
    videos.mkdir(exist_ok=True)
    output = videos / prefix
    env = dict(ENV, TERRAIN_CASE=case, TERRAIN_DIFFICULTY=str(difficulty), TERRAIN_OUTPUT=str(output))
    run([PYTHON, "-u", str(ROOT / "evaluate.py"), *MIXED_TASK, "--num_envs", "1", "--seed", "7", "--checkpoint", str(trained), "--headless", "--video", "--video_length", "2400"], f"录制_{prefix}", 1800, env)
    webm = Path(str(output) + ".webm")
    run([FFMPEG, "-y", "-v", "error", "-i", str(output) + "_video/rl-video-step-0.mp4", "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-an", str(webm)], f"编码_{prefix}", 900)
    run([FFMPEG, "-v", "error", "-i", str(webm), "-f", "null", "-"], f"校验_{prefix}", 180)


def publish(results, flat_checkpoint, mixed_checkpoint=None):
    rows = ["| 测试条件 | V34严格从头训练 |", "|---|---:|"]
    for case, (label, _) in CASES.items():
        data = results.get(case)
        rows.append(f'| {label} | {data["success"] if data else "待评估"}/32 |')
    (BATCH / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (BATCH / "结果.md").write_text(
        "# V34 严格从头训练：四地形统一策略\n\n"
        + "\n".join(rows)
        + f"\n\n平地基础模型：{flat_checkpoint}\n四地形模型：{mixed_checkpoint}\n"
        + "\n未加载V24/V26或任何外部checkpoint；V24/V26仅为论文对照。\n"
    )
    cards = []
    for case, (label, _) in CASES.items():
        video = BATCH / "videos" / f"V34_{case}.webm"
        if video.exists():
            score = results.get(case, {}).get("success", "?")
            cards.append(f'<article><h2>{html.escape(label)}（{score}/32）</h2><video controls loop muted preload="metadata" src="videos/{video.name}"></video></article>')
    page = BATCH / "V34严格从头训练四地形报告.html"
    page.write_text(
        '<!doctype html><meta charset="utf-8"><title>V34严格从头训练四地形报告</title>'
        '<style>body{font-family:sans-serif;margin:24px;background:#eee}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article{background:white;padding:16px}video{width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>V34：严格从头训练的四地形统一策略</h1>'
        '<p>全新Actor、Critic、normalizer、optimizer、alpha与replay；先平地基础，再训练楼梯、斜坡、随机方块、随机粗糙面。</p>'
        '<p>历史基线：V24楼梯10 cm 31/32；V26 Rough hard 32/32、Boxes hard 21/32、上坡16/32、下坡17/32。</p>'
        '<pre>' + html.escape("\n".join(rows)) + '</pre><main>' + "".join(cards) + "</main>"
    )
    return page


def main():
    BATCH.mkdir(parents=True)
    (ROOT / "latest.txt").write_text(str(BATCH) + "\n")
    snapshot = BATCH / "source_snapshot"
    snapshot.mkdir()
    for path in ROOT.iterdir():
        if path.suffix in (".py", ".sh", ".md"):
            shutil.copy2(path, snapshot / path.name)

    protected_hashes = {"V24": sha256(V24), "V26": sha256(V26)}
    provenance = {
        "strict_from_scratch": True,
        "external_checkpoint_loaded": False,
        "training_seed": 1,
        "evaluation_seed": 7,
        "observation_dim": 62,
        "action_dim": 12,
        "protected_reference_hashes": protected_hashes,
    }
    (BATCH / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    results = {}
    try:
        smoke_name = f"smoke_{STAMP}"
        run(
            [PYTHON, "-u", str(ROOT / "train.py"), *FLAT_TASK, "--num_envs", "64", "--max_iterations", "29", "--seed", "1", "--run_name", smoke_name, "--headless", "agent.algorithm.replay_buffer_size=8192", "agent.algorithm.num_mini_batches=2", "agent.algorithm.mini_batch_size=128"],
            "V34严格从头接口与更新检查", 900,
        )

        flat_name = f"flat_{STAMP}"
        run(
            [PYTHON, "-u", str(ROOT / "train.py"), *FLAT_TASK, "--num_envs", "4096", "--max_iterations", "1000", "--seed", "1", "--run_name", flat_name, "--headless"],
            "V34平地基础_1000轮", 14400,
        )
        flat_model = checkpoint("terrain_v34_scratch_flat_foundation", flat_name)
        evaluate(flat_model, "plane", 0.0, "V34_flat_gate", results)
        publish(results, flat_model)
        flat_success = results["plane"]["success"]
        if flat_success < 24:
            page = publish(results, flat_model)
            status("flat_gate_failed", checkpoint=str(flat_model), success=flat_success, page=str(page))
            subprocess.Popen(["firefox", "--new-window", page.as_uri()], env=ENV)
            print(f"[门禁停止] 平地仅 {flat_success}/32，不进入四地形训练。", flush=True)
            return

        mixed_name = f"mixed_{STAMP}"
        resume_env = dict(ENV, V34_OWN_FLAT_CHECKPOINT=str(flat_model))
        run(
            [PYTHON, "-u", str(ROOT / "train.py"), *MIXED_TASK, "--num_envs", "4096", "--max_iterations", "800", "--seed", "1", "--resume", "--run_name", mixed_name, "--headless"],
            "V34四地形课程_800轮", 14400, resume_env,
        )
        mixed_model = checkpoint("terrain_v34_scratch_four_terrain", mixed_name)
        results = {}
        for case, (_, difficulty) in CASES.items():
            evaluate(mixed_model, case, difficulty, f"V34_{case}", results)
            publish(results, flat_model, mixed_model)
        for case, (_, difficulty) in CASES.items():
            record(mixed_model, case, difficulty, f"V34_{case}")
            publish(results, flat_model, mixed_model)

        assert sha256(V24) == protected_hashes["V24"]
        assert sha256(V26) == protected_hashes["V26"]
        page = publish(results, flat_model, mixed_model)
        subprocess.Popen(["firefox", "--new-window", page.as_uri()], env=ENV)
        status("complete", checkpoint=str(mixed_model), flat_checkpoint=str(flat_model), page=str(page))
        print("[全部完成] Firefox 已打开：", page, flush=True)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
