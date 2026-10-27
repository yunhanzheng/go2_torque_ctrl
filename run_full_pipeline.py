#!/usr/bin/env python3
"""End-to-end orchestration: V34 from-scratch → V50 final model.

Usage:
    # Full pipeline from scratch
    python run_full_pipeline.py

    # Resume from a specific version (requires prior checkpoints)
    python run_full_pipeline.py --start-from v38

    # Dry-run: print all commands without executing
    python run_full_pipeline.py --dry-run

    # Skip GPU preflight (for debugging)
    python run_full_pipeline.py --no-gpu-check

Prerequisites:
    - Ubuntu 22.04, Python 3.11, NVIDIA GPU + driver + CUDA
    - Isaac Sim 5.1 + IsaacLab-SAC + RSL-RL-SAC
    - PYTHONPATH must include the source directories
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Configurable paths — update these for your machine
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPO_ROOT / "final" / "source"
DEPENDENCY_ROOT = SOURCE_ROOT / "dependency_chain"
V50_ROOT = SOURCE_ROOT / "terrain_v50_curriculum_recovery"

PYTHON = os.environ.get(
    "ISAACLAB_PYTHON",
    "/opt/conda/envs/isaac_sac/bin/python",
)
LOG_BASE = Path("/home/ubuntu22/logs/rsl_rl")
PIPELINE_LOG_DIR = REPO_ROOT / "pipeline_logs"

# Core training entry point (shared by all versions)
CORE_TRAIN_SCRIPT = "/data/sac/scripts/train_eth_fixed05_v2.py"


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline stage definition
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Stage:
    """One training stage in the pipeline."""
    name: str                           # e.g. "v34_flat"
    version: str                        # e.g. "v34"
    task: str                           # gym task id
    train_script: str                   # path to version's train.py
    experiment: str                     # RSL-RL experiment name (log dir)
    iterations: int
    env_var: str                        # env var for source checkpoint
    needs_resume: bool                  # whether to pass --resume
    num_envs: int = 4096
    seed: int = 1
    timeout_hours: float = 5.0
    run_name_prefix: str = ""
    extra_env: dict = field(default_factory=dict)
    description: str = ""


def build_pipeline() -> list[Stage]:
    """Define the full V34→V50 training pipeline."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stages: list[Stage] = []

    # ── V34 Phase 1: flat foundation (from scratch, no checkpoint) ──
    stages.append(Stage(
        name="v34_flat",
        version="v34",
        task="Isaac-Go2-V34-ScratchFlat-v0",
        train_script=str(DEPENDENCY_ROOT / "terrain_v34_from_scratch_four_terrain" / "train.py"),
        experiment="terrain_v34_scratch_flat_foundation",
        iterations=1000,
        env_var="V34_OWN_FLAT_CHECKPOINT",  # not used for phase 1
        needs_resume=False,
        timeout_hours=4.0,
        run_name_prefix=f"flat_{stamp}",
        description="V34 平地基础训练（全新权重，1000 轮）",
    ))

    # ── V34 Phase 2: four-terrain curriculum ──
    stages.append(Stage(
        name="v34_mixed",
        version="v34",
        task="Isaac-Go2-V34-ScratchFourTerrain-v0",
        train_script=str(DEPENDENCY_ROOT / "terrain_v34_from_scratch_four_terrain" / "train.py"),
        experiment="terrain_v34_scratch_four_terrain",
        iterations=800,
        env_var="V34_OWN_FLAT_CHECKPOINT",
        needs_resume=True,  # resumes from V34 flat checkpoint
        timeout_hours=4.0,
        run_name_prefix=f"mixed_{stamp}",
        description="V34 四地形课程训练（加载平地 checkpoint，800 轮）",
    ))

    # ── V35: straight-line fine-tuning ──
    stages.append(Stage(
        name="v35",
        version="v35",
        task="Isaac-Go2-V35-StraightFourTerrain-v0",
        train_script=str(DEPENDENCY_ROOT / "terrain_v35_straight_finetune" / "train.py"),
        experiment="terrain_v35_straight_finetune",
        iterations=300,
        env_var="V35_SOURCE_CHECKPOINT",
        needs_resume=True,
        timeout_hours=3.0,
        run_name_prefix=f"straight_from_v34_{stamp}",
        description="V35 直线微调（从 V34 四地形模型，300 轮）",
    ))

    # ── V36: straight four-terrain strengthening ──
    stages.append(Stage(
        name="v36",
        version="v36",
        task="Isaac-Go2-V36-StraightFourTerrain-v0",
        train_script=str(DEPENDENCY_ROOT / "terrain_v36_straight_four_terrain" / "train.py"),
        experiment="terrain_v36_straight_four_terrain",
        iterations=200,
        env_var="V36_SOURCE_CHECKPOINT",
        needs_resume=True,
        timeout_hours=3.0,
        run_name_prefix=f"four_terrain_from_v35_{stamp}",
        description="V36 直线四地形强化（从 V35，200 轮）",
    ))

    # ── V37: stair-focused training ──
    stages.append(Stage(
        name="v37",
        version="v37",
        task="Isaac-Go2-V37-StairsFocus-v0",
        train_script=str(DEPENDENCY_ROOT / "terrain_v37_stairs_focus" / "train.py"),
        experiment="terrain_v37_stairs_focus",
        iterations=250,
        env_var="V37_SOURCE_CHECKPOINT",
        needs_resume=True,
        timeout_hours=3.0,
        run_name_prefix=f"stairs_5to10cm_from_v36_{stamp}",
        description="V37 楼梯专项 5-10 cm（从 V36，250 轮）",
    ))

    # ── V38: official stair curriculum 5-23 cm ──
    stages.append(Stage(
        name="v38",
        version="v38",
        task="Isaac-Go2-V38-OfficialStairs-v0",
        train_script=str(DEPENDENCY_ROOT / "terrain_v38_official_stairs" / "train.py"),
        experiment="terrain_v38_official_stairs",
        iterations=1000,
        env_var="V38_SOURCE_CHECKPOINT",
        needs_resume=True,
        timeout_hours=5.0,
        run_name_prefix=f"official_5to23cm_from_v37_{stamp}",
        description="V38 官方楼梯课程 5-23 cm（从 V37，1000 轮）",
    ))

    # ── V44: 10-18 cm continuation ──
    stages.append(Stage(
        name="v44",
        version="v44",
        task="Isaac-Go2-V44-18cmFromV38-v0",
        train_script=str(DEPENDENCY_ROOT / "terrain_v44_18cm_from_v38" / "train.py"),
        experiment="terrain_v44_18cm_from_v38",
        iterations=250,
        env_var="V44_SOURCE_CHECKPOINT",
        needs_resume=True,
        timeout_hours=3.0,
        run_name_prefix=f"height_10to18cm_from_v38_{stamp}",
        description="V44 向 18 cm 推进（从 V38，250 轮）",
    ))

    # ── V45: 12-23 cm with 18 cm anchor ──
    stages.append(Stage(
        name="v45",
        version="v45",
        task="Isaac-Go2-V45-23cmFrom18cm-v0",
        train_script=str(DEPENDENCY_ROOT / "terrain_v45_23cm_from_18cm" / "train.py"),
        experiment="terrain_v45_23cm_from_18cm",
        iterations=200,
        env_var="V45_SOURCE_CHECKPOINT",
        needs_resume=True,
        timeout_hours=3.0,
        run_name_prefix=f"height_12to23cm_anchor18_from_v44_{stamp}",
        description="V45 保 18 cm 冲 23 cm（从 V44，200 轮）",
    ))

    # ── V46: 18-23 cm with 20 cm anchor ──
    stages.append(Stage(
        name="v46",
        version="v46",
        task="Isaac-Go2-V46-23cmFrom20cm-v0",
        train_script=str(DEPENDENCY_ROOT / "terrain_v46_23cm_from_20cm" / "train.py"),
        experiment="terrain_v46_23cm_from_20cm",
        iterations=300,
        env_var="V46_SOURCE_CHECKPOINT",
        needs_resume=True,
        timeout_hours=4.0,
        run_name_prefix=f"height_18to23cm_anchor20_from_v45_{stamp}",
        description="V46 从 20 cm 续训到 23 cm（从 V45，300 轮）",
    ))

    # ── V47: focused 22 cm breakthrough ──
    stages.append(Stage(
        name="v47",
        version="v47",
        task="Isaac-Go2-V47-22cmBreakthrough-v0",
        train_script=str(DEPENDENCY_ROOT / "terrain_v47_22cm_breakthrough" / "train.py"),
        experiment="terrain_v47_22cm_breakthrough",
        iterations=300,
        env_var="V47_SOURCE_CHECKPOINT",
        needs_resume=True,
        timeout_hours=4.0,
        run_name_prefix=f"focused_20to22cm_from_v46_{stamp}",
        description="V47 集中突破 22 cm（从 V46，300 轮）",
    ))

    # ── V48: final 23 cm sprint ──
    stages.append(Stage(
        name="v48",
        version="v48",
        task="Isaac-Go2-V48-23cmFinal-v0",
        train_script=str(DEPENDENCY_ROOT / "terrain_v48_23cm_final" / "train.py"),
        experiment="terrain_v48_23cm_final",
        iterations=300,
        env_var="V48_SOURCE_CHECKPOINT",
        needs_resume=True,
        timeout_hours=4.0,
        run_name_prefix=f"focused_215to23cm_from_v47_{stamp}",
        description="V48 最终冲刺 23 cm（从 V47，300 轮）",
    ))

    # ── V49: multi-terrain generalization ──
    stages.append(Stage(
        name="v49",
        version="v49",
        task="Isaac-Go2-V49-Generalize23cm-v0",
        train_script=str(DEPENDENCY_ROOT / "terrain_v49_generalize_23cm" / "train.py"),
        experiment="terrain_v49_generalize_23cm",
        iterations=400,
        env_var="V49_SOURCE_CHECKPOINT",
        needs_resume=True,
        timeout_hours=5.0,
        run_name_prefix=f"mixed55upstairs_from_v48_{stamp}",
        description="V49 多地形泛化，保留 23 cm（从 V48，400 轮）",
    ))

    # ── V50: curriculum recovery (final) ──
    stages.append(Stage(
        name="v50",
        version="v50",
        task="Isaac-Go2-V50-CurriculumRecovery-v0",
        train_script=str(V50_ROOT / "train.py"),
        experiment="terrain_v50_curriculum_recovery",
        iterations=120,
        env_var="V50_SOURCE_CHECKPOINT",
        needs_resume=True,
        timeout_hours=1.0,
        run_name_prefix=f"level6_trial_from_v49_{stamp}",
        description="V50 课程恢复微调（从 V49，120 轮，最终模型）",
    ))

    return stages


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_latest_checkpoint(experiment: str, run_name: str) -> Optional[Path]:
    """Find the latest model checkpoint from a training run."""
    run_dirs = list(LOG_BASE.glob(f"{experiment}/*_{run_name}"))
    if not run_dirs:
        return None
    if len(run_dirs) > 1:
        print(f"  [WARNING] Multiple run dirs found, using latest: {run_dirs}")
    run_dir = run_dirs[-1]
    models = sorted(run_dir.glob("model_*.pt"),
                    key=lambda p: int(p.stem.split("_")[-1]))
    return models[-1] if models else None


def gpu_preflight() -> bool:
    """Check that at least one NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False
        print("[GPU 状态]")
        for line in result.stdout.strip().split("\n"):
            print(f"  {line.strip()}")
        return True
    except Exception as e:
        print(f"[GPU] preflight failed: {e}")
        return False


def check_training_process() -> bool:
    """Check if another Isaac/SAC training process is running."""
    result = subprocess.run(
        ["pgrep", "-f", "train_eth_fixed05_v2.py"],
        capture_output=True,
    )
    return result.returncode == 0


def run_training_stage(stage: Stage, source_checkpoint: Optional[Path],
                       dry_run: bool = False) -> Optional[Path]:
    """Run one training stage and return the output checkpoint path."""
    print(f"\n{'='*80}")
    print(f"[{stage.name.upper()}] {stage.description}")
    print(f"{'='*80}")

    # Build environment
    env = dict(
        os.environ,
        PYTHONPATH=f"{SOURCE_ROOT}:{DEPENDENCY_ROOT}:{SOURCE_ROOT / 'base_adapter'}:{SOURCE_ROOT / 'base_adapter' / 'initial_stair_source'}",
        # WANDB_MODE set to online by default; set WANDB_MODE=offline env var to disable
        # WANDB_API_KEY=os.environ["WANDB_API_KEY"],
        WANDB_API_KEY=os.environ["WANDB_API_KEY"],
    )
    env.update(stage.extra_env)

    # Build command
    run_name = f"{stage.run_name_prefix}"
    cmd = [
        PYTHON, "-u", stage.train_script,
        "--task", stage.task,
        "--agent", "rsl_rl_sac_cfg_entry_point",
        "--num_envs", str(stage.num_envs),
        "--max_iterations", str(stage.iterations),
        "--seed", str(stage.seed),
        "--run_name", run_name,
        "--logger", "wandb",
        "--log_project_name", "isaac_sac_pipeline",
        "--logger", "wandb", "--log_project_name", "isaac_sac_pipeline", "--headless",
    ]

    # Handle checkpoint passing
    if stage.needs_resume and source_checkpoint is not None:
        cmd.append("--resume")
        env[stage.env_var] = str(source_checkpoint)
        print(f"  来源 checkpoint: {source_checkpoint}")
        if source_checkpoint.exists():
            print(f"  来源 SHA-256: {sha256(source_checkpoint)}")
        else:
            print(f"  [WARNING] checkpoint 文件不存在（dry-run 模式）")
    elif stage.needs_resume:
        print(f"  [ERROR] 需要来源 checkpoint 但未提供！")
        return None
    else:
        print(f"  从全新权重开始训练")

    print(f"  训练轮次: {stage.iterations}")
    print(f"  并行环境: {stage.num_envs}")
    print(f"  实验名称: {stage.experiment}")
    print(f"  命令: {' '.join(cmd)}")

    if dry_run:
        print(f"  [DRY-RUN] 跳过实际训练")
        # Return a fake path for dry-run
        return Path(f"/tmp/dry_run_{stage.name}_checkpoint.pt")

    # Execute training
    log_dir = PIPELINE_LOG_DIR / stage.name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage.name}_{run_name}.log"
    status_path = log_dir / "status.json"

    print(f"  日志文件: {log_path}")
    start_time = time.monotonic()
    timeout_seconds = int(stage.timeout_hours * 3600)

    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(SOURCE_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        try:
            # Stream log output to console
            with open(log_path) as reader:
                while True:
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end="", flush=True)
                    elapsed = round(time.monotonic() - start_time)

                    # Update status
                    status_path.write_text(json.dumps({
                        "stage": stage.name,
                        "status": "training",
                        "pid": process.pid,
                        "elapsed_seconds": elapsed,
                        "checkpoint": str(source_checkpoint) if source_checkpoint else None,
                    }, indent=2))

                    if process.poll() is not None:
                        # Print remaining output
                        print(reader.read(), end="", flush=True)
                        if process.returncode != 0:
                            raise RuntimeError(
                                f"{stage.name} 训练失败，退出码 {process.returncode}"
                            )
                        break

                    if elapsed > timeout_seconds:
                        raise TimeoutError(
                            f"{stage.name} 训练超时（{stage.timeout_hours}h）"
                        )

                    time.sleep(5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()

    # Find output checkpoint
    checkpoint = find_latest_checkpoint(stage.experiment, run_name)
    elapsed_min = round((time.monotonic() - start_time) / 60, 1)

    if checkpoint is None:
        print(f"  [ERROR] 未找到输出 checkpoint！")
        return None

    status_path.write_text(json.dumps({
        "stage": stage.name,
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "elapsed_minutes": elapsed_min,
    }, indent=2))

    print(f"\n  [完成] {stage.name} → {checkpoint}")
    print(f"  [耗时] {elapsed_min} 分钟")
    print(f"  [SHA-256] {sha256(checkpoint)}")
    return checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline provenance tracking
# ──────────────────────────────────────────────────────────────────────────────
def save_provenance(stages_run: list[tuple[str, Optional[Path], Optional[Path]]]):
    """Save pipeline provenance showing the checkpoint chain."""
    provenance = {
        "pipeline": "V34→V50 full training chain",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": [],
    }
    for name, source, output in stages_run:
        provenance["stages"].append({
            "stage": name,
            "source_checkpoint": str(source) if source else None,
            "source_sha256": sha256(source) if source and source.exists() else None,
            "output_checkpoint": str(output) if output else None,
            "output_sha256": sha256(output) if output and output.exists() else None,
        })
    provenance_path = PIPELINE_LOG_DIR / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    print(f"\n[溯源] 已保存到 {provenance_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="V34→V50 全流程训练自动化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--start-from", type=str, default=None,
        help="从指定版本开始（如 v38），跳过之前的版本",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印命令，不实际执行训练",
    )
    parser.add_argument(
        "--no-gpu-check", action="store_true",
        help="跳过 GPU 预检",
    )
    parser.add_argument(
        "--python", type=str, default=None,
        help="Python 解释器路径（覆盖默认值）",
    )
    parser.add_argument(
        "--list-stages", action="store_true",
        help="列出所有阶段并退出",
    )
    args = parser.parse_args()

    global PYTHON
    if args.python:
        PYTHON = args.python

    # Build pipeline
    stages = build_pipeline()

    if args.list_stages:
        print(f"{'序号':>4}  {'名称':<12} {'版本':<6} {'轮次':>6}  {'描述'}")
        print("-" * 70)
        for i, s in enumerate(stages, 1):
            print(f"{i:>4}  {s.name:<12} {s.version:<6} {s.iterations:>6}  {s.description}")
        print(f"\n共 {len(stages)} 个阶段")
        total_iter = sum(s.iterations for s in stages)
        print(f"总训练轮次: {total_iter}")
        return

    # Create log directory
    PIPELINE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  V34 → V50 全流程训练管道")
    print(f"  共 {len(stages)} 个阶段，总 {sum(s.iterations for s in stages)} 轮迭代")
    print(f"  Python: {PYTHON}")
    print(f"  源码根目录: {SOURCE_ROOT}")
    print(f"  日志目录: {PIPELINE_LOG_DIR}")
    print("=" * 80)

    # GPU preflight
    if not args.dry_run and not args.no_gpu_check:
        if not gpu_preflight():
            print("[ERROR] GPU 预检失败。请检查 NVIDIA 驱动。")
            sys.exit(1)
        if check_training_process():
            print("[ERROR] 已有 Isaac/SAC 训练进程在运行。请先停止它。")
            sys.exit(1)

    # Determine start point
    start_idx = 0
    if args.start_from:
        for i, s in enumerate(stages):
            if s.name == args.start_from or s.version == args.start_from:
                start_idx = i
                break
        else:
            print(f"[ERROR] 未找到版本 '{args.start_from}'")
            print(f"可用版本: {', '.join(s.name for s in stages)}")
            sys.exit(1)
        print(f"\n[起始点] 从 {stages[start_idx].name} 开始")

    # Run pipeline
    provenance: list[tuple[str, Optional[Path], Optional[Path]]] = []
    current_checkpoint: Optional[Path] = None
    failed = False

    for i, stage in enumerate(stages):
        if i < start_idx:
            # For skipped stages, try to find their checkpoint in logs
            # This allows --start-from to work by finding the last available checkpoint
            continue

        print(f"\n[{i+1}/{len(stages)}] 准备执行: {stage.name}")

        # For stages after start_idx that need a checkpoint, try to find it
        # if we don't have one yet (e.g., starting from v38)
        if stage.needs_resume and current_checkpoint is None and i == start_idx:
            # Try to find checkpoint from previous experiment
            # Look for the latest run in the previous version's experiment
            prev_experiments = {
                "v34_mixed": "terrain_v34_scratch_flat_foundation",
                "v35": "terrain_v34_scratch_four_terrain",
                "v36": "terrain_v35_straight_finetune",
                "v37": "terrain_v36_straight_four_terrain",
                "v38": "terrain_v37_stairs_focus",
                "v44": "terrain_v38_official_stairs",
                "v45": "terrain_v44_18cm_from_v38",
                "v46": "terrain_v45_23cm_from_18cm",
                "v47": "terrain_v46_23cm_from_20cm",
                "v48": "terrain_v47_22cm_breakthrough",
                "v49": "terrain_v48_23cm_final",
                "v50": "terrain_v49_generalize_23cm",
            }
            prev_exp = prev_experiments.get(stage.name)
            if prev_exp:
                # Find latest run directory
                exp_dirs = sorted(LOG_BASE.glob(f"{prev_exp}/2*"))
                if exp_dirs:
                    latest_dir = exp_dirs[-1]
                    models = sorted(
                        latest_dir.glob("model_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1]),
                    )
                    if models:
                        current_checkpoint = models[-1]
                        print(f"  [自动发现] 使用 {prev_exp} 的最新 checkpoint: {current_checkpoint}")

        source = current_checkpoint
        try:
            output = run_training_stage(stage, source, dry_run=args.dry_run)
        except (RuntimeError, TimeoutError) as e:
            print(f"\n[FAILED] {stage.name}: {e}")
            provenance.append((stage.name, source, None))
            failed = True
            break

        if output is None:
            print(f"\n[FAILED] {stage.name}: 未产生输出 checkpoint")
            provenance.append((stage.name, source, None))
            failed = True
            break

        provenance.append((stage.name, source, output))
        current_checkpoint = output

    # Save provenance
    if not args.dry_run:
        save_provenance(provenance)

    if failed:
        print("\n" + "=" * 80)
        print("  [管道中断] 训练在某个阶段失败")
        print("  请检查日志并修复后使用 --start-from 从失败点继续")
        print("=" * 80)
        sys.exit(1)
    else:
        print("\n" + "=" * 80)
        print("  [管道完成] 所有阶段训练成功！")
        if current_checkpoint:
            print(f"  最终模型: {current_checkpoint}")
            print(f"  SHA-256: {sha256(current_checkpoint)}")
        print("=" * 80)


if __name__ == "__main__":
    main()
