#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# V34 → V50 全流程训练启动脚本
#
# 用法:
#   # 完整训练（从 V34 平地基础开始）
#   bash run_full_pipeline.sh
#
#   # 从指定版本继续（需要之前版本的 checkpoint 已存在）
#   bash run_full_pipeline.sh --start-from v38
#
#   # 预览模式（只打印命令，不训练）
#   bash run_full_pipeline.sh --dry-run
#
#   # 列出所有阶段
#   bash run_full_pipeline.sh --list-stages
#
# 环境变量:
#   ISAACLAB_PYTHON  - Python 解释器路径（默认 miniforge3 环境）
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ISAACLAB_PYTHON:-/opt/conda/envs/isaac_sac/bin/python}"

# ── Preflight checks ──
echo "[预检] Python: $PYTHON_BIN"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] Python 不存在或不可执行: $PYTHON_BIN" >&2
    echo "  请设置 ISAACLAB_PYTHON 环境变量指向正确的解释器" >&2
    exit 1
fi

echo "[预检] 检查 CUDA..."
"$PYTHON_BIN" -c '
import torch
if not torch.cuda.is_available():
    print("[ERROR] CUDA 不可用，请检查 NVIDIA 驱动")
    exit(1)
print(f"[OK] CUDA 可用: {torch.cuda.get_device_name(0)}")
print(f"[OK] GPU 数量: {torch.cuda.device_count()}")
' || exit 1

echo "[预检] 检查 IsaacLab 导入..."
"$PYTHON_BIN" -c '
try:
    import isaaclab
    print(f"[OK] IsaacLab: {isaaclab.__file__}")
except ImportError as e:
    print(f"[ERROR] IsaacLab 未安装: {e}")
    exit(1)
' || exit 1

echo "[预检] 检查 RSL-RL..."
"$PYTHON_BIN" -c '
try:
    import rsl_rl
    print(f"[OK] RSL-RL: {rsl_rl.__file__}")
except ImportError as e:
    print(f"[ERROR] RSL-RL 未安装: {e}")
    exit(1)
' || exit 1

echo ""
echo "══════════════════════════════════════════════════════════════════════════"
echo "  V34 → V50 全流程训练管道"
echo "  共 13 个阶段（V34 含平地和四地形两阶段）"
echo "  预计总训练时间: 8-12 小时（取决于 GPU 性能）"
echo "══════════════════════════════════════════════════════════════════════════"
echo ""

# ── Check for existing training process ──
if pgrep -f "train_eth_fixed05_v2.py" > /dev/null 2>&1; then
    echo "[WARNING] 检测到正在运行的 Isaac/SAC 训练进程:"
    pgrep -af "train_eth_fixed05_v2.py" || true
    echo ""
    read -p "是否继续？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "[中止] 请等待现有训练完成后再运行"
        exit 0
    fi
fi

# ── Create pipeline log directory ──
mkdir -p "$SCRIPT_DIR/pipeline_logs"

# ── Launch pipeline ──
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/run_full_pipeline.py" "$@"
