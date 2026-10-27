#!/usr/bin/env bash
# V56 launcher: position control vs direct torque, compared through the ground reaction forces.
#
#   ./run_trial.sh smoke                       # 3 tasks, asserts tau math + sum(GRF)/mg, ~1 min
#   ./run_trial.sh train pos                   # reference: stock JointPositionAction + PD
#   ./run_trial.sh train torque                # tau = clamp(10 * a), no PD, 50 Hz
#   ./run_trial.sh train torque200             # same at 200 Hz (bandwidth ablation)
#   ./run_trial.sh eval pos  <ckpt.pt>         # forces.csv / summary.json / figures (+ Kp/Kd for torque)
#   ./run_trial.sh compare                     # merge the three summaries into one table
#   ./run_trial.sh identify <ckpt.pt>          # is the torque policy an equivalent joint PD?
#
# Everything is overridable from the environment:
#   NUM_ENVS (4096) SEED (1) ITERS (1000) ACTION_SCALE (10.0) PYTHON (python) CUDA_VISIBLE_DEVICES (0)
#   EVAL_ENVS (4) EVAL_SECONDS (6.0) PD_ENVS (8) PD_SECONDS (6.0) PD_EXCITE (0.15)
#   PD_PROBE (auto) STOCK_KP (25) STOCK_KD (0.5)
# WANDB_API_KEY must be set for `train` (never store it in this file).
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_ROOT=$(cd "$ROOT/.." && pwd)
PYTHON=${PYTHON:-python}

NUM_ENVS=${NUM_ENVS:-4096}
SEED=${SEED:-1}
ITERS=${ITERS:-1000}
ACTION_SCALE=${ACTION_SCALE:-10.0}
WANDB_PROJECT=${WANDB_PROJECT:-isaac_sac_pipeline}
EVAL_ENVS=${EVAL_ENVS:-4}
EVAL_SECONDS=${EVAL_SECONDS:-6.0}
PD_ENVS=${PD_ENVS:-8}
PD_SECONDS=${PD_SECONDS:-6.0}
PD_EXCITE=${PD_EXCITE:-0.15}
PD_PROBE=${PD_PROBE:-auto}
STOCK_KP=${STOCK_KP:-25}
STOCK_KD=${STOCK_KD:-0.5}

export PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/base_adapter:$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CMD=${1:-smoke}
shift || true

task_of() {
  case "$1" in
    pos) echo "Isaac-Go2-V56-StandPos-v0" ;;
    torque) echo "Isaac-Go2-V56-StandTorque-v0" ;;
    torque200) echo "Isaac-Go2-V56-StandTorque200-v0" ;;
    *) echo "unknown variant '$1' (pos|torque|torque200)" >&2; return 1 ;;
  esac
}

mkdir -p "$ROOT/results"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

case "$CMD" in
  smoke)
    # No wandb, no checkpoint. Fails loudly if the torque path is not exactly clamp(scale * a).
    exec "$PYTHON" -u "$ROOT/smoke_test.py" --headless "$@"
    ;;

  train)
    VARIANT=${1:-torque}
    shift || true
    TASK=$(task_of "$VARIANT")
    export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY first}"
    LOG="$ROOT/results/V56_${VARIANT}_${STAMP}.log"

    echo "V56 train [$VARIANT] -> $TASK"
    echo "  baseline : IsaacLab UnitreeGo2FlatEnvCfg (plane, 48-D obs, stock PPO runner)"
    echo "  envs=$NUM_ENVS seed=$SEED iters=$ITERS action_scale=$ACTION_SCALE"
    echo "  log      : $LOG"

    cd "$ROOT"
    # train.py exits with os._exit(0) (the render teardown hangs on this host); pipefail is
    # satisfied because the python exit status stays 0.
    exec "$PYTHON" -u "$ROOT/train.py" \
      --task "$TASK" \
      --agent rsl_rl_cfg_entry_point \
      --num_envs "$NUM_ENVS" --max_iterations "$ITERS" --seed "$SEED" \
      --action_scale "$ACTION_SCALE" \
      --run_name "V56_${VARIANT}_seed${SEED}_${STAMP}" \
      --logger wandb --log_project_name "$WANDB_PROJECT" \
      --headless "$@" 2>&1 | tee "$LOG"
    ;;

  eval)
    VARIANT=${1:?usage: run_trial.sh eval <pos|torque|torque200> <checkpoint.pt>}
    shift
    CHECKPOINT=${1:?usage: run_trial.sh eval <variant> <checkpoint.pt>}
    shift || true
    TASK=$(task_of "$VARIANT")
    cd "$ROOT"
    exec "$PYTHON" -u "$ROOT/evaluate_force.py" \
      --task "$TASK" \
      --checkpoint "$CHECKPOINT" \
      --label "$VARIANT" \
      --num_envs "$EVAL_ENVS" \
      --duration "$EVAL_SECONDS" \
      --pd_probe "$PD_PROBE" \
      --stock_kp "$STOCK_KP" \
      --stock_kd "$STOCK_KD" \
      --output_dir "$ROOT/results/force_eval/$VARIANT" \
      --headless "$@"
    ;;

  identify)
    # Fit tau = Kp (q* - q) - Kd qdot to the torque policy, then close the loop with the fitted
    # PD and with the stock PD (Kp 25 / Kd 0.5) to see whether the forces survive the swap.
    CHECKPOINT=${1:?usage: run_trial.sh identify <checkpoint.pt> [extra identify_pd.py flags]}
    shift || true
    cd "$ROOT"
    exec "$PYTHON" -u "$ROOT/identify_pd.py" \
      --task "Isaac-Go2-V56-StandTorque-v0" \
      --checkpoint "$CHECKPOINT" \
      --num_envs "$PD_ENVS" \
      --duration "$PD_SECONDS" \
      --excite "$PD_EXCITE" \
      --output_dir "$ROOT/results/pd_eval" \
      --headless "$@"
    ;;

  compare)
    # Pure post-processing: no sim app, runs on any machine with numpy.
    shift || true
    ARGS=()
    for variant in pos torque torque200; do
      dir="$ROOT/results/force_eval/$variant"
      [[ -f "$dir/summary.json" ]] && ARGS+=("--run" "$variant=$dir")
    done
    if [[ ${#ARGS[@]} -eq 0 ]]; then
      echo "No summary.json found under $ROOT/results/force_eval - run 'eval' first." >&2
      exit 1
    fi
    exec "$PYTHON" -u "$ROOT/compare_force.py" "${ARGS[@]}" --out "$ROOT/results/force_eval/compare" "$@"
    ;;

  *)
    echo "usage: $0 {smoke|train|eval|compare|identify} ..." >&2
    exit 2
    ;;
esac
