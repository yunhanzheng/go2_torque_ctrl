#!/usr/bin/env bash
# V56 launcher: position control vs direct torque, compared through the ground reaction forces.
#
#   ./run_trial.sh smoke                        # all four tasks, asserts tau math + sum(GRF)/mg, ~2 min
#   ./run_trial.sh train pos                    # reference: stock JointPositionAction + PD, 50 Hz
#   ./run_trial.sh train torque                 # tau = clamp(10 * a), no PD, 50 Hz
#   ./run_trial.sh train pos200                 # same, policy at the physics rate (200 Hz)
#   ./run_trial.sh train torque200              # same, 200 Hz - the bandwidth-matched pair
#   ./run_trial.sh bc torque200                 # DAgger behaviour cloning, label = stock PD torque
#   ./run_trial.sh eval pos  <ckpt.pt>          # forces.csv / summary.json / figures (+ Kp/Kd for torque)
#   ./run_trial.sh eval torque200 <ckpt.pt>     # a .pt written by bc_train.py works here too
#   ./run_trial.sh compare                      # merge the summaries into one table
#   ./run_trial.sh identify [torque|torque200] <ckpt.pt>   # is the torque policy an equivalent joint PD?
#
# The two pairs differ in one setting only: ``decimation`` 4 -> 1.  The 200 Hz runners quadruple
# ``num_steps_per_env`` (24 -> 96) so an iteration still covers 0.48 s of physical time; ITERS
# therefore means the same budget for all four arms and does not need to be scaled by hand.
#
# Everything is overridable from the environment:
#   NUM_ENVS (4096) SEED (1) ITERS (1000) BC_ITERS (agent config) ACTION_SCALE (10.0) PYTHON (python)
#   CUDA_VISIBLE_DEVICES (0)
#   EVAL_ENVS (4) EVAL_SECONDS (6.0) PD_ENVS (8) PD_SECONDS (6.0) PD_EXCITE (0.15)
#   PD_PROBE (auto) STOCK_KP (25) STOCK_KD (0.5)
# WANDB_API_KEY must be set for `train` (never store it in this file).
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_ROOT=$(cd "$ROOT/.." && pwd)
REPO_ROOT=$(cd "$SOURCE_ROOT/../.." && pwd)
PYTHON=${PYTHON:-python}

NUM_ENVS=${NUM_ENVS:-4096}
SEED=${SEED:-1}
ITERS=${ITERS:-1000}
BC_ITERS=${BC_ITERS:-}
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

# ``rsl_rl`` has to come from the SAC fork under ``isaac_sac_setup``: the top-level ``rsl_rl/``
# is upstream and has no ``OffPolicyRunner``.  The pip editable install in this environment points
# at a directory that no longer exists, so the checkout has to win.
export PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/base_adapter:$ROOT:$REPO_ROOT/isaac_sac_setup/rsl_rl_sac:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CMD=${1:-smoke}
shift || true

task_of() {
  case "$1" in
    pos) echo "Isaac-Go2-V56-StandPos-v0" ;;
    torque) echo "Isaac-Go2-V56-StandTorque-v0" ;;
    pos200) echo "Isaac-Go2-V56-StandPos200-v0" ;;
    torque200) echo "Isaac-Go2-V56-StandTorque200-v0" ;;
    *) echo "unknown variant '$1' (pos|torque|pos200|torque200)" >&2; return 1 ;;
  esac
}

VARIANTS=(pos torque pos200 torque200)

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
    VARIANT=${1:?usage: run_trial.sh eval <pos|torque|pos200|torque200> <checkpoint.pt>}
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

  bc)
    # The training signal is the stock PD torque of reference.py, not the reward table; both arms
    # of a pair get the same network, optimiser, seed, sample budget and update count.
    VARIANT=${1:-torque200}
    shift || true
    TASK=$(task_of "$VARIANT")
    LOG="$ROOT/results/V56_bc_${VARIANT}_${STAMP}.log"
    BC_OUT="$ROOT/results/bc/$VARIANT"
    ARGS=()
    [[ -n "$BC_ITERS" ]] && ARGS+=(--iterations "$BC_ITERS")

    echo "V56 bc [$VARIANT] -> $TASK"
    echo "  envs=$NUM_ENVS seed=$SEED iterations=${BC_ITERS:-agent config} (identical for both arms)"
    echo "  log  : $LOG"
    echo "  model: $BC_OUT/final.pt"

    cd "$ROOT"
    # ``--output_dir`` is passed explicitly so a checkpoint never lands outside $ROOT/results,
    # whatever default bc_train.py ships with.
    exec "$PYTHON" -u "$ROOT/bc_train.py" \
      --task "$TASK" \
      --num_envs "$NUM_ENVS" --seed "$SEED" "${ARGS[@]}" \
      --output_dir "$BC_OUT" \
      --headless "$@" 2>&1 | tee "$LOG"
    ;;

  identify)
    # Fit tau = Kp (q* - q) - Kd qdot to the torque policy, then close the loop with the fitted
    # PD and with the stock PD (Kp 25 / Kd 0.5) to see whether the forces survive the swap.
    # The opt-in variant only matters for the 200 Hz arm, whose commanded torque is held for 5 ms.
    VARIANT=torque
    case "${1:-}" in
      torque|torque200) VARIANT=$1; shift ;;
    esac
    CHECKPOINT=${1:?usage: run_trial.sh identify [torque|torque200] <checkpoint.pt> [extra flags]}
    shift || true
    TASK=$(task_of "$VARIANT")
    cd "$ROOT"
    exec "$PYTHON" -u "$ROOT/identify_pd.py" \
      --task "$TASK" \
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
    for variant in "${VARIANTS[@]}"; do
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
    echo "usage: $0 {smoke|train|bc|eval|compare|identify} ..." >&2
    exit 2
    ;;
esac
