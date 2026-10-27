#!/usr/bin/env bash
# Record one 5 s clip per (sub-terrain, row) pair from a single checkpoint.
#
# Clips run one after another, not in parallel: four Isaac Sim renderers on one GPU blew up
# (one of the four died with SIGSEGV on 2026-10-25 while the other three finished), and a
# single instance only needs ~2.5 min including terrain generation, so serialising costs
# little and never truncates a run.
#
# Same Vulkan workaround and GPU pinning as run_video_server.sh: the container has no Vulkan
# runtime, so the loader + NVIDIA ICD fanlineng unpacked under his work directory are borrowed,
# and both the renderer (``--/renderer/activeGpu``) and PhysX (``--device cuda:``) go on the
# same card, selected with ``GPU=`` (default 1: GPU0 usually runs the training job).
#
# One instance peaks at ~6 GB, so a *second* card is the only way to run two streams at once:
# ``GPU=0 CLIPS=... bash run_clips.sh & GPU=1 CLIPS=... bash run_clips.sh & wait``.  Four
# instances on a single 24 GB card is what segfaulted, never do that.
#
#   bash run_clips.sh                               # the four clips below
#   CLIPS="boxes:4:my_clip" bash run_clips.sh
#
# Each entry is ``<sub_terrain>:<row>:<name>``.  The row is the difficulty: the generator maps
# row r onto ``difficulty = (r + U(0,1)) / num_rows``, so row r *is* difficulty r/10 -- 0 ->
# 0.0, 2 -> 0.2, 4 -> 0.4.  Columns are handed out per type in sub_terrains order by proportion
# (20 columns: 4 pyramid_stairs, 4 pyramid_stairs_inv, 4 boxes, 4 random_rough, 2 + 2 slopes),
# which is what ``--terrain_type`` resolves; without it num_envs=1 always lands on column 0
# (pyramid_stairs).
#
# 1000 policy steps at --video_fps 200 = a 5 s mp4.  The policy itself runs at ~150-200 Hz
# (SATA's growth curve), so the clip shows ~6 s of simulated time.
set -euo pipefail

FAN=/root/bayes-tmp/xiaoyulong/fanlineng
WORK=/root/bayes-tmp/xiaoyulong/yunhanzheng

export HOME="$WORK"
export TMPDIR="$WORK/tmp"
mkdir -p "$TMPDIR"
export LD_LIBRARY_PATH="$FAN/tmp/vulkan-loader/usr/lib/x86_64-linux-gnu:$FAN/tmp/vulkan-nvidia-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="$FAN/tmp/nvidia_egl_icd.json"
# Isaac Sim's Vulkan device enumeration disagrees with a restricted CUDA_VISIBLE_DEVICES list.
unset CUDA_VISIBLE_DEVICES

cd "$WORK/go2_torque_ctrl/final/source/terrain_v54_sata_ppo_rough"

RUN=${RUN:-logs/rsl_rl/terrain_v54_sata_ppo_rough/2026-10-25_11-58-19_V54_sata_ppo_rough_20261025T035807Z}
CKPT=${CKPT:-$RUN/model_5000.pt}
OUT_ROOT=${OUT_ROOT:-$RUN/videos_model5000_5s}
# 5000 iterations x 24 steps/iter: SATA's Gompertz clock sits where this checkpoint was trained
# (gs = exp(-exp(-3e-5 * (120000 - 24000))) = 0.945, i.e. ~22.5 Nm and ~194 Hz).
STEPS=${STEPS:-120000}
SPEED=${SPEED:-0.8}
LENGTH=${LENGTH:-1000}
FPS=${FPS:-200}
# Card for both the renderer and PhysX.  Must be set before KIT_ARGS, whose default embeds it.
GPU=${GPU:-1}

KIT_ARGS="${KIT_ARGS:---allow-root --/renderer/multiGpu/enabled=false --/renderer/activeGpu=$GPU --/plugins/carb.tasking.plugin/threadCount=4}"

if [[ -n "${CLIPS:-}" ]]; then
    read -r -a clips <<< "$CLIPS"
else
    clips=(
        pyramid_stairs_inv:0:stairs_inv_d0
        pyramid_stairs_inv:2:stairs_inv_d2
        boxes:4:boxes_d4
        boxes:2:boxes_d2
    )
fi

mkdir -p "$OUT_ROOT"

pids=0
rc=0
for clip in "${clips[@]}"; do
    IFS=: read -r tname row name <<< "$clip"
    log="$OUT_ROOT/$name.log"
    echo "== $name: $tname row $row -> $OUT_ROOT/$name ($log)"
    /opt/conda/envs/isaac_sac/bin/python evaluate_video.py \
        --task Isaac-Go2-V54-SataPpoRough-v0 \
        --agent rsl_rl_sac_cfg_entry_point \
        --num_envs 1 \
        --seed 42 \
        --device cuda:$GPU \
        --kit_args "$KIT_ARGS" \
        --checkpoint "$CKPT" \
        --common_step_counter "$STEPS" \
        --fix_terrain_level "$row" \
        --terrain_type "$tname" \
        --freeze_terrain \
        --command "$SPEED" 0 0 0 \
        --no_time_out \
        --no_push \
        --log_command \
        --sky_plain \
        --video \
        --video_length "$LENGTH" \
        --video_fps "$FPS" \
        --video_folder "$OUT_ROOT/$name" \
        --headless > "$log" 2>&1 || rc=1
    pids=$((pids + 1))
done

echo "== done $pids clip(s) (rc=$rc); mp4s:"
find "$OUT_ROOT" -name '*.mp4' -printf '%p %s bytes\n' 2>/dev/null | sort
exit "$rc"
