#!/usr/bin/env bash
# Same 7 clips as clips_m3000.sh, but from model_10000 (iter 10000 = step 240000).
# One stream per card: GPU_LIST="0 1" splits the clip list round-robin across two GPUs,
# GPU_LIST="0" is plain serial.  A single instance peaks at ~6 GB, so only add a card that
# actually has that much free -- four instances on one card segfaults.
set -euo pipefail

cd /root/bayes-tmp/xiaoyulong/yunhanzheng/go2_torque_ctrl/final/source/terrain_v54_sata_ppo_rough

R=logs/rsl_rl/terrain_v54_sata_ppo_rough/2026-10-25_19-12-07_V54_sata_ppo_rough_20261025T111153Z
export CKPT=$R/model_10000.pt
export STEPS=240000     # 10000 iterations x 24 steps
export LENGTH=2000      # 2000 frames @ 200 fps = 10 s
export OUT_ROOT=$R/videos_model10000_10s
GPU_LIST=${GPU_LIST:-0}

clips=(
    pyramid_stairs:8:stairs_d8
    pyramid_stairs_inv:4:stairs_inv_d4
    pyramid_stairs_inv:8:stairs_inv_d8
    boxes:8:boxes_d8
    random_rough:8:rough_d8
    hf_pyramid_slope:8:slope_d8
    hf_pyramid_slope_inv:8:slope_inv_d8
)

read -r -a gpus <<< "$GPU_LIST"
n=${#gpus[@]}
mkdir -p "$OUT_ROOT"

pids=()
for i in "${!gpus[@]}"; do
    g=${gpus[$i]}
    stream=""
    for j in "${!clips[@]}"; do
        if (( j % n == i )); then
            stream+="${clips[$j]} "
        fi
    done
    if [[ -z "$stream" ]]; then
        continue
    fi
    echo "== GPU $g stream: $stream"
    GPU=$g CLIPS="$stream" bash run_clips.sh > "$OUT_ROOT/stream_gpu$g.log" 2>&1 &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done
echo "== done (rc=$rc); mp4s:"
find "$OUT_ROOT" -name '*.mp4' -printf '%p %s bytes\n' 2>/dev/null | sort
exit "$rc"
