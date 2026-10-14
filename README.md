# Semester Project — Final SAC Result for Unitree Go2

This repository contains **only the final V50 result** of the semester project:
the final source code, trained checkpoint, evaluation evidence, training log, and
three demonstration videos. Earlier experimental models, repeated checkpoints,
and obsolete result folders are intentionally excluded.

## What to open first

| Path | Content |
|---|---|
| `final/model/model_5099.pt` | Final PyTorch SAC checkpoint |
| `final/results/evaluation/summary.json` | Final seven-terrain summary |
| `final/results/videos/` | Three browser-playable WebM demonstrations |
| `final/source/terrain_v50_curriculum_recovery/` | Final V50 experiment code |
| `final/source/dependency_chain/` | Source-only config inheritance required by V50; not archived old results |
| `open_source/README.md` | Clearly marked open-source starting points, commits, and licences |

Large model/video files use Git LFS. Clone with:

```bash
git lfs install
git clone git@gitlab.lrz.de:00000000014C2063/semeterarbeit-haomin.git
cd semeterarbeit-haomin
git lfs pull
```

## Final result: V50

Checkpoint: `model_5099.pt`

- deterministic policy;
- evaluation seed 7;
- 32 parallel environments;
- first episode only;
- terminal training curriculum level: **6.3861**.

| Evaluation case | Physical traversal |
|---|---:|
| Fixed upstairs, 23 cm | 24/32 |
| Plane | 32/32 |
| Downstairs, 18 cm | 26/32 |
| Uphill | 32/32 |
| Downhill | 32/32 |
| Boxes | 7/32 |
| Random rough | 26/32 |

Mean over the six generalisation terrains is **25.83/32**; the minimum is
**7/32**. The raw per-environment data and logs are under
`final/results/evaluation/`.

These are fixed-protocol simulation results, not a claim of real-robot
reliability or statistical significance across multiple training seeds.

## Videos

Open these directly in Firefox, Chrome, or VLC:

- `V50_model5099_upstairs_23cm_seed7_env0.webm`
- `V50_model5099_slope_up_seed7_env0.webm`
- `V50_model5099_boxes_seed7_env5.webm`

## Code structure

V50 was developed through Python config inheritance. Therefore the small
source-only dependency chain is retained so imports resolve, while all older
models, evaluations, and videos are omitted. The relevant line is:

```text
V34 -> V35 -> V36 -> V37 -> V38 -> V44 -> V45 -> V46 -> V47 -> V48 -> V49 -> V50
```

`final/source/base_adapter/` contains the Go2/SAC adapter and the earlier stair
config modules imported by this chain. This is supporting source, not a second
set of submitted results.

## Reproduction requirements

- Ubuntu 22.04;
- Python 3.11;
- NVIDIA GPU, driver, CUDA, and Isaac Sim 5.1;
- IsaacLab-SAC and RSL-RL-SAC at the commits documented in
  `open_source/README.md`.

The archived scripts retain absolute paths from the original machine
(`/home/ubuntu22` and `/data/sac`) for auditability. Replace them with the local
checkout paths before running. The original final evaluator is:

```text
final/source/terrain_v50_curriculum_recovery/evaluate_final.py
```

Point its model path to `final/model/model_5099.pt`, and place repository root,
`final/source/dependency_chain`, `final/source/base_adapter`, and
`final/source/base_adapter/initial_stair_source` on `PYTHONPATH`. Isaac Sim is
not redistributed here because it is an external NVIDIA dependency.

## Model integrity

```text
SHA-256: 3291bb0fdf9063b1bb5ae3438fea3717a600aa14a08d2276bf30dfefa3d70227
```

PyTorch checkpoints may contain pickle data; load only in a trusted environment.

