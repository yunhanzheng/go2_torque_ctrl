# Semester Project — Final SAC Result for Unitree Go2

This repository contains the **final V50 result** of the semester project — the
final source code, trained checkpoint, evaluation evidence, training log, and
three demonstration videos — plus the source code of the follow-up experiments
V51–V55. Earlier experimental models, repeated checkpoints, training artefacts,
and obsolete result folders are intentionally excluded; see
[Repository scope](#repository-scope-and-ignored-artefacts).

## What to open first

| Path | Content |
|---|---|
| `final/model/model_5099.pt` | Final PyTorch SAC checkpoint |
| `final/results/evaluation/summary.json` | Final seven-terrain summary |
| `final/results/videos/` | Three browser-playable WebM demonstrations |
| `final/source/terrain_v50_curriculum_recovery/` | Final V50 experiment code |
| `final/source/dependency_chain/` | Source-only config inheritance required by V50; not archived old results |
| `final/source/` | V51–V55 follow-up experiments, source and configs only (see below) |
| `open_source/README.md` | Clearly marked open-source starting points, commits, and licences |

Large model/video files use Git LFS. Clone with:

```bash
git lfs install
git clone git@github.com:yunhanzheng/go2_torque_ctrl.git
cd go2_torque_ctrl
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

### Additional experiments (V51–V55)

After V50 the work continued by porting the dynamics of the `SATA` `go2_torque`
reference project into IsaacLab and then isolating one factor per experiment.
Only source and configs are committed for these runs; their checkpoints, logs and
videos stay local (see [Repository scope](#repository-scope-and-ignored-artefacts)).

| Path | Experiment | Change against its baseline |
|---|---|---|
| `final/source/terrain_v51_native_terrain/` | V51 | Ports SATA dynamics (TG torque model, Gompertz growth curriculum, 60-D observation) onto SAC + IsaacLab, keeping V50's 16 rewards and the native terrain/curriculum. Details in `final/source/README.md`. |
| `final/source/terrain_v52_sata_ppo_flat/` | V52 | SATA-identical verification of that port: PPO + flat ground + SATA's nine growth-coupled rewards. |
| `final/source/terrain_v53_sata_sac_flat/` | V53 | Algorithm-only control against V52: identical environment, SAC instead of PPO. |
| `final/source/terrain_v54_sata_ppo_rough/` | V54 | Terrain-only control against V52: IsaacLab's official Go2 rough terrain, 12 reward terms and revised joint limits; see `V54修改说明.md`. |
| `final/source/terrain_v55_sata_warmstart_isaac_rl/` | V55 | Warm start from SATA's `model_3000.pt` and continue with IsaacLab's stock Go2 reward table plus a bounded `gs` ramp; see `实验说明.md`. |

`pipeline_logs/` holds the shell helpers used to cut the 10 s clips from those
runs, and `run_full_pipeline.py` / `run_full_pipeline.sh` drive the V34→V50
chain.

## Repository scope and ignored artefacts

Only source code, configurations and documentation are versioned. `.gitignore`
keeps the following local, so a fresh clone is small:

| Ignored | Reason |
|---|---|
| `*/logs/`, `*/outputs/`, `tmp_tb/`, `wandb/`, `events.out.tfevents.*`, `*.log` | Training run output; the V51–V55 runs alone are ≈1.5 GB |
| `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `_ckpt_server/` | Checkpoints; only the V50 `model_5099.pt` is committed, through Git LFS |
| `*.mp4`, `*.webm`, `final/**/video*/` | Rendered videos and extracted frames; only the three V50 WebM demos are committed, through Git LFS |
| `isaac_sac_setup/isaaclab-sac/`, `isaac_sac_setup/rsl_rl_sac/`, `rsl_rl/` | Third-party clones; install them at the commits pinned in `open_source/README.md` and in [Environment setup](#environment-setup) |

Final evidence logs are explicitly kept tracked (`final/results/training.log`,
`final/results/evaluation/*.log`) so the V50 numbers stay auditable. New
artefacts that must be committed despite these rules need `git add -f`.

## Reproduction requirements

- Ubuntu 22.04;
- Python 3.11;
- NVIDIA GPU, driver, CUDA, and Isaac Sim 5.1;
- IsaacLab-SAC and RSL-RL-SAC at the commits documented in
  `open_source/README.md`.

The V51–V55 experiments expect the same IsaacLab environment plus the SATA
dynamics in `sata_alignment.py`; each folder has its own `train.py`,
`smoke_test.py` and `run_trial.sh`. Their training was run on a remote GPU host
with absolute paths, so paths inside `run_trial.sh` and the log helpers must be
adapted before reuse.

The final evaluator and its dynamic source chain use repository-relative paths.
The original final evaluator is:

```text
final/source/terrain_v50_curriculum_recovery/evaluate_final.py
```

It automatically loads `final/model/model_5099.pt`. Isaac Sim is not
redistributed here because it is an external NVIDIA dependency.

## Environment setup

### 1. Create conda environment

```bash
conda create -n isaac_sac python=3.11 -y
conda activate isaac_sac
```

### 2. Install Isaac Sim 5.1 (complete version with all extensions)

```bash
pip install "isaacsim[all]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

### 3. Install PyTorch and dependencies

```bash
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install "tensordict==0.6.2"  # Compatible with torch 2.7.0
```

### 4. Install IsaacLab-SAC

```bash
git config --global http.version HTTP/1.1  # Avoid HTTP2 errors
mkdir -p ~/isaac_sac_setup && cd ~/isaac_sac_setup
git clone https://github.com/sabagian/isaaclab-sac.git
cd isaaclab-sac
git checkout bbd05d8031d8c0a78d372da198863ca38d6b8c5d
bash isaaclab.sh --install
```

### 5. Install RSL-RL-SAC

```bash
cd ~/isaac_sac_setup
git clone https://github.com/leggedrobotics/rsl_rl_sac.git
cd rsl_rl_sac
git checkout e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521
pip install -e .
```

### 6. Create symlink for base_adapter

```bash
sudo mkdir -p /data
sudo ln -s $(pwd)/final/source/base_adapter /data/sac
```

### 7. Verify installation

```bash
python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
from rsl_rl.algorithms import SAC
from rsl_rl.runners import OffPolicyRunner
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
print('All components ready!')
app.close()
"
```

## Quick start

For a quick end-to-end check on a machine with the Isaac Lab environment:

```bash
ISAACLAB_PYTHON=/home/ubuntu22/.holosoma_deps/miniconda3/envs/isaac_sac/bin/python bash run_smoke_test.sh
```

This creates the V50 environment, loads the final checkpoint, and runs one
deterministic headless plane episode. Full seven-case reproduction can be run
with `evaluate_final.py`. The submitted final model is directly runnable for
inference/evaluation; retraining the transfer stage from scratch would also
require its V49 source checkpoint, which is intentionally not included in this
archive.

### Running the V51 native-terrain experiment

To train V51 — IsaacLab's built-in `ROUGH_TERRAINS_CFG` terrain and curriculum,
with the ported SATA torque dynamics and V50's reward set:

```bash
cd final/source/terrain_v51_native_terrain
python smoke_test.py --headless
python train.py --task Isaac-Go2-V51-NativeTerrain-v0 --agent rsl_rl_sac_cfg_entry_point
```

`run_trial.sh` wraps the same command with the paths of the original host.
Per-folder details are in `final/source/README.md`.

## Full training pipeline

To reproduce the complete training from V34 to V50 (13 stages, ~5420 iterations):

```bash
ISAACLAB_PYTHON=/home/ubuntu22/.holosoma_deps/miniconda3/envs/isaac_sac/bin/python bash run_full_pipeline.sh
```

This runs the entire curriculum chain: flat terrain → stairs → slopes → boxes → generalization.
See `run_full_pipeline.py` for details and options (--start-from, --dry-run, etc.).

## Model integrity

```text
SHA-256: 3291bb0fdf9063b1bb5ae3438fea3717a600aa14a08d2276bf30dfefa3d70227
```

PyTorch checkpoints may contain pickle data; load only in a trusted environment.

