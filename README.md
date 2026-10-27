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

### Additional experiments

- `final/source/native_terrain_sata/` — A baseline experiment using IsaacLab's
  native terrain curriculum with SATA rewards and algorithm, serving as a comparison to V50.

## Reproduction requirements

- Ubuntu 22.04;
- Python 3.11;
- NVIDIA GPU, driver, CUDA, and Isaac Sim 5.1;
- IsaacLab-SAC and RSL-RL-SAC at the commits documented in
  `open_source/README.md`.

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
final-only repository.

### Running the native curriculum baseline

To train with IsaacLab's native terrain curriculum and rewards:

```bash
cd final/source/native_terrain_sata
python supervisor.py
```

This experiment uses the built-in ROUGH_TERRAINS_CFG curriculum (6 terrain
types, 10 difficulty levels) with default IsaacLab rewards, while keeping
the project's core torque control settings (base_height_l2, feet_slide,
low_base termination).

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

