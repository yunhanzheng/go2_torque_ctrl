# Semester Project: SAC Locomotion for Unitree Go2

This repository is the complete hand-off package for the semester project on
Soft Actor-Critic (SAC) locomotion for the Unitree Go2. It contains the
open-source starting points, our experiment code, selected reproducible model
checkpoints, evaluation data, training logs, and videos.

> **Start here:** the final result is V50, while V48 is the strongest dedicated
> 23 cm stair policy and V49 is the multi-terrain transfer stage. Large `.pt`,
> `.mp4`, and `.webm` files are stored with Git LFS; clone with Git LFS enabled.

## Repository map

| Folder | Ownership / purpose |
|---|---|
| `01_open_source_baselines/` | **Third-party open-source starting code**: SATA, IsaacLab-SAC, and RSL-RL-SAC. See its README for URLs, commits, and licences. |
| `02_our_work/legacy_sac_workspace/` | **Our earlier SAC adaptation**, scripts, evaluation outputs, and notes. |
| `02_our_work/terrain_experiments/` | **Our experiment sequence V25--V50**, including source, logs, JSON results, plots, and videos. |
| `03_final_models/` | **Selected final checkpoints** for V48, V49, and V50, with checksums and intended use. |
| `04_archived_milestones/` | Earlier verified milestones: V14 flat locomotion, V23 8 cm stairs, and V24 10 cm stairs. |
| `05_documentation/` | Experiment hand-off notes and detailed result documentation. |

Generated caches, duplicate intermediate checkpoints, local Python environments,
Isaac Sim/Isaac Gym installations, and proprietary simulator binaries are not
included. They are not project source and would make the repository several
gigabytes larger. The representative checkpoints and all final evaluation
artifacts are included.

## Final results

All reported tests use deterministic policies, seed 7, 32 parallel environments,
and first-episode physical traversal counts. These are simulation results for the
specified test setup, not claims of real-robot reliability or statistical
significance across training seeds.

| Model | Main purpose | Fixed 23 cm upstairs | Other terrain result |
|---|---|---:|---|
| V48 `model_4778.pt` | Dedicated 23 cm stair policy | 21/32 | Baseline before generalisation |
| V49 `model_4980.pt` | Transfer source retained for V50 | 24/32 | Mean 26.33/32 across six generalisation terrains; minimum 8/32 |
| V49 `model_5177_recommended.pt` | Best V49 model under the documented ranking rule | 20/32 | Mean 26.67/32; minimum 8/32 |
| V50 `model_5099.pt` | Final curriculum-recovery model | 24/32 | Mean 25.83/32; minimum 7/32; terminal curriculum level 6.3861 |

For V50 the six generalisation counts are: plane 32/32, 18 cm downstairs
26/32, uphill 32/32, downhill 32/32, boxes 7/32, and random rough 26/32.
The raw evidence is in
`02_our_work/terrain_experiments/terrain_v50_curriculum_recovery/evaluations/`.

Earlier verified milestones are also retained: V23 achieved 30/32 at 8 cm, and
V24 achieved 31/32 at 10 cm under their documented fixed protocol. See
`05_documentation/SAC_STAIRS_V23_V24_README.md`.

## Quick review (no simulator required)

1. Read this file and `05_documentation/FINAL_HANDOFF.md`.
2. Open the `.webm` videos under the V48--V50 `video_reports/` folders in any
   modern browser (Firefox/Chrome) or VLC.
3. Inspect `summary.json` in the V49 and V50 evaluation folders for the raw
   per-terrain counts.
4. Check the model hashes in `03_final_models/README.md`.

## Clone correctly

```bash
git lfs install
git clone git@gitlab.lrz.de:00000000014C2063/semeterarbeit-haomin.git
cd semeterarbeit-haomin
git lfs pull
```

If Git LFS is not installed, the checkout contains small pointer files instead
of the actual models and videos.

## Reproducing the final experiment

The final work uses Ubuntu 22.04, Python 3.11, NVIDIA Isaac Sim/Isaac Lab 5.1,
the included IsaacLab-SAC fork, and the included RSL-RL-SAC fork. A CUDA-capable
NVIDIA GPU is required. Isaac Sim itself is not redistributed here; install it
under NVIDIA's licence.

The experiment modules form an inheritance chain. Keep the whole
`02_our_work/terrain_experiments/` directory on `PYTHONPATH`; do not copy only
V50. The original scripts contain absolute paths from the experiment machine for
auditability. Before running on another computer, replace `/home/ubuntu22` and
`/data/sac` with the local clone path, or create equivalent paths.

Recommended installation outline:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e 01_open_source_baselines/rsl_rl_sac
pip install -e 01_open_source_baselines/isaaclab_sac/source/isaaclab_rl
export PYTHONPATH="$PWD:$PWD/02_our_work/legacy_sac_workspace:$PWD/02_our_work/legacy_sac_workspace/scripts:$PYTHONPATH"
```

Install Isaac Sim/Isaac Lab dependencies according to the bundled upstream
README before running. Exact final-stage commands and path notes are in
`05_documentation/REPRODUCING.md`.

## Provenance and licences

Third-party code is deliberately separated from our work and retains its
original licence files. The principal starting points are:

- SATA: <https://github.com/PeiZhuoLi/SATA>
- IsaacLab-SAC fork: <https://github.com/sabagian/isaaclab-sac>, local source
  commit `bbd05d8031d8c0a78d372da198863ca38d6b8c5d`
- RSL-RL-SAC: <https://github.com/leggedrobotics/rsl_rl_sac>, local source
  commit `e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521`

See `01_open_source_baselines/README.md` for the detailed boundary between
open-source inputs and project contributions. No claim of authorship is made
over third-party code.

## Integrity and limitations

- Model files are PyTorch checkpoints and should only be loaded in a trusted
  environment; PyTorch checkpoints may contain pickle data.
- Absolute paths in archived logs and JSON files document the original machine.
- Isaac Sim, NVIDIA drivers, CUDA, and local Conda environments are external
  dependencies and are not included.
- Results come from fixed simulation protocols and mostly one training seed.

