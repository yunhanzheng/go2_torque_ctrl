# Open-source baselines (third-party code)

Everything in this folder is a third-party starting point, not an original
contribution of this semester project. It is included so that the code boundary
is explicit and the project can be inspected without guessing which upstream
repositories were used.

## `SATA_original/`

- Project: **SATA: Safe and Adaptive Torque-Based Locomotion Policies Inspired
  by Animal Learning** (RSS 2025)
- Upstream: <https://github.com/PeiZhuoLi/SATA>
- Paper: <https://arxiv.org/abs/2502.12674>
- Role here: original torque-locomotion task and reward reference; also the
  original PPO baseline from which the early comparison workspace was derived.
- Licence information: retained inside the `legged_gym/` and `rsl_rl/` trees.
- Provenance limitation: the local snapshot from which this copy was made did
  not contain its `.git` metadata, so an exact upstream commit cannot be proven
  from the available files. The files and original README are preserved.

## `isaaclab_sac/`

- Upstream: <https://github.com/sabagian/isaaclab-sac>
- Commit represented by the local checkout:
  `bbd05d8031d8c0a78d372da198863ca38d6b8c5d`
- Role here: Isaac Lab 5.1 environment and training integration for SAC.
- Licences: BSD-3-Clause for Isaac Lab and Apache-2.0 for the mimic extension;
  see the included `LICENSE`, `LICENSE-mimic`, and dependency licence files.
- Local note: `scripts/reinforcement_learning/rsl_rl/play.py` had an uncommitted
  local modification in the experiment checkout. It is preserved because the
  project used that checkout; it must not be mistaken for pristine upstream.

Training logs, output folders, Git metadata, and caches were excluded from this
vendor copy. They are generated data rather than baseline source.

## `rsl_rl_sac/`

- Upstream: <https://github.com/leggedrobotics/rsl_rl_sac>
- Commit: `e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521`
- Role here: SAC algorithm, replay storage, network modules, and off-policy
  runner used with IsaacLab-SAC.
- Licence: BSD-3-Clause; see the included `LICENSE` and dependency licences.

## What is ours?

The project-specific environments, curricula, transfer logic, supervisors,
evaluation protocols, result selection, and analysis live under
`../02_our_work/`. Selected trained weights live under `../03_final_models/`.

