# Go2 SAC comparison workspace

## Purpose

This directory is the independent workspace for the SAC side of the Go2
SAC-versus-PPO comparison.

## Directory boundaries

- `/data/sata`: existing SATA/PPO code, environments, logs, and checkpoints.
- `/data/eth_sac`: upstream paper implementation kept as a read-only reference.
- `/data/sac`: SAC adaptation, experiment configs, logs, checkpoints, and notes.

## Fair-comparison rule

The SAC experiment must reuse the same Go2 observation space, torque action
interface, reward functions, termination rules, simulation parameters, and
evaluation protocol as the selected SATA/PPO baseline. Algorithm-specific
differences such as replay, entropy optimization, timeout-aware targets, and
n-step returns are documented separately.

## Planned layout

- `src/`: SAC implementation and SATA environment adaptation.
- `configs/`: immutable experiment configurations.
- `scripts/`: training, evaluation, plotting, and playback entry points.
- `logs/`: training logs and checkpoints.
- `results/`: evaluation tables, plots, and videos.
- `docs/`: experiment notes and paper material.
- `upstream/`: references or patches derived from official RSL-RL-SAC.

No SAC experiment should write into `/data/sata` or `/data/eth_sac`.
