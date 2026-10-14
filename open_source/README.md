# Open-source starting points

The projects below are **third-party open-source foundations**, not original
work of this semester project. They are linked instead of copied wholesale so
the final submission stays focused and their authorship remains unambiguous.

## SATA

- Repository: <https://github.com/PeiZhuoLi/SATA>
- Paper: <https://arxiv.org/abs/2502.12674>
- Role: original torque-locomotion task/reward reference and initial PPO
  comparison baseline.
- The available local snapshot did not retain `.git` metadata, so no exact SATA
  commit is asserted here. Consult the upstream repository for its licences.

## IsaacLab-SAC

- Repository: <https://github.com/sabagian/isaaclab-sac>
- Commit used: `bbd05d8031d8c0a78d372da198863ca38d6b8c5d`
- Role: Isaac Lab 5.1 environment and off-policy training integration.
- Licence: BSD-3-Clause for Isaac Lab; Apache-2.0 for the mimic extension. See
  the upstream repository for dependency and asset licences.

## RSL-RL-SAC

- Repository: <https://github.com/leggedrobotics/rsl_rl_sac>
- Commit used: `e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521`
- Role: SAC algorithm, replay buffer, network modules, and off-policy runner.
- Licence: BSD-3-Clause; see the upstream repository.

## Project-authored layer

The project-specific curricula, terrain distributions, transfer logic,
evaluation protocol, and final V50 result are under `../final/`.

