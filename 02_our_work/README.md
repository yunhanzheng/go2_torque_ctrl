# Our work

This folder contains the project-authored adaptation and experiment sequence.

## `legacy_sac_workspace/`

The earlier Go2 SAC comparison workspace. It contains the adapted SATA/Go2
environment, SAC scripts, evaluation data, and experiment notes. Generated
checkpoint directories, caches, and the 494 MB duplicate archive tarball were
excluded; the verified milestone models are retained separately in
`../04_archived_milestones/`.

## `terrain_experiments/`

The later experiment lineage V25--V50. Each directory is named by experiment
stage and normally contains environment/runner configs, transfer logic, training
entry points, evaluation scripts, JSON outputs, logs, figures, and videos.

The folders intentionally remain in version order because later configs inherit
from earlier ones. The final chain is V47 -> V48 -> V49 -> V50, with older
modules also needed transitively. Keep this directory intact on `PYTHONPATH`.

The original experiment paths and timestamps are preserved in source and logs
for traceability. They are not portable installation defaults.

