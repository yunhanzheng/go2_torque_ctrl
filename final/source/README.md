# Final source package

- `terrain_v50_curriculum_recovery/`: final experiment entry points and config.
- `dependency_chain/`: only the source modules inherited by V50; no old models
  or archived experiment results.
- `base_adapter/`: Go2 SAC adapter, train/play entry points, and stair config
  modules needed by the inheritance chain.

The source preserves original absolute paths. Update those paths for a new
machine as described in the top-level README.

