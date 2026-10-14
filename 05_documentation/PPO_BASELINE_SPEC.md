# PPO Baseline Specification for the SAC Comparison

Status: frozen baseline, 2026-08-12

## Baseline identity

- Algorithm: PPO in the existing SATA/Legged Gym stack.
- Task: `go2_torque_gait` (Stage 3 diagonal-gait task).
- Seed: 1.
- Training iterations: 3000.
- Checkpoint: `/data/sata/SATA_flat_reproduction/legged_gym/logs/SATA_flat_gait_coordination/Aug05_19-46-44_stage3_diagonal_gait_seed1_visible/model_3000.pt`.
- Terrain: `plane`, without height measurements or terrain curriculum.
- Robot: Unitree Go2, 12 torque-controlled joints.
- Observation dimension: 62.
- Number of training environments: 4096.
- Episode duration configured as 10 s.

This checkpoint is the approved PPO reference. Later fixed-speed PPO runs from
2026-08-07 are diagnostic experiments and do not replace it.

## Speed protocol (important)

### Training commands

Stage 3 was **not trained at a fixed 0.5 m/s**. Its configured forward-command
range was `0.0--0.8 m/s`, with zero lateral and yaw commands. Commands were
resampled every 5 s.

The SATA implementation additionally applies a global `general_scale` growth
schedule. At each resampling, the actual forward-command interval is

```text
[max(0.4 - 0.8*general_scale, 0.0),
 min(0.4 + 0.8*general_scale, 0.8)] m/s.
```

Therefore training begins in a narrower interval around 0.4 m/s (approximately
0.30--0.50 m/s at initialization with the current growth parameters) and
gradually expands to the full 0.0--0.8 m/s configured range. The forward reward
also blends its early curriculum target around 0.4 m/s into the sampled command
as `general_scale` grows.

For concise reporting, describe this as:

> The policy was trained with a curriculum that progressively expanded the
> forward-speed commands to 0.0--0.8 m/s; commands were resampled every 5 s.

Do not describe Stage 3 as fixed-speed training.

### Primary evaluation command

The primary comparison is a fixed command of exactly:

```text
vx = 0.5 m/s, vy = 0.0 m/s, yaw rate = 0.0 rad/s.
```

The existing `PLAY_STAGE3.sh` loads the frozen `model_3000.pt`, while
`play_flat.py` overrides the command to 0.5 m/s. Thus **0.5 m/s is the primary
evaluation condition, not the Stage 3 training condition**.

### Additional speed evaluation

Use the same trained policy without fine-tuning at each command:

- 0.25, 0.50, and 0.75 m/s: interpolation within the final training range.
- 1.00 m/s: out-of-range extrapolation/generalization.
- 0.00 m/s: optional standing test, also within the final configured range.

For each speed, report commanded speed, mean measured forward speed, signed
tracking error, MAE, RMSE, standard deviation, fall rate, and valid tracking
duration. Do not use the training reward alone as a speed-tracking metric.

## Environment and control conditions to match in SAC

- Initial base position: `[0.0, 0.0, 0.30]` m.
- Observation noise: disabled.
- Friction randomization: disabled.
- Base-mass randomization: disabled.
- External pushes: disabled.
- Dropped action observations: disabled.
- Simulation step: 0.005 s.
- Control type during training: `TG` (torque control with growth).
- Action scale: 5.
- Nominal joint torque limit used by the task: 23.5 Nm, further affected by
  the SATA torque-growth schedule and motor model.
- Action interface: 12-dimensional policy output converted into motor torque by
  SATA's activation process, Hill model, fatigue model, torque clipping, and
  growth schedule. SAC must reuse this pipeline.
- Terminations: joint position outside the extended joint limits, upside-down
  orientation (`projected_gravity[:, 2] > 0`), or episode timeout.
- Timeout is already exposed as `extras["time_outs"]`, but the current step API
  returns observations after reset. SAC therefore needs the pre-reset terminal
  observation for correct timeout-aware replay targets.

## Stage 3 gait rewards

Relative to the flat Stage 2 task, Stage 3 uses:

- non-foot collision penalty: `-5.0`;
- feet air-time reward disabled: `0.0`;
- diagonal gait-contact reward: `3.0`;
- swing-clearance reward: `1.5`;
- feet-slip penalty: `-0.5`;
- gait frequency: 1.5 Hz;
- target swing height: 0.065 m;
- clearance sigma: 0.025 m.

All inherited SATA rewards and their curriculum behavior must remain identical
for the first SAC/PPO comparison.

## PPO algorithm configuration

- Actor/Critic hidden layers: `[512, 256, 128]`.
- Activation: ELU.
- Initial policy noise std: 1.0.
- Rollout steps per environment: 24.
- Learning rate: `1e-3`, adaptive schedule.
- Discount factor: 0.99.
- GAE lambda: 0.95.
- PPO clip: 0.2.
- Learning epochs: 5.
- Mini-batches: 4.
- Entropy coefficient: 0.01.

## Existing training summary

- Final mean reward: approximately 204.97.
- Highest logged mean reward: approximately 220.01 around iteration 1933.
- Final mean episode length: approximately 2045.83 logged simulation steps.
- Final policy noise std: approximately 0.828.
- Final forward-reward component: approximately 9.50.
- Final gait-contact component: approximately 1.8509.
- Final swing-clearance component: approximately 0.3625.
- Final feet-slip component: approximately -0.0277.
- Final non-foot collision component: approximately -0.0001.

These are training-log statistics, not results from a controlled fixed-speed
evaluation. They must not be presented as measured 0.5 m/s tracking accuracy.

## Fair SAC comparison rule

SAC must use the same environment, command curriculum, observations, torque
pipeline, reward functions, reset conditions, and evaluation scripts. Only the
learning algorithm and its necessary off-policy machinery may differ. Compare
both algorithms by environment interactions and by wall-clock time, and use the
same fixed-speed evaluation protocol.
