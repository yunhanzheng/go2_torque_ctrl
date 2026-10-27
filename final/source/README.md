# Final source package

- `terrain_v51_native_terrain/`: V51 experiment — SATA `go2_torque` dynamics (TG torque + growth + 60-D obs) ported onto SAC + IsaacLab, with V50 rewards + IsaacLab native terrain.
- `terrain_v52_sata_ppo_flat/`: V52 verification run — same SATA dynamics as V51, but SATA's *original* recipe: PPO + flat ground + SATA's 9 growth-coupled rewards. Reuses V51's `sata_alignment.py`; V51 stays untouched.
- `terrain_v53_sata_sac_flat/`: V53 control run — V52's environment unchanged, only the algorithm differs: SAC (OffPolicyRunner) instead of PPO.
- `terrain_v54_sata_ppo_rough/`: V54 control run — V52's setup on IsaacLab's official Go2 rough terrain, with 12 reward terms and revised joint limits. See `V54修改说明.md` inside the folder.
- `terrain_v55_sata_warmstart_isaac_rl/`: V55 — warm start from SATA's `model_3000.pt` (gs 0.789) and continue on V54's terrain with **IsaacLab's stock Go2 reward table only** (11 terms, `undesired_contacts` off). gs becomes a bounded 0.789→1.0 ramp over 500 iters (`v55_growth.py`) instead of SATA's Gompertz. See `实验说明.md` inside the folder.
- `terrain_v56_stand_force_compare/`: V56 — position control vs **direct torque** on a flat *standing* task, compared through the ground reaction forces. Baseline is IsaacLab's stock `UnitreeGo2FlatEnvCfg` (not the V50/V54 chain); the torque variant applies `tau = clamp(10 * a, +-23.5 N.m)` with the actuator PD removed (no SATA tanh / filter / hill / fatigue / growth), plus a 200 Hz ablation. GRF / CoP / torque metrics reach wandb through `extras["log"]`. Entry point: `run_trial.sh smoke|train|eval|compare|identify`. `eval` reports the *instantaneous* equivalent gains of the torque policy (`pd_probe.py`): every control step takes `Kp_j = -d tau_j / d q_j`, `Kd_j = -d tau_j / d qdot_j` and `q*_j = q_j + (tau_j + Kd_j qdot_j) / Kp_j` from the policy Jacobian, plots them over time and per joint (`pd_gains.png`, `pd_distribution.png`, `pd_coupling.png`, the last one showing `d tau_i / d q_j` for `i != j`), and scores one constant PD rebuilt from the mean gains (`fixed_pd_r2`). `identify` is the offline counterpart: it regresses constant gains over a rollout and replays the fitted and the stock PD in closed loop to test whether the learned law is realisable as a joint PD.
- `terrain_v50_curriculum_recovery/`: V50 experiment entry points and config.
- `dependency_chain/`: source modules inherited by V50; no old models or archived results.
- `base_adapter/`: Go2 SAC adapter, train/play entry points, and stair config modules.

The source preserves original absolute paths. Update those paths for a new machine as described in the top-level README. Checkpoints, run logs and videos of the V51–V56 experiments are not committed (see "Repository scope" in the top-level README).

## V51 — V50 Rewards + IsaacLab Native Terrain + SATA Dynamics

V51 **keeps** SAC + IsaacLab/IsaacSim + V50's 16 rewards + IsaacLab native terrain/curriculum, and **ports the robot dynamics & task interface** from `/home/ubuntu22/data/SATA` `go2_torque` (TG torque model + Gompertz growth curriculum + 60-D observation). The port lives in `sata_alignment.py`; `config.py` wires it in.

### Changes from V50

| 修改项 | V50 | V51 | 说明 |
|--------|-----|-----|------|
| **控制方式** | 位置控制 (scale=0.25) | `SataTorqueAction` TG 力矩 (action_scale=5) | tanh 激活 + hill 模型 + 电机疲劳，匹配 SATA |
| **decimation** | 4 (50 Hz) | 1 (200 Hz) | 匹配 SATA 的控制频率 |
| **摩擦力随机** | [0.8, 0.8] | [0.5, 1.25] | 匹配 SATA |
| **质量随机** | [-1, 3] kg | [-1, 5] kg | 匹配 SATA |
| **COM 偏移** | x±0.05, y±0.05 | x±0.2, y±0.1, z±0.1 | 匹配 SATA |
| **外部推力** | 每 10-15s, ±0.5 m/s | 每 4s, ±1.5 m/s, ±1.0 rad/s | 匹配 SATA |
| **观测** | V50 观测 | SATA 60-D (真实 motor_fatigue，无步态时钟) | 匹配 SATA compute_observations |
| **观测噪声** | 无 (enable_corruption=False) | 分维度 uniform (noise_level=1.5) + 10% action/obs loss | 匹配 SATA _get_noise_scale_vec / loss_rate |
| **episode 长度** | 20 s | 10 s | 匹配 SATA |
| **力矩上限课程** | — | 23.5 Nm × Gompertz(0.3→1.0) | general_scale 同时缩放力矩上限与命令范围 |
| **地形配置** | V49 的 12 种自定义地形 | IsaacLab 原生 ROUGH_TERRAINS_CFG (6 种) | 使用标准地形系统 |
| **课程函数** | terrain_levels_forward_survival | terrain_levels_vel | IsaacLab 原生课程函数 |
| **出生高度** | 0.40m | 0.10m | 匹配 SATA 的低高度出生 |
| **关节姿态** | IsaacLab 默认 | SATA 默认 (thigh 1.45, calf -2.5) | 匹配 SATA 的弯曲姿态 |
| **奖励函数** | 16 项 | 相同 | 完全继承 V50 |
| **速度命令** | 固定 0.5 m/s 前进 | growth 缩放 x[-0.5,1.5]/y[-0.5,0.5]/yaw[-1.5,1.5], resampling 5s | 匹配 SATA _resample_commands |

**出生姿态问题说明：**
V50 的出生高度 (0.40m) 和关节姿态 (IsaacLab 默认) 导致机器人从较高位置直接摔落到地面，着陆时无法保持平衡，也就无法有效前进。V51 改为 SATA 的出生配置 (0.10m + 弯曲关节姿态)，机器人几乎贴地出生，能够立即保持稳定并开始前进。

### Key Configuration

| Item | Value |
|------|-------|
| Control | `SataTorqueAction` — TG torque (tanh activation + hill model + motor fatigue), action_scale=5, limit 23.5 Nm |
| Decimation | 1 → 200 Hz control (matches SATA) |
| Spawn height | 0.10m above terrain (matches SATA) |
| Joint pose | hip ±0.1, thigh 1.45, calf -2.5 rad (matches SATA) |
| Terrain | IsaacLab ROUGH_TERRAINS_CFG (6 types, kept) |
| Curriculum | terrain_levels_vel (IsaacLab native) |
| Rewards | V50's 16 reward terms (kept) |
| Observations | SATA 60-D + per-channel noise (noise_level=1.5) + 10% obs/action loss |
| Growth curriculum | Gompertz `general_scale = exp(-exp(-3e-5·(step-24000)))` scales torque limit (0.3→1.0) + command ranges |
| Episode length | 10 s (matches SATA) |
| Domain rand | Friction [0.5,1.25], mass [-1,5]kg, COM shift, push every 4s |

### SATA TG 力矩模型（`sata_alignment.py`）

每 env step（decimation=1，200 Hz）计算：

```
general_scale = exp(-exp(-3e-5 · (step - 24000)))            # Gompertz 生长曲线
torque_scale  = 0.3 + general_scale · (1.0 - 0.3)
limits        = torque_scale · 23.5
activation    = tanh(action·5 / limits)                      # 一阶滤波 (cur-prev)·0.6+prev
              ↪ 10% 概率保持上一帧 (action loss)
torque        = activation · limits · (1 - sign(activation)·vel / dof_vel_limits)   # hill 模型
motor_fatigue += |torque|·dt ;  motor_fatigue ·= 0.9          # 电机疲劳
```

关键实现细节：

- **DCMotor PD 中性化**：Go2 的执行器是 `DCMotorCfg`（stiffness=25, damping=0.5），其
  `compute = stiffness·err_pos + damping·err_vel + effort`。为得到 SATA 的纯力矩，`apply_actions`
  把位置/速度目标设为当前状态（err_pos=err_vel=0），只输出 TG 力矩。
- **hill 模型速度归一化硬编码**：SATA 从 `go2_torque.urdf` 读取 `dof_vel_limits = [30.1, 30.1, 20.07]`
  （hip/thigh/calf）。IsaacLab 的 Go2 USD 报告的小腿速度上限是 **15.7**（≠ 20.07），若直接读 USD 会
  让小腿力矩被多削减 ~28%。因此 `SATA_DOF_VEL_LIMITS` 硬编码为 SATA 的值（smoke test 已验证该差异）。
- **力矩上限 = 常数 23.5**：SATA 在 `_init_buffers` 把 `torque_limits` 覆写为常数 23.5（而非 URDF 的
  23.7/35.55），本移植同样用标量 `torque_limit=23.5`。
- **命令 growth 缩放**：`SataVelocityCommand._resample_command` 用 general_scale 收缩命令范围，训练初期
  ≈ 直行 0.5 m/s，后期展开到完整 SATA 范围。

**唯一近似**：SATA 通过改变每策略步的物理子步数把控制频率从 100 Hz 渐变到 200 Hz；IsaacLab 的
`decimation` 在构造时固定，无法复刻该频率渐变，故固定为 200 Hz（SATA 的目标频率）。生长曲线*形状*一致，
但达到某一 general_scale 所需的训练时间因 SAC/PPO 的 step 语义不同而略有差异。

### Terrain Types (ROUGH_TERRAINS_CFG)

| Type | Proportion | Parameters |
|------|-----------|------------|
| pyramid_stairs | 20% | step_height 5-23cm, step_width 0.3m |
| pyramid_stairs_inv | 20% | (descending stairs) |
| boxes | 20% | height 5-20cm, width 45cm |
| random_rough | 20% | noise 2-10cm |
| hf_pyramid_slope | 10% | slope up |
| hf_pyramid_slope_inv | 10% | slope down |

- Grid: 8m × 8m per sub-terrain
- Layout: 10 rows × 20 columns = 200 sub-terrains
- Seed: 42 (reproducible)

### Running

```bash
# Smoke test: build env + step, verify 60-D obs / TG torque bound / fatigue / growth.
# Uses a single flat plane to skip the slow native rough-terrain mesh generation.
python smoke_test.py --headless

# Training
python train.py --task Isaac-Go2-V51-NativeTerrain-v0 --agent rsl_rl_sac_cfg_entry_point

# Video recording (5 seeds × 500 frames)
python record_videos.py --checkpoint logs/rsl_rl/terrain_v51_native_terrain/<run>/model_1499.pt
```

---

## V52 — SATA-Identical Verification (PPO + Flat Ground + SATA's 9 Rewards)

V52 is a **separate, non-invasive** experiment that sanity-checks the SATA dynamics port
(`sata_alignment.py`, shared with V51) by reproducing the *original* SATA `go2_torque`
recipe as closely as IsaacLab allows. It answers: “with SATA's own algorithm, rewards and
the easiest terrain, does the ported dynamics learn to walk?” If yes, the port is faithful.

V51 (SAC + native rough terrain + 16 rewards) is **left completely untouched** — V52 only
imports `sata_alignment.py` and `V51NativeTerrainEnvCfg` from it.

### What V52 changes vs V51 (everything else is reused)

| 修改项 | V51 | V52 | 说明 |
|--------|-----|-----|------|
| **算法** | SAC (`OffPolicyRunner`) | PPO (`OnPolicyRunner`) | SATA 原版算法 |
| **地形** | IsaacLab 原生 rough (6 种) + 课程 | 单一平地 (`plane`)，无课程 | 最易验证步态涌现 |
| **奖励** | V50 的 16 项 | SATA `go2_torque` 的 9 项 | growth 耦合，见下 |
| 动力学/观测/命令/域随机化/出生 | SATA 同款 | **相同（复用 V51）** | TG 力矩、60-D 观测、growth 命令 |

### SATA 的 9 项奖励（`sata_rewards.py`，逐条对齐源码）

| 奖励 | 权重 | 公式（gs = general_scale） |
|------|------|------|
| forward | +10 | `exp(-|vx - target|/0.25)`，`target = 0.5·max(1-2gs,0) - cmd_x·min(2gs,1)` |
| head_height | +5 | `mean(base_z - terrain_z).clip(max=0.3)·(1+gs) + head_up` |
| moving_y | +5 | `exp(-|vy - cmd_y|/0.25)·gs` |
| moving_yaw | +5 | `exp(-|wz - cmd_yaw|/0.25)·gs` |
| soft_dof_pos_limits | -5 | 关节超出 90% 软限位的部分求和 |
| motor_fatigue | -0.05 | `Σ(motor_fatigue·|torques_action|)` |
| dof_acc | -1e-6 | `Σ(joint_acc²)` |
| roll | -5 | `|projected_gravity_y|` |
| lin_vel_z | -5 | `base_lin_vel_z²` |

> `forward` 的 target 中命令项带负号（`- cmd_x·min(2gs,1)`），这是 SATA 源码的逐字实现，
> 为保持“同款”未作修正（已在 `sata_rewards.py` 注释标注）。

### PPO 超参（= SATA `GO2TorqueCfgPPO` = IsaacLab 官方 Go2 PPO）

`num_steps_per_env=24`, `max_iterations=3000`, actor/critic `[512,256,128]` elu,
`lr=1e-3` adaptive, `desired_kl=0.01`, `gamma=0.99`, `lam=0.95`, `clip=0.2`, `entropy=0.01`.

### Running

```bash
cd terrain_v52_sata_ppo_flat

# Smoke test: verify PPO runner + exactly 9 SATA rewards + flat ground + 60-D obs.
python smoke_test.py --headless

# Training (PPO). Both --agent keys map to the PPO runner.
python train.py --task Isaac-Go2-V52-SataPpoFlat-v0 --agent rsl_rl_sac_cfg_entry_point
```

