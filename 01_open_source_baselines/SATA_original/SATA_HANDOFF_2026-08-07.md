# SATA 实验交接（2026-08-07）

## 导师结论与下一步

- 导师认可当前 PPO 结果；SATA 的 PPO 部分到此作为可用基线。
- 2026-08-09 开始 SAC 实验。
- SAC 实验应新建独立任务、日志和模型目录，不覆盖 PPO 结果。

## 最终保留的 PPO 模型

### 平地 Stage 3

- 模型：`legged_gym/logs/SATA_flat_gait_coordination/Aug05_19-46-44_stage3_diagonal_gait_seed1_visible/model_3000.pt`
- 任务：`go2_torque_gait`
- 训练速度指令：`0.0–0.8 m/s` 随机采样，每 5 s 重采样。
- 用途：平地最佳 PPO 基线，以及 `0.25/0.5/0.75 m/s` 范围内速度对照。
- 演示：`./PLAY_STAGE3.sh`

### 粗糙地形 Stage 4A

- 模型：`legged_gym/logs/SATA_flat_gait_coordination/Aug06_19-21-04_stage4a_rough_curriculum_seed1_visible/model_4900.pt`
- 任务：`go2_torque_rough_gait`
- 用途：粗糙地形 PPO 最佳基线。
- 演示：`./PLAY_STAGE4A.sh`

### 固定 0.75 m/s 对照

- 使用同一个 Stage 3 `model_3000.pt`，不是单独训练模型。
- 只把演示指令设为固定 `0.75 m/s`。
- 演示：`./PLAY_STAGE3_075.sh`

## 2026-08-07 固定 0.5 m/s 实验记录

所有实验都从 0 开始，运行 1000 轮，日志与旧模型分开。

1. `SATA_flat_gait_coordination/Aug07_12-22-36_aug07_flat_fixed_0p5_from_zero_seed1`
   - 固定 `0.5 m/s` 基线。
   - 能前进，但采用低重心趴行策略。
   - 奖励峰值约 `178.02`（约第 863 轮），候选 `model_850.pt`。
   - 仅作实验记录，不替代 Stage 3 最佳模型。

2. `SATA_flat_fixed_speed_posture/Aug07_14-13-54_aug07_fixed_0p5_posture_from_zero_seed1`
   - 加入机身高度误差惩罚 `-10.0`。
   - 仍趴行，且前进奖励明显下降。
   - 判定失败，不作最佳模型。

3. `SATA_flat_fixed_speed_height_termination/Aug07_14-31-12_aug07_fixed_0p5_height_term_from_zero_seed1`
   - 机身低于 `0.22 m` 立即终止回合。
   - 平均回合约 19 步，机器人频繁重置，无法学习前进。
   - 判定失败。

4. `SATA_flat_fixed_speed_nominal_posture/Aug07_14-46-52_aug07_fixed_0p5_nominal_posture_from_zero_seed1`
   - 加入正常四足关节姿态软惩罚 `-1.0`。
   - 仍趴行且几乎不前进，前进奖励约 `1.34`。
   - 判定失败。

## 速度实验结论

使用同一个已训练的 Stage 3 PPO 策略，仅在评估时切换固定速度指令。这样模型、奖励、地形和控制参数不变，唯一自变量是速度。

- 范围内：`0.25/0.5/0.75 m/s`
- 范围外泛化：`1.0 m/s`

## 下次直接使用

```bash
cd /data/sata/SATA_flat_reproduction
./PLAY_STAGE3.sh       # 最佳平地，固定 0.5 m/s
./PLAY_STAGE4A.sh      # 最佳粗糙地形
./PLAY_STAGE3_075.sh   # 最佳平地，固定 0.75 m/s
./OPEN_PPO_MODEL_FOLDERS.sh
```

## SAC 开始前检查

1. 保留 PPO 任务注册和检查点，不在原目录直接改为 SAC。
2. 确认现有 `rsl_rl` 是否包含 SAC；如果没有，创建独立 SAC runner/algorithm 和回放缓冲区。
3. 首个 SAC 基线应使用与 PPO Stage 3 相同的平地环境、观测、动作和奖励，便于公平比较。
