# SATA 平地复现实验

## 目标

在 Unitree Go2 的纯平地环境中复现 SATA 力矩控制策略，先验证稳定站立和基础速度跟踪，再逐步恢复噪声、域随机化与复杂指令。

## 目录说明

- `legged_gym/`：环境、Go2 模型、训练与播放脚本
- `rsl_rl/`：PPO 强化学习实现
- `legged_gym/logs/`：新实验日志和模型（首次训练时自动创建）
- Isaac Gym 共用路径：`/data/sata/isaacgym`
- 原始 SATA 保留路径：`/data/sata/SATA`

## 已发现的旧实验问题

- 旧实验并非严格纯平地训练，配置使用 `trimesh`，以平滑/粗糙坡面为主。
- `model_3500.pt` 播放时机器人翻倒；`model_1200.pt` 也无法正常站起，左腿支撑异常。
- 旧训练平均奖励在约 1202 次迭代达到 32.4，随后下降到约 -76.4。
- 策略动作噪声标准差从约 1.38 增长到 5.93，表明后期训练发生策略退化。
- 因此旧模型不作为有效平地基线，新实验将从零训练。

## 平地实验计划

1. 创建独立任务 `go2_torque_flat`。
2. 地形设为 `plane`，使用正常站立初始姿态。
3. 第一阶段仅训练低速前进，关闭外部推力、质量随机化和观测丢失。
4. 验证站立、前进、机身高度和左右腿动作对称性。
5. 通过后逐步加入横移、转向、噪声和域随机化。
6. 保存训练曲线、最佳检查点和演示视频，用于周五讨论。

## 当前状态

- 代码副本已创建，原始代码与旧模型未修改。
- Conda 环境：`sata`
- GPU：NVIDIA GeForce RTX 4080 Laptop GPU
- PyTorch：2.0.1+cu118，CUDA 调用正常

## 2026-08-05 实验结果

- Stage 1（`SATA_flat_baseline`）：能够前进，但采用膝盖/小腿着地的奖励捷径。
- Stage 2（`SATA_flat_foot_contact`）：消除了非足端碰撞，但后足拖曳、前后腿不协调。
- Stage 3（`SATA_flat_gait_coordination`）：加入步态时钟、对角接触、摆动足高度和足端滑移约束，获得协调步态。
- Stage 3 最终模型：`legged_gym/logs/SATA_flat_gait_coordination/Aug05_19-46-44_stage3_diagonal_gait_seed1_visible/model_3000.pt`
- Stage 3 最终平均奖励约 204.97，对角接触奖励约 1.85，非足端碰撞接近 0。

## 一键脚本

- `TRAIN_STAGE3.sh`：从零启动 Stage 3 的 3000 轮训练；每次运行会新建日志。
- `PLAY_STAGE3.sh`：播放已保存的 Stage 3 `model_3000.pt`，用于演示。
- `OPEN_TENSORBOARD.sh`：启动 Stage 1–3 全部训练曲线；浏览器打开 `http://localhost:6006`。

运行方式：

```bash
cd /data/sata/SATA_flat_reproduction
./PLAY_STAGE3.sh
```

训练命令：

```bash
cd /data/sata/SATA_flat_reproduction
./TRAIN_STAGE3.sh
```

## 2026-08-06 Stage 4A 结果与明日计划

- Stage 4A 使用 Stage 3 `model_3000.pt` warm-start，在 `trimesh` 斜坡+随机起伏地形上追加训练。
- 最终可用检查点：`legged_gym/logs/SATA_flat_gait_coordination/Aug06_19-21-04_stage4a_rough_curriculum_seed1_visible/model_4900.pt`。
- 最终平均奖励约 190.97，平均回合长度约 2054 步，平均课程等级约 2.93/5，前进奖励约 9.58，非足端碰撞约 -0.0076。
- 演示地形配置：8x8 m，0.1 m 网格，基础坡度约 7.5%，随机起伏配置 -0.06～+0.06 m；实际演示地形高度范围 -0.135～+0.060 m。
- TensorBoard 静态图：`/home/ubuntu22/SATA_flat_paper_assets/stage4a_training_dashboard.png`。

### 明日执行顺序

1. 先评估 Stage 3 与 Stage 4A：平地、上坡、下坡和 0.25/0.5/0.75/1.0 m/s 指令。
2. 统计速度误差、跌倒率、机身高度、横滚/俯仰波动和非足端碰撞率。
3. 启动 Stage 4B：低台阶+随机起伏，继续从 Stage 4A `model_4900.pt` warm-start。
4. Stage 4B 稳定后再加入离散障碍、踏脚石、浅沟和浅坑。
5. 最后恢复观测噪声、摩擦随机化和外力扰动，并更新小论文结果表。
