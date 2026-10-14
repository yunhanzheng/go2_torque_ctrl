# ETH-SAC v16 地形实验记录

日期：2026-09-02

## 1. 实验目的

以周末表现稳定的 SAC v15 为基线，在不破坏 v15 策略结构和奖励设计的前提下加入 Isaac Lab 程序化地形，验证已有平地策略能否通过微调适应非平整地面。

本次 v16 不是直接使用 Isaac Lab 官方 rough SAC 策略。v16 继承 v15 的：

- 62 维 SATA actor observation；
- 固定前进速度 0.5 m/s；
- SATA 风格步态、摆腿和稳定性奖励；
- 200 Hz policy/control loop；
- critic-only warm-up 25 iterations；
- warm-up 后 `policy_frequency=2`；
- actor-only checkpoint transfer。

初始化模型为：

```text
/data/sac/logs/rsl_rl/eth_sac_isaaclab_sata_reward_stable_v15/
2026-08-29_18-54-16_isaaclab_ethsac_sata_stable_v15_300iter_seed1_retry1/model_299.pt
```

## 2. v16 地形设计

启用的 Isaac Lab 地形包括：

- random rough（随机粗糙面）；
- boxes（随机方块）；
- pyramid slope（斜坡）；
- pyramid stairs（正台阶）；
- inverted pyramid stairs（反向台阶）。

为保持 v15 checkpoint 的网络输入兼容，height scanner 没有加入 actor observation。Actor 仍是原来的 62 维盲走策略。

Height scanner 仅用于奖励计算：v15 原来的足端高度奖励使用世界坐标 z，在起伏地形上会失真；v16 改为足端相对局部地面的 clearance，再计算摆腿高度奖励和过高抬腿惩罚。

## 3. 训练配置

```text
task: Isaac-Velocity-Rough-Unitree-Go2-ETHSAC-SATAReward-Terrain-v0
num_envs: 4096
iterations: 300
seed: 1
steps_per_env: 24
actor initialization: v15 model_299.pt
critic/optimizer/alpha/replay: reset
```

训练约 12 分 56 秒，总环境交互量约 29,491,200 steps。

训练过程中：

- 没有出现 NaN；
- 没有发生 CUDA OOM；
- critic loss 从约 2 以上逐渐降到约 0.5；
- terrain curriculum level 从早期低点约 1.62 回升到最终约 2.20；
- 后期累计 time-out 比例接近 48%；
- 后期单轮 mean reward 多次达到约 230–250；
- 最终单轮指标存在正常的 episode 批次波动，不应只用最后一轮 reward 判断模型。

## 4. 保存位置

### 代码

```text
/home/ubuntu22/eth_sata_reward_v16_terrain/__init__.py
/home/ubuntu22/train_eth_sata_reward_v16_terrain.py
/home/ubuntu22/play_eth_sata_reward_v16_terrain_right_camera.py
/home/ubuntu22/TRAIN_ETH_SAC_SATA_TERRAIN_V16_VISIBLE.sh
/home/ubuntu22/RECORD_ETH_SAC_TERRAIN_V16_SHOWCASE.sh
```

### 训练结果

```text
/data/sac/logs/rsl_rl/eth_sac_isaaclab_sata_reward_terrain_v16/
2026-09-02_19-40-58_isaaclab_ethsac_sata_terrain_v16_300iter_seed1/
```

最终 checkpoint：

```text
/data/sac/logs/rsl_rl/eth_sac_isaaclab_sata_reward_terrain_v16/
2026-09-02_19-40-58_isaaclab_ethsac_sata_terrain_v16_300iter_seed1/model_299.pt
```

完整训练日志：

```text
/home/ubuntu22/isaaclab_ethsac_sata_terrain_v16_300iter_seed1_training.log
```

### 五段 8 秒 WebM 演示

```text
/home/ubuntu22/terrain_v16_1_random_rough.webm
/home/ubuntu22/terrain_v16_2_boxes.webm
/home/ubuntu22/terrain_v16_3_slope.webm
/home/ubuntu22/terrain_v16_4_stairs.webm
/home/ubuntu22/terrain_v16_5_inverse_stairs.webm
```

## 5. 当前结论

前三类地形的回放结果较好：

- 随机粗糙面：能够持续前进并保持步态；
- 随机方块：能够适应离散的小高度变化；
- 斜坡：能够稳定运动。

楼梯结果目前不应作为成功案例：

- 正台阶能够靠近并偶尔踏上前几级，但容易卡住或失稳 reset；
- 反向台阶没有形成稳定有效的通过行为；
- 当前 62 维 actor 看不到前方几何，只能依靠接触后的被动反馈；
- 官方台阶高度最高约 23 cm，而 v15/v16 的目标摆腿 clearance 为 6.5 cm，二者能力范围不匹配。

因此，当前可以得出的谨慎结论是：

> SAC v15 经过 actor-only terrain fine-tune 后，能够适应随机粗糙面、随机方块和斜坡；现有盲策略对台阶的泛化仍然不足。

## 6. 下一步建议

明天优先进行独立的 stair curriculum，不修改今天已经保存的 v16：

1. 新建 v17 stair 配置，保留 v16 作为对照组；
2. 初始台阶高度设为约 2.5–8 cm；
3. 将相对地面的 swing clearance 目标提高到适合台阶的范围；
4. 先验证低台阶稳定通过，再逐步增加到 10–15 cm；
5. 分别评估上楼梯和下楼梯，固定出生方向和相机；
6. 如果盲策略仍无法稳定通过，再建立带 height scan 的 perceptive actor；
7. perceptive actor 输入维度不同，不能直接加载 v15 actor，需要作为新的实验分支报告。

## 7. 论文表述注意事项

- 不应把当前楼梯视频描述为成功通过楼梯；
- v16 可作为“从平地策略迁移到连续/低幅离散地形”的初步结果；
- v15 与 v16 之间不仅地形发生变化，v16 的 clearance 奖励也改为 terrain-relative，因此需要在论文中明确说明；
- 正式结果建议至少运行 3 个 seeds，并使用统一评估地形和固定难度统计成功率、平均存活时间、速度误差和 terrain level。

 codex resume 01a09705-cc0e-7892-a215-3d6742f9e7fa
Or run codex resume and select 检查小论文楼梯学习表述.

