# SATA PPO 与 SAC v15 对比及展示指南

更新日期：2026-08-29  
用途：明天现场展示 Unitree Go2 的 SATA PPO 基线与 SAC v15 结果。

## 1. 先说明比较对象

SATA 和 SAC 并不是同一层级的概念：

- **SATA** 是执行器、肌肉动力学和任务体系，包含 torque activation、Hill model、fatigue、torque growth 与步态奖励。
- **SAC** 是强化学习算法。

严格的论文主对比应写成 **SATA + PPO vs SATA + SAC**。目前冻结的 SATA 基线使用 PPO 和完整 SATA torque pipeline；当前 SAC v15 使用 Isaac Lab 的关节位置动作及 SATA 风格奖励，因此属于很有价值的预实验，但还不是完全公平的算法消融。

## 2. 当前结果对比

| 项目 | SATA PPO Stage 3 | SAC v15 stable |
|---|---:|---:|
| 算法 | PPO | SAC |
| 后端 | Isaac Gym / Legged Gym | Isaac Lab / Isaac Sim |
| 动作接口 | SATA torque + Hill + fatigue | 12-D joint-position target |
| 训练命令 | 课程扩展至 0.0--0.8 m/s | 固定 0.5 m/s |
| Observation | 62-D，包含 SATA 状态 | 62-D SATA layout；fatigue 通道目前为零 |
| 环境数 | 4096 | 4096 |
| Episode 时长 | 约 10 s | 20 s / 4000 steps |
| 环境交互量 | 约 294,912,000 steps | 29,491,200 steps |
| Wall-clock | 约 24 min 15 s | 7 min 17 s |
| 最终 mean reward | 204.97 | 495.75 |
| 训练历史最高 reward | 220.01 | 495.75（最终） |
| 最终 episode length | 约 2045.83 logged steps | 4000 / 4000 |
| 最终 timeout | 不宜跨实现直接比较 | 100% |
| 最终失败终止 | 需统一脚本重测 | base contact 0%，invalid upright 0% |
| 视觉特点 | 对角步态成熟、左右较对称 | 行走稳定，右后腿偶尔抬得偏高 |

### Reward 为什么是 204.97 对 495.75？

两个 reward 主体接近，但 episode 时长不同。SATA PPO 约 10 秒，SAC v15 为 20 秒；累计 reward 不能直接用绝对值比较。

粗略按时长归一化：

```text
SATA PPO final: 204.97 / 10 s = 20.50 reward/s
SATA PPO best:  220.01 / 10 s = 22.00 reward/s
SAC v15 final:  495.75 / 20 s = 24.79 reward/s
SAC v15 换算到 10 秒：约 247.88
```

这说明 v15 的训练回报表现很好，但由于控制接口、训练速度范围和步态权重仍不同，不能把差异全部归因于 SAC。

### 主要 reward 权重

| Reward 项 | SATA PPO | SAC v15 |
|---|---:|---:|
| Forward tracking 前进速度跟踪| 约 10 | 10 |
| Yaw tracking  偏航角速度| 约 5 | 5 |
| Head/base height 头部 机身高度| 约 5 | 5 |
| Lateral motion  侧向运动惩罚  左右| 约 5 | 5 |
| Roll penalty 横滚  前后| 约 -5 | -5 |
| Gait contact 步态触地 | 3.0 | 5.0 |
| Swing clearance 摆动腿离地高度 | 1.5 | 2.0 |
| Feet slip 足端打滑 | -0.5 | -0.5 |
| Non-foot contact  非足端接触 | -5.0 | -5.0 |
| Excessive foot height  足端过高| 无 | -0.03 |

## 3. SATA PPO 展示路径

### 项目目录

```text
/data/sata/SATA_flat_reproduction
```

用文件管理器打开：

```bash
xdg-open /data/sata/SATA_flat_reproduction
```

### 冻结模型

```text
/data/sata/SATA_flat_reproduction/legged_gym/logs/SATA_flat_gait_coordination/Aug05_19-46-44_stage3_diagonal_gait_seed1_visible/model_3000.pt
```

### 一键现场运行

```bash
bash /data/sata/SATA_flat_reproduction/PLAY_STAGE3.sh
```

该脚本运行 `go2_torque_gait`，加载冻结的 `model_3000.pt`；`play_flat.py` 在评估时将目标速度固定为 0.5 m/s。

## 4. SAC v15 展示路径

### 项目与文档目录

```text
/data/sac
/data/sac/docs
```

用文件管理器打开：

```bash
xdg-open /data/sac
```

### 最终模型

```text
/data/sac/logs/rsl_rl/eth_sac_isaaclab_sata_reward_stable_v15/2026-08-29_18-54-16_isaaclab_ethsac_sata_stable_v15_300iter_seed1_retry1/model_299.pt
```

### Firefox 直接打开回放视频

Snap Firefox 建议打开主目录副本，避免 `/data` 权限问题：

```bash
firefox file:///home/ubuntu22/eth_sac_sata_stable_v15_model299_right.mp4
```

视频副本：

```text
/home/ubuntu22/eth_sac_sata_stable_v15_model299_right.mp4
```

原始视频：

```text
/data/sac/logs/rsl_rl/eth_sac_isaaclab_sata_reward_stable_v15/2026-08-29_18-54-16_isaaclab_ethsac_sata_stable_v15_300iter_seed1_retry1/videos/play/rl-video-step-0.mp4
```

### 重新生成 SAC v15 右侧回放

```bash
cd /data/sac
PYTHONPATH=/home/ubuntu22:/data/sac:/data/sac/scripts \
/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python \
/home/ubuntu22/play_measure_eth_sata_reward_v15_right_camera.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-ETHSAC-SATAReward-Stable-v0 \
  --agent rsl_rl_sac_cfg_entry_point \
  --num_envs 1 \
  --checkpoint /data/sac/logs/rsl_rl/eth_sac_isaaclab_sata_reward_stable_v15/2026-08-29_18-54-16_isaaclab_ethsac_sata_stable_v15_300iter_seed1_retry1/model_299.pt \
  --video --video_length 1200 --headless
```

## 5. 明天推荐展示顺序

1. 先说明公平边界：这是 SATA PPO 完整动力学基线与 SAC v15 稳定性预实验。
2. 运行 SATA PPO：

   ```bash
   bash /data/sata/SATA_flat_reproduction/PLAY_STAGE3.sh
   ```

3. 打开 SAC v15 已录视频：

   ```bash
   firefox file:///home/ubuntu22/eth_sac_sata_stable_v15_model299_right.mp4
   ```

4. 展示上面的对比表，特别解释 episode 10 秒与 20 秒造成的累计 reward 差异。
5. 总结：v15 在约十分之一环境交互量下实现稳定满回合，但严格结论仍需在同一完整 SATA torque 环境中只替换 PPO/SAC，并至少运行 3 个 seeds。

## 6. 论文正式对比下一步

在同一个 SATA Stage-3 环境中统一以下条件，只更换 PPO/SAC：

- 完整 torque/Hill/fatigue/growth pipeline；
- 真实 62-D observation，不能使用全零 fatigue placeholder；
- 相同 0.0--0.8 m/s command curriculum；
- 相同 reward、episode、termination、randomization 与 action limits；
- seed 1、2、3；
- 同时按相同 environment interactions 和相同 wall-clock 两种预算比较；
- 固定 0.25、0.50、0.75、1.00 m/s 统一评估；
- 报告速度 MAE/RMSE、跌倒率、有效跟踪时间、对角接触准确率、左右对称性、足端滑移、非足端碰撞、能耗及扭矩。

训练 reward 只作为诊断指标，正式结论以统一评估结果为准。

## 7. Episode 是什么，为什么一个 10 秒、一个 20 秒？

Episode 是强化学习中的一局或一个回合。机器人从初始状态开始行走，直到达到时间上限，或者因为摔倒、身体触地、姿态失效而提前结束。

SAC v15 的控制频率为 200 Hz，最大 4000 steps：

```text
4000 steps / 200 Hz = 20 s
```

因此：

- `episode length = 4000`：完整走满 20 秒；
- `timeout = 100%`：回合全部因为到达时间上限结束，而不是摔倒；
- episode reward：该回合每一步 reward 的累计值。

| Episode 长度 | 优点 | 局限 |
|---|---|---|
| 10 秒 | 重置频繁，训练早期更快看到初始和恢复状态 | 不容易暴露长期漂移；累计 reward 较小 |
| 20 秒 | 更能检验持续行走和后半程稳定性 | 训练早期可能更慢；累计 reward 天然更大 |

PPO 并不要求 10 秒，SAC 也不要求 20 秒；当前差异来自两个项目继承的环境配置。正式公平对比应统一 episode，例如训练都用 20 秒，并另外进行 60 秒或更长的独立稳定性评估。

## 8. PPO 与 SAC 的算法结构区别

### 总览

| 项目 | PPO | SAC |
|---|---|---|
| 类型 | On-policy | Off-policy |
| 数据使用 | 使用当前策略新采集的轨迹，更新后通常丢弃 | 存入 replay buffer，历史数据可重复学习 |
| Actor | 输出动作分布 | 输出随机动作分布 |
| Critic | 通常一个 `V(s)` 网络 | 两个 `Q(s,a)` 网络及两个 target Q 网络 |
| 稳定机制 | Probability-ratio clipping | 双 Q 取最小值、慢速 target update、replay |
| 探索机制 | Policy noise + entropy bonus | 最大熵目标及温度参数 alpha |
| 数据效率 | 通常较低 | 通常较高 |
| 调参特点 | 相对简单、稳定 | 对 critic、replay、alpha 和更新频率更敏感 |

### PPO 数据流

```text
Observation
   ├── Actor  ──> Action distribution ──> Action
   └── Critic ──> V(s)

最新 rollout ──> Return / Advantage ──> PPO clipped update
```

PPO 比较动作的实际结果与 Critic 预期。如果结果更好，就提高该动作概率；如果更差，就降低概率。它通过 clipping 限制新旧策略概率比，避免单次更新幅度过大：

```text
ratio = new_policy_probability / old_policy_probability
clip range ≈ [1 - epsilon, 1 + epsilon]
```

当前 PPO 基线使用 `epsilon = 0.2`。

### SAC 数据流

```text
Observation ──> Actor ──> Action ──> Environment
                    │
                    ├──> Q1(s,a)
                    └──> Q2(s,a)

(s, a, r, s_next, done) ──> Replay Buffer ──> random mini-batches
                                            ├──> Critic update
                                            ├──> Actor update
                                            └──> Alpha update

Q1/Q2 ──slow update──> Target Q1/Q2
```

SAC 使用两个 Critic，并取较小的估计以降低价值高估：

```text
Q target uses min(Q1_target, Q2_target)
```

Target Critic 缓慢更新：

```text
Q_target <- tau * Q + (1 - tau) * Q_target
```

当前 `tau = 0.003`。

SAC 最大化 reward 与 entropy 的组合：

```text
maximize reward + alpha * entropy
```

alpha 太大时动作过于随机，太小时探索不足；自动 alpha 会根据目标熵调节探索强度。

### 为什么 v15 比 v14 稳定？

v14 从成熟 actor 继续训练时清空了 critic、optimizer、alpha 和 replay buffer，却立即更新 actor。尚未学准的新 critic 可能把已有好策略带偏。

v15 的稳定化流程：

```text
加载 v14 actor
    └──> 前 25 轮冻结 actor 与 alpha，只训练 critic
         └──> actor lr 降至 5e-5
              └──> 每两次 critic update 才更新一次 actor
                   └──> target entropy scale 降至 0.12
```

| 参数 | v14 | v15 |
|---|---:|---:|
| Actor learning rate | `2e-4` | `5e-5` |
| Critic learning rate | `2e-4` | `2e-4` |
| Policy frequency | 1 | 2 |
| Target entropy scale | 0.167 | 0.12 |
| Actor/alpha warm-up freeze | 无 | 25 iterations |

一句话总结：

- **PPO**：只用当前策略的新数据，通过 clipping 限制策略变化，稳定但样本复用较少。
- **SAC**：用 replay buffer 重复学习历史数据，采用双 Q 和最大熵探索，样本效率通常更高，但更依赖稳定的 critic。

## 9. 导师问答速查

### 为什么 SAC reward 495，而 SATA PPO 只有 205？

因为 SAC episode 是 20 秒，PPO 约 10 秒。累计 reward 会随时长增长；粗略归一化后分别约为 `24.79 reward/s` 和 `20.50 reward/s`。同时两边 gait 权重和控制接口仍有差异，不能只凭累计 reward 判断算法优劣。

### 当前能否说 SAC 优于 PPO？

不能。可以说 SAC v15 在固定 0.5 m/s 的简化 Isaac Lab 任务上，以约 2950 万环境步实现稳定满回合，并表现出较好的样本效率。要证明算法优势，必须在同一个完整 SATA torque 环境里只替换 PPO/SAC。

### v15 最重要的结果是什么？

- 300 iterations 完成；
- 29,491,200 environment steps；
- 训练约 437 秒；
- final reward `495.75`；
- episode length `4000/4000`；
- timeout `100%`；
- base-contact 和 invalid-upright termination 均为 `0%`；
- 后半程没有出现 v14 的策略坍塌。

### 下一步实验是什么？

把 SAC 接入完整 SATA torque/Hill/fatigue/growth pipeline，在统一 episode、reward、速度课程、环境随机化和交互预算下，与 PPO 各跑至少 3 个 seeds，并用统一速度测试报告 tracking、跌倒率、步态对称性、滑移、扭矩和能耗。

## 10. 相关资料

- `/data/sac/docs/PPO_BASELINE_SPEC.md`
- `/data/sac/docs/SAC_PAPER_HANDOFF_2026-08-15.md`
- `/home/ubuntu22/ETH_SAC_SATA_BEST_V14_README.md`
- `/home/ubuntu22/isaaclab_ethsac_sata_stable_v15_300iter_seed1_retry1_training.log`
