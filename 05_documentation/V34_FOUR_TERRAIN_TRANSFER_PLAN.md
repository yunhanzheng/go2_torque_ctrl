# Go2 ETH-SAC V34：最佳策略迁移与四地形联合训练方案（训练前评审稿）

> 状态：**仅完成方案设计，尚未启动训练。** 只有确认本 README 后，才会创建 V34 代码、打开可见终端并开始训练。
>
> 目标：保留 V24 的楼梯能力与正常步态，继承 V26 的 Rough 地形能力，在同一个 62 维盲策略中联合训练四类目标地形，最后自动生成批量评估、WebM 视频和 Firefox 报告。

## 1. 当前版本结论

### 1.1 已确认的最好模型

| 模型 | 作用 | 已确认结果 | Checkpoint |
|---|---|---|---|
| V24 | 楼梯能力最强的基准 | 平地 32/32；8 cm 32/32；9 cm 31/32；10 cm 31/32 | `/data/sac/saved_models/ETH_SAC_V24_10CM/model_299.pt` |
| V26 | 综合 Rough 地形起点；它本身由 V24 迁移而来 | Rough hard 32/32；Boxes hard 21/32；上坡 hard 16/32；下坡 hard 17/32 | `/home/ubuntu22/logs/rsl_rl/terrain_v26_official3_from_v24/2026-09-17_21-18-06_v26_20260917T211724Z/model_599.pt` |

V24 SHA-256：`6d41e21f2dc7bc7a6dd9f9c490a535507724a5e7eaa7c1dea76d1fbd20dcc5dc`

V26 SHA-256：`8c1f588b9aaa39449bca185df986e079ab4a017d914ed772c04796d66477fb4e`

V29–V32 的五帧历史策略均未超过 V24/V26：从头训练容易在早期探索中摔倒，不能作为这轮初始化。V33 只有设计和源码，**没有启动训练**。

### 1.2 V34 的初始化决定

默认从 **V26 model_599.pt** 迁移，因为：

1. V26 已完整继承 V24 的 Actor、Critic、优化器和温度参数；
2. V26 与 V24 网络结构、62 维输入和 12 维输出完全一致，可无损加载；
3. V26 在三类非楼梯 Rough 地形上均优于 V24，是四地形联合训练更合理的综合起点。

但 V26 训练时没有规则楼梯，现有归档中也没有 V26 的楼梯复测。因此正式训练前先对 V24、V26 做相同 seed 的楼梯与平地预检：

- 若 V26 平地不少于 30/32、8 cm 上楼不少于 30/32，则使用 V26 初始化；
- 若 V26 未通过门禁，则回退到 V24 初始化；
- 两个源模型只读保存，V34 输出到全新目录，绝不覆盖 V24/V26。

这不是“按版本号选最新”，而是用统一门禁选择真正适合四地形任务的最佳起点。

## 2. 四类目标地形

四类指论文中的四个主要地形族；上/下方向属于同一地形族。

| 地形族 | 内部子类 | 参数范围 | 训练占比（建议） |
|---|---|---|---:|
| 规则楼梯 | 上楼、下楼 | 台阶高 0.05–0.12 m，宽 0.40 m | 22.5%（各 11.25%） |
| 斜坡 | 上坡、下坡 | 坡度 0.00–0.30 | 22.5%（各 11.25%） |
| 随机方块 | Boxes / 不规则台阶 | 高度 0.025–0.10 m，网格宽 0.45 m | 22.5% |
| 随机粗糙面 | Random rough | 高度噪声 0.01–0.06 m，步长 0.01 m | 22.5% |

另外保留 **10% 平地锚点**，只用于防止联合微调破坏正常平地步态；它不算第五个目标地形。总占比为 90% 四类目标地形 + 10% 平地。

地形生成器使用 10 个难度等级。初始化只在等级 0–2，随后由 Isaac Lab curriculum 根据机器人前进表现逐步升高或降低难度，不让模型一开始直接落在最高楼梯或最陡斜坡上。

## 3. Policy 输入：精确 62 维

V34 保持 V24/V26 的 SATA 62 维盲策略输入，不加入高度扫描，因此可以完整迁移已有权重。

| 顺序 | 观测项 | 维度 | 预处理 / 含义 |
|---:|---|---:|---|
| 1 | Base linear velocity | 3 | 机体坐标系线速度，乘 2.0 |
| 2 | Base angular velocity | 3 | 机体坐标系角速度，乘 0.25 |
| 3 | Projected gravity | 3 | 重力方向投影到机体坐标系 |
| 4 | Joint position error | 12 | `joint_pos - default_joint_pos`，按 FL/FR/RL/RR，每腿 hip/thigh/calf 排列 |
| 5 | Joint velocity | 12 | 关节速度，乘 0.05 |
| 6 | Velocity command | 3 | `[vx, vy, yaw_rate]`，分别乘 `[2.0, 2.0, 0.25]` |
| 7 | Applied joint torque | 12 | 12 个关节当前施加力矩 |
| 8 | SATA fatigue placeholder | 12 | Isaac Lab 无对应肌肉疲劳状态，固定为 0，保留 SATA 输入布局 |
| 9 | Gait clock | 2 | `[sin(phase), cos(phase)]`，频率 1.5 Hz |
|  | **合计** | **62** | Actor 和 Critic 各自维护 62 维经验归一化统计 |

这是一种 **blind policy**：高度扫描器只服务于地形相对抬脚奖励和评估，不输入 Actor。优点是能与 V24/V26 精确兼容；限制是策略不能直接“看见”前方几何，只能通过本体感觉、力矩与运动历史的瞬时状态适应地形。

## 4. Policy 输出：12 维关节位置动作

Actor 最后一层输出 24 个数：

- 前 12 维：12 个动作的 Gaussian mean；
- 后 12 维：12 个动作的 state-dependent `log_std`；
- 训练时从 Gaussian 采样并经过 `tanh`；
- 评估时使用 mean，再经过 `tanh`；
- SAC 内部再用动作上下界做 affine scaling；
- 环境动作最终解释为 12 个关节相对默认站姿的位置目标，`JointPositionActionCfg scale = 0.5`。

因此网络的策略输出不是直接电机力矩。12 个动作对应 Go2 的四条腿，每腿 hip、thigh、calf 三个关节。

## 5. Actor / Critic 网络结构

### 5.1 Actor

```text
62-D observation
  -> EmpiricalNormalization(62)
  -> Linear(62, 512) + SiLU/Swish
  -> Linear(512, 256) + SiLU/Swish
  -> Linear(256, 128) + SiLU/Swish
  -> Linear(128, 24)
       ├─ mean[12]
       └─ log_std[12]
  -> Gaussian sample (train) / mean (eval)
  -> tanh + action-range scaling
  -> 12-D joint-position action
```

- Actor 可训练参数：199,576；
- observation normalization：开启；
- layer normalization：关闭；
- V34 不改网络宽度、不换激活函数，确保 checkpoint 精确兼容。

### 5.2 Twin Critic

每个 Q 网络输入为 `62-D observation + 12-D action = 74-D`：

```text
Q1: 74 -> 512 -> 256 -> 128 -> 1
Q2: 74 -> 512 -> 256 -> 128 -> 1
```

- 每个 Q 网络参数：202,753；
- 在线 Q1 + Q2 可训练参数：405,506；
- 另有同结构 Q1-target、Q2-target，通过 Polyak soft update 更新，不参与梯度下降；
- Actor + 在线双 Critic 共 605,082 个网络可训练参数，另外有 1 个自动温度参数 `log_alpha`。

## 6. SAC 与迁移训练设置

| 参数 | V34 计划值 | 说明 |
|---|---:|---|
| 并行环境 | 4096 | 与 V24/V26 保持一致 |
| 正式训练 | 600 iterations | 每 50 轮保存一次 checkpoint |
| 每环境每轮步数 | 24 | 200 Hz 下对应 0.12 s 物理时间 |
| 控制 / policy 频率 | 200 Hz | `decimation = 1`，物理步长约 5 ms |
| Replay capacity | 5,000,000 | 新建 replay，不把旧地形分布的 replay 带入新任务 |
| Mini-batch | 8192 | 每次 SAC 更新 |
| Updates / iteration | 200 | 1 epoch × 200 mini-batches |
| Actor LR | **1.0e-5** | 比 V26 的 2.5e-5 更保守，降低灾难性遗忘 |
| Critic LR | 2.0e-4 | 让 Q 网络较快适应联合地形分布 |
| Alpha LR | 2.0e-5 | 自动熵温度 |
| Gamma | `0.96**0.25 ≈ 0.989846` | 200 Hz 与原 50 Hz 的物理折扣时间对齐 |
| n-step | 20 | 200 Hz 下约 100 ms，与原 50 Hz 的 5-step 对齐 |
| Tau | 0.003 | target critic 软更新 |
| Policy frequency | 2 | Critic 两次更新对应一次 Actor 更新 |
| Target entropy | `-0.12 × 12 = -1.44` | 继承稳定版 V15→V24 配置 |
| Gradient clip | 1.0 | Actor/Critic 最大梯度范数 |
| Seed | 1 | 训练；评估固定 seed=7 |

### 6.1 无损迁移内容

从选中的源 checkpoint 加载：

- Actor 权重与 Actor observation normalizer；
- Twin Critic、target Critic 与 Critic observation normalizer；
- Actor/Critic optimizer state；
- `log_alpha` 与 alpha optimizer state。

明确不加载：

- iteration 计数（V34 从 0 开始）；
- replay buffer（新任务必须使用新的四地形数据）；
- 原日志目录或原实验名称。

### 6.2 防遗忘保护

1. 前 25 次 SAC update 只训练 Critic，冻结 Actor 与自动 alpha；
2. 25 次以后 Actor 使用 1.0e-5 小学习率；
3. 10% 平地锚点持续约束正常步态；
4. 每 50 轮保留 checkpoint，不只保留最后一轮；
5. 最终从候选 checkpoint 中按统一评估选最好模型，不默认 `model_599.pt` 一定最好；
6. V24/V26 源 checkpoint 做 SHA-256 校验，训练前后都确认没有被修改。

## 7. 可见 Terminal 与自动监管

正式执行时由一个独立的 GNOME Terminal 打开 `run_visible.sh`。终端实时显示：

- 当前阶段（预检、smoke、正式训练、评估、录像、编码）；
- iteration、总步数、steps/s、ETA；
- reward、episode length；
- Actor/Critic/alpha loss；
- action noise std 与 alpha；
- base-contact termination、速度误差等指标；
- 当前 checkpoint 与结果目录。

Supervisor 同时写 `status.json` 和逐阶段日志。监管规则：

- 检查是否已有 Isaac/训练进程，避免双开抢 GPU；
- 检查 GPU 可见、显存和磁盘空间；
- 检查所有 loss、参数和动作是否 finite；
- 出现 NaN/Inf 立即停止子进程并保留最后 checkpoint；
- base-contact termination 连续异常升高时停止，防止把好策略继续训坏；
- 每个阶段设置独立 timeout；
- 关闭 Terminal 不应静默留下孤儿训练进程；
- 用户手动 Ctrl+C 时优雅终止并写明 `interrupted` 状态。

训练命令不会隐藏到后台日志里：**用户能在 Terminal 里看到全过程，Supervisor 负责自动监管和落盘。**

## 8. 训练前检查与 Smoke Test

在 4096 环境正式训练前，依次执行：

1. 校验 V24/V26 checkpoint 路径、大小和 SHA-256；
2. 对 V24、V26 做平地与楼梯门禁，决定真正的初始化模型；
3. 生成四类地形，确认每类难度上下界和实际 mesh 可生成；
4. 用 64 环境、小 replay、29 iterations 做 smoke test；
5. 验证源 Actor 在加载后的 deterministic output 数值一致；
6. 验证前 25 次 update Actor 完全未变化，之后 Actor 确实开始变化；
7. 验证 checkpoint 能重新加载，且所有参数 finite；
8. Smoke 全通过后才进入 4096 环境正式训练。

Smoke 使用独立 run name 和输出目录，不能被误当成最终模型。

## 9. 统一评估与最终模型选择

评估对象：V24、V26、V34 候选 checkpoint（优先比较 199、399、599；必要时加入中间最好轮次）。

统一规则：

- 固定 seed=7；
- 每个条件 32 个环境；
- 使用 deterministic Actor；
- 只看首回合，不允许摔倒重置后“补过”；
- base 前进超过 3.5 m、四脚位置通过终点、机体保持直立 0.5 s；
- 全程横向偏移 `abs(y) <= 0.75 m`。

评估矩阵：

| 类别 | 条件 |
|---|---|
| 平地锚点 | 平地 |
| 规则楼梯 | 5、8、9、10 cm 上楼；8、10 cm 下楼 |
| 随机粗糙面 | easy / hard |
| 随机方块 | easy / hard |
| 斜坡 | easy / hard，上坡与下坡分别统计 |

建议的最低验收门槛：

- 平地 ≥ 30/32；
- 8 cm 上楼 ≥ 30/32；
- 10 cm 上楼 ≥ 24/32；
- Rough hard ≥ 30/32；
- Boxes hard ≥ 20/32；
- 上坡 hard ≥ 16/32；
- 下坡 hard ≥ 17/32。

模型选择采用“先过门禁、再比较综合分数”的原则。未通过任一关键门禁的 checkpoint 不会因为单项很高而被宣布为最好版本。

## 10. WebM 视频与 Firefox 报告

最终选出的 V34 模型生成以下可视化：

- 楼梯上行 hard WebM；
- 楼梯下行 hard WebM；
- 随机粗糙面 hard WebM；
- 随机方块 hard WebM；
- 斜坡上行 hard WebM；
- 斜坡下行 hard WebM；
- 可选平地步态参考 WebM。

录像要求：

- 仿真仍按 200 Hz，视频按 50 fps 采样；
- 保留真实失败和 reset，不剪掉失败片段；
- FFmpeg 编码为 VP9 WebM；
- 编码后用 FFmpeg 再解码校验，损坏文件不得发布；
- HTML 报告同时展示 V24、V26、V34 的表格、最终 checkpoint、SHA-256 和全部视频；
- 全部流程完成后由 Firefox 自动打开本地 HTML 报告。

## 11. 计划中的目录结构

```text
/home/ubuntu22/terrain_v34_best_transfer_four_terrain/
├── __init__.py
├── config.py                 # 四类地形、curriculum、runner 参数
├── training.py               # checkpoint 迁移、冻结、审计
├── train.py                  # 注册 V34 并启动 ETH-SAC
├── preflight.py              # GPU/源模型/地形/维度检查
├── evaluate.py               # 固定 seed、32 环境统一评估
├── catalog.py                # 评估条件与难度表
├── supervisor.py             # 可见日志、门禁、监管、录像和报告
├── run_visible.sh            # GNOME Terminal 入口
├── latest.txt                # 最近一次结果目录
└── results/<UTC时间戳>/
    ├── status.json
    ├── source_snapshot/
    ├── provenance.json
    ├── training.log
    ├── audit.json
    ├── evaluations/
    ├── videos/*.webm
    ├── results.json
    ├── 结果.md
    └── V34四地形报告.html
```

训练日志与 checkpoint 继续放在：

```text
/home/ubuntu22/logs/rsl_rl/terrain_v34_best_transfer_four_terrain/<run_name>/
```

## 12. 执行边界

当前只批准了“写设计 README 并打开查看”。因此本阶段：

- 不创建 V34 训练代码；
- 不启动 Isaac Sim；
- 不占用 GPU；
- 不运行 smoke test；
- 不进行任何正式训练。

确认本方案后，下一步才是按此 README 实现 V34、做静态检查，然后在可见 Terminal 中启动预检与训练。
