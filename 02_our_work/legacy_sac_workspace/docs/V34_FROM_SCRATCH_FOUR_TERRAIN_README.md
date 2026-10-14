# Go2 ETH-SAC V34：严格从头训练四地形方案

## 实验定义

V34 是严格的 from-scratch 实验：

- 不加载 V24、V26 或其他外部 checkpoint；
- Actor、Critic、target Critic、normalizer、optimizer、alpha 和 replay 全部新建；
- V24/V26 只作为论文中的历史对照，不参与 V34 训练；
- 平地阶段结束后，四地形阶段只允许加载本次 V34 自己产生的平地 checkpoint。

本次运行 UTC 时间戳：`20260920T135347Z`。

## 两阶段训练

1. 平地基础：4096 环境、1000 iterations、seed=1；
2. 平地固定评估：seed=7、32 环境，至少 24/32 通过才继续；
3. 四地形 curriculum：4096 环境、800 iterations，继续同一个 V34 模型；
4. 七项固定评估；
5. 生成并校验 WebM；
6. Firefox 自动打开最终 HTML 报告。

四类目标地形为：

- 规则楼梯（上/下）；
- 斜坡（上/下）；
- 随机方块；
- 随机粗糙面。

训练分布为四类目标地形各 22.5%，另含 10% 平地锚点。地形 curriculum 共 10 个难度等级，初始等级为 0–2。

## Policy 输入输出

Actor 输入为 SATA 62 维本体感觉：

| 观测 | 维度 |
|---|---:|
| 机体线速度 | 3 |
| 机体角速度 | 3 |
| 投影重力 | 3 |
| 关节位置相对默认站姿 | 12 |
| 关节速度 | 12 |
| 速度指令 | 3 |
| 已施加关节力矩 | 12 |
| SATA fatigue 占位零向量 | 12 |
| 1.5 Hz gait clock（sin/cos） | 2 |
| 合计 | 62 |

Actor 最后一层输出 24 维：12 个 Gaussian mean 和 12 个 state-dependent log-std。经过 Gaussian 采样（训练）或 mean（评估）、tanh 和动作范围映射后，形成 12 维关节位置动作。

## 网络结构

```text
Actor:
62 -> 512 -> 256 -> 128 -> 24
                         ├─ mean[12]
                         └─ log_std[12]

Twin Critic:
(62 observation + 12 action = 74)
Q1: 74 -> 512 -> 256 -> 128 -> 1
Q2: 74 -> 512 -> 256 -> 128 -> 1
```

隐藏层激活为 SiLU/Swish，Actor 和 Critic 均开启经验 observation normalization。

## 从头初始化方式

网络参数来自新的随机数状态，不复制旧模型。为避免随机 Actor 在第一步把机器人踢倒，只把 Actor 最后一层的 mean 初始化为默认站姿对应动作；这个值由当前环境 action bias/range 解析计算，不来自 checkpoint。

训练日志必须出现：

```text
[V34_FRESH_INIT] no checkpoint loaded
[V34_FRESH_INIT] initial deterministic action: [...接近全零...]
```

这属于安全的算法初始化，仍是严格从头训练。

## 关键 SAC 参数

| 参数 | 值 |
|---|---:|
| 控制频率 | 200 Hz |
| 训练环境 | 4096 |
| 每轮每环境步数 | 24 |
| Actor LR | 平地 5.0e-5；四地形 2.5e-5 |
| Critic LR | 2.0e-4 |
| Replay capacity | 5,000,000 |
| Mini-batch | 8192 |
| Updates/iteration | 200 |
| Gamma | `0.96**0.25` |
| n-step | 20 |
| Tau | 0.003 |
| Policy frequency | 2 |
| Target entropy | `-0.12 × 12` |
| 平地探索噪声上限 | 0.05 |
| 四地形探索噪声上限 | 0.05 |

### 首次运行后的监管修正

首次正式运行在约第 200 轮出现“稳定站立但不前进”的局部最优。监管中止后使用该运行自己的 checkpoint 做平地评估，结果为 0/32，证明继续训练没有意义。该失败运行保留，不用于第二次初始化。

第二次运行仍从全新随机权重、全新 normalizer/optimizer/alpha/replay 开始，并仅修改 V34 专用奖励：

- 静止时 forward tracking reward 严格为 0；
- 0.5 m/s 附近 forward tracking reward 接近 1，权重 20；
- 横向速度由正奖励改为 `-2 * vy^2`；
- 偏航误差由正奖励改为 `-2 * yaw_error^2`；
- 探索噪声提高到 0.05，使新策略能采到有效迈步动作。

## 监管与输出

训练在标题为 `V34严格从头四地形训练` 的 GNOME Terminal 中可见运行。Supervisor 负责阶段日志、PID、timeout、status.json、checkpoint 定位、评估、视频编码和 Firefox 报告。

代码：

```text
/home/ubuntu22/terrain_v34_from_scratch_four_terrain/
```

本次结果：

```text
/home/ubuntu22/terrain_v34_from_scratch_four_terrain/results/20260920T135347Z/
```

训练 checkpoint：

```text
/home/ubuntu22/logs/rsl_rl/terrain_v34_scratch_flat_foundation/
/home/ubuntu22/logs/rsl_rl/terrain_v34_scratch_four_terrain/
```

最终固定评估条件：平地、随机粗糙面 hard、随机方块 hard、上坡 hard、下坡 hard、10 cm 上楼、10 cm 下楼。每项使用 seed=7、32 环境、deterministic policy、首回合穿越判据。

每项生成 50 fps VP9 WebM，FFmpeg 解码校验后写入 HTML；全流程完成后 Firefox 自动打开报告。
