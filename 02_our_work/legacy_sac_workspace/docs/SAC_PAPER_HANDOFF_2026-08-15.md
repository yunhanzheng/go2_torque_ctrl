# SAC 小论文实验交接（2026-08-15）

## 下次恢复口令

用户会说：

> 继续小论文，读取今天的进度和训练并继续

收到后，先完整阅读本文件与：

- `/data/sac/docs/PPO_BASELINE_SPEC.md`
- `/data/sata/SATA_flat_reproduction/SATA_HANDOFF_2026-08-07.md`

不要重新运行已经完成的 v7，也不要覆盖现有日志和模型。

## 今日完成结果：SAC v7

- 算法：SAC（双 critic、自动 alpha、replay buffer）。
- 任务：`Isaac-Velocity-Flat-Unitree-Go2-ETHOfficial-Fixed05-Policy200Hz-TimeAligned-SATAObs-v0`
- 实验目录：`eth_official_fixed05_policy200hz_timealigned_sataobs_v7`
- run：`2026-08-15_20-11-23_eth_official_policy200hz_timealigned_safe_seed1_restart`
- seed：1。
- 环境数：4096。
- 训练轮数：2400（日志最终显示 `2399/2400`，从 0 计数）。
- 总环境步数：235,929,600。
- wall-clock：01:08:14。
- 最终吞吐量：约 56,548 steps/s。
- 最终 mean reward：40.72。
- 最终线速度误差：0.0427。
- 最终 yaw 误差：0.0427。
- 最终 base-contact termination：0.0002（0.02%）。

最终 checkpoint：

`/data/sac/logs/rsl_rl/eth_official_fixed05_policy200hz_timealigned_sataobs_v7/2026-08-15_20-11-23_eth_official_policy200hz_timealigned_safe_seed1_restart/model_2399.pt`

训练日志：

`/data/sac/eth_official_policy200hz_timealigned_sataobs_v7_restart_training.log`

最终视频：

`/data/sac/logs/rsl_rl/eth_official_fixed05_policy200hz_timealigned_sataobs_v7/2026-08-15_20-11-23_eth_official_policy200hz_timealigned_safe_seed1_restart/videos/play/rl-video-step-0.mp4`

Web 展示副本：

- `/home/ubuntu22/go2_web_view/go2_model2399.mp4`
- `/home/ubuntu22/go2_web_view/index.html`

临时 HTTP 服务已关闭。需要重开时：

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory /home/ubuntu22/go2_web_view
```

然后 Firefox 打开 `http://127.0.0.1:8765/`。

## v7 实际 Observation 和频率

- 62-D SATA 顺序 Observation。
- 组成：base linear velocity 3、base angular velocity 3、projected gravity 3、joint position error 12、joint velocity 12、command 3、applied torque 12、fatigue placeholder 12、gait clock 2。
- fatigue 的 12 个通道目前全为零，并非真实 SATA fatigue state。
- 关节顺序：FL/FR/RL/RR，每条腿 hip/thigh/calf。
- sim dt：0.005 s。
- decimation：1。
- policy/action/observation/control：200 Hz。
- gait clock：1.5 Hz。
- episode：20 s / 4000 steps。
- SAC n-step：20 steps = 100 ms。
- gamma：`0.96 ** 0.25 = 0.989846400767953`。
- 固定命令：vx=0.5 m/s、vy=0、yaw=0。

实现入口：`/data/sac/eth_fixed05_v2/__init__.py`。

## 视觉结论与已知问题

- 策略已能稳定、直立向前行走，不再采用早期低体趴行策略。
- 用户观察到右前腿（FR）明显向身体内侧收，整体体态仍不如 SATA PPO。
- Observation 的 FR 顺序未发现排错。
- 当前 `symmetry_cfg: null`，没有左右镜像约束。
- 当前 reward 没有足端横向位置、步宽、左右镜像或 hip deviation 约束。
- 当前动作是 12-D joint-position target（scale 0.25），不是 SATA torque/Hill/fatigue pipeline。

若后续做“改进 SAC”消融，候选方案是：

1. 为 62-D SATA Observation 和 12-D action 实现严格左右镜像函数。
2. SAC symmetry data augmentation + mirror loss，初始系数建议 0.05。
3. 四腿统一 hip-deviation penalty（建议从 -0.1 开始），不要只惩罚 FR。
4. 必要时再加入 body-frame 足端横向越界惩罚。

这些改动不能混入主公平对比，必须标为 improved-SAC/ablation。

## 与 SATA PPO 的现状

冻结 PPO 基线：

`/data/sata/SATA_flat_reproduction/legged_gym/logs/SATA_flat_gait_coordination/Aug05_19-46-44_stage3_diagonal_gait_seed1_visible/model_3000.pt`

SATA PPO Stage 3：

- 4096 environments，3000 iterations。
- 约 294,912,000 环境步。
- 训练约 24 分 15 秒。
- 推算吞吐量约 203k steps/s。
- 最终 mean reward 约 204.97，最高约 220.01；此 reward 与 v7 不同，禁止直接数值比较。
- 训练速度课程逐步扩展到 0.0--0.8 m/s，每 5 s 重采样。
- 评估时固定 0.5 m/s。
- 使用 SATA torque activation、Hill model、fatigue、torque growth、diagonal-contact、swing-clearance、feet-slip 和 non-foot-collision 条件。
- 视觉上对角步态和左右对称性优于当前 SAC v7。

训练速度结论：当前实现下 SATA PPO throughput 约为 SAC v7 的 3.6 倍；总 wall-clock 约快 2.8 倍。这个差异同时混有 Isaac Gym 与 Isaac Sim 后端影响，不能全部归因于 PPO/SAC 算法。

## 研究结论边界

当前 v7 只能表述为：

> SAC 在 Isaac Lab、200-Hz、62-D SATA-layout observation 和固定 0.5 m/s 的简化 joint-position 控制任务上学会了稳定行走。

不能表述为：

- SAC 已优于 SATA/PPO；
- SAC 与 SATA 已完成公平对比；
- reward 40.72 优于或劣于 SATA reward 204.97。

## 下一阶段推荐顺序

### A. 主实验：严格公平 PPO vs SAC

优先在同一个 SATA/Isaac Gym任务中，仅替换算法：

1. 原样复用 Stage 3 torque/Hill/fatigue/growth pipeline。
2. fatigue observation 必须是真实状态，不能是 12 个零。
3. 原样复用 Stage 3 rewards：
   - non-foot collision `-5.0`
   - feet air-time `0.0`
   - diagonal gait-contact `3.0`
   - swing-clearance `1.5`
   - feet-slip `-0.5`
   - gait frequency `1.5 Hz`
   - swing height `0.065 m`
   - clearance sigma `0.025 m`
4. 原样复用 0.0--0.8 m/s command curriculum 与 5-s resampling。
5. 统一初始化、termination、randomization、action limits、evaluation script。
6. 特别检查 off-policy replay 是否保存 pre-reset terminal observation；不能把 reset observation 当作 timeout transition 的 next observation。
7. 至少 3 seeds；按相同 environment interactions 与 wall-clock 两种预算报告。

### B. 统一评估

固定 vx=0.5 m/s、vy=0、yaw=0，报告：

- mean measured vx
- signed error
- MAE / RMSE / std
- fall rate 与有效跟踪时长
- feet slip
- non-foot collision
- base height
- gait symmetry / diagonal contact accuracy
- energy/torque 指标
- environment interactions 与 wall-clock

训练 reward 只作诊断，不能作为主要跨任务比较指标。

### C. 改进版 SAC

完成严格公平主实验后，再独立加入 symmetry、hip posture 或 foot-lateral constraint，作为 ablation；不要改变 PPO/SAC 主对比的 MDP。

## 今日决定

- 今天不再临时增加 reward 或重训。
- v7 作为简化 SAC baseline 冻结保留。
- 下一次从公平 SATA Stage 3 SAC 迁移与统一评估设计继续。
