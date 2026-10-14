# SAC 楼梯模型：V23 保留版与10cm后续实验

更新：2026-09-16。

## 必须保留的8cm版本

用户认可V23的体态及8cm上楼表现。该版本独立归档，后续训练使用新目录，不覆盖。

- 主归档：`/data/sac/saved_models/ETH_SAC_V23_APPROVED_8CM/`
- 模型：`model_299.pt`；视频：`8cm.webm`、`5cm.webm`、`flat.webm`
- 第二份备份：`/home/ubuntu22/saved_models/V23_APPROVED_8CM/`
- 原始训练：`/home/ubuntu22/logs/rsl_rl/stairs_v23_from_v22/2026-09-16_01-04-07_v23_20260916T010341Z/`
- 原始评估及Firefox对比页：`/home/ubuntu22/stair_v23/results/20260916T010341Z/视频对比.html`
- 归档内含参数YAML、训练日志、来源审计、评估JSON、源码快照和`SHA256.json`校验表。

## 已确认结果

固定评估seed=7，32环境，仅统计首回合。0.5m/s，台阶宽40cm。成功要求机身x>3.5m、全部足端x>3.05m、重力z<-0.85且维持0.5秒，无重置。平地同样统计规定行走距离，不代表完整20秒无失稳。

| 条件 | V22 | V23 |
|---|---:|---:|
| 平地 |32/32|32/32|
|5cm上楼|32/32|32/32|
|6cm上楼|29/32|32/32|
|7cm上楼|3/32|32/32|
|8cm上楼|0/32|30/32|
|10cm上楼|0/32|1/32|

8cm演示连续约12秒，无重置并达到通过标准。单训练种子及单评估种子，不能据此宣称泛化或跨种子显著提升。10cm尚未稳定学会。

## 训练来源与方法

V14平地actor→V22平地/低楼梯混合训练300轮→V23更高楼梯训练300轮。V23继承V22 actor、critic、优化器和alpha；经验缓存重建，阶段计数重新开始，前25次SAC.update冻结actor与alpha。4096环境，seed=1，actor学习率2.5e-5。62维单帧本体输入、12维关节位置动作。高度扫描只参与奖励，不输入actor。

V23地形为25%平地、约58%上楼、约17%下楼，楼梯5–9.5cm。摆腿奖励目标在有地面高度变化时从6.5cm提高至10cm，sigma=3.5cm，过高抬腿阈值14cm。当前结果不区分各改动的单独作用。

## 使用

在Firefox打开主归档中的WebM即可查看。恢复或复现实验需本机Isaac Lab环境：`/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python`，以及`/data/sac`、`/data/eth_sac`相关依赖。源码快照保留原路径，不能作为脱离本机依赖的一键安装包。

## 今日10cm结果：V24

从V23继续训练300轮，4096环境，seed1。加载actor、critic、优化器及alpha，经验缓存重新建立，前25次SAC.update仅更新critic；actor lr=2.5e-5。楼梯课程7.5–12cm，初始难度0–2；保持25%平地练习。附近地面有起伏时摆腿目标最高12cm，平地仍6.5cm；过高抬腿阈值14cm。该实验改变了课程与奖励目标，不是单因素消融。

| 条件 | V23保留版 | V24 |
|---|---:|---:|
|平地规定距离|32/32|32/32|
|5cm上楼|32/32|32/32|
|8cm上楼|30/32|32/32|
|9cm上楼|18/32|31/32|
|10cm上楼|1/32|31/32|

固定最终model_299.pt。协议与V23一致，seed7、32环境。V23已有条件复用已保存评估，9cm额外评估。10cm达到96.875%通过率，仍有1个失败；这是当前固定仿真条件的结果，不代表任意楼梯或实机可靠性。

- V24独立归档位置：`/data/sac/saved_models/ETH_SAC_V24_10CM/`
- 入口：`/home/ubuntu22/stair_v24/run_visible.sh`
- 实时日志：`/home/ubuntu22/stair_v24/visible.log`
- 本轮原始结果：`/home/ubuntu22/stair_v24/results/20260916T074323Z/`
- Firefox页面：本轮目录中的`视频对比.html`（已生成，已用Firefox打开）

V23归档保持原样；V24使用独立归档。6段WebM均已通过完整解码校验。V24的10cm演示连续约12秒，达到通过标准且全程无重置。

V24模型SHA256：`6d41e21f2dc7bc7a6dd9f9c490a535507724a5e7eaa7c1dea76d1fbd20dcc5dc`。

归档内：`model_299.pt`、`视频对比.html`、`videos/V24_up_0.10m.webm`、参数、训练日志、JSON评估、源码快照及`SHA256.json`。

## 快速打开

- V23 8cm保留视频：`/data/sac/saved_models/ETH_SAC_V23_APPROVED_8CM/8cm.webm`
- V24 10cm视频：`/data/sac/saved_models/ETH_SAC_V24_10CM/videos/V24_up_0.10m.webm`
- V23/V24对比：`/data/sac/saved_models/ETH_SAC_V24_10CM/视频对比.html`
