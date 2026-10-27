# 快照：V52「每物理子步对齐 SATA」版本

创建时间：2026-10-21 21:11（本地）
对应对照 run：`logs/rsl_rl/terrain_v52_sata_ppo_flat/2026-10-21_19-35-31_V52_sata_ppo_flat_20261021T113518Z`（3000 iter 跑完）

## 这一版的特征
- `sata_alignment.py::SataControlFreqEnv.step` -> `_sata_substep_step(action, n)`
  每 5 ms 物理子步执行一次 `termination_manager.compute()` /
  `reward_manager.compute(dt=physics_dt)` / `_reset_idx`，力矩每子步重算。
- `config.py` 关闭 `base_contact` 终止（`self.terminations.base_contact = None`），
  终止集合 = time_out + dof_pos_limit + fell_over。

## 文件清单（md5 见 MD5SUMS.txt）
| 快照文件 | 原路径 |
| --- | --- |
| config.py / sata_rewards.py / sata_terrain.py / evaluate_video.py / train.py / run_trial.sh / smoke_test.py | `final/source/terrain_v52_sata_ppo_flat/` |
| v51/sata_alignment.py | `final/source/terrain_v51_native_terrain/` |

## 恢复方法
```bash
cd /home/ubuntu22/data/go2_torque_ctrl/final/source
S=terrain_v52_sata_ppo_flat/_snapshot_2026-10-21_2110_substep_aligned
cp $S/{config.py,sata_rewards.py,sata_terrain.py,evaluate_video.py,train.py,run_trial.sh,smoke_test.py} terrain_v52_sata_ppo_flat/
cp $S/v51/sata_alignment.py terrain_v51_native_terrain/
cd terrain_v52_sata_ppo_flat && md5sum -c $S/MD5SUMS.txt   # 校验（路径相对 V52 目录）
```
服务器端同名快照目录已同步，可原样执行。

## 回滚目标（用户要求复现）
`2026-10-21_19-02-15_V52_sata_ppo_flat_20261021T110204Z`（只跑到 341 iter）
- 终止集合包含 `base_contact`（`illegal_contact` on `base`，threshold 1.0）
- reward / termination / observation 每**控制周期**一次，而不是每物理子步一次
- 其余配置（9 项奖励权重、SATA 地形 negative_prob 0.375、DR、60D obs、PPO 超参）与该 run 的
  `params/env.yaml` 一致，未改动
