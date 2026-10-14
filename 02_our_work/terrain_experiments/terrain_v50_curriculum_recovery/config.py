"""Recover a meaningful terrain level while retaining V49 terrain balance."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import gymnasium as gym
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass

from terrain_v49_generalize_23cm.config import (
    V49Generalize23CmEnvCfg,
    V49Generalize23CmRunnerCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_forward_survival(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Advance on forward progress/survival; demote only clear early failures.

    Promotion stops at level 7.  This avoids IsaacLab's level-9 completion
    wrap-around randomly sending successful robots back to levels 0--8 and
    gives the requested stable operating band around level 6.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)

    forward = asset.data.root_pos_w[ids, 0] - env.scene.env_origins[ids, 0]
    survival = env.episode_length_buf[ids].float() / float(env.max_episode_length)
    levels = terrain.terrain_levels[ids]

    demonstrated_progress = forward > 2.0
    survived_with_progress = (survival >= 0.80) & (forward > 0.75)
    move_up = (demonstrated_progress | survived_with_progress) & (levels < 7)

    early_failure = survival < 0.35
    move_down = early_failure & (forward < 0.75) & (~move_up)
    terrain.update_env_origins(ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


@configclass
class V50CurriculumRecoveryEnvCfg(V49Generalize23CmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Initial levels 0--6 average 3; the policy must earn promotion toward
        # the stable 6--7 band under the new criterion.
        self.scene.terrain.max_init_terrain_level = 6
        self.curriculum.terrain_levels.func = terrain_levels_forward_survival


@configclass
class V50CurriculumRecoveryRunnerCfg(V49Generalize23CmRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v50_curriculum_recovery"
        self.save_interval = 20
        self.algorithm.class_name = "terrain_v50_curriculum_recovery.models:V50GuardedTransferSAC"


gym.register(
    id="Isaac-Go2-V50-CurriculumRecovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V50CurriculumRecoveryEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V50CurriculumRecoveryRunnerCfg",
    },
)
