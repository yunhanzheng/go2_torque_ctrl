"""V53 - SAC control experiment: SAC + flat ground + SATA's 9 rewards.

Purpose
-------
Controlled comparison against V52 (``terrain_v52_sata_ppo_flat``).  The *environment*
is V52's, unchanged: flat ground, SATA's 9 rewards, SATA-aligned TG torque dynamics,
Gompertz growth curriculum, 60-D observation, SATA joint limits, and SATA's terminations
(``invalid_upright_state`` disabled).  The only change is the learning algorithm:

  - V52 : PPO (rsl_rl OnPolicyRunner, fresh init, 3000 iters)
  - V53 : SAC (rsl_rl OffPolicyRunner, fresh init, 1500 iters)

SAC hyperparameters come from this repo's existing *flat, from-scratch* SAC recipe
(``V34FlatRunnerCfg``): CenteredStartSAC fresh init, actor_lr=5e-5, critic_lr=2e-4,
gamma=0.96, n_steps=5, 5M replay buffer.  Nothing on the env side is overridden here -
V53 inherits ``V52SataPpoFlatEnvCfg`` verbatim.
"""
from __future__ import annotations

import os
import sys

# --- Path bootstrap: mirrors V52/V51.  ``dependency_chain`` is required for
# ``terrain_v34_from_scratch_four_terrain`` (the flat SAC runner recipe), and the V51
# directory because V52 reuses V51's ``sata_alignment`` (imported as a bare module).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.path.dirname(_HERE)
for _p in (
    _SOURCE_ROOT,
    os.path.join(_SOURCE_ROOT, "dependency_chain"),
    os.path.join(_SOURCE_ROOT, "base_adapter"),
    os.path.join(_SOURCE_ROOT, "base_adapter", "initial_stair_source"),
    os.path.join(_SOURCE_ROOT, "terrain_v51_native_terrain"),
    _HERE,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gymnasium as gym
from isaaclab.utils import configclass

from terrain_v52_sata_ppo_flat.config import V52SataPpoFlatEnvCfg
from terrain_v34_from_scratch_four_terrain.config import V34FlatRunnerCfg


@configclass
class V53SataSacFlatEnvCfg(V52SataPpoFlatEnvCfg):
    """Identical to V52's environment; only the runner (algorithm) differs."""

    def __post_init__(self):
        # Deliberately no overrides: the SAC control run must see EXACTLY V52's env
        # (flat plane, SATA's 9 rewards, joint limits, disabled invalid_upright_state).
        super().__post_init__()


@configclass
class V53SataSacFlatRunnerCfg(V34FlatRunnerCfg):
    """SAC counterpart of V52's PPO runner.

    ``V34FlatRunnerCfg`` is this repo's validated flat, from-scratch SAC recipe
    (CenteredStartSAC fresh init + actor_lr=5e-5 / critic_lr=2e-4 / gamma=0.96 /
    n_steps=5).  Only the experiment name and budget are retargeted for V53.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "terrain_v53_sata_sac_flat"
        self.max_iterations = 1500
        self.save_interval = 50


gym.register(
    id="Isaac-Go2-V53-SataSacFlat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:V53SataSacFlatEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:V53SataSacFlatRunnerCfg",
    },
)
