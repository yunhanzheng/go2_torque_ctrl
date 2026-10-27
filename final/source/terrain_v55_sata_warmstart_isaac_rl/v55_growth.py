"""V55 growth clock: SATA's ``general_scale`` as a bounded start-up ramp.

Why this file exists
--------------------
SATA couples everything to one Gompertz scalar (``go2_torque.py:183-188``)::

    general_scale = exp(-exp(-k * (step_count - x0)))          # k=3e-5, x0=24000
    current_freq              = gs*(200 - 100) + 100           # 100 -> 200 Hz
    current_torque_limit_scale= gs*(1.0 - 0.3) + 0.3           # 0.3 -> 1.0 x 23.5 Nm
    commands                  = ranges * gs                    # x centred, y/yaw +-

V55 warm-starts from ``SATA/logs/SATA/Oct14_16-33-32_/model_3000.pt``.  That checkpoint was
written at ``step_count = 24 * 3000 = 72 000``, i.e.

    gs(72 000) = exp(-exp(-3e-5 * (72000 - 24000))) = 0.789

so the policy was trained at **179 Hz, 20.0 Nm** (0.3 + 0.7 * 0.789 = 0.8524 x 23.5),
**vx in [-0.29, 1.29]**.  Restarting the
Gompertz from zero (what the inherited code does on a fresh run) would drop it straight back
to gs = 0.128 -- 113 Hz and 7 Nm, which destroys the warm start.  Continuing the Gompertz
instead would need another ~50 k steps to reach 0.95, during which the control frequency
keeps drifting.

V55 therefore replaces the clock with an explicit ramp:

    gs(n) = 0.789 + (1.0 - 0.789) * min(n / gs_ramp_steps, 1)      n = policy steps

- starts exactly at the pretrained operating point (no distribution shift at step 0),
- reaches full capability (200 Hz, 23.5 Nm, full command range) after
  ``gs_ramp_steps`` = 12 000 policy steps = 500 iterations of ``num_steps_per_env = 24``,
- then stays at 1.0, which is also the regime V54 runs at once its own Gompertz saturates,
  so V55 and V54 stay numerically comparable.

The reward side of gs vanishes in V55 (all rewards are IsaacLab defaults and none of them
reads gs); gs now only describes **actuator capability, control frequency and command
range**.  Nothing else in the SATA actuator model changes -- the tanh activation, the hill
model, the activation loss and the fatigue buffer are inherited verbatim.

Evaluation / play must pin gs at the trained value, because the counter starts at 0::

    env.unwrapped._v55_freeze_gs = 1.0

One clock, three consumers
--------------------------
SATA's gs is read in three places (all through the module-level ``sata_growth_scale``):

1. ``SataTorqueAction`` -> torque ceiling (0.3 + 0.7 gs) x 23.5 Nm;
2. ``SataVelocityCommand`` -> command ranges (x anchored at 0.5 m/s, y/yaw scaled by gs);
3. ``SataControlFreqEnv.step`` -> control frequency 100 + 100 gs Hz (hence decimation).

If only (1) followed the ramp, a V55 run would drive 20 Nm at 113 Hz (the stale Gompertz
value at step 0) instead of the 179 Hz the pretrained actor expects.  The module function is
therefore replaced with :func:`v55_growth_scale` once, at import time, so all three read the
same number.  ``SataTorqueAction._update_growth`` is overridden too, but only to call the
same function and to keep the substep counter alive.
"""
from __future__ import annotations

import os
import sys

# Same path bootstrap as V51/V52/V54 so this module also imports outside train.py.
_SOURCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SOURCE_ROOT, os.path.join(_SOURCE_ROOT, "terrain_v51_native_terrain")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.utils import configclass

import terrain_v51_native_terrain.sata_alignment as _sata
from terrain_v51_native_terrain.sata_alignment import (
    SataControlFreqEnv,
    SataTorqueAction,
    SataTorqueActionCfg,
)

# gs at SATA model_3000 (step_count = 24 * 3000 = 72 000, k = 3e-5, x0 = 24 000).
SATA_MODEL_3000_GS = 0.7891


def v55_growth_scale(env, k: float = 3e-5, x0: float = 24000.0) -> float:
    """V55 clock.  ``k``/``x0`` are accepted (and ignored) to match the inherited signature.

    Policy steps, not physics substeps: the control frequency doubles over the ramp, so a
    substep counter would make the schedule run ~1.3x faster early on.  Both endpoints live on
    the env (``_v55_gs_start`` / ``_v55_ramp_steps``), copied there from the action cfg by
    :class:`V55ControlFreqEnv`.
    """
    frozen = float(getattr(env, "_v55_freeze_gs", -1.0))
    if frozen >= 0.0:
        return frozen
    start = float(getattr(env, "_v55_gs_start", SATA_MODEL_3000_GS))
    ramp = max(float(getattr(env, "_v55_ramp_steps", 12000.0)), 1.0)
    n = float(getattr(env, "_v55_policy_steps", 0))
    return start + (1.0 - start) * min(max(n / ramp, 0.0), 1.0)


# Replace SATA's Gompertz for every consumer in this process (see the module docstring).
# V55 always runs in its own process, so this cannot leak into a V54 run.
_sata.sata_growth_scale = v55_growth_scale


class V55TorqueAction(SataTorqueAction):
    """SATA TG actuation with a linear growth ramp instead of the Gompertz clock.

    Everything except :meth:`_update_growth` is inherited: the action scale (5.0), tanh
    activation with the first-order filter, the velocity-dependent hill model, the 10 %
    activation-loss dropout and the motor-fatigue buffer.
    """

    def _update_growth(self) -> None:
        # Keep the inherited substep counter alive: it is what SATA's logs and the
        # `--growth_step_offset` style bookkeeping refer to.
        self._env._sata_physics_steps += 1

        gs = v55_growth_scale(self._env)
        self.general_scale = gs
        self.current_torque_limit_scale = (
            gs * (self.cfg.max_torque_scale - self.cfg.start_torque_scale) + self.cfg.start_torque_scale
        )


@configclass
class V55TorqueActionCfg(SataTorqueActionCfg):
    """``SataTorqueActionCfg`` plus the V55 ramp endpoints."""

    class_type: type = V55TorqueAction

    gs_start: float = SATA_MODEL_3000_GS
    """gs at policy step 0 (SATA model_3000's operating point)."""
    gs_ramp_steps: int = 12000
    """Policy steps over which gs climbs from ``gs_start`` to 1.0 (500 x 24)."""


class V55ControlFreqEnv(SataControlFreqEnv):
    """``SataControlFreqEnv`` that also counts *policy* steps for the V55 ramp.

    ``SataControlFreqEnv.step`` is the only place that knows a whole control period has
    elapsed (it runs the substep loop internally), so the counter lives here rather than in
    the action term, which is entered once per physics substep.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        term_cfg = cfg.actions.joint_pos
        # Single source of truth for the ramp: the action cfg.
        self._v55_gs_start = float(term_cfg.gs_start)
        self._v55_ramp_steps = float(term_cfg.gs_ramp_steps)
        self._v55_policy_steps = 0
        self._v55_freeze_gs = -1.0

    def step(self, action):
        self._v55_policy_steps += 1
        return super().step(action)
