"""Direct joint-torque action term: the network output *is* the torque.

This is deliberately *not* a modification of V51's ``SataTorqueAction``: V51-V55 are
frozen results and their TG model (tanh activation, 0.6 first-order filter, hill model,
motor fatigue, growth clock) must stay byte-identical.  The comparison run needs the
opposite: the smallest possible actuation layer so that what the ground reaction forces
show is the controller, not the actuator model.

    tau = clamp(action_scale * raw_action, -torque_limit, +torque_limit)

No tanh, no filter, no action loss, no hill model, no fatigue, no growth.
"""
from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.utils import configclass


class DirectTorqueAction(JointAction):
    """Joint action term that applies ``clamp(action_scale * a)`` as a pure joint effort.

    The articulation must run with zero stiffness/damping (see ``config.py``), otherwise
    the implicit PD gain adds a second, hidden control loop on top of the network.  The
    torque that reaches the simulation can be read back from ``robot.data.applied_torque``.
    """

    cfg: "DirectTorqueActionCfg"

    def __init__(self, cfg: "DirectTorqueActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._torque_limit = float(cfg.torque_limit)
        self._action_scale = float(cfg.action_scale)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Scale, clamp and store the torque target.

        The clamp is the only non-linearity: it is what keeps an unconstrained policy
        output inside the actuator's torque budget without tanh saturation.
        """
        self._raw_actions[:] = actions
        self._processed_actions = (self._raw_actions * self._action_scale).clamp(
            -self._torque_limit, self._torque_limit
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)


@configclass
class DirectTorqueActionCfg(JointActionCfg):
    """Configuration for :class:`DirectTorqueAction`."""

    class_type: type = DirectTorqueAction

    action_scale: float = 10.0
    """N.m per unit of network output.  Calibrate, do not guess.

    Go2 in stance needs ~5-10 N.m per leg joint (m ~ 15 kg, 37 N per foot, ~0.2 m lever)
    and peaks near 23.5 N.m while walking.  With ``init_noise_std=1.0`` the policy starts
    at ``action_scale`` N.m of white noise, so pick the p95 of a trained policy's torque
    distribution (typically 8-12) and watch ``Act/sat_frac`` in the logs.
    """
    torque_limit: float = 23.5
    """Hard clamp in N.m.  Matches UNITREE_GO2_CFG's ``DCMotorCfg.effort_limit``."""
