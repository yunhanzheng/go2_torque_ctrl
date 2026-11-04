"""Base-relative standing reference, plus the analytic torque teacher that defines it.

Why relative
------------
V56 originally judged "standing" against an absolute body height (``base_height_l2`` with a
hand-picked ``target_height=0.30``).  Measurements (``diag_spawn_pose.py``) show that number cannot
be derived from the spawn pose: at reset the four foot bodies sit at z = 0.0998 (FL/FR) and 0.1090
(RL/RR) m, i.e. the nominal joint angles - front hip pitch 0.8 rad, rear 1.0 rad, every knee -1.5
rad - are *not* a level stance.  The rear soles are 9.2 mm above the front ones, so no vertical
translation of the spawn frame puts all four feet on the plane at once, and any absolute target
height either penalises a geometric fact or hard-codes a choice of controller gains.

Everything in this module is therefore expressed in the base frame and relative to the frame the
episode started in, the way motion-tracking stacks such as BeyondMimic do it:

* joint pose      ``dq = q - q_0``                       (no ground, no gains)
* uprightness     angle between ``g`` and ``g_0``,       ``g = R^T [0,0,-1]`` (yaw-free)
* leg extension   ``ell = -(mean(foot) - base) . z_body``,  ``z_body = -g``   (ground-free)

``ell`` replaces absolute height: it is the distance from the body to its own feet along the body
z axis.  A robot that crouches, one that stands on a 5 cm slope and one that stands on the plane
all have different absolute base z but the same ``ell`` when their legs are equally extended.

The teacher
-----------
The imitation label is the stock IsaacLab actuator law evaluated on the reference pose,

    tau* = Kp (q_0 - q) + Kd (0 - qdot),      Kp = 25.0, Kd = 0.5 N.m/(rad/s),

which are the ``DCMotorCfg`` numbers of ``UNITREE_GO2_CFG`` - not a tuned controller.  It needs
only ``(q, qdot)``, so a label exists for *every* state, independently of whether the episode is
still alive; that is what makes DAgger cheap here and what removes the reward table as the source
of the position/torque asymmetry.

Note the structural asymmetry this exposes, which is the point of the experiment rather than a
bug: on the position arm ``tau*`` is produced by its own built-in PD with the action held at 0
(because ``q_0`` equals the default joint angles), so the label is a constant the network can emit
without learning anything.  On the torque arm the same label is a state feedback the network has to
recover from data.
"""
from __future__ import annotations

import torch

TEACHER_KP: float = 25.0
"""``UNITREE_GO2_CFG`` leg ``DCMotorCfg.stiffness`` - the reference gain, not a tuned one."""

TEACHER_KD: float = 0.5
"""``UNITREE_GO2_CFG`` leg ``DCMotorCfg.damping``."""

TEACHER_TORQUE_LIMIT: float = 23.5
"""``DCMotorCfg.effort_limit`` in N.m; the labels are clipped to it because no actuator can exceed it."""

FOOT_ORDER: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def _body_up(robot) -> torch.Tensor:
    """``[num_envs, 3]`` base z axis in world coordinates.

    ``projected_gravity_b`` is ``R^T [0,0,-1]`` (the gravity vector seen from the body), so the body
    "up" direction is its negation - on the level spawn frame that is exactly ``[0,0,1]``.
    """
    return -robot.data.projected_gravity_b


def leg_extension(robot, foot_body_ids: list[int]) -> torch.Tensor:
    """``[num_envs]`` distance from the base to the mean foot position along the body z axis (m).

    Positive, and it grows as the legs straighten.  Ground-free by construction: it uses only the
    feet relative to the base, so the 9.2 mm front/rear offset of the nominal pose and the height of
    the plane never enter.
    """
    up = _body_up(robot)
    rel = robot.data.body_pos_w[:, foot_body_ids, :] - robot.data.root_pos_w[:, None, :]
    return -(rel.mean(dim=1) * up).sum(dim=1)


def foot_extensions(robot, foot_body_ids: list[int]) -> torch.Tensor:
    """``[num_envs, 4]`` per-foot version of :func:`leg_extension`, in ``FOOT_ORDER``."""
    up = _body_up(robot)
    rel = robot.data.body_pos_w[:, foot_body_ids, :] - robot.data.root_pos_w[:, None, :]
    return -(rel * up[:, None, :]).sum(dim=2)


class StandingReference:
    """Per-environment snapshot of the frame each episode started in, and the relative errors."""

    def __init__(self, env, foot_body_ids: list[int]) -> None:
        self.env = env
        self.foot_body_ids = foot_body_ids
        num_envs = env.num_envs
        joint_count = env.scene["robot"].num_joints
        self.q_0 = torch.zeros(num_envs, joint_count, device=env.device)
        self.g_0 = torch.zeros(num_envs, 3, device=env.device)
        self.g_0[:, 2] = -1.0
        self.ell_0 = torch.zeros(num_envs, device=env.device)
        self.foot_ell_0 = torch.zeros(num_envs, len(foot_body_ids), device=env.device)
        self._captured = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    def capture(self, env_ids: torch.Tensor) -> None:
        """Store the *current* base-relative state as the reference for these environments.

        Called on ``reset()`` for every environment and, inside ``step()``, for the environments
        whose ``episode_length_buf`` came back 0 from an auto-reset - at that point
        ``scene.reset()`` has already written the spawn state into the asset's own buffers, so
        this reads the true first frame and not one step of motion after it.
        """
        if env_ids.numel() == 0:
            return
        robot = self.env.scene["robot"]
        self.q_0[env_ids] = robot.data.joint_pos[env_ids]
        self.g_0[env_ids] = robot.data.projected_gravity_b[env_ids]
        self.ell_0[env_ids] = leg_extension(robot, self.foot_body_ids)[env_ids]
        self.foot_ell_0[env_ids] = foot_extensions(robot, self.foot_body_ids)[env_ids]
        self._captured[env_ids] = True

    def missing(self) -> torch.Tensor:
        """Environments that have never been captured (a fresh env before its first reset)."""
        return ~self._captured

    # -- relative state and errors ---------------------------------------------------------------

    def relative_state(self) -> dict[str, torch.Tensor]:
        robot = self.env.scene["robot"]
        return {
            "dq": robot.data.joint_pos - self.q_0,
            "dqdot": robot.data.joint_vel,
            "gravity_b": robot.data.projected_gravity_b,
            "ell": leg_extension(robot, self.foot_body_ids),
            "foot_ell": foot_extensions(robot, self.foot_body_ids),
        }

    def tilt_error(self) -> torch.Tensor:
        """``[num_envs]`` angle (rad) between the current and the reference gravity direction."""
        cos = torch.nn.functional.cosine_similarity(
            self.env.scene["robot"].data.projected_gravity_b, self.g_0, dim=1
        )
        return torch.acos(cos.clamp(-1.0, 1.0))

    def tracking_error(self) -> dict[str, torch.Tensor]:
        """Per-environment scalars that answer "is it still in the frame it started in?".

        All three are dimensionless-or-physical and none of them refers to the world frame, so a
        different spawn height or a different plane elevation cannot move them.
        """
        state = self.relative_state()
        return {
            # rms over the 12 joints, rad - how far the leg configuration drifted.
            "dq_rms": state["dq"].square().mean(dim=1).sqrt(),
            # worst single joint, rad - crouching shows up mostly in the knees.
            "dq_max": state["dq"].abs().amax(dim=1),
            "tilt": self.tilt_error(),
            # signed, m: negative means the body sank towards its own feet.
            "d_ell": state["ell"] - self.ell_0,
            # largest per-foot drift of the base-to-foot distance, m.  Catches a pose that keeps
            # the mean extension but tucks the rear legs and stretches the front ones.
            "d_foot_ell_max": (state["foot_ell"] - self.foot_ell_0).abs().amax(dim=1),
        }


def teacher_torque(robot, ref: StandingReference, limit: float = TEACHER_TORQUE_LIMIT) -> torch.Tensor:
    """``[num_envs, num_joints]`` joint torque of the stock PD law aimed at ``ref.q_0``."""
    tau = TEACHER_KP * (ref.q_0 - robot.data.joint_pos) - TEACHER_KD * robot.data.joint_vel
    return tau.clamp(-limit, limit)


def teacher_action_position(
    robot, ref: StandingReference, action_scale: float, kp: float = TEACHER_KP
) -> torch.Tensor:
    """The same torque expressed as a position-arm action.

    The position arm applies ``tau = kp (q_cmd - q) - damping qdot`` with
    ``q_cmd = default_joint_pos + action_scale * a``, so reproducing ``tau*`` needs

        a = (q_0 - q_default) / action_scale + tau* / (kp * action_scale).

    ``q_0`` equals ``q_default`` with the stock reset events, so the first term vanishes and the
    label is identically zero - see the module docstring for why that is the null hypothesis here
    rather than a mistake.
    """
    tau = teacher_torque(robot, ref)
    offset = (ref.q_0 - robot.data.default_joint_pos) / action_scale
    return offset + tau / (kp * action_scale)


def teacher_action_torque(robot, ref: StandingReference, action_scale: float) -> torch.Tensor:
    """``tau*`` expressed as an action of :class:`DirectTorqueAction` (``tau = action_scale * a``)."""
    return teacher_torque(robot, ref) / action_scale
