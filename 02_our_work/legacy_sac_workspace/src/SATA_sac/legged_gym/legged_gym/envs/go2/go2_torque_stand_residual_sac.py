"""Standing task with a model-based posture torque plus learned SAC residual."""

import torch

from legged_gym.envs.go2.go2_torque_stand_sac import GO2TorqueStandSAC


class GO2TorqueStandResidualSAC(GO2TorqueStandSAC):
    def _compute_torques(self, residual_actions):
        # Convert a conventional default-posture PD support torque back into the
        # SATA policy-action units, then let the unchanged TG motor pipeline
        # apply activation dynamics, Hill model, fatigue, and torque clipping.
        support_torque = (
            self.cfg.residual_support.stiffness * (self.default_dof_pos - self.dof_pos)
            - self.cfg.residual_support.damping * self.dof_vel
        )
        support_action = support_torque / self.cfg.control.action_scale
        combined_action = support_action + residual_actions
        self.residual_actions = residual_actions
        self.support_torque_command = support_torque
        return super()._compute_torques(combined_action)

    def _reward_residual_action(self):
        return torch.mean(torch.square(self.residual_actions), dim=1)
