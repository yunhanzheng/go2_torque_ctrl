"""Standard Go2 walking rewards and safety checks; no SATA dependencies."""

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot


class GO2WalkSAC(LeggedRobot):
    def check_termination(self):
        super().check_termination()
        self.reset_buf |= self.root_states[:, 2] < self.cfg.safety.min_base_height
        if hasattr(self.cfg.safety, "max_tilt"):
            tilt = torch.linalg.vector_norm(self.projected_gravity[:, :2], dim=1)
            self.reset_buf |= tilt > self.cfg.safety.max_tilt

    def _reward_forward_progress(self):
        # Signed body-frame forward speed: standing still earns exactly zero.
        return torch.clamp(self.base_lin_vel[:, 0], min=-0.2, max=0.6)

    def _reward_low_speed(self):
        return (self.base_lin_vel[:, 0] < self.cfg.safety.min_forward_speed).float()

    def _reward_speed_error(self):
        return torch.square(self.base_lin_vel[:, 0] - self.commands[:, 0])

    def _reward_nominal_posture(self):
        return torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
