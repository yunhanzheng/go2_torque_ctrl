"""SAC Stage A environment for learning a normal static posture."""

import torch

from legged_gym.envs.go2.go2_torque_gait import GO2TorqueGait


class GO2TorqueStandSAC(GO2TorqueGait):
    def check_termination(self):
        super().check_termination()
        self.reset_buf |= self.root_states[:, 2] < self.cfg.safety.min_base_height

    def _reward_base_height_error(self):
        return torch.square(self.root_states[:, 2] - self.cfg.rewards.base_height_target)

    def _reward_nominal_posture(self):
        return torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_stand_still(self):
        linear = torch.sum(torch.square(self.base_lin_vel), dim=1)
        angular = torch.sum(torch.square(self.base_ang_vel), dim=1)
        return linear + 0.25 * angular
