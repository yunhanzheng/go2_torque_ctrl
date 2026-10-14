"""Gait environment with a soft nominal quadruped-posture objective."""

import torch

from legged_gym.envs.go2.go2_torque_gait import GO2TorqueGait


class GO2TorqueNominalPosture(GO2TorqueGait):
    def _reward_nominal_posture(self):
        # Mean squared joint displacement keeps all four legs near the normal
        # standing configuration while still allowing the learned gait motion.
        return torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
