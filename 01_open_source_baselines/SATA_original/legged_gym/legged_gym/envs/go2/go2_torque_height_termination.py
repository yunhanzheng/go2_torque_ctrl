"""Gait environment that rejects low-body locomotion."""

from legged_gym.envs.go2.go2_torque_gait import GO2TorqueGait


class GO2TorqueHeightTermination(GO2TorqueGait):
    def check_termination(self):
        super().check_termination()
        too_low = self.root_states[:, 2] < self.cfg.safety.min_base_height
        self.reset_buf |= too_low
