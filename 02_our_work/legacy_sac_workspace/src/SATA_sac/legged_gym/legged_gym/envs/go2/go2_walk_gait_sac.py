"""Clock-conditioned diagonal gait for the standard Go2 environment."""

import math

import torch
from isaacgym import gymtorch

from legged_gym.envs.go2.go2_walk_sac import GO2WalkSAC


class GO2WalkGaitSAC(GO2WalkSAC):
    def _init_buffers(self):
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, -1, 13
        )

    def _post_physics_step_callback(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        super()._post_physics_step_callback()

    def _gait_phase(self):
        return (
            2.0 * math.pi * self.cfg.gait.frequency
            * self.episode_length_buf.float() * self.dt
        )

    def _desired_contacts(self):
        first_diagonal = torch.sin(self._gait_phase()) >= 0.0
        desired = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device
        )
        # Standard Go2 body order is FL, FR, RL, RR.
        desired[:, 0] = first_diagonal
        desired[:, 3] = first_diagonal
        desired[:, 1] = ~first_diagonal
        desired[:, 2] = ~first_diagonal
        return desired

    def compute_observations(self):
        super().compute_observations()
        phase = self._gait_phase()
        clock = torch.stack((torch.sin(phase), torch.cos(phase)), dim=1)
        self.obs_buf = torch.cat((self.obs_buf, clock), dim=1)

    def _reward_gait_contact(self):
        actual = self.contact_forces[:, self.feet_indices, 2] > 1.0
        return (actual == self._desired_contacts()).float().mean(dim=1)

    def _reward_swing_clearance(self):
        swing = ~self._desired_contacts()
        height = self.rigid_body_states[:, self.feet_indices, 2]
        score = torch.exp(-torch.square(
            (height - self.cfg.gait.swing_height) / self.cfg.gait.clearance_sigma
        ))
        return (score * swing.float()).sum(dim=1) / 2.0

    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        xy_speed_sq = torch.sum(
            torch.square(self.rigid_body_states[:, self.feet_indices, 7:9]), dim=2
        )
        return torch.sum(xy_speed_sq * contact.float(), dim=1)

    def _reward_contact_count_error(self):
        contact_count = (self.contact_forces[:, self.feet_indices, 2] > 1.0).sum(dim=1)
        return torch.square(contact_count.float() - 2.0)
