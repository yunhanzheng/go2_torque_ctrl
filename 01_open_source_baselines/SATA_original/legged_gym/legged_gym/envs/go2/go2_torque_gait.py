"""Torque locomotion environment with explicit diagonal gait coordination."""

import numpy as np
import torch

from legged_gym.envs.go2.go2_torque.go2_torque import GO2Torque


class GO2TorqueGait(GO2Torque):
    """Adds an observable gait clock and per-foot coordination rewards."""

    def _post_physics_step_callback(self):
        # SATA acquires this tensor for playback but does not refresh it during
        # training. Stage 3 needs current foot positions and velocities.
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        super()._post_physics_step_callback()

    def _gait_phase(self):
        frequency = self.cfg.gait.frequency
        return 2.0 * np.pi * frequency * self.episode_length_buf.float() * self.dt

    def _desired_contacts(self):
        first_diagonal = torch.sin(self._gait_phase()) >= 0.0
        desired = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device
        )
        # Isaac Gym foot order for this GO2 asset: FL, FR, RL, RR.
        desired[:, 0] = first_diagonal
        desired[:, 3] = first_diagonal
        desired[:, 1] = ~first_diagonal
        desired[:, 2] = ~first_diagonal
        return desired

    def compute_observations(self):
        phase = self._gait_phase()
        gait_clock = torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)
        obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.commands[:, :3] * self.commands_scale,
            self.torques,
            self.motor_fatigue.detach(),
            gait_clock,
        ), dim=-1)
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec
        self.obs_buf = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1)
            > self.cfg.domain_rand.loss_rate,
            obs_buf,
            self.obs_buf,
        )

    def _reward_gait_contact(self):
        actual = self.contact_forces[:, self.feet_indices, 2] > 1.0
        desired = self._desired_contacts()
        match = (actual == desired).float().mean(dim=1)
        moving = torch.norm(self.commands[:, :2], dim=1) > 0.1
        return match * moving * self.general_scale

    def _reward_swing_clearance(self):
        desired_swing = ~self._desired_contacts()
        foot_height = self.rigid_body_states[:, self.feet_indices, 2]
        error = (foot_height - self.cfg.gait.swing_height) / self.cfg.gait.clearance_sigma
        score = torch.exp(-torch.square(error))
        moving = (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        return (score * desired_swing.float()).sum(dim=1) / 2.0 * moving * self.general_scale

    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        foot_xy_speed_sq = torch.sum(
            torch.square(self.rigid_body_states[:, self.feet_indices, 7:9]), dim=-1
        )
        return torch.sum(foot_xy_speed_sq * contact.float(), dim=1)
