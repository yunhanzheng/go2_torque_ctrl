"""Soft Actor-Critic for the legacy tensor-only RSL-RL interface."""

from typing import Dict

import torch
import torch.nn.functional as F

from rsl_rl.modules.sac_actor_critic import SACActor, DoubleQCritic
from rsl_rl.storage.sac_replay_buffer import ReplayBuffer


class SAC:
    def __init__(self, obs_dim: int, action_dim: int, num_envs: int, device="cpu", **cfg):
        self.device = device
        hidden = cfg.get("hidden_dims", [512, 256, 128])
        activation = cfg.get("activation", "silu")
        normalize_obs = cfg.get("normalize_obs", True)
        self.actor = SACActor(
            obs_dim, action_dim, hidden_dims=hidden, activation=activation,
            init_noise_std=cfg.get("init_noise_std", 0.15),
            action_limit=cfg.get("action_limit", 4.7), normalize_obs=normalize_obs,
        ).to(device)
        self.critic = DoubleQCritic(
            obs_dim, action_dim, hidden_dims=hidden, activation=activation,
            normalize_obs=normalize_obs,
        ).to(device)
        self.gamma = cfg.get("gamma", 0.97)
        self.tau = cfg.get("tau", 0.003)
        self.n_steps = cfg.get("n_steps", 5)
        self.batch_size = cfg.get("mini_batch_size", 8192)
        self.num_updates = cfg.get("num_mini_batches", 200) * cfg.get("num_learning_epochs", 1)
        self.max_grad_norm = cfg.get("max_grad_norm", 1.0)
        self.policy_frequency = cfg.get("policy_frequency", 1)
        self.update_step = 0
        self.target_entropy = -cfg.get("target_entropy_scale", 0.167) * action_dim

        self.replay = ReplayBuffer(
            num_envs, obs_dim, action_dim, cfg.get("replay_buffer_size", 5_000_000),
            self.n_steps, self.gamma, device,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.get("actor_learning_rate", 2e-4)
        )
        critic_params = list(self.critic.q1.parameters()) + list(self.critic.q2.parameters())
        self.critic_optimizer = torch.optim.Adam(
            critic_params, lr=cfg.get("critic_learning_rate", 2e-4)
        )
        initial_alpha = cfg.get("alpha", 0.001)
        self.log_alpha = torch.tensor(initial_alpha, device=device).log().requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=cfg.get("alpha_learning_rate", 2e-5)
        )
        self.auto_alpha = cfg.get("auto_alpha", True)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs, deterministic=False)

    @torch.no_grad()
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs, deterministic=True)

    @torch.no_grad()
    def store(self, obs, actions, rewards, next_obs, dones, timeouts, terminal_obs) -> None:
        timeout_mask = timeouts.bool().unsqueeze(-1)
        corrected_next_obs = torch.where(timeout_mask, terminal_obs, next_obs)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        self.replay.add(obs, actions, rewards, corrected_next_obs, dones, timeouts)

    def update(self) -> Dict[str, float]:
        if not self.replay.ready(self.batch_size):
            return {}
        totals = {"critic_loss": 0.0, "actor_loss": 0.0, "alpha_loss": 0.0}
        actor_updates = 0
        for _ in range(self.num_updates):
            batch = self.replay.sample(self.batch_size)
            with torch.no_grad():
                next_actions, next_logp = self.actor.sample(batch["next_obs"])
                tq1, tq2 = self.critic.target(batch["next_obs"], next_actions)
                next_value = torch.min(tq1, tq2) - self.alpha.detach() * next_logp
                bootstrap_mask = batch["timeouts"] + 1.0 - batch["dones"]
                discount = self.gamma ** batch["steps"].float()
                target_q = batch["rewards"] + discount * bootstrap_mask * next_value

            q1, q2 = self.critic(batch["obs"], batch["actions"])
            critic_loss = 0.5 * (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_params = list(self.critic.q1.parameters()) + list(self.critic.q2.parameters())
            torch.nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm)
            self.critic_optimizer.step()

            actor_loss = torch.zeros((), device=self.device)
            alpha_loss = torch.zeros((), device=self.device)
            if self.update_step % self.policy_frequency == 0:
                new_actions, logp = self.actor.sample(batch["obs"])
                aq1, aq2 = self.critic(batch["obs"], new_actions)
                actor_loss = (self.alpha.detach() * logp - torch.min(aq1, aq2)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                actor_updates += 1

                if self.auto_alpha:
                    alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()

            self.critic.soft_update(self.tau)
            self.update_step += 1
            totals["critic_loss"] += critic_loss.item()
            totals["actor_loss"] += actor_loss.item()
            totals["alpha_loss"] += alpha_loss.item()

        totals["critic_loss"] /= self.num_updates
        denom = max(actor_updates, 1)
        totals["actor_loss"] /= denom
        totals["alpha_loss"] /= denom
        totals["alpha"] = self.alpha.item()
        return totals

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_step": self.update_step,
        }

    def load_state_dict(self, state, load_optimizer=True):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.log_alpha.data.copy_(state["log_alpha"])
        self.update_step = state.get("update_step", 0)
        if load_optimizer:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
