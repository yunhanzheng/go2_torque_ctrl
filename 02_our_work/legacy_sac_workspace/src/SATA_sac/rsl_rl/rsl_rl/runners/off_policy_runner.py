"""Legacy RSL-RL runner for tensor-based SAC training."""

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms.sac import SAC


class OffPolicyRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.device = device
        self.cfg = train_cfg["runner"]
        self.alg = SAC(
            env.num_obs, env.num_actions, env.num_envs, device=device,
            **train_cfg["algorithm"]
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.log_dir = log_dir
        self.writer = None
        self.current_learning_iteration = 0
        self.total_steps = 0
        self.total_time = 0.0
        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        reward_window = deque(maxlen=100)
        length_window = deque(maxlen=100)
        running_reward = torch.zeros(self.env.num_envs, device=self.device)
        running_length = torch.zeros(self.env.num_envs, device=self.device)

        end_iteration = self.current_learning_iteration + num_learning_iterations
        for iteration in range(self.current_learning_iteration, end_iteration):
            start = time.time()
            episode_infos = []
            for _ in range(self.num_steps_per_env):
                with torch.no_grad():
                    actions = self.alg.act(obs)
                    next_obs, _, rewards, dones, extras = self.env.step(actions)
                next_obs = next_obs.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                timeouts = extras.get("time_outs", torch.zeros_like(dones)).to(self.device)
                terminal_obs = extras["terminal_observation"].to(self.device)
                self.alg.store(obs, actions, rewards, next_obs, dones, timeouts, terminal_obs)

                running_reward += rewards
                running_length += 1
                done_ids = dones.nonzero(as_tuple=False).flatten()
                if len(done_ids) > 0:
                    reward_window.extend(running_reward[done_ids].cpu().tolist())
                    length_window.extend(running_length[done_ids].cpu().tolist())
                    running_reward[done_ids] = 0
                    running_length[done_ids] = 0
                if "episode" in extras and extras["episode"]:
                    episode_infos.append(extras["episode"])
                obs = next_obs

            collect_time = time.time() - start
            learn_start = time.time()
            losses = self.alg.update()
            learn_time = time.time() - learn_start
            self.current_learning_iteration = iteration + 1
            self.total_steps += self.num_steps_per_env * self.env.num_envs
            self.total_time += collect_time + learn_time

            if self.writer is not None:
                self._log(iteration, losses, reward_window, length_window, episode_infos,
                          collect_time, learn_time, actions)
                if iteration % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, "model_{}.pt".format(iteration)))

        if self.writer is not None:
            self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))
            self.writer.close()

    def _log(self, iteration, losses, rewards, lengths, episode_infos, collect_time, learn_time, actions):
        for key, value in losses.items():
            self.writer.add_scalar("Loss/{}".format(key), value, iteration)
        if rewards:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(rewards), iteration)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(lengths), iteration)
        for info in episode_infos:
            for key, value in info.items():
                if torch.is_tensor(value):
                    value = value.float().mean().item()
                self.writer.add_scalar("Episode/{}".format(key), value, iteration)
        self.writer.add_scalar("Perf/collection_time", collect_time, iteration)
        self.writer.add_scalar("Perf/learning_time", learn_time, iteration)
        self.writer.add_scalar("Perf/environment_steps", self.total_steps, iteration)
        self.writer.add_scalar("State/mean_base_height", self.env.root_states[:, 2].mean().item(), iteration)
        self.writer.add_scalar("State/mean_abs_forward_speed", self.env.base_lin_vel[:, 0].abs().mean().item(), iteration)
        self.writer.add_scalar("Policy/mean_abs_action", actions.abs().mean().item(), iteration)
        mean_reward = statistics.mean(rewards) if rewards else float("nan")
        print(
            "SAC iteration {}/{} | steps {} | reward {:.3f} | alpha {:.6f} | collect {:.2f}s | learn {:.2f}s".format(
                iteration, self.current_learning_iteration, self.total_steps, mean_reward,
                self.alg.alpha.item(), collect_time, learn_time
            )
        )

    def save(self, path, infos=None):
        torch.save({"sac": self.alg.state_dict(), "iter": self.current_learning_iteration, "infos": infos}, path)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.alg.load_state_dict(checkpoint["sac"], load_optimizer=load_optimizer)
        self.current_learning_iteration = checkpoint["iter"]
        return checkpoint.get("infos")

    def get_inference_policy(self, device=None):
        if device is not None:
            self.alg.actor.to(device)
        self.alg.actor.eval()
        return self.alg.act_inference
