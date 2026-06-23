from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .buffers.replay_buffer import ReplayBatch, ReplayBuffer
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym

from ..models.agents import BaseAgent, RecurrentAgent
from ..models.heads import DoubleQNet


@dataclass
class SACUpdateStats:
    actor_loss: float | list[float]
    critic_loss: float | list[float]
    alpha_loss: float | list[float]
    alpha: float | list[float]

    @staticmethod
    def init_lists():
        return SACUpdateStats(
            actor_loss=[],
            critic_loss=[],
            alpha_loss=[],
            alpha=[],
        )

    def append(self, other) -> None:
        self.actor_loss.append(other.actor_loss)
        self.critic_loss.append(other.critic_loss)
        self.alpha_loss.append(other.alpha_loss)
        self.alpha.append(other.alpha)

    def mean(self):
        return SACUpdateStats(
            actor_loss=float(np.mean(self.actor_loss)),
            critic_loss=float(np.mean(self.critic_loss)),
            alpha_loss=float(np.mean(self.alpha_loss)),
            alpha=float(np.mean(self.alpha)),
        )


class SAC(ABC):

    # -- System --
    buffer = None
    device = "cpu"
    env = None

    # -- Training Iterations --
    n_iterations = None
    mini_batch_size = None
    n_epochs = None

    # -- Architecture --
    agent = None
    critic_target : DoubleQNet = None

    # -- Constants
    gamma = None
    tau = None
    init_alpha = None
    target_entropy = None
    actor_lr = None
    critic_lr = None
    alpha_lr = None
    train_freq = None


    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def sample_mini_batch(self, batch_size: int) -> ReplayBatch:
        pass

    @abstractmethod
    def train(self):
        pass

class MLPSAC(SAC):
    def __init__(self, buffer: ReplayBuffer, device: torch.device, env: gym.Env, eval_env: gym.Env,
                n_iterations: int, mini_batch_size: int, n_gradient_updates: int, agent: BaseAgent | RecurrentAgent,
                gamma: float, tau: float, init_alpha: float, target_entropy, warm_start_steps: int, 
                train_freq: int, eval_freq: int, actor_lr: float, critic_lr: float, alpha_lr: float):
        
        super().__init__()
        
        # -- System --
        self.buffer = buffer
        self.device = device
        self.env = env
        self.eval_env = eval_env

        # -- Training Iterations --
        self.n_iterations = n_iterations
        self.mini_batch_size = mini_batch_size
        self.n_gradient_updates = n_gradient_updates
        self.warm_start_steps = warm_start_steps
        self.train_freq = train_freq
        self.eval_freq = eval_freq


        # -- Architecture --
        self.agent = agent
        self.critic_target = self.agent.critic.copy()
        self.log_apha = nn.Parameter(torch.log(init_alpha), dtype=torch.float32, device=device)

        # -- Constants
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy

        # Optimizer initialization
        self.actor_optimizer = optim.Adam(params=self.agent.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(params=self.agent.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = optim.Adam(params=self.log_apha.parameters(), lr=alpha_lr)

    
    def sample_mini_batch(self, batch_size: int) -> ReplayBuffer:
        return self.buffer.get(batch_size)
    
    def alpha(self):
        return self.log_apha.exp()
    
    def compute_actor_loss(self, obs_batch):
        
        actions, log_a = self.agent.select_action(obs_batch)
        value = 0.0

        q_online, q_target = self.agent.get_state_action_value(obs_batch, actions)
        value = torch.min(q_online, q_target)

        loss = torch.mean(self.alpha() * log_a - value)

        return loss

    def compute_critic_loss(self, obs_batch, next_obs_batch, rew_batch, act_batch, done_batch):

        with torch.no_grad(): # Dont want to track gradient during target value computation
            next_act, next_log_a = self.agent.select_action(next_obs_batch)
            q1_target, q2_target = self.critic_target(next_obs_batch, next_act)
            q_val_next = torch.min(q1_target, q2_target) - self.alpha() * next_log_a
            q_target_next = rew_batch + self.gamma * (1 - done_batch) * q_val_next

        # Here we track the gradients for the doubleQNets
        q_online, q_target = self.agent.get_state_action_value(obs_batch, act_batch)
        loss = (F.mse_loss(q_online, q_target_next) + F.mse_loss(q_target, q_target_next)) * 0.5

        return loss

    def compute_alpha_loss(self, obs_batch):
        
        _, log_a = self.agent.select_action(obs_batch)

        loss = torch.mean(- self.log_apha  * (log_a + self.target_entropy).detach())

        return loss

    def soft_update_targets(self):

        with torch.no_grad():
            for target_param, param in zip(
                self.critic_target.parameters(), self.agent.critic.parameters()
            ):
                target_param.data.copy_( (1 - self.tau) * target_param + self.tau * param)

    def update(self, replay_batch: ReplayBatch):

        obs_batch = replay_batch.obs
        act_batch = replay_batch.act
        rew_batch = replay_batch.rew
        done_batch = replay_batch.done
        next_obs_batch = replay_batch.next_obs


        # Update critic
        critic_loss = self.compute_critic_loss(obs_batch, next_obs_batch, rew_batch,
                                               act_batch, done_batch)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = self.compute_actor_loss(obs_batch)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss = self.compute_alpha_loss(obs_batch)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update target network weights
        self.soft_update_targets()

        return SACUpdateStats(critic_loss=critic_loss.item(),
                              actor_loss=actor_loss.item(),
                              alpha_loss=alpha_loss.item(),
                              aux_loss=0.0,
                              alpha=self.alpha().item())
    
    
    def evaluate_policy(self,num_episodes=5):
        returns = []
        lengths = []

        self.agent.eval_mode()
        with torch.inference_mode():
            for _ in range(num_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                episode_return = 0.0
                episode_length = 0

                while not done:
                    obs = torch.as_tensor(obs, dtype=torch.float, device=self.buffer.device).unsqueeze(0)
                    action = self.agent.predict_action(obs)
                    next_obs, reward, terminated, truncated, info = self.eval_env.step(action.cpu().numpy().squeeze(0))

                    obs = next_obs
                    episode_return += reward
                    episode_length += 1
                    done = terminated or truncated

                returns.append(float(episode_return))
                lengths.append(int(episode_length))

        return float(np.mean(returns)), float(np.mean(lengths))

    
    def train(self):

        self.agent.train()
        obs, _ = self.env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float, device=self.device)
        steps = 0

        for iter in range(self.n_iterations):
            with torch.no_grad():
                steps += 1
                if steps < self.warm_start_steps:
                    # Uniform initilaized action of shape (1, num_envs, act_dim)
                    actions = torch.empty((self.buffer.num_envs, self.buffer.act_dim),
                                          dtype=torch.float32, device=self.buffer.device).uniform_(-1.0, 1.0).unsqueeze(0)
                else:
                    actions = self.agent.sample_action(obs)
                
                next_obs, reward, terminated, truncated, info = self.env.step(actions)
                next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.buffer.device)
                done = terminated or truncated

                self.buffer.store(obs, actions, reward, next_obs, done)
                
                obs = next_obs

                if done:
                    obs, _ = self.env.reset()
                    obs = torch.as_tensor(obs, dtype=torch.float32, device=self.buffer.device)
            
            if steps >= self.warm_start_steps and steps % self.train_freq == 0:
                mean_stats = SACUpdateStats.init_lists()

                for _ in range(self.n_gradient_updates):

                    mini_batch = self.buffer.get(self.mini_batch_size)
                    stats = self.update(mini_batch)
                    mean_stats.append(stats)
                
                mean_stats = mean_stats.mean()
            
            if steps % self.eval_freq == 0:
                mean_returns, mean_episode_lengths = self.evaluate_policy()
            # TODO: Logging stats and evaluation on wandb


        

    

