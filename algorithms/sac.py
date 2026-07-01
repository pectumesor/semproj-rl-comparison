from typing import Optional, Tuple
import copy
import torch
import torch.nn as nn
import numpy as np
import wandb
from abc import ABC, abstractmethod
from dataclasses import dataclass
from omegaconf import DictConfig
from .buffers.replay_buffer import ReplayBatch, ReplayBuffer
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm

from models.agents import BaseAgent, RecurrentAgent
from models.heads import DoubleQNet


@dataclass
class SACUpdateStats:
    actor_loss: float | list[float]
    critic_loss: float | list[float]
    alpha_loss: float | list[float]
    alpha: float | list[float]
    train_rew: float | list[float]

    @staticmethod
    def init_lists():
        return SACUpdateStats(
            actor_loss=[],
            critic_loss=[],
            alpha_loss=[],
            alpha=[],
            train_rew=[]
        )

    def append(self, other) -> None:
        self.actor_loss.append(other.actor_loss)
        self.critic_loss.append(other.critic_loss)
        self.alpha_loss.append(other.alpha_loss)
        self.alpha.append(other.alpha)
        self.train_rew.append(other.train_rew)

    def mean(self):
        return SACUpdateStats(
            actor_loss=float(np.mean(self.actor_loss)),
            critic_loss=float(np.mean(self.critic_loss)),
            alpha_loss=float(np.mean(self.alpha_loss)),
            alpha=float(np.mean(self.alpha)),
            train_rew=float(np.mean(self.train_rew)),
        )


class SAC(ABC):

    # -- System --
    buffer = None
    device = "cpu"
    env = None
    eval_env = None

    # -- Architecture --
    agent = None
    target_agent = None

    # -- Algorithm Configuration --
    cfg = None

    @abstractmethod
    def sample_mini_batch(self, batch_size: int) -> ReplayBatch:
        pass

    @abstractmethod
    def train(self):
        pass

class MLPSAC(SAC):
    def __init__(self, buffer: ReplayBuffer, device: torch.device,
                env: gym.Env, eval_env: gym.Env, agent: BaseAgent | RecurrentAgent,
                cfg:DictConfig):
        
        super().__init__()
        
        # -- System --
        self.buffer = buffer
        self.device = device
        self.env = env
        self.eval_env = eval_env

        # -- Training Iterations --
        self.n_iterations = cfg.algorithm.n_iterations
        self.mini_batch_size = cfg.algorithm.mini_batch_size
        self.n_gradient_updates = cfg.algorithm.n_gradient_update
        self.warm_start_steps = cfg.algorithm.warm_start_steps
        self.train_freq = cfg.algorithm.train_freq
        self.eval_freq = cfg.algorithm.eval_freq
        self.save_interval = cfg.algorithm.save_interval


        # -- Architecture --
        self.agent = agent
        self.target_agent = copy.deepcopy(self.agent)
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(cfg.algorithm.init_alpha, dtype=torch.float32, device=device)))

        # -- Constants
        self.gamma = cfg.env.gamma
        self.tau = cfg.algorithm.tau
        self.target_entropy = cfg.algorithm.target_entropy

        # Optimizer initialization
        encoder_params = list(self.agent.obs_embed_model.parameters()) + list(self.agent.backbone_model.parameters())
        self.actor_optimizer = optim.Adam(params=encoder_params + list(self.agent.actor.parameters()), lr=cfg.algorithm.actor_lr)
        self.critic_optimizer = optim.Adam(params=encoder_params + list(self.agent.critic.parameters()), lr=cfg.algorithm.critic_lr)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=cfg.algorithm.alpha_lr)

    
    def sample_mini_batch(self, batch_size: int) -> ReplayBuffer:
        return self.buffer.get(batch_size)
    
    def alpha(self):
        return self.log_alpha.exp()
    
    def compute_actor_loss(self, obs_batch, actions, log_action):
        q_online, q_target = self.agent.get_state_action_value(obs_batch, actions)
        actor_loss = torch.mean(self.alpha().detach() * log_action
                                 - torch.min(q_online, q_target).squeeze(-1))
        return actor_loss

    def compute_alpha_loss(self, log_action):

        alpha_loss = torch.mean(-self.log_alpha * (log_action + self.target_entropy).detach())

        return alpha_loss

    def compute_critic_loss(self, obs_batch, next_obs_batch, rew_batch, act_batch, done_batch):

        with torch.no_grad():
            next_act, next_log_a = self.agent.sample_action(next_obs_batch)
            q1_target, q2_target = self.target_agent.get_state_action_value(next_obs_batch, next_act)
            q_val_next = torch.min(q1_target, q2_target).squeeze(-1) - self.alpha() * next_log_a
            q_target_next = rew_batch + self.gamma * (1 - done_batch.float()) * q_val_next

        # Here we track the gradients for the doubleQNets
        q_online, q_target = self.agent.get_state_action_value(obs_batch, act_batch)
        loss = (F.mse_loss(q_online.squeeze(-1), q_target_next) + F.mse_loss(q_target.squeeze(-1), q_target_next)) * 0.5

        return loss

    def soft_update_targets(self):

        with torch.no_grad():
            target_params = (list(self.target_agent.obs_embed_model.parameters()) +
                             list(self.target_agent.backbone_model.parameters()) +
                             list(self.target_agent.critic.parameters()))
            live_params   = (list(self.agent.obs_embed_model.parameters()) +
                             list(self.agent.backbone_model.parameters()) +
                             list(self.agent.critic.parameters()))
            for target_param, param in zip(target_params, live_params):
                target_param.data.copy_((1 - self.tau) * target_param + self.tau * param)

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

        # Update alpha

        action, log_action = self.agent.sample_action(obs_batch)

        alpha_loss = self.compute_alpha_loss(log_action)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update actor
        actor_loss = self.compute_actor_loss(obs_batch, action, log_action)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # Update target network weights
        self.soft_update_targets()

        return SACUpdateStats(critic_loss=critic_loss.item(),
                              actor_loss=actor_loss.item(),
                              alpha_loss=alpha_loss.item(),
                              alpha=self.alpha().item(),
                              train_rew=rew_batch.mean().item())
    
    def evaluate_policy(self,num_episodes=5):
        returns = []
        lengths = []

        self.agent.eval()
        with torch.inference_mode():
            for _ in tqdm(range(num_episodes), desc="Evaluating"):
                obs, _ = self.eval_env.reset()
                done = False
                episode_return = 0.0
                episode_length = 0

                while not done:
                    action = self.agent.predict_action(obs)
                    next_obs, reward, terminated, truncated, info = self.eval_env.step(action)

                    obs = next_obs
                    episode_return += reward
                    episode_length += 1
                    done = terminated or truncated

                returns.append(float(episode_return))
                lengths.append(int(episode_length))

        self.agent.train()
        return float(np.mean(returns)), float(np.mean(lengths))
    
    def train(self, run_dir = None):

        self.agent.train()
        obs, _ = self.env.reset()
        steps = 0
        eval_step = 0

        for iter in tqdm(range(self.n_iterations), desc="Training"):

            with torch.no_grad():
                steps += 1
                if steps < self.warm_start_steps:
                    # Uniform initilaized action of shape (num_envs, act_dim)
                    actions = torch.empty((self.buffer.num_envs, self.buffer.act_dim),
                                          dtype=torch.float32, device=self.buffer.device).uniform_(-1.0, 1.0)
                else:
                    actions, _ = self.agent.sample_action(obs)
                
                next_obs, reward, terminated, truncated, info = self.env.step(actions)
                done = terminated | truncated

                self.buffer.store(obs, actions, reward, next_obs, done)
                
                obs = next_obs

                new_obs, _ = self.env.reset(done=done)
                obs["rays"][done]    = new_obs["rays"][done]
                obs["proprio"][done] = new_obs["proprio"][done]
            
            if steps >= self.warm_start_steps and steps % self.train_freq == 0:
                mean_stats = SACUpdateStats.init_lists()

                for _ in range(self.n_gradient_updates):

                    mini_batch = self.buffer.get(self.mini_batch_size)
                    stats = self.update(mini_batch)
                    mean_stats.append(stats)
                
                mean_stats = mean_stats.mean()
              
            
            if steps >= self.warm_start_steps and steps % self.eval_freq == 0:
                eval_step += 1
                mean_returns, mean_episode_lengths = self.evaluate_policy()

                print(
                f"[SAC] eval_step={eval_step} "
                f"iteration={iter}/{self.n_iterations} "
                f"step={steps} "
                f"critic_loss={mean_stats.critic_loss:.4f} "
                f"actor_loss={mean_stats.actor_loss:.4f} "
                f"alpha_loss={mean_stats.alpha_loss:.4f} "
                f"alpha={mean_stats.alpha:.4f} "
                f"mean_train_ret={mean_stats.train_rew:.4f} "
                f"eval_return={mean_returns:.4f} "
                f"eval_length={mean_episode_lengths:.2f}"
            )
                
            if wandb.run is not None:
                    wandb.log({
                        "Critic Loss": {mean_stats.critic_loss},
                        "Actor Loss": {mean_stats.actor_loss},
                        "Alpha Loss": {mean_stats.alpha_loss},
                        "Alpha": {mean_stats.alpha},
                        "Mean Train Return": {mean_stats.train_rew},
                        "Mean Eval Return": {mean_returns},
                        "Mean Eval Length": {mean_episode_lengths}
                    })

            if run_dir is not None:
                if iter % self.save_interval == 0 or iter == self.n_iterations:
                    model_path = run_dir / f"iter_{iter}.pt"
                    optimizers = {
                        "actor": self.actor_optimizer,
                        "critic": self.critic_optimizer,
                        "alpha": self.alpha_optimizer
                    }
                    self.agent.save_model(model_path, optimizers)


        

    

