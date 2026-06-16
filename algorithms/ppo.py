from typing import Sequence
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .buffers.rollout_buffer import RolloutBatch, RolloutBuffer
import torch.optim as optim
import gymnasium as gym



@dataclass
class PPOUpdateStats:
    mean_kl: float
    mean_surrogate_loss: float
    mean_value_loss: float
    mean_entropy: float
    mean_aux_loss: float = 0

class PPO(ABC):

    # -- System --
    buffer = None
    device = "cpu"
    env = None
    optimizer = None

    # -- Training Iterations --
    n_iterations = None
    mini_batch = None
    n_epochs = None

    # -- Architecture --
    actor = None
    critic = None
    auxiliary = None

    # -- Constants    
    gamma = None
    gae_lambda = None
    clip_esilon = None
    entropy_coeff = None
    val_coeff = None
    aux_coeff = None
    task_coeff = None
    intr_coeff = None
   

    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def collect_rollout(self):
        pass
    
    @abstractmethod
    def sample_mini_batch(self) -> RolloutBatch:
        pass

    @abstractmethod
    def train(self):
        pass
        
class MLPPPO(PPO):

    def __init__(self,
                buffer: RolloutBuffer, device: torch.device, env: gym.Env, optimizer: optim.Optimizer,
                n_iterations: int,  mini_batch: int,  n_epochs: int,
                model: nn.Module, gamma: float, gae_lambda: float, clip_epsilon: float, entropy_coeff: float,
                val_coeff: float, aux_coeff: float, task_coeff: float, intr_coeff: float
                ):
        
        # -- System --
        self.buffer = buffer
        self.device = device
        self.env = env
        self.eval_env = env.copy()
        self.optimizer = optimizer

        # -- Training Iterations --
        self.n_iterations = n_iterations
        self.mini_batch = mini_batch
        self.n_epochs = n_epochs

        # -- Architecture --
        self.model = model
        self.actor = self.model.actor
        self.critic = self.model.critic

        # -- Constants    
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_esilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.val_coeff = val_coeff
        self.aux_coeff = aux_coeff
        self.task_coeff = task_coeff
        self.intr_coeff = intr_coeff

    
    def select_action(self, obs: torch.Tensor):

        h = self.model(obs)

        with torch.no_grad():
            action = self.actor.act(h)
            action_log_prob = self.actor.log_prob_action(action)
            action_mu = self.ctor.action_mean
            action_std = self.actor.action_std
            value = self.critic(h)

        return action, action_log_prob, action_mu, action_std, value
    

    def collect_rollout(self, obs: torch.Tensor, done: bool):

        for _ in range(self.buffer.num_steps):
            if done:
                obs = _, self.env.reset()
                obs = torch.as_tensor(obs, dtype=torch.float, device=self.device)
            
            action, action_log_prob, action_mu, action_std, value = self.select_action(obs)
            next_obs, reward, terminated, timeout, info = self.env.step(action.cpu().numpy())

            next_obs = torch.as_tensor(obs, dtype=torch.float, device=self.device)
            done = terminated or timeout

            if timeout:
                reward += self.gamma * self.critic(self.model(next_obs)).item()

            self.buffer.store(
                obs=next_obs,
                act=action,
                logp=action_log_prob,
                mu=action_mu,
                std=action_std,
                val=value,
                done=terminated,
                rew=reward
            )

            obs = next_obs

        last_val = self.critic(self.model(obs)).item()
        self.buffer.compute_returns(last_val)

        return obs, done
    
    def sample_mini_batch(self, batch: RolloutBatch):

        for _ in range(self.n_epochs):

            indices = torch.randperm(
                n = self.buffer.num_steps,
                requires_grad=False,
                device=self.device
            )

            for start in range(0, self.buffer.num_steps, self.mini_batch):
                end = start + self.mini_batch
                batch_indices = indices[start:end]
                yield RolloutBatch(
                    obs=batch.obs[batch_indices],
                    act=batch.act[batch_indices],
                    logp=batch.logp[batch_indices],
                    mu=batch.mu[batch_indices],
                    std=batch.std[batch_indices],
                    val=batch.val[batch_indices],
                    ret=batch.ret[batch_indices],
                    adv=batch.adv[batch_indices]
                )
    
    def compute_surrogate_loss(self, logp_batch, old_logp_batch, adv_batch):
        ratio = torch.exp(logp_batch - old_logp_batch)
        cliped_ratio = torch.clamp(ratio, 1 - self.clip_esilon, 1 + self.clip_esilon)
        surrogate_loss = -torch.min(ratio * adv_batch, cliped_ratio * adv_batch).mean()
        return surrogate_loss

    def compute_value_loss(self, val_batch, old_val_batch, ret_batch):
        value_loss_unclipped = (val_batch - ret_batch) ** 2
        value_clipped = old_val_batch + torch.clamp(val_batch - old_val_batch, -self.clip_esilon, self.clip_esilon)
        value_loss_clipped = (value_clipped - ret_batch) ** 2
        return torch.max(value_loss_unclipped, value_loss_clipped).mean()
        
    def compute_entropy_loss(self, entropy_batch):
        entropy_scalar = torch.sum(entropy_batch, dim=-1).mean().item()
        return -entropy_scalar

    def compute_aux_loss(self):
        # TODO: See what kind of inputs each auxiliary task needs to compute the loss
        # TODO: Implement reward computation inside the auxiliary head architecture, call it here
        return 0

    def compute_kl_mean(self, old_mu_batch, old_std_batch, mu_batch, std_batch):
        
        kl_per_dim = (
            torch.log(std_batch / old_std_batch) + (old_std_batch.pow(2) + (old_mu_batch - mu_batch).pow(2))
            / (2 * std_batch.pow(2)) - 0.5 )
        
        kl_per_sample = torch.sum(kl_per_dim, dim=-1)

        return kl_per_sample.mean()

    def update(self):
        
        rollout_batch = self.buffer.get()

        mean_kl = 0
        mean_entropy = 0
        mean_surrogate_loss = 0
        mean_val_loss = 0
        mean_aux_loss = 0
        num_updates = 0

        for mini_batch in self.sample_mini_batch(self, rollout_batch):

            obs_batch = mini_batch.obs
            act_batch = mini_batch.act
            old_logp_batch = mini_batch.logp
            old_mu_batch = mini_batch.mu
            old_std_batch = mini_batch.std
            old_val_batch = mini_batch.val
            ret_batch = mini_batch.ret
            adv_batch = mini_batch.adv

            h = self.model(obs_batch)
            self.actor.update_distribution(h)
            logp_batch = self.actor.log_prob_action(act_batch)
            mu_batch = self.actor.action_mean
            std_batch = self.actor.action_std
            entropy_batch = self.actor.entropy
            val_batch = self.critic(h)

            kl = self.compute_kl_mean(old_mu_batch, old_std_batch, mu_batch, std_batch)
            surrogate_loss = self.compute_surrogate_loss(logp_batch, old_logp_batch, adv_batch)
            value_loss = self.compute_value_loss(val_batch, old_val_batch, ret_batch)
            entropy_los = self.compute_entropy_loss(entropy_batch)
            intr_loss = self.compute_aux_loss()
            task_loss = surrogate_loss + self.val_coeff * value_loss + self.entropy_coeff * entropy_los
            loss = self.task_coeff * task_loss + self.intr_coeff * intr_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_kl += kl.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_val_loss += value_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_aux_loss += intr_loss.item()
            num_updates += 1
        
        mean_kl /= num_updates
        mean_surrogate_loss /= num_updates
        mean_val_loss /= num_updates
        mean_entropy /= num_updates
        mean_aux_loss /= num_updates

        return PPOUpdateStats(
            mean_kl=mean_kl,
            mean_surrogate_loss=mean_surrogate_loss,
            mean_value_loss=mean_val_loss,
            mean_entropy=mean_entropy,
            mean_aux_loss=mean_aux_loss
        )
    
    def predict_action(self, obs: torch.Tensor):

        h = self.model(obs)
        action = self.actor.act_inference(h)
        return action

    def evaluate_policy(self, num_episodes=5):
        returns = []
        lengths = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(num_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                episode_return = 0.0
                episode_length = 0

                while not done:
                    obs = torch.as_tensor(obs, dtype=torch.float, device=self.device).unsqueeze(0)
                    action = self.predict_action(obs)
                    next_obs, reward, terminated, truncated, info = self.eval_env.step(action.cpu().numpy())

                    obs = next_obs
                    episode_return += reward
                    episode_length += 1
                    done = terminated or truncated
            
                returns.append(float(episode_return))
                lengths.append(int(episode_length))

        return float(np.mean(returns)), float(np.mean(lengths))

    def train(self):

        self.model.train()
        obs, _ = self.env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float, device=self.device)
        done = False

        for iter in range(self.n_iterations):

            obs, done = self.collect_rollout(obs, done)
            stats = self.update(self)
            mean_eval_return, mean_eval_length = self.evaluate_policy()
            iteration = iter + 1

            # TODO: Logging stats and evaluation on wandb


class RecuurentPPO(MLPPPO):
    def __init__(self, num_layers: int, hidden_size: int, num_minibatches: int, **kwargs):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if self.buffer.num_envs % num_minibatches != 0:
            raise ValueError(
                f"For RecurrentPPO, it must hold that num_evs % num_minibatches == 0 "
                f"Current num_envs: {self.buffer.num_envs}, num_minibatches: {num_minibatches}"
            )
        else:
              self.mini_batch = self.buffer.num_envs // num_minibatches

    def select_action(self, obs, lstm_state, done):
        
        hidden, lstm_state = self.model(obs, lstm_state, done)

        with torch.no_grad():
            action = self.actor.act(hidden)
            action_log_prob = self.actor.log_prob_action(action)
            action_mu = self.ctor.action_mean
            action_std = self.actor.action_std
            value = self.critic(hidden)

        return action, action_log_prob, action_mu, action_std, value, lstm_state
    
    def collect_rollout(self, obs, lstm_state, done):

        for _ in range(self.buffer.num_steps):
            if done:
                obs = _, self.env.reset()
                obs = torch.as_tensor(obs, dtype=torch.float, device=self.device)
            
            action, action_log_prob, action_mu, action_std, value, lstm_state = self.select_action(obs, lstm_state, done)
            next_obs, reward, terminated, timeout, info = self.env.step(action.cpu().numpy())

            next_obs = torch.as_tensor(obs, dtype=torch.float, device=self.device)
            done = terminated or timeout

            if timeout:
                reward += self.gamma * self.critic(self.model(next_obs)).item()

            self.buffer.store(
                obs=next_obs,
                act=action,
                logp=action_log_prob,
                mu=action_mu,
                std=action_std,
                val=value,
                done=terminated,
                rew=reward
            )

            obs = next_obs

        last_val = self.critic(self.model(obs)).item()
        self.buffer.compute_returns(last_val)

        return obs, done, lstm_state
    
    def sample_mini_batch(self, batch):
        # TODO: Add lstm states to the batch

        buffer_ids = torch.arange(self.mini_batch).resize(self.buffer.num_steps, self.buffer.num_envs)

        for _ in range(self.n_epochs):

            env_ids = torch.randperm(
                n = self.buffer.num_envs,
                device=self.device
            )

            for start in range(0, self.buffer.num_envs, self.mini_batch):
                end = start + self.mini_batch
                mini_batch_envs_ids = env_ids[start:end]
                mini_batch_buffer_ids = buffer_ids[:, mini_batch_envs_ids]

                yield (RolloutBatch(
                    obs=batch.obs[mini_batch_buffer_ids],
                    act=batch.act[mini_batch_buffer_ids],
                    logp=batch.logp[mini_batch_buffer_ids],
                    mu=batch.mu[mini_batch_buffer_ids],
                    std=batch.std[mini_batch_buffer_ids],
                    val=batch.val[mini_batch_buffer_ids],
                    ret=batch.ret[mini_batch_buffer_ids],
                    adv=batch.adv[mini_batch_buffer_ids]
                )
                , mini_batch_envs_ids)
    
    
    def update(self, initial_lstm_state):

        rollout_batch = self.buffer.get()

        mean_kl = 0
        mean_entropy = 0
        mean_surrogate_loss = 0
        mean_val_loss = 0
        mean_aux_loss = 0
        num_updates = 0

        for mini_batch, mini_batch_env_ids in self.sample_mini_batch(self, rollout_batch):

            obs_batch = mini_batch.obs
            act_batch = mini_batch.act
            old_logp_batch = mini_batch.logp
            old_mu_batch = mini_batch.mu
            old_std_batch = mini_batch.std
            old_val_batch = mini_batch.val
            ret_batch = mini_batch.ret
            adv_batch = mini_batch.adv
            done_batch = mini_batch.done

            h, _ = self.model(obs_batch, 
                              (initial_lstm_state[0][:, mini_batch_env_ids], initial_lstm_state[1][:, mini_batch_env_ids]),
                              done_batch)
            
            self.actor.update_distribution(h)
            logp_batch = self.actor.log_prob_action(act_batch)
            mu_batch = self.actor.action_mean
            std_batch = self.actor.action_std
            entropy_batch = self.actor.entropy
            val_batch = self.critic(h)

            kl = self.compute_kl_mean(old_mu_batch, old_std_batch, mu_batch, std_batch)
            surrogate_loss = self.compute_surrogate_loss(logp_batch, old_logp_batch, adv_batch)
            value_loss = self.compute_value_loss(val_batch, old_val_batch, ret_batch)
            entropy_los = self.compute_entropy_loss(entropy_batch)
            intr_loss = self.compute_aux_loss()
            task_loss = surrogate_loss + self.val_coeff * value_loss + self.entropy_coeff * entropy_los
            loss = self.task_coeff * task_loss + self.intr_coeff * intr_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_kl += kl.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_val_loss += value_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_aux_loss += intr_loss.item()
            num_updates += 1
        
        mean_kl /= num_updates
        mean_surrogate_loss /= num_updates
        mean_val_loss /= num_updates
        mean_entropy /= num_updates
        mean_aux_loss /= num_updates

        return PPOUpdateStats(
            mean_kl=mean_kl,
            mean_surrogate_loss=mean_surrogate_loss,
            mean_value_loss=mean_val_loss,
            mean_entropy=mean_entropy,
            mean_aux_loss=mean_aux_loss)
    
    def predict_action(self, obs, lstm_state, done):

        h, lstm_state = self.model(obs, lstm_state, done)
        action = self.actor.act_inference(h)
        return action, lstm_state
    
    def evaluate_policy(self, num_episodes=5):
        returns = []
        lengths = []

        lstm_state = (
            torch.zeros((self.num_layers, self.buffer.num_envs, self.hidden_size), 
                        dtype=torch.float, device=self.device),
            torch.zeros((self.num_layers, self.buffer.num_envs, self.hidden_size),
                         dtype=torch.float, device=self.device)
        )

        self.model.eval()
        with torch.no_grad():
            for _ in range(num_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                episode_return = 0.0
                episode_length = 0

                while not done:
                    obs = torch.as_tensor(obs, dtype=torch.float, device=self.device).unsqueeze(0)
                    action, lstm_state = self.predict_action(obs, lstm_state, done)
                    next_obs, reward, terminated, truncated, info = self.eval_env.step(action.cpu().numpy())

                    obs = next_obs
                    episode_return += reward
                    episode_length += 1
                    done = terminated or truncated
            
                returns.append(float(episode_return))
                lengths.append(int(episode_length))

        return float(np.mean(returns)), float(np.mean(lengths))

    def train(self):

        self.model.train()
        obs, _ = self.env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float, device=self.device)
        done = False

        initial_lstm_state = (
            torch.zeros((self.num_layers, self.buffer.num_envs, self.hidden_size), 
                        dtype=torch.float, device=self.device),
            torch.zeros((self.num_layers, self.buffer.num_envs, self.hidden_size),
                         dtype=torch.float, device=self.device)
        )

        for iter in range(self.n_iterations):

            obs, _, done = self.collect_rollout(obs, initial_lstm_state, done)
            stats = self.update(initial_lstm_state)
            mean_eval_return, mean_eval_length = self.evaluate_policy()
            iteration = iter + 1

            # TODO: Logging stats and evaluation on wandb





        





            






