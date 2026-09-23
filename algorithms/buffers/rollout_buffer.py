from dataclasses import dataclass
import torch
from omegaconf import DictConfig

@dataclass
class RolloutBatch:
    rays:    torch.Tensor
    proprio: torch.Tensor
    act:     torch.Tensor
    logp:    torch.Tensor
    mu:      torch.Tensor
    std:     torch.Tensor
    val:     torch.Tensor
    ret:     torch.Tensor
    adv:     torch.Tensor
    done:    torch.Tensor

@dataclass
class RecurrentRolloutBatch:
    rollout:      RolloutBatch
    ep_start:     torch.Tensor
    hidden_state: torch.Tensor
    cell_state:   torch.Tensor

class RolloutBuffer:
    def __init__(
            self,
            ray_dim: tuple,
            proprio_dim: int,
            device: torch.device,
            cfg: DictConfig
            ):
        
        self.act_dim    = cfg.env.act_dim
        self.gamma      = cfg.env.gamma
        self.gae_lambda = cfg.algorithm.gae_lambda
        self.num_steps  = cfg.algorithm.num_steps
        self.num_envs   = cfg.env.num_envs
        self.device     = device
        self.ptr        = 0

        self.rays_buf     = torch.zeros((self.num_steps, self.num_envs, *ray_dim),        dtype=torch.float, device=device)
        self.proprio_buf  = torch.zeros((self.num_steps, self.num_envs, proprio_dim),     dtype=torch.float, device=device)
        self.act_buf      = torch.zeros((self.num_steps, self.num_envs, self.act_dim),    dtype=torch.float, device=device)
        self.logp_buf     = torch.zeros((self.num_steps, self.num_envs),                  dtype=torch.float, device=device)
        self.mu_buf       = torch.zeros((self.num_steps, self.num_envs, self.act_dim),    dtype=torch.float, device=device)
        self.std_buf      = torch.zeros((self.num_steps, self.num_envs, self.act_dim),    dtype=torch.float, device=device)
        self.val_buf      = torch.zeros((self.num_steps, self.num_envs),                  dtype=torch.float, device=device)
        self.done_buf     = torch.zeros((self.num_steps, self.num_envs),                  dtype=torch.bool,  device=device)
        self.rew_buf      = torch.zeros((self.num_steps, self.num_envs),                  dtype=torch.float, device=device)
        self.ret_buf      = torch.zeros((self.num_steps, self.num_envs),                  dtype=torch.float, device=device)
        self.adv_buf      = torch.zeros((self.num_steps, self.num_envs),                  dtype=torch.float, device=device)
        self.ep_start_buf = torch.zeros((self.num_steps, self.num_envs),                  dtype=torch.bool,  device=device)


    def store(self, obs: dict, act, logp, mu, std, val, rew, done):
        if self.ptr >= self.num_steps:
            raise ValueError("RolloutBuffer is full. Call get() first")

        self.rays_buf[self.ptr]    = obs["rays"]
        self.proprio_buf[self.ptr] = obs["proprio"]
        self.act_buf[self.ptr]     = act
        self.logp_buf[self.ptr]    = logp
        self.mu_buf[self.ptr]      = mu
        self.std_buf[self.ptr]     = std
        self.val_buf[self.ptr]     = val
        self.rew_buf[self.ptr]     = rew
        self.done_buf[self.ptr]    = done
        self.ptr += 1

    def compute_returns(self, last_val):
        advantage = 0
        for step in reversed(range(self.num_steps)):
            next_val = last_val if step == self.num_steps - 1 else self.val_buf[step + 1]
            not_terminal = 1.0 - self.done_buf[step].float()
            delta      = self.rew_buf[step] + self.gamma * next_val * not_terminal - self.val_buf[step]
            advantage  = delta + self.gamma * self.gae_lambda * advantage * not_terminal
            self.adv_buf[step] = advantage
            self.ret_buf[step] = advantage + self.val_buf[step]


    def get(self) -> RolloutBatch:
        if self.ptr != self.num_steps:
            raise ValueError(
                f"RolloutBuffer must be full before calling get(). "
                f"Current size: {self.ptr}, expected: {self.num_steps}"
            )
        batch = RolloutBatch(
            rays=self.rays_buf,
            proprio=self.proprio_buf,
            act=self.act_buf,
            logp=self.logp_buf,
            mu=self.mu_buf,
            std=self.std_buf,
            val=self.val_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            done=self.done_buf,
        )
        self.ptr = 0
        return batch

class RecurrentRolloutBuffer(RolloutBuffer):

    def __init__(self, ray_dim, proprio_dim, device, cfg):
        super().__init__(ray_dim, proprio_dim, device, cfg)

        hidden_state_shape = (self.num_steps, cfg.backbone.lstm_num_layers, 
                              self.num_envs, cfg.backbone.lstm_backbone_feature_dim)

        self.hidden_states_buf  = torch.zeros(hidden_state_shape, dtype=torch.float, device=self.device)
        self.cell_states_buf    = torch.zeros(hidden_state_shape, dtype=torch.float, device=self.device)

        self.recurrent_ptr = 0

    def store(self, obs: dict, act, logp, mu, std, val, rew, done, ep_starts, hidden_states, cell_states):

        super().store(obs, act, logp, mu, std, val, rew, done)

        self.ep_start_buf[self.recurrent_ptr] = ep_starts
        self.hidden_states_buf[self.recurrent_ptr] = hidden_states
        self.cell_states_buf[self.recurrent_ptr] = cell_states

        self.recurrent_ptr += 1

    def get(self) -> RecurrentRolloutBatch:

        rollout = super().get()

        recurrent_batch = RecurrentRolloutBatch(
            rollout=rollout,
            ep_start=self.ep_start_buf,
            hidden_state=self.hidden_states_buf,
            cell_state=self.cell_states_buf
        )

        self.recurrent_ptr = 0

        return recurrent_batch

class FrameStackRolloutBuffer(RolloutBuffer):
    """
    RolloutBuffer variant for frame-stacked observations: rays/proprio carry an
    extra stack-size dim right after num_envs, i.e. (num_steps, num_envs, stack_size, ...).
    """

    def __init__(self, ray_dim, proprio_dim, stack_size: int, device, cfg):
        self.act_dim    = cfg.env.act_dim
        self.gamma      = cfg.env.gamma
        self.gae_lambda = cfg.algorithm.gae_lambda
        self.num_steps  = cfg.algorithm.num_steps
        self.num_envs   = cfg.env.num_envs
        self.device     = device
        self.ptr        = 0

        self.rays_buf     = torch.zeros((self.num_steps, self.num_envs, stack_size, *ray_dim),    dtype=torch.float, device=device)
        self.proprio_buf  = torch.zeros((self.num_steps, self.num_envs, stack_size, proprio_dim),  dtype=torch.float, device=device)
        self.act_buf      = torch.zeros((self.num_steps, self.num_envs, self.act_dim),    dtype=torch.float, device=device)
        self.logp_buf     = torch.zeros((self.num_steps, self.num_envs),                  dtype=torch.float, device=device)
        self.mu_buf       = torch.zeros((self.num_steps, self.num_envs, self.act_dim),    dtype=torch.float, device=device)
        self.std_buf      = torch.zeros((self.num_steps, self.num_envs, self.act_dim),   dtype=torch.float, device=device)
        self.val_buf      = torch.zeros((self.num_steps, self.num_envs),                  dtype=torch.float, device=device)
        self.done_buf      = torch.zeros((self.num_steps, self.num_envs),                 dtype=torch.bool,  device=device)
        self.rew_buf       = torch.zeros((self.num_steps, self.num_envs),                 dtype=torch.float, device=device)
        self.ret_buf       = torch.zeros((self.num_steps, self.num_envs),                 dtype=torch.float, device=device)
        self.adv_buf       = torch.zeros((self.num_steps, self.num_envs),                 dtype=torch.float, device=device)
        self.ep_start_buf  = torch.zeros((self.num_steps, self.num_envs),                 dtype=torch.bool,  device=device)

class FrameStackRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    """RecurrentRolloutBuffer variant that stores frame-stacked rays/proprio."""

    def __init__(self, ray_dim, proprio_dim, stack_size: int, device, cfg):
        FrameStackRolloutBuffer.__init__(self, ray_dim, proprio_dim, stack_size, device, cfg)

        hidden_state_shape = (self.num_steps, cfg.backbone.lstm_num_layers,
                              self.num_envs, cfg.backbone.lstm_backbone_feature_dim)

        self.hidden_states_buf  = torch.zeros(hidden_state_shape, dtype=torch.float, device=self.device)
        self.cell_states_buf    = torch.zeros(hidden_state_shape, dtype=torch.float, device=self.device)

        self.recurrent_ptr = 0






