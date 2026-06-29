from dataclasses import dataclass
import torch
from omegaconf import DictConfig

#TODO: Adjust replay buffer for LSTM Policy backbone

@dataclass
class ReplayBatch:
    obs: dict
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: dict
    done: torch.Tensor


class ReplayBuffer:
    def __init__(
            self,
            ray_dim: tuple,
            proprio_dim: int,
            device:torch.device,
            cfg:DictConfig
            ):
        
        self.gamma = cfg.env.gamma
        self.num_steps = cfg.algorithm.num_steps
        self.num_envs = cfg.env.num_envs
        self.device = device
        self.act_dim = cfg.env.act_dim

        self.ptr = 0
        self.size = 0
        
        self.rays_buf   = torch.zeros((self.num_steps, self.num_envs, *ray_dim),   dtype=torch.float, device=device)
        self.proprio_buf = torch.zeros((self.num_steps, self.num_envs, proprio_dim), dtype=torch.float, device=device)
        self.act_buf = torch.Tensor((self.num_steps, self.num_envs, self.act_dim), dtype=torch.float, device=device)
        self.next_rays_buf   = torch.zeros((self.num_steps, self.num_envs, *ray_dim),   dtype=torch.float, device=device)
        self.next_proprio_buf = torch.zeros((self.num_steps, self.num_envs, proprio_dim), dtype=torch.float, device=device)
        self.rew_buf = torch.Tensor((self.num_steps, self.num_envs), dtype=torch.float, device=device)
        self.done_buf = torch.Tensor((self.num_steps,  self.num_envs), dtype=torch.bool, device=device)


        
      
    
    def store(
            self,
            obs: dict,
            act: torch.Tensor,
            rew: torch.Tensor,
            next_obs: dict,
            done: torch.Tensor
            ):
        
        self.rays_buf[self.ptr] = obs["rays"]
        self.proprio_buf[self.ptr] = obs["proprio"]
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_rays_buf[self.ptr] = next_obs["rays"]
        self.next_proprio_buf[self.ptr] = next_obs["proprio"]
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.num_steps
        self.size = min(self.size + 1, self.num_steps)

    
    def get(self, batch_size:int) -> ReplayBatch:

        env_idx = torch.randint(low=0, high=self.num_envs, size=batch_size)
        batch_idx = torch.randint(low=0, high=self.size, size=batch_size)

        return  ReplayBatch(
            obs={
                "rays": self.rays_buf[batch_idx, env_idx],
                "proprio": self.proprio_buf[batch_idx, env_idx]
                },
            act=self.act_buf[batch_idx, env_idx],
            rew=self.rew_buf[batch_idx, env_idx],
            next_obs={
                "rays": self.next_rays_buf[batch_idx, env_idx],
                "proprio": self.next_proprio_buf[batch_idx, env_idx]
                },
            done=self.done_buf[batch_idx, env_idx],
        )
