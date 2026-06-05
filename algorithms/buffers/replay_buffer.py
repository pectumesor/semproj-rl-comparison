from dataclasses import dataclass
import torch

@dataclass
class ReplayBatch:
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    


class ReplayBuffer:
    def __init__(
            self,
            obs_dim:int,
            act_dim:int,
            size: int,
            gamma:int,
            device:torch.device
            ):
        
        self.obs_buf = torch.Tensor((obs_dim, size), dtype=torch.float, device=device)
        self.act_buf = torch.Tensor((act_dim, size), dtype=torch.float, device=device)
        self.next_obs_buf = torch.Tensor((obs_dim, size), dtype=torch.float, device=device)
        self.rew_buf = torch.Tensor(size, dtype=torch.float, device=device)
        self.done_buf = torch.Tensor(size, dtype=torch.bool, device=device)

        self.gamma = gamma
        self.max_size = size
        
        self.ptr = 0
        self.size = 0
    
    def store(
            self,
            obs: torch.Tensor,
            act: torch.Tensor,
            rew: torch.Tensor,
            next_obs: torch.Tensor,
            done: torch.Tensor
            ):
        
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    
    def sample_batch(self, batch_size:int) -> ReplayBatch:
        idxs = torch.randint(0, self.size, size=[batch_size])

        return  ReplayBatch(
            obs=self.obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            done=self.done_buf[idxs],
        )
