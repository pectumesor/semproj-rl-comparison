from dataclasses import dataclass
import torch

@dataclass
class RolloutBatch:
    obs: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    mu: torch.Tensor
    std: torch.Tensor
    val: torch.Tensor
    ret: torch.Tensor
    adv: torch.Tensor
    done: torch.Tensor

class RolloutBuffer:
    def __init__(
            self,
            obs_dim: tuple,
            act_dim: int,
            num_steps: int,
            num_envs: int,
            gamma: float,
            gae_lambda: float,
            device: torch.device,
            ):
        
        self.obs_buf = torch.zeros((num_steps, num_envs, *obs_dim), dtype=torch.float, device=device)
        self.act_buf = torch.zeros((num_steps, num_envs, act_dim), dtype=torch.float, device=device)
        self.logp_buf = torch.zeros((num_steps, num_envs), dtype=torch.float, device=device)
        self.mu_buf = torch.zeros((num_steps, num_envs, act_dim), dtype=torch.float, device=device)
        self.std_buf = torch.zeros((num_steps, num_envs, act_dim), dtype=torch.float, device=device)
        self.val_buf = torch.zeros((num_steps, num_envs), dtype=torch.float, device=device)

        self.done_buf = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.rew_buf = torch.zeros((num_steps,num_envs), dtype=torch.float, device=device) # Rewards

        self.ret_buf = torch.zeros((num_steps, num_envs), dtype=torch.float, device=device) # Reward to go
        self.adv_buf = torch.zeros((num_steps,num_envs), dtype=torch.float, device=device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.ptr=0

    def store(
            self,
            obs: torch.Tensor,
            act: torch.Tensor,
            logp: torch.Tensor,
            mu: torch.Tensor,
            std: torch.Tensor,
            val: torch.Tensor,
            rew: torch.Tensor,
            done: torch.Tensor,
    ):
        if self.ptr >= self.num_steps:
            raise ValueError("RolloutBuffer is full. Call get() first")
        
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.logp_buf[self.ptr] = logp
        self.mu_buf[self.ptr] = mu
        self.std_buf[self.ptr] = std
        self.val_buf[self.ptr] = val
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr += 1

    def compute_returns(self, last_val):
        """
        Compute GAE for the advantage function and Reward-To-Go
        """

        advantage = 0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_val = last_val
            else:
                next_val = self.val_buf[step + 1]
            
            not_terminal = 1.0 - self.done_buf[step].float()
            # \delta_t^V = r + \gamma * V(s_{t+1}) - V(s_{t})
            delta = self.rew_buf[step] + self.gamma * next_val * not_terminal - self.val_buf[step]
            # A_t = delta + \gamma*\lambda * A_{t+1}
            advantage = delta + self.gamma * self.gae_lambda * advantage * not_terminal
            self.adv_buf[step] = advantage
            # Reward-to-go = G_t ==> A_t = G_t - V_t ==> G_t = A_t + V_t
            self.ret_buf[step] = advantage + self.val_buf[step]
        
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-8)

    def get(self) -> RolloutBatch:

        if self.ptr != self.num_steps:
            raise ValueError(
                f"RolloutBuffer must be full before calling get(). "
                f"Current size: {self.ptr}, expected: {self.num_steps}"
            )
    
        batch = RolloutBatch(
            obs=self.obs_buf,
            act=self.act_buf,
            logp=self.logp_buf,
            mu=self.mu_buf,
            std=self.std_buf,
            val=self.val_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            done=self.done_buf
        )

        self.ptr=0

        return batch
