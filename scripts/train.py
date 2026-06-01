# scripts/train.py
import hydra
from utils.logging import init_wandb, log, finish

@hydra.main(config_path="../configs", config_name="base")
def train(cfg):
    init_wandb(cfg)

    for update in range(cfg.train.n_updates):
        # --- rollout + update ---
        metrics = {
            "train/reward":       mean_reward,
            "train/policy_loss":  policy_loss,
            "train/value_loss":   value_loss,
            "train/entropy":      entropy,
            "train/kl":           approx_kl,    # PPO-specific
        }
        log(metrics, step=update)

    finish()
