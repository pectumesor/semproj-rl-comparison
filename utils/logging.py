# utils/logging.py
import wandb
from pathlib import Path
from omegaconf import OmegaConf

def save_video(frames: list, path: Path, fps: int = 10):
    import imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps)
    print(f"Saved video to {path}")
