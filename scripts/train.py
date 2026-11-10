"""
Training script for the whole ablation

"""
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import warnings
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb

from utils import (create_ppo_agent, evaluate_model_on_metrics,
save_video, create_buffer, create_algorithm, create_environment, seed_metric_summary)
from envs import (NavigationEnv, FrameStackWrapper, compute_num_rays, compute_proprio_dim)


import torch
import numpy as np

device = torch.device( "mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available()
                      else "cpu" )

device = "cpu"

print(f"My device: {device}")

def variant_name(cfg: DictConfig):
    """
    Every setting the experiments (E0-E4) vary besides the architecture, so two configs of the
    same sweep never write their seeds into the same folder.
    """
    info = cfg.env.information
    depth = len(cfg.backbone.backbone_hidden_sizes)
    recurrent = cfg.backbone.name != "mlp"
    return (
        f"colour:{cfg.env.colour_mode}_goal:{cfg.env.goal_mode}_"
        f"ginfo:{info.goal_info}_pinfo:{info.position_info}_ggrid:{info.goal_grid}_"
        f"fov:{cfg.env.fov}_act:{cfg.env.action_type}"
        + (f"{cfg.env.discrete_actions}" if cfg.env.action_type == "discrete" else "")
        + f"_cpen:{cfg.env.collision_penalty}_gamma:{cfg.env.gamma}_ep:{cfg.algorithm.max_steps}_"
        f"mask:{'-'.join(map(str, cfg.observation.masked_channels)) or 'none'}_"
        f"width:{cfg.backbone.backbone_hidden_sizes[-1]}_lr:{cfg.algorithm.lr}"
        + (f"_aux:{cfg.head.auxiliary_head.depth_coeff}-{cfg.head.auxiliary_head.loop_closure_coeff}"
           if cfg.head.auxiliary_head.enabled else "")
        # Only when they differ from the default, so the folders of earlier sweeps keep their names
        + (f"_depth:{depth}" if depth != 2 else "")
        + (f"_jshape:{cfg.env.junction_shaping}" if cfg.env.get("junction_shaping", 0.0) != 0.0 else "")
        + ("_modact:True" if cfg.module_output_activation else "")
        # Minibatch sizes set the number of updates per iteration: envs for a recurrent backbone (default 64),
        # samples for the MLP (default 256)
        + (f"_mb:{cfg.algorithm.minibatch_size}" if recurrent and cfg.algorithm.minibatch_size != 64 else "")
        + (f"_bs:{cfg.algorithm.mini_batch_size}" if not recurrent and cfg.algorithm.mini_batch_size != 256 else "")
    )

def run_names(cfg: DictConfig):
    """
    (architecture_name, storage_name) of a run. The storage name is the folder of its seeds under
    <storage>/{models,eval_metrics}/<timestamp>/<room>/, so other scripts (e.g. e2_diagnostics.py) can find them.
    """
    frame_stack = cfg.observation.frame_stack.enabled
    architecture_name = (
        f"encoder:{cfg.observation.name}_backbone:{cfg.backbone.name}_"
        f"frame_stack:{frame_stack}"
        + (f"{cfg.observation.frame_stack.size}" if frame_stack else "")
        + f"_grid_cell:{cfg.observation.grid_cell.enabled}_"
        f"auxiliary_head:{cfg.head.auxiliary_head.enabled}"
    )

    # The BPTT window and the env/observation variant aren't part of the architecture name,
    # but a sweep over them must not write every config's seeds into the same folder
    recurrent = cfg.backbone.name != "mlp"
    storage_name = (architecture_name + (f"_bptt:{cfg.backbone.bptt_window_size}" if recurrent else "")
                    + "_" + variant_name(cfg))
    return architecture_name, storage_name

@hydra.main( config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # The proprio vector grows by the information slots (goal / position / goal grid code),
    # resolve it once here so the agent, buffers and env all agree on its size
    with open_dict(cfg):
        cfg.env.proprio_dim = compute_proprio_dim(cfg)

    wandb.login()

    frame_stack = cfg.observation.frame_stack.enabled
    grid_cell = cfg.observation.grid_cell.enabled
    auxiliary_head = cfg.head.auxiliary_head.enabled

    if auxiliary_head and cfg.head.auxiliary_head.depth_coeff > 0 and 3 not in cfg.observation.masked_channels:
        warnings.warn("Depth aux loss is on but ray channel 3 (distance) is still a policy input, so the depth head "
                      "regresses its own input. Set observation.masked_channels=[3]")

    architecture_name, storage_name = run_names(cfg)
    room = Path(cfg.env.room_path).stem

    # storage_name in the config links the W&B run to its metric folder (scripts/log_metrics_to_wandb.py)
    run_config = OmegaConf.to_container(cfg, resolve=True)
    run_config.update({"architecture_name": architecture_name, "variant_name": variant_name(cfg),
                       "storage_name": storage_name})

    with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=run_config,
                        group=cfg.wandb.group, name=f"{room}_{cfg.observation.name}_{cfg.backbone.name}_{cfg.algorithm.name}_seed_{cfg.seed}",
                        tags=[f"backbone:{cfg.backbone.name}", f"encoder:{cfg.observation.name}",f"algorithm:{cfg.algorithm.name}",
                              f"frame_stack:{frame_stack}", f"grid_cell:{grid_cell}",
                              f"auxiliary_head:{auxiliary_head}", f"room:{cfg.env.room_path}",
                              f"colour:{cfg.env.colour_mode}", f"goal:{cfg.env.goal_mode}",
                              f"goal_info:{cfg.env.information.goal_info}",
                              f"position_info:{cfg.env.information.position_info}",
                              f"fov:{cfg.env.fov}", f"action:{cfg.env.action_type}"],
                        sync_tensorboard=True, reinit=True, mode=cfg.wandb.mode):

        sweep_storage = Path(cfg.storage)
        run_dir = sweep_storage / "models" / str(cfg.timestamp) / room / storage_name / f"seed_{cfg.seed}"
        metrics_dir = sweep_storage / "eval_metrics" / str(cfg.timestamp) / room / storage_name / f"seed_{cfg.seed}"

        num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
        ray_dim = np.array([cfg.env.ray_encoding, num_rays])

        agent = create_ppo_agent(ray_dim= ray_dim, cfg=cfg).to(device)

        env      = create_environment(cfg=cfg, agent=agent, num_rays=num_rays, ray_dim=ray_dim,
                                      num_envs=cfg.env.num_envs, device=device)
        eval_env = create_environment(cfg=cfg, agent=agent, num_rays=num_rays, ray_dim=ray_dim,
                                      num_envs=cfg.env.num_eval_envs, device=device)

        buffer = create_buffer(backbone_type=cfg.backbone.name, algorithm_name=cfg.algorithm.name,
                               ray_dim=ray_dim, proprio_dim=cfg.env.proprio_dim,
                               device=device, cfg=cfg)

        algorithm = create_algorithm(cfg=cfg, algorithm_name=cfg.algorithm.name, backbone_type=cfg.backbone.name,
                                     buffer=buffer, device=device, env=env, eval_env=eval_env, agent=agent)

        algorithm.train(run_dir=run_dir)

        # Learning curve + budget-padded AUC, for offline aggregation (rliable) next to the other metrics
        metrics_dir.mkdir(parents=True, exist_ok=True)
        if algorithm.auc is not None:
            np.save(metrics_dir / "auc.npy", np.array([algorithm.auc]))
            np.save(metrics_dir / "success_curve.npy", algorithm.eval_success_curve)

        agent.load_model(run_dir / f"best.pt", device, algorithm.optimizer)
        agent.eval()

        # --- Rollout + video ---
        render_env = create_environment(cfg=cfg, agent=agent, num_rays=num_rays,
                                        ray_dim=ray_dim, num_envs=1, device=device)
        with torch.no_grad():
            frames = render_env.record_rollout(cfg.backbone.name, agent, 200, cfg)
        custom_video_path = run_dir / "videos" / f"rollout.mp4"
        save_video(frames, custom_video_path)

        wandb.log({
                        f"Rollout": wandb.Video(str(custom_video_path), fps=10, format="mp4")
                    })
          
        # Evaluate metrics on a fixed starting and ending goal when the goal is fixed (one start/goal pair,
        # as before). With per-episode goals the starts stay random too, and every episode is scored
        # against the shortest path of its own start/goal pair
        if cfg.env.goal_mode == "fixed":
            env.unwrapped.random_pos_flag = False

        evaluate_model_on_metrics(agent=agent, env=env,
                                        episodes=cfg.env.completion_rate_eps,
                                        nr_runs=cfg.env.mean_of_means_runs,
                                        json_path=cfg.env.room_path,
                                        storage_path=metrics_dir,
                                        backbone_type=cfg.backbone.name)

        # Final results on the run itself (AUC, CR, SPL, nDTW, ...), so runs can be compared in the W&B table.
        # In the summary, not logged: one value per run would plot as a dot at the step the run happened to end
        wandb.run.summary.update(seed_metric_summary(metrics_dir))
        
if __name__ == "__main__":
    main()
