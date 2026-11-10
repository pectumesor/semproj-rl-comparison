# Codebase reference

This describes the code as of October 29, 2026. Sections 1–7 describe the original pipeline, and section 0 lists the experiment settings added on top of it. Grep before relying on any detail, because the code keeps changing.

## Contents
0. Experiment knobs (E0 implementation)
1. Training run, step by step
2. Environment
3. Ray casting and colour
4. Agents, models, factory
5. PPO and buffers
6. Metrics and navmesh
7. Room JSON format

## 0. Experiment knobs

All of these live in `configs/env/base_env.yaml` unless noted otherwise. Defaults reproduce the original behaviour.

| Key | Values | Used by |
|---|---|---|
| `env.colour_mode` | global / room_local / off / per_wall; plus `colour_period` and `colour_seed` (fixed across RL seeds; null = old behaviour) | E1, E2 |
| `env.goal_mode` | fixed / random (with `min_goal_distance`) / regions (twin rooms: `goal_points`, `goal_regions`, `cue_colours` in the JSON; entering a wrong region ends the episode as a failure) | E1–E3 |
| `env.goal_visual_radius` | the radius of the goal disc the rays can see. `env.goal_radius` remains the success radius. | E2, E3 |
| `env.information.goal_info` | none / oracle / last_seen_updated / last_seen_frozen → 5-dim slot | E3 |
| `env.information.position_info` | none / xy / rff / grid / noisy_grid; `goal_grid`; `reserve_info_slots`; `grid.*`; `path_integration.*` | E3 |
| `env.max_turn_deg` | 90 (turning is no longer tied to FOV) | E1b, E4 |
| `env.action_type`, `env.discrete_actions` | continuous / discrete with 3 or 5 actions (see the table in `envs/env_utils.py`) | E4 |
| `env.collision_penalty` | subtracted on every step a wall blocks the movement | E4 |
| `env.junction_shaping` | potential-based shaping on the path distance to the twin rooms' junction (room JSON `junction_path`), 0 from the junction on, so cue-free; storage suffix `_jshape:` | E2 |
| `observation.masked_channels` | ray channels zeroed before embedding, e.g. `[3]` with depth aux | E4 |
| `backbone=gru_backbone` | GRU with the LSTM's tuple-state interface (it reuses the `lstm_*` keys) | E2, trigger rule |

Rules that come with these knobs:
- `train.py` sets `cfg.env.proprio_dim = compute_proprio_dim(cfg)` once, so the agent, buffers and env agree on the size of the information slots.
- `info["success"]` (not `terminated`) marks success.
- The success AUC is padded to the full budget.
- Twin rooms come from `scripts/generate_twin_rooms.py` as `rooms/twin_rooms_L{L}.json`.
- `scripts/aliasing_score.py` computes the E1a aliasing score.
- `scripts/e2_diagnostics.py`: E2 memory checks on trained checkpoints (reset at the junction, confusion matrix, linear probe of the cue). It uses `env.in_junction()`, `info["room"]` and `env.cue_visible`, and `e2_twin_rooms.sh <timestamp>` runs it alone on a finished sweep.
- `scripts/seed_stats.py`: seed-level statistics for the paper (IQM + bootstrap CI, Welch + Holm, probability of improvement). `+stats.vary=backbone +stats.reference=mlp` compares backbones within every condition. The E1–E3 scripts call it at the end.
- `scripts/paper_figures.py`: the paper's data figures (`+figure.name=e1|e1b|e2|e3|e4`) plus the E2 diagnostics table and the E4 Kendall τ, saved under `<storage>/figures/<timestamp>/` with a .csv of the plotted numbers. Memory gain = LSTM − MLP, never best of LSTM and GRU. E4 reads `<storage>/promotion/<timestamp>.json`, and `+figure.compare=<json>` adds τ between stages.
- `scripts/e4_promotion.py`: E4 survivors between stages. `seed_stats.py` imports its `iqm`, `stratified_bootstrap_ci` and `parse_storage_name`.
- `run_names(cfg)` in `scripts/train.py` gives a run's (architecture name, storage name). Import it to locate a config's seed folders instead of rebuilding the name.
- Sweep scripts follow the pattern `scripts/e{0,1,1b,2,3,4}_*.sh`.
- `tests/test_experiment_features.py` holds the current tests. Most other files in `tests/` are stale.

## 1. Training run (`scripts/train.py`)

1. `torch.manual_seed(cfg.seed)`; build `architecture_name` from encoder, backbone, frame stack, grid cell and aux flags. The storage name adds `_bptt:<w>` for recurrent backbones.
2. `num_rays = compute_num_rays(fov, ray_density)` (forced odd); `ray_dim = [ray_encoding, num_rays]`.
3. `create_ppo_agent` builds the agent; `create_environment` builds `env` (num_envs) and `eval_env` (num_eval_envs), wrapped in `FrameStackWrapper` if enabled.
4. `create_buffer` and `create_algorithm`: `MLPPPO` for the mlp backbone, `RecurrentPPO` otherwise.
5. `algorithm.train(run_dir)` saves `best.pt` whenever the eval-success EMA improves, plus `iter_<n>.pt` every `save_interval`.
6. Load `best.pt`, record a 200-step rollout video, set `random_pos_flag=False`, and run `evaluate_model_on_metrics` into `<storage>/eval_metrics/<timestamp>/<room>/<name>/seed_<s>/`.

Device is forced to `"cpu"` in `train.py`.

## 2. Environment (`envs/navigation_env.py`)

- **State tensors:** `agent_pos (E,2)`, `facing_direction (E,)`, `steps (E,)`, `last_speed`, `last_turning`, and `goal_pos`. Originally `goal_pos` was a single `(2,)`; check whether it is now `(E,2)`.
- **Action** `(E,2)`: `[turn ∈ [-1,1], speed ∈ [0,1]]`, clamped to `env.action_low/high`. Movement is a single-ray `intersect`, so the agent stops just before walls (`safe_t = min_t - 1e-3`) without a penalty.
- **Reward:** `-1/max_steps` per step, `+1` on reaching the goal (`dist <= goal_radius`).
- **Truncation:** `steps >= max_steps`. `step()` zeroes the step counter but does *not* reset position; the algorithm calls `reset(done=mask)` at the start of the next loop iteration.
- **Rays `(E, 7, R)`:**
  - channel 0: no-hit
  - channel 1: goal hit (ray–disc test against the goal, occluded by walls)
  - channel 2: wall hit
  - channel 3: normalised distance
  - channels 4–6: RGB colour at the hit point
- **Proprio `(E, 2)`:** last speed, last turn. `NavigationEnvEasy` adds sin/cos heading and distance shaping; it exists to compare against SB3.
- **Spawning:** `rand_pos: "True"` (a string) means a uniform point in a random spawn region, or the wall-offset bounding box if there are none. Heading is uniform random.
- **Rendering:** `w2s()` assumes room coordinates start at 0 and are positive.

## 3. Ray casting and colour (`envs/env_utils.py`)

- `RayCast.intersect(positions (E,2), d (E,R,2))` → `min_t (E,R)` as a fraction of `|d|`, `inf` if nothing is hit. This is the standard 2D segment intersection over all walls at once (an `(E,R,W)` tensor).
- `RayCast.scan(pos, facing)` → `intersections (E,R,2)`, `distances (E,R)` (`inf` on a miss) and `d_unit`. Ray offsets are `linspace(-fov/2, fov/2, R)`.
- `max_range` comes from `range_type`: `diagonal` (bounding-box diagonal) or `horizontal`.
- `PerlinColor(seed, scale=0.15, octaves=4)` gives 3 decorrelated fBm channels in [0,1], queried at world (x, y).
- `LoopLabeler`: a position counts as a loop closure if it is within `eta1` of an earlier position *and* the path went more than `eta2` away in between.

## 4. Agents, models, factory

- **Observation models** (`models/embeddings`):
  - `MLPObservationEmbeddings`: flatten rays and concatenate proprio.
  - `CNNObservationEmbeddings`: Conv1d over rays (channels = ray channels), max-pool by 2, flattened so bearing is kept, plus a proprio MLP, then fusion.
  - Frame-stack variants put the stack dim after E.
- **Backbones:**
  - `MLPBackbone`.
  - `SimpleLSTM(input_dim, feature_dim, num_layers)`: steps through time, zeroing state where `done` (the episode start). State is a tuple `(h, c)` of shape `(layers, B, H)`.
  - `NAVA3C` exists but is unused.
- **Heads:** `GuassianPolicyHead` (that spelling) has a state-independent `log_std` clamped to [-5, 1]; `act_inference` returns the mean. Also `ValueNet`, plus `DepthHead` (sigmoid) and `LoopClosureHead` (logit).
- **Agents:**
  - `PPOAgent` (BaseAgent): `select_action` returns action, clipped action, logp, mu, std and value; `predict_action` is deterministic and clamped.
  - `RecurrentPPOAgent`: the same API, plus the LSTM state and an episode-start mask.
  - Aux heads are saved and loaded in the checkpoint when present.
- **Factory** (`utils/factory.py`): `create_backbone_model` branches on `cfg.backbone.name`. Heads take `backbone_hidden_sizes[-1]` as input size, which matches `lstm_backbone_feature_dim` (64) by default. Keep them equal.

## 5. PPO and buffers

- **`MLPPPO.collect_rollout`:** reset done envs, act, step, then bootstrap truncations into the reward (`reward += γ·V(next)·(truncated & ~terminated)`) before storing. GAE runs in `RolloutBuffer.compute_returns` with `not_terminal = 1 - done`.
- **Update:**
  - The advantage is normalised per minibatch.
  - Value clipping uses the same ε as the policy.
  - Entropy is averaged; summing it was a past bug that pinned `log_std`.
  - KL is the Gaussian KL from the stored mu/std. Training breaks out of the epochs when the running-mean KL exceeds 1.5·target_kl.
  - LR anneals linearly, and when it does the KL-adaptive LR is off.
- **`RecurrentPPO`:**
  - Minibatches are over env indices, and each is split into BPTT windows of `bptt_window_size`. Each window starts from the stored hidden state at `w_start`.
  - `num_envs % minibatch_size == 0` is required.
  - It prints a warning if the first-minibatch ratio is far from 1, which signals that the replay doesn't match collection.
- **Early stopping** (`ConvergenceMonitor`): an EMA of eval success (α = 0.2), with patience counted in evals, and only once the best EMA ≥ `min_converged_value` (0.95). `improved` also triggers `best.pt`.
- **Eval** (`evaluate_policy`): `num_eval_envs` parallel episodes, each counted until its first termination or truncation.

## 6. Metrics (`utils/metrics.py`, `utils/navigation_mesh.py`)

- **Extraction:** `extract_trajectory_{mlp,recurrent}` rolls out until `nr_runs` goal-reaching trajectories are collected, with a cutoff of `3·(max_steps+1)` steps.
- **CR:** `completion_rate_*` gives a percentage over `completion_rate_eps` episodes, repeated `mean_of_means_runs` times.
- **DTW** resamples the reference to the test trajectory's length. NPL = path length / reference length.
- **Shortest path:** the reference is a Delaunay triangulation (`triangle`, flags `'pn'`), then A* over the triangle adjacency graph (wall crossings cost 1e6), then the funnel algorithm (`string_pull`). It is logged as a W&B image.
- **Aggregation:** `scripts/evaluate_metrics.py` turns every `<room>/<architecture>` folder into a mean-of-means plot and a W&B summary table.

## 7. Room JSON

```json
{
  "edges": [{"from": {"x": 0, "y": 0}, "to": {"x": 400, "y": 0}}],
  "start_pos": {"x": 360, "y": 360},
  "goal_pos": {"x": 20, "y": 20},
  "spawn_regions": [[{"from": …, "to": …}, …]]
}
```

- **Scale:** about 100 units per cell, `max_speed` 13 units per step.
- **Spawn regions:** axis-aligned rectangles given as their four edges. Use them whenever the bounding box contains space outside the walkable area (C_maze and cross_room needed them).
- **Rooms by tier:**
  - smoke: one_room
  - Tier 1: two_room, T_maze_room, I_maze_room
  - Tier 2: four_room, cross_room, C_maze
  - Tier 3: medium_maze, hard_maze
- **Not navigation rooms:** `grid_cell_training.json` and `test_room.json` have no start or goal.
- **Hospital plans:** `convert_hospital_json_format` converts Arxitect plans, but none have been converted yet.
