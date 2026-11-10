---
name: semproj-rl-navigation
description: Working knowledge for the semproj-rl-comparison repo, Emerson's ETH IVIA Lab semester project comparing RL design choices for indoor navigation. It covers PPO agents with 2D ray-cast observations, MLP/CNN encoders, MLP/LSTM/GRU/frame-stack backbones, Hydra configs, Euler SLURM sweeps, and the workshop paper "Perceive, Remember, or Be Told". Use this skill for any task in this repo, even if the user doesn't name it. That includes changing env/model/algorithm code, adding a config knob or room JSON, writing a sweep .sh script, debugging a training run, questions about AUC, success rate, SPL, DTW or mean-of-means metrics, and planning or analysing ablations and experiments E0–E4.
---

# semproj-rl-comparison: working in this repo

This repo is a controlled comparison of RL navigation design choices: observation, memory architecture, auxiliary losses, action space and reward. Agents are PPO-trained and egocentric, and they see only ray casts in 2D floorplans. Results go into a semester report and a 4-page workshop paper.

The source of truth for *what* to run is `reports/Final paper and experiment plan.md`. Read the relevant section before designing or changing an experiment. The literature behind it is in `research_notes/` and `reports/Workshop paper experiments RL navigation.md`.

## Ground rules from the user

- **No git commits or pushes** unless the user asks in that conversation. They review diffs themselves.
- **Match the existing code style** (see "Style" below). This is the user's own codebase, and they want it to still read like theirs.
- **Fix bugs you come across, and list each one** in your final message with file:line. Several important results here were changed by quiet bugs (AUC horizon, depth aux leakage, CNN pooling over rays), so the user wants to know about every fix.
- **Training runs belong on Euler by default.** Euler is ETH's cluster (CPU, about 16 concurrent SLURM jobs). Locally, run smoke tests (`scripts/smoke_test.sh` in this skill) and short timing probes freely.
  - Before any longer local run, state its cost and ask. A full run is 1000 iterations × 256 envs × 256 steps; the twin room at L=32 takes about 2.9 s per iteration, so about an hour per seed on the laptop. Sweeps never run locally.
  - Whenever you suggest Euler, give a **ready-to-run command or script** (the matching `scripts/e*.sh` call or a `train.py --multirun hydra/launcher=euler …` line with the right `max_steps`). Don't stop at offering to write one.
- **LSTM is the default recurrent backbone.** GRU runs in the twin-room experiment E2, and elsewhere only under the pre-registered trigger rule in the plan (LSTM's IQM below MLP's with non-overlapping CIs). Never report "best of LSTM and GRU".

## Code map (details in `references/codebase.md`)

| Area | File | What lives there |
|---|---|---|
| Entry point | `scripts/train.py` | Hydra main: builds agent → env/eval_env → buffer → algorithm; trains; loads `best.pt`; video; `evaluate_model_on_metrics` |
| Env | `envs/navigation_env.py` | Vectorised `NavigationEnv` (all state is `(num_envs, …)` torch tensors), reward, obs construction, rendering |
| Ray casting | `envs/env_utils.py` | `RayCast` (vectorised ray–wall intersection), `PerlinColor`, `LoopLabeler` (loop-closure labels) |
| Wrappers | `envs/wrappers.py` | `FrameStackWrapper` (batched, done-masked reset), SB3 single-env wrapper |
| Geometry | `utils/geometry.py` | Room JSON parsing, `InitialPositionSampler` (spawn regions), `compute_num_rays` |
| Factory | `utils/factory.py` | `create_observation_model / backbone / ppo_agent / algorithm / buffer / environment` |
| PPO | `algorithms/ppo.py` | `MLPPPO`, `RecurrentPPO` (truncated BPTT windows), eval, AUC, early stopping hookup |
| Buffers | `algorithms/buffers/rollout_buffer.py` | Rollout / recurrent / frame-stack buffers, GAE |
| Models | `models/` | embeddings (MLP, CNN, frame-stack), backbones (MLP, LSTM, GRU), heads (Gaussian policy, value, aux), agents |
| Metrics | `utils/metrics.py`, `utils/navigation_mesh.py` | Trajectory extraction, CR/DTW/NPL (+SPL/nDTW), Delaunay navmesh + funnel shortest path |
| Configs | `configs/` | `base.yaml` defaults: algorithm=ppo, env=base_env, observation, backbone, head |
| Rooms | `rooms/*.json` | `edges` (from/to), `start_pos`, `goal_pos`, optional `spawn_regions` (and goal/cue fields for twin rooms) |
| Sweeps | `scripts/*.sh` | Self-contained Euler launchers (conda setup + `train.py --multirun` + `evaluate_metrics.py`) |

The plan's experiment settings are implemented as config keys. Their defaults reproduce the original behaviour. They include `env.colour_mode`, `env.colour_seed`, `env.goal_mode`, `env.information.*` (goal_info, position_info, goal_grid, reserve_info_slots), `env.max_turn_deg`, `env.action_type`, `env.collision_penalty`, `observation.masked_channels` and the `gru_backbone` config. The full list is in `references/codebase.md` § Experiment knobs. Before relying on a key, check `configs/` for it: the code keeps changing.

`variant_name(cfg)` in `scripts/train.py` builds the storage/W&B suffix from every varied setting. **Any new knob must be added there.**

## Validity pitfalls

These are the ways a run in this repo can produce a clean-looking but wrong result. Check each one that touches your change.

1. **Colour as a place code.** Ray channels 4–6 are a Perlin colour sampled at the hit point's *world* coordinates. A memoryless MLP can read its position from them, which is the leading explanation for "MLP+MLP wins". Identical-looking rooms are only visually identical if colour is off, room-local, or hard-coded per wall. The colour seed must not follow the RL seed, or seed variance mixes with colour-map variance.
2. **A fixed goal makes memory unnecessary.** With one goal per room, every room becomes a reactive lookup. Memory, goal-information and grid-code experiments need the goal randomised per episode (or chosen from goal regions).
3. **`terminated` is not the same as success** once wrong-room failures exist. Eval, completion rate and trajectory extraction must count success explicitly.
4. **The AUC must cover the full budget.** Early-stopped runs have fewer evals; pad the success curve with its last value up to `n_iterations // eval_interval` evals before integrating. Otherwise fast and slow learners are scored on different horizons.
5. **Depth aux needs a hidden target.** The depth head regresses ray channel 3, which the policy also sees unless it's masked. Compare aux-on vs aux-off with the *same* masking, or the comparison also changes what the policy sees.
6. **FOV used to set the turn rate** (`turning = action * half_fov`). Any FOV sweep must use a FOV-independent max turn.
7. **Recurrent checks.** Code that branches on the backbone must treat every recurrent backbone (`lstm`, `gru`) the same: rollout recording, metrics hidden-state sizes, storage names, `create_algorithm`. Grep for `"lstm"` after adding a backbone.
8. **Shape bookkeeping.** `cfg.env.proprio_dim` and `ray_dim = (ray_encoding, num_rays)` size the agent *and* the buffers. Any new observation feature must update both, and frame stacking adds a stack dim after `num_envs`.
9. **Storage collisions.** Seeds are written to `<storage>/{models,eval_metrics}/<timestamp>/<room>/<architecture name>/seed_<s>`. Every setting a sweep varies must appear in that name, or runs silently overwrite each other.
10. **Cue leaks in memory tasks.** A task only needs memory if the cue is out of sight once the agent leaves the start. Rays reach the whole room (`range_type: diagonal`), and a 180° FOV includes the rays straight to the sides, so any straight corridor back to a cue wall leaks it. The twin rooms use a bent shaft for this reason. For any new or changed memory room, check line of sight with 360° rays from every reachable point past the start, as `test_twin_rooms_cue_hidden_past_the_bend` does.
11. **Seed sets.** Use 0–2 for screening, 100–104 for stage 2 and E1/E1b/E2/E3, and 1000–1009 for final claims. Never report screening seeds as results. The literature behind this (Henderson 2018, Patterson 2024, Eimer 2023, Colas 2019, Agarwal 2021) is in the plan's "Why these seed sets".

## Adding an experiment knob (checklist)

1. Add the config key with a default that **reproduces current behaviour**, and a comment explaining what it does and why.
2. Implement it vectorised over `num_envs`, on the env's device, in `reset()` (masked by `done`) and/or `step()`.
3. Update the observation shapes (`proprio_dim`, `ray_dim`), the buffers, and the frame-stack path if the observation changes.
4. Add it to the storage/architecture name and the W&B tags in `train.py`.
5. Run a smoke test with each backbone it interacts with (`scripts/smoke_test.sh`).
6. Add or extend the sweep `.sh` script, following `references/euler.md`.
7. Update the plan doc if a design decision changed.

## Running things

- **Smoke test:** `bash .claude/skills/semproj-rl-navigation/scripts/smoke_test.sh <python> [hydra overrides…]`. It runs 2 PPO iterations with tiny env counts, W&B disabled, and output in a temp dir. Run it for MLP and LSTM at least, plus GRU, CNN and frame stacking if you touched shared paths.
- **Local Python:** the system Anaconda has torch 2.1 but lacks `hydra-core`, `gymnasium`, `triangle`, `pygame`, `stable-baselines3` and `sb3-contrib`. Create a venv in the scratchpad: `python3 -m venv --system-site-packages <dir>/venv && <dir>/venv/bin/pip install "gymnasium>=0.29" omegaconf "hydra-core>=1.3" triangle pygame stable-baselines3 sb3-contrib`.
  - **Check isolation before any `pip install`:** run `<venv>/bin/python -c "import sys; assert sys.prefix != sys.base_prefix"`. The temp cleaner can delete an old venv's `pyvenv.cfg`; its `python` then silently is base Anaconda, and pip changes the user's base environment. If the check fails, make a new venv, never reinstall into the old one.
- **The tests in `tests/` are stale.** Many use old config keys like `cfg.model`. Don't treat their failures as regressions; write a focused new check instead.
- **Euler sweeps:** see `references/euler.md` for the script template, timeouts and per-run costs.

## Evaluation and statistics conventions

- **Primary metric:** budget-padded success AUC. Secondary: success rate, SPL, nDTW (exp(−DTW/(N·success_radius)), Ilharco et al. 2019), plus normalised path length and raw DTW for continuity.
- **Supervisor's scheme:** 10 eval runs × 10 episodes per seed, with per-seed means and standard errors ("mean of means"). The statistical unit is the **training seed**.
- **Paper statistics:** IQM with stratified-bootstrap 95% CIs (rliable), probability of improvement, and Welch's t at α = 0.01 with Holm correction for final claims.

## Style

Read the neighbouring code and copy it. The recurring patterns are:

- **Shape comments** on tensor lines: `# (E, R, 2)`, using E = envs, R = rays, W = walls, T = time, F = stack, C = channels.
- **Comments say why** ("Bootstrap time-limit cutoffs only: …"), placed above or beside the line. Docstrings are triple-quoted and short, sometimes with an ASCII formula.
- **Fully vectorised torch**, with no Python loops over envs. `mask = done` style resets, `torch.where` for branches.
- **Hydra configs** are grouped by folder (`algorithm/`, `env/`, `observation/`, `backbone/`, `head/`) with long explanatory comments on non-obvious values, and a factory function per component.
- **Logging:** `print(f"[PPO] key={value:.4f} …")` and W&B keys in Title Case ("Eval Success Rate").
- **Sweep scripts** are self-contained bash with the conda/Euler header, `join()` for Hydra comma lists, one `NOW` timestamp per sweep, `|| echo "Some training jobs failed…"`, and `evaluate_metrics.py` at the end.
