# Past chats and repo state: what is planned, what was found, what the code can do now

Sources: the four transcripts in `/Users/emers/.claude/projects/-Users-emers-projects-semproj-rl-comparison/` (text messages only), the two PDFs the user attached to the current session (`/Users/emers/Downloads/Semester Project - Weekly Progress.pdf`, `/Users/emers/Downloads/Emerson_L__Azevedo_Aguiar___ETH_Zurich___IVIA_Lab___Semester_Project (3).pdf`, read with pdftotext), and the repo at `/Users/emers/projects/semproj-rl-comparison` at commit 7975b66 (2026-10-20). The memory dir `/Users/emers/.claude/projects/-Users-emers-projects-semproj-rl-comparison/memory/` is empty. Repo paths below are relative to the repo root.

Transcript coverage is thin. a23eccea (2026-10-06) is trivial: it locates `ppo_agents.py` and adds a hello-world script. 072ca653 (2026-10-09) is an AUC code review. f65e7faf (2026-10-09) is the auxiliary-loss debugging session. 3e28ab7d (2026-10-29) holds only the current request. Most results, plans and paper ideas are in the Weekly Progress PDF, not the chats.

## Q1. What experiment ideas, ablation plans, paper-framing ideas and supervisor feedback appear in past conversations and project documents?

### Takeaway
The user wrote down five workshop-paper ideas at the 20.10 meeting:
- goal conditioning
- colour in similar rooms
- comparison with a small VLA
- the minimum parameter count before learning degrades
- when an auxiliary loss helps without adding parameters

Earlier notes add more:
- an aliased "two identical rooms" layout
- randomized or generated layouts
- an extra goal ray cast
- compute-based metrics
- two curricula from the proposal

### Cited Findings
**Workshop ideas from the 20.10 meeting notes**, verbatim list:
- "Test goal conditioned and how that affects the model"
- "Test color in similar rooms"
- "Compare with a small VLA"
- "Show tests of how little parameters we need until learning degrades"
- "Show when auxiliary loss helps policy without adding extra layers + parameters. When the trade-off happens"

[Weekly Progress PDF, "Meeting 20.10" → "Ideas for publishing in a workshape"](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)

**Aliased-room idea (pre-12.10 "Next Week" list):** "Create a complicated room, with many rooms, but two of these looks exactly the same. Ablation on this room. Use two walls with a small space to bias the Perlin Noise." [Weekly Progress PDF](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)

**Other items on the same list:**
- Prio 2: Area under the Curve.
- "Possible metrics to compare algorithms: Steps, FLOPs, Wallclock".
- Prio 3: "Make harder environments. Use Learning to navigate paper as a guide".
- "Randomized environments, create or pick from layouts".
- "On hold, literature reading: Extend observation with an extra ray cast representing the goal".

[Weekly Progress PDF](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)

**Open questions from 14.09–21.09:**
- "Finish grid cell activation replication? Use grid cell training as described in paper or use only an grid cell activation function (similar to SIFT)".
- SAC: "Failed to do so. Would suggest to do most of PPO's ablation now and try to fix SAC later".

[Weekly Progress PDF](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)

**Comparability question (Week 11.08–17.08):** "Need to think if we should adjust nr_iterations for different architectures to compare them with similar trained performances". This later became early stopping on success rate. [Weekly Progress PDF](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)

**Supervisor feedback (24.08–31.08 TODO):**
- Report metrics per seed and the mean.
- Use the standard error.
- Do 10 runs per seed and take means of means.
- Make the BPTT window adjustable.
- Run until good-enough convergence.

The note "Kasch au 10 runs pro seed und means of means macha" is Swiss German for "you can also do 10 runs per seed and do means of means", which suggests supervisor phrasing. [Weekly Progress PDF](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)

**Proposal framing:**
- Central question: "What is the impact of reinforcement learning design parameters on downstream policy performance in RL-based navigation?"
- Novelty: a systematic comparison under one shared evaluation framework.
- Applications: the Arxitect tool and hospital floorplan assessment.

[Proposal PDF §1.1.2, §1.3.1](/Users/emers/Downloads/Emerson_L__Azevedo_Aguiar___ETH_Zurich___IVIA_Lab___Semester_Project%20(3).pdf)

**Proposal assumption: undirected search.** The agent "has no information of where the goal is… should find the goal without any information in its observations or through reward shaping". Goal encodings for directed search are only "if time permits". This makes goal conditioning a natural extension the proposal already anticipated. [Proposal PDF §1.2.2, §1.3](/Users/emers/Downloads/Emerson_L__Azevedo_Aguiar___ETH_Zurich___IVIA_Lab___Semester_Project%20(3).pdf)

**Proposal main goals:**
- Reliably reach goals in increasingly complex environments, up to hospital blueprints.
- Reach a sequence of goals.
- Simulate many agent types (doctors, nurses, visitors).
- Investigate design decisions: action space, FOV, architecture, RL algorithm.

[Proposal PDF §1.4.1](/Users/emers/Downloads/Emerson_L__Azevedo_Aguiar___ETH_Zurich___IVIA_Lab___Semester_Project%20(3).pdf)

**Proposal curricula:**
- "Starting Position Progression": spawn on a circle around the goal and grow the radius R*.
- "Small Rewards Sprinkle": K small rewards spread through the maze.

[Proposal PDF §1.4.1 Curriculum Training](/Users/emers/Downloads/Emerson_L__Azevedo_Aguiar___ETH_Zurich___IVIA_Lab___Semester_Project%20(3).pdf)

**Proposal design-dimension table (Table 1.1):**

| Dimension | Range in the proposal |
|---|---|
| Field of view | 90–360° |
| Ray density | 0.1–1.0 rays/° |
| Ray range | local to full room |
| Action space | discrete/continuous; fixed vs learned speed |
| Memory | none / frame stack (2–16) / LSTM (hidden 64–512, BPTT 16–256) |
| Algorithm | PPO (clip 0.1–0.3) / SAC (α 0.01–0.5) |
| Curriculum | none / radius growth (r0 0.5–3 m, linear or success-triggered) |
| Reward | sparse +1; time penalty 0 to −1/T; collision penalty 0 to −0.5 |
| Architecture | depth 1–5 layers, width 64–512 |
| Environments | T-Maze, I-Maze, FourRoom, CrossRoom, FloorPlans |

[Proposal PDF Table 1.1](/Users/emers/Downloads/Emerson_L__Azevedo_Aguiar___ETH_Zurich___IVIA_Lab___Semester_Project%20(3).pdf)

**Proposal observation design:** a one-hot object type per ray (empty, wall, door, goal) plus "the distance from the observed position to the goal location". The implemented env has no door class and no goal-distance feature (see Q3). [Proposal PDF §1.4.1 Agent Design](/Users/emers/Downloads/Emerson_L__Azevedo_Aguiar___ETH_Zurich___IVIA_Lab___Semester_Project%20(3).pdf)

**Current-session request (3e28ab7d, 2026-10-29):**
- Find the most promising experiments to raise the chance of a workshop paper.
- Explicitly named: goal conditioning, and "similar rooms but only 1 has the goal (needs to learn to reach the goal via memory + information along the way)".
- A structured way to cut down the ablation: "test models on easier rooms up to middle difficulty, keep the best and go on".

[transcript 3e28ab7d…jsonl, user message 2026-10-29T11:08:42Z](/Users/emers/.claude/projects/-Users-emers-projects-semproj-rl-comparison/3e28ab7d-47d5-4892-bfe4-c8d3e63c4c8a.jsonl)

### Inferences
- Two ideas, "similar rooms" and "parameters until learning degrades", need no new learning machinery. The first needs room JSONs plus a colour/cue mechanism. The second needs only config sweeps over `obs_embed_hidden_sizes`, `backbone_hidden_sizes` and the head sizes, which the configs already expose.
- The aux-loss idea ("helps without extra parameters") is blocked by a design issue, not by missing code. The depth target is still part of the input (see Q2/Q3), so the depth aux loss currently cannot help by construction.

### Gaps
- I found no written reviewer or supervisor reaction to the five workshop ideas. The 20.10 notes list them but record no decision.
- I found no record of which workshop or venue was targeted.

## Q2. What results have been obtained so far, and what bugs and open problems were found?

### Takeaway
All results are qualitative and stored on W&B; the repo and chats contain no success-rate numbers. The findings so far:
- MLP encoder + MLP backbone works best.
- The LSTM struggles to learn the loop-closure aux loss on complex rooms.
- The depth aux loss gives almost no signal because depth is already an input.
- Loop closure helps once learned.
- Cross room is hard unless episodes are longer.

Several bugs were fixed along the way: CNN average pooling, agents spawning outside the room, AUC, unclamped LSTM actions, and the aux-loss plumbing.

### Cited Findings
**Results at the 20.10 meeting:**
- "MLP+MLP seems to work the best".
- "LSTM Backbone struggles to learn loop closure aux loss on complicated environments. Might require more layers".

[Weekly Progress PDF, Meeting 20.10](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)

**Findings at 12.10:**
- "The depth aux loss gives almost no learning signal, due to distance already being present". Proposed fix: keep the distances as labels but drop them from the observation.
- "Loop closure takes some iterations to learn, but once it does, it helps training."
- "Agents have a hard time learning in cross room". Idea: longer episodes.
- Early stopping should switch from eval return to success rate.

[Weekly Progress PDF, 12.10](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)

**Bugs found at 20.10:**
- The CNN encoder average-pooled over the ray dimension, "effectively removing features at certain directions". It now uses a sliding-window max pool.
- "Room Topology can make agents spawn outside". Fixed with hand-made spawn regions; this affected only C_Maze and CrossRoom.

[Weekly Progress PDF, Meeting 20.10](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf). The current CNN is Conv1d(3)→ReLU→Conv1d(3)→ReLU→MaxPool1d(2)→Flatten, which keeps the ray axis: [models/network_utils.py `build_cnn1d`](/Users/emers/projects/semproj-rl-comparison/models/network_utils.py)

**Earlier fixes:**
- LSTM actions were not clamped (31.08–07.09).
- LayerNorm/GroupNorm replaced BatchNorm in the CNN encoder because batch statistics drift during training.
- Observation embeddings are now updated only by the actor optimizer in SAC, to avoid exploding gradients.

[Weekly Progress PDF](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf); [git log 9a0c893, 2810f66, c8f9b87](/Users/emers/projects/semproj-rl-comparison/.git)

**Experiments run (W&B groups named in the notes):**
- `four_room_ablation` (5 seeds)
- `lstm_fix_check(_v2)`
- `ablation_sweep_test` (one_room, two_room, I and T mazes, four room, cross room)
- `lstm_tbptt` (BPTT windows)
- `aux_loss_test`
- `hard_rooms_v2`

[Weekly Progress PDF](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf); scripts: [scripts/ablation_sweep_test.sh:40-59](/Users/emers/projects/semproj-rl-comparison/scripts/ablation_sweep_test.sh), [scripts/test_truncdated_bptt.sh:40-59](/Users/emers/projects/semproj-rl-comparison/scripts/test_truncdated_bptt.sh), [scripts/test_auc_and_aux_loss.sh:40-60](/Users/emers/projects/semproj-rl-comparison/scripts/test_auc_and_aux_loss.sh), [scripts/hard_rooms.sh:40-76](/Users/emers/projects/semproj-rl-comparison/scripts/hard_rooms.sh)

**Last hard-room sweep:** `hard_rooms_v2` submitted 32 jobs (#0–#31: cross_room, C_maze, medium_maze, hard_maze × mlp/cnn × lstm/mlp × seeds 0 and 4; gamma 0.999, max_steps 4096, goal_radius 16). The launcher job was "CANCELLED AT 2026-10-20T10:27:48 DUE to SIGNAL Terminated", about 23 minutes after launch. [slurm-17612235.out (tail)](/Users/emers/projects/semproj-rl-comparison/slurm-17612235.out)

**AUC review (2026-10-09).** Claude flagged four problems:
1. Early stopping truncates the curve, so fast learners get a smaller AUC.
2. The x-axis counts evals, not env steps.
3. `None` values can enter the history.
4. `wandb.log` is called without a run.

Claude added `log_eval_auc`, which padded the curve with the last value and integrated over env steps. [transcript 072ca653…jsonl, 2026-10-09T16:00–16:03Z](/Users/emers/.claude/projects/-Users-emers-projects-semproj-rl-comparison/072ca653-d781-4db0-aaea-f40e670a48e6.jsonl)

**AUC as it stands now.** This was later replaced (commit 0892c45, "Simplified AUC computation"). The current code computes `np.trapezoid(success_rates)/(n_evals-1)` with unit spacing and no padding to the full budget, logged against `auc/seed` as the x-axis. In other words, it is the mean eval success rate over the evaluations that actually ran. [algorithms/ppo.py:399-416](/Users/emers/projects/semproj-rl-comparison/algorithms/ppo.py#L399). The 20.10 notes say "AUC now is normalized, such that runs that take more or less iterations have comparable values". [Weekly Progress PDF](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)

**Aux-loss session (2026-10-09).** The user implemented the depth and loop-closure aux losses from Mirowski et al. 2017. Claude explained that only the loop-closure head uses a softmax, and found these bugs:
- The 2-logit head was paired with a BCE loss.
- The depth target is part of the input.
- `compute_aux_loss` passed a dict, indexed the wrong dimension, and returned nothing.
- Loop labels were off by one, and the spawn position was never recorded.
- A `NameError` from the `h`→`hidden` rename.
- The agent class did not inherit the aux heads.
- `sample_mini_batch` did not pass `loop_labels`.
- `super().__init__` received `cfg` positionally.
- A broken `env_utils` import.
- The frame-stack buffer was missing `loopclo_buf`.

The user asked for a 1-logit sigmoid loop head and a sigmoid-bounded depth head, then for all fixes. The smoke test (8 envs × 32 steps × 3 updates on four_room) ran with aux on and off, with frame stacking, and with the CNN encoder. The aux loss fell from about 0.83 to 0.75, and 15–30% of the loop labels were positive. [transcript f65e7faf…jsonl, 2026-10-09T06:33–06:49Z](/Users/emers/.claude/projects/-Users-emers-projects-semproj-rl-comparison/f65e7faf-b624-4079-bee0-69dc8f52abb8.jsonl)

**Left open after the aux-loss session:**
- "Depth is still in the input… the depth head mostly reconstructs its own input. Until you remove that channel, and the no-hit channel 0, from the input, this auxiliary loss gives little training signal."
- "Checkpoints don't include the aux heads."
- `eta1: 20.0`, `eta2: 40.0` are guesses: rooms are about 100 units per cell and the agent moves up to 13 units per step.
- Whether depth can be inferred from Perlin colours on a 1-D ray cast "is worth checking".

[transcript f65e7faf…jsonl, 2026-10-09T06:34Z and 06:49Z](/Users/emers/.claude/projects/-Users-emers-projects-semproj-rl-comparison/f65e7faf-b624-4079-bee0-69dc8f52abb8.jsonl); [configs/head/auxiliary_head/base.yaml:1-16](/Users/emers/projects/semproj-rl-comparison/configs/head/auxiliary_head/base.yaml)

**The depth target is still the input channel in the current code.** The MLP and recurrent aux losses both regress `rays[..., 3, :]`. No code masks channel 3 out of the policy input. [algorithms/ppo.py:222-236](/Users/emers/projects/semproj-rl-comparison/algorithms/ppo.py#L222), [algorithms/ppo.py:545-566](/Users/emers/projects/semproj-rl-comparison/algorithms/ppo.py#L545), [models/embeddings/simple.py:18-22](/Users/emers/projects/semproj-rl-comparison/models/embeddings/simple.py#L18)

**Seeds used:**
- ablation_sweep_test, the four_room tests, tbptt and aux: 5 seeds (0–4).
- hard_rooms_v2: 2 seeds (0, 4).
- Metrics: 10 runs × 10 episodes per seed (`completion_rate_eps: 10`, `mean_of_means_runs: 10`).

[scripts/hard_rooms.sh:42-43](/Users/emers/projects/semproj-rl-comparison/scripts/hard_rooms.sh#L42), [configs/env/base_env.yaml:25-26](/Users/emers/projects/semproj-rl-comparison/configs/env/base_env.yaml#L25)

### Inferences
- The current AUC normalizes each run by its own number of evals and does not pad to `n_iterations`, so it measures the mean success over the run's own curve. A fast learner that early-stops at 95% and a slow learner that crawls to 95% over 1000 iterations can get similar or reversed AUCs. If sample efficiency is a paper claim, a budget-padded AUC (Claude's 10-09 version) or "env steps to X% success" would be more defensible.
- Pieces exist to probe the LSTM-vs-MLP gap (BPTT window, number of LSTM layers, the NavA3C-style two-LSTM class), but only the BPTT sweep has been run. On the current rooms the goal is fixed and the start is random, so a reactive MLP policy can memorize a per-room map from Perlin colours. That may explain why MLP+MLP wins: the tasks do not require memory.

### Gaps
- I found no numeric success rates, AUCs, DTW or NPL values. They live on W&B (links in the PDF) and in Euler scratch (`/cluster/scratch/eazevedo/semproj/...`), which is not local.
- I found no logged result for the `hard_rooms_v2` sweep. The local slurm log ends with the launcher being cancelled.
- Results of the BPTT-window sweep and the frame-stacking sweep are not recorded outside W&B.

## Q3. How is the observation built? Is the goal observed, fixed or randomized? What would goal conditioning and cue/colour objects take?

### Takeaway
The observation is 37 rays × 7 channels plus 2 proprioceptive values:
- channels 0–2: one-hot no-hit / goal / wall
- channel 3: normalized distance
- channels 4–6: Perlin RGB from a global-position colour field seeded by `cfg.seed`
- proprio: last speed and last turn

There is a single goal per room, read from the JSON and shared by all envs. It is observed only when a ray passes within `goal_radius` of it in line of sight. There is no goal vector, no goal randomization, no per-wall object type and no per-wall colour.

Goal conditioning and cue rooms are both feasible. Each needs changes in about four to six places: the env, the ray cast, the configs, the buffers/observation size, and the navmesh reference used by the metrics.

### Cited Findings
**Observation layout.** The docstring is at [envs/navigation_env.py:20-30](/Users/emers/projects/semproj-rl-comparison/envs/navigation_env.py#L20):
- channel 0 no_hit
- channel 1 goal_hit (ray-circle test)
- channel 2 wall_hit
- channel 3 normalized distance in [0, 1]
- channels 4–6 Perlin RGB at the hit point
- proprio: last speed, last turning

It is built in `get_observations` at [envs/navigation_env.py:202-240](/Users/emers/projects/semproj-rl-comparison/envs/navigation_env.py#L202).

**Ray count.** Rays = int(fov × ray_density), made odd: with fov 180 and density 0.2 that is 37 rays. `ray_encoding: 7`, `proprio_dim: 2`, `num_classes: 3`. [utils/geometry.py `compute_num_rays`](/Users/emers/projects/semproj-rl-comparison/utils/geometry.py), [configs/env/base_env.yaml:4-16](/Users/emers/projects/semproj-rl-comparison/configs/env/base_env.yaml#L4)

**Encoders.**
- MLP: flattens rays and proprio into one vector, 7×37+2 = 261 inputs. [models/embeddings/simple.py:7-22](/Users/emers/projects/semproj-rl-comparison/models/embeddings/simple.py#L7)
- CNN: 1-D conv over rays plus a proprio MLP, then fusion. [models/embeddings/simple.py:25-60](/Users/emers/projects/semproj-rl-comparison/models/embeddings/simple.py#L25)

**Goal handling.**
- The goal position is a single (2,) tensor read from the room JSON `goal_pos`. [envs/navigation_env.py:61-66](/Users/emers/projects/semproj-rl-comparison/envs/navigation_env.py#L61), [utils/geometry.py `start_goal_pos_from_json`](/Users/emers/projects/semproj-rl-comparison/utils/geometry.py)
- It is broadcast to all envs as `goal_pos[None, None, :]` in the goal ray test. [envs/navigation_env.py:213-219](/Users/emers/projects/semproj-rl-comparison/envs/navigation_env.py#L213)
- Termination is `dist_to_goal <= goal_radius`. [envs/navigation_env.py:190-191](/Users/emers/projects/semproj-rl-comparison/envs/navigation_env.py#L190)
- The goal is never resampled in `reset`. [envs/navigation_env.py:139-158](/Users/emers/projects/semproj-rl-comparison/envs/navigation_env.py#L139)

**Goal radius has two jobs.** `goal_radius` (default 4.0; sweeps use 16) is both the success threshold and the visible disk for the goal ray test. [configs/env/base_env.yaml:23](/Users/emers/projects/semproj-rl-comparison/configs/env/base_env.yaml#L23), [scripts/hard_rooms.sh:47](/Users/emers/projects/semproj-rl-comparison/scripts/hard_rooms.sh#L47)

**Start handling.**
- With `rand_pos: "True"` (the default), every reset samples a uniform position from the room's `spawn_regions` rectangles, or from the bounding box shrunk by WALL_OFFSET=10 if there are none. Heading is uniform in [0, 2π). [envs/navigation_env.py:55, 147-151](/Users/emers/projects/semproj-rl-comparison/envs/navigation_env.py#L147), [utils/geometry.py `InitialPositionSampler`](/Users/emers/projects/semproj-rl-comparison/utils/geometry.py), [configs/env/base_env.yaml:18](/Users/emers/projects/semproj-rl-comparison/configs/env/base_env.yaml#L18)
- Final metric evaluation switches to the fixed JSON start. [scripts/train.py:103-111](/Users/emers/projects/semproj-rl-comparison/scripts/train.py#L103)

**Colour field.**
- `PerlinColor(seed=cfg.seed)`: three fBm noise channels over global (x, y) with scale 0.15 and 4 octaves. [envs/navigation_env.py:68](/Users/emers/projects/semproj-rl-comparison/envs/navigation_env.py#L68), [envs/env_utils.py:93-178](/Users/emers/projects/semproj-rl-comparison/envs/env_utils.py#L93)
- Colour is sampled at the ray hit point, at the goal centre for goal rays, and is black for no-hit rays. [envs/navigation_env.py:228-232](/Users/emers/projects/semproj-rl-comparison/envs/navigation_env.py#L228)

**Ray cast keeps only the distance.** `RayCast.intersect` takes the min over walls and throws away which wall was hit (`min_t, _ = t_hits.min(dim=-1)`). [envs/env_utils.py:64-70](/Users/emers/projects/semproj-rl-comparison/envs/env_utils.py#L64)

**Reward and dynamics.**
- Reward is −1/max_steps per step and +1 at the goal, with no collision penalty. Movement is clipped at walls by intersecting the motion ray. [envs/navigation_env.py:130-137, 175-182](/Users/emers/projects/semproj-rl-comparison/envs/navigation_env.py#L130)
- `NavigationEnvEasy` adds sin/cos heading to the proprio (proprio_dim 4), a potential-based distance-to-goal shaping term, and a fixed start and heading. It is used only to check against SB3. [envs/navigation_env.py:365-454](/Users/emers/projects/semproj-rl-comparison/envs/navigation_env.py#L365), [configs/env/easy_env.yaml](/Users/emers/projects/semproj-rl-comparison/configs/env/easy_env.yaml)
- Action: turn in [−1, 1] × half-FOV (±90°), speed in [0, 1] × 13 units. [configs/env/base_env.yaml:5-13](/Users/emers/projects/semproj-rl-comparison/configs/env/base_env.yaml#L5), [envs/navigation_env.py:172-173](/Users/emers/projects/semproj-rl-comparison/envs/navigation_env.py#L172)

**Metrics reference path.** The reference shortest path is built only from the JSON start and goal (Delaunay triangulation → A* → funnel). [utils/navigation_mesh.py:282-297](/Users/emers/projects/semproj-rl-comparison/utils/navigation_mesh.py#L282), [utils/metrics.py:243-286](/Users/emers/projects/semproj-rl-comparison/utils/metrics.py#L243). Commit 00dfab6 describes the navmesh script as able "to compute shortest path trajectories from arbitrary start and end goal positions", but the entry point reads only the JSON. [git log 00dfab6](/Users/emers/projects/semproj-rl-comparison/.git)

**Planned features not wired in:**
- `envs/curriculum.py` is empty (0 lines).
- `observation.grid_cell.enabled` appears only in the run name and tags. [scripts/train.py:40-47](/Users/emers/projects/semproj-rl-comparison/scripts/train.py#L40)
- The grid-cell network is a separate supervised model with its own env and config, and the PPO factory never builds it. [models/embeddings/grid_cell.py](/Users/emers/projects/semproj-rl-comparison/models/embeddings/grid_cell.py), [envs/traj_gen_env.py](/Users/emers/projects/semproj-rl-comparison/envs/traj_gen_env.py), [configs/train_grid_cell.yaml](/Users/emers/projects/semproj-rl-comparison/configs/train_grid_cell.yaml), [utils/factory.py:60-70](/Users/emers/projects/semproj-rl-comparison/utils/factory.py#L60)
- A NavA3C-style two-LSTM backbone class exists, but the factory builds only `SimpleLSTM`. [models/backbones/lstm_backbone.py:46-71](/Users/emers/projects/semproj-rl-comparison/models/backbones/lstm_backbone.py#L46), [utils/factory.py:64-69](/Users/emers/projects/semproj-rl-comparison/utils/factory.py#L64)
- `train.py` always calls `create_ppo_agent`. [scripts/train.py:72](/Users/emers/projects/semproj-rl-comparison/scripts/train.py#L72)

### Inferences
**Goal visibility.** The goal is not hidden: it is observable whenever it is in the FOV and in line of sight, through channel 1 and its Perlin colour. A large `goal_radius` (16) makes it easier to see. The task is "undirected search with a visible goal beacon", not pure undirected search.

**Colour cue is seed-dependent.** The Perlin field is seeded by `cfg.seed`, so each training seed sees a different colour map of the same room. Because colour depends on global position, it acts as a strong place code that a memoryless MLP can exploit for localization. Two consequences:
- It is a plausible reason MLP+MLP wins.
- Rooms that are geometrically identical are not visually aliased unless Perlin is disabled, replaced with room-local or periodic colours, or overridden per wall.

**Steps to add goal conditioning (relative goal vector):**
- (a) Make `goal_pos` an (E, 2) tensor and resample it in `reset(done)` from spawn regions or free space, with a minimum distance from the start.
- (b) Add an egocentric goal feature such as distance/max_range, sin and cos of the relative bearing, or a goal ray. Do this by extending `proprio`, plus `proprio_dim` in `base_env.yaml` and the `observation_space`.
- (c) Update the per-env goal broadcasting at lines 213-229 and the termination check.
- (d) For metrics, generalize `generate_reference_trajectory` to take a start and goal instead of reading the JSON, and evaluate over sampled start-goal pairs.
- (e) Optionally add a "goal-visible" ablation that masks channel 1.

The buffers already size off `proprio_dim`, so the change is moderate and confined mostly to `navigation_env.py`, `navigation_mesh.py`, `metrics.py` and the env config. This is an estimate from reading the code, not a tested change.

**Steps to add cue or colour object types:**
- (a) Add per-edge attributes in the room JSON, such as `"type": "cue"` or `"color": [r, g, b]`.
- (b) Return the argmin wall index from `RayCast.intersect`.
- (c) Add one-hot channels, raising `ray_encoding` and `num_classes`, or override the RGB channels with per-wall colours.
- (d) Add a flag to disable or replace the Perlin colours.

For "two identical rooms, one with the goal, a cue on the way", the goal must also be randomized between the two rooms per episode, with the cue colour indicating which room. Otherwise a fixed goal is solvable without memory. That needs per-episode goal sampling from a discrete set of goal positions, which is the same machinery as goal conditioning item (a).

### Gaps
- Not verified by running code: the parameter counts per architecture. hydra is not installed in the local system Python, so I could not instantiate the agents. A rough hand estimate from the configs: MLP encoder [64, 64] on 261 inputs ≈ 25k parameters; 2×64 MLP backbone ≈ 8k; 2-layer 64-unit LSTM ≈ 66k; policy and value heads [64, 64] ≈ 8k each. That puts MLP+MLP at roughly 50k parameters and MLP+LSTM at roughly 110k. These are unverified estimates.

## Q4. How expensive is a training run, and how big are the current sweeps?

### Takeaway
One PPO iteration is 256 envs × 256 steps = 65,536 env steps. The cap is 1000 iterations, about 65.5M steps, with early stopping once the EMA of eval success reaches at least 0.95 and plateaus for 5 evals. Runs are CPU-only on Euler: 1 CPU and 2 GB per job, about 16 parallel jobs.

The 4096-step-episode hard-room runs take about 30 s per iteration, so up to about 8.3 h, and need a 720-min timeout. The current sweep scripts are 20–120 jobs each; the full cross product is much larger.

### Cited Findings
**PPO settings:**
- num_steps 256, num_envs 256, n_iterations 1000, max_steps (episode length) 512
- n_epochs 2, mini_batch_size 256
- lr 3e-4, linearly annealed
- entropy 0.005, target_kl 0.04
- eval every 5 iterations on 50 eval envs
- early stopping: patience 5, min_delta 0.01, EMA α 0.2, min_converged_value 0.95 on eval success rate

[configs/algorithm/ppo.yaml:11-79](/Users/emers/projects/semproj-rl-comparison/configs/algorithm/ppo.yaml#L11), [configs/env/base_env.yaml:20-21](/Users/emers/projects/semproj-rl-comparison/configs/env/base_env.yaml#L20)

**Memory settings:**
- LSTM: 2 layers, 64 hidden, BPTT window 16 by default; the sweep covers 16, 64, 128 and 256.
- Frame stack size 8.

[configs/backbone/lstm_backbone.yaml](/Users/emers/projects/semproj-rl-comparison/configs/backbone/lstm_backbone.yaml), [configs/observation/frame_stack/base.yaml](/Users/emers/projects/semproj-rl-comparison/configs/observation/frame_stack/base.yaml), [scripts/test_truncdated_bptt.sh:55](/Users/emers/projects/semproj-rl-comparison/scripts/test_truncdated_bptt.sh#L55)

**Euler resources:**
- Launcher: submitit SLURM, timeout_min 300, cpus_per_task 1, mem_per_cpu 2G, gpus 0. [configs/hydra/launcher/euler.yaml:7-16](/Users/emers/projects/semproj-rl-comparison/configs/hydra/launcher/euler.yaml#L7)
- `train.py` forces `device = "cpu"`. [scripts/train.py:24-28](/Users/emers/projects/semproj-rl-comparison/scripts/train.py#L24)
- "Using CPU only, it manages 16 Tasks simultaneously". [Weekly Progress PDF, 17.08–24.08](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)

**Hard-room run time.** "EPISODE_LENGTH=4096 runs take ~30 s/iteration, so 1000 iterations plus the final metrics don't fit euler.yaml's 300 min", hence `timeout_min=720`. That sweep also uses gamma 0.999. [scripts/hard_rooms.sh:48-65](/Users/emers/projects/semproj-rl-comparison/scripts/hard_rooms.sh#L48). Frame-stacking runs were skipped on hard rooms because "these take long for each run". [Weekly Progress PDF, 12.10](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)

**Sweep sizes:**

| Script | Rooms | Configs | Seeds | Jobs |
|---|---|---|---|---|
| `ablation_sweep_test.sh` | 6 (one, T, I, two, four, cross) | obs {mlp, cnn} × backbone {mlp, lstm} | 5 | 120 |
| `hard_rooms.sh` | 4 (cross, C, medium, hard) | 2 × 2 | 2 | 32 |
| `test_auc_and_aux_loss.sh` | four_room | 2 × 2 × frame_stack {F, T}, aux on | 5 | 40 |
| `test_frame_stacking.sh` | four_room | 2 × 2 | 5 | 20 |
| `test_truncdated_bptt.sh` | four_room | 4 BPTT windows, MLP encoder + LSTM | 5 | 20 |

[scripts/ablation_sweep_test.sh:51-56](/Users/emers/projects/semproj-rl-comparison/scripts/ablation_sweep_test.sh#L51), [scripts/hard_rooms.sh:41-46](/Users/emers/projects/semproj-rl-comparison/scripts/hard_rooms.sh#L41), [scripts/test_auc_and_aux_loss.sh:51-57](/Users/emers/projects/semproj-rl-comparison/scripts/test_auc_and_aux_loss.sh#L51), [scripts/test_frame_stacking.sh:51-56](/Users/emers/projects/semproj-rl-comparison/scripts/test_frame_stacking.sh#L51), [scripts/test_truncdated_bptt.sh:51-56](/Users/emers/projects/semproj-rl-comparison/scripts/test_truncdated_bptt.sh#L51)

**Final evaluation per seed.** After training, each seed loads `best.pt`, records a 200-step video, and runs CR/DTW/NPL over 10 runs × 10 episodes. `evaluate_metrics.py` then builds the mean-of-means plots and a W&B summary table with 95% CI half-widths. [scripts/train.py:86-111](/Users/emers/projects/semproj-rl-comparison/scripts/train.py#L86), [scripts/evaluate_metrics.py:27-63](/Users/emers/projects/semproj-rl-comparison/scripts/evaluate_metrics.py#L27)

### Inferences
- Upper bound for one hard-room run: about 1000 iterations × 30 s ≈ 8.3 h CPU. The default 512-step rooms are presumably much cheaper per iteration, but I found no figure for them.
- A full factorial over the implemented switches is roughly 2 encoders × 3 memory options × 4 BPTT windows (LSTM only) × 2 aux settings × 10 rooms × 5 seeds, which is hundreds to over a thousand jobs. With about 16 concurrent CPU jobs this supports the user's plan for a staged, successive-halving style protocol.
- `one_room_run.sh`, `two_room_run.sh` and `euler_multirun_gpu_test.sh` still override `env.goal_pos.x/y`. That key no longer exists in `base_env.yaml` (the goal now comes from JSON), so these scripts are likely stale and would fail in Hydra's strict mode. I have not run them to check. [scripts/one_room_run.sh:22-23](/Users/emers/projects/semproj-rl-comparison/scripts/one_room_run.sh#L22), [configs/env/base_env.yaml](/Users/emers/projects/semproj-rl-comparison/configs/env/base_env.yaml)

### Gaps
- I found no wall-clock time per iteration for the default 512-step rooms, and no record of how many iterations runs needed before early stopping.
- GPU timing is unknown. A GPU test script exists, but no result is recorded.

## Q5. What is the room JSON format, and how hard is it to author new rooms (two identical rooms, cue rooms)?

### Takeaway
Rooms are hand-written JSON:
- a list of wall segments, `edges: [{from: {x, y}, to: {x, y}}]`
- a `start_pos` and a `goal_pos`
- optional axis-aligned `spawn_regions` (lists of edges forming rectangles)

Authoring two identical rooms is easy geometrically: copy the edges with an offset. The schema has no per-wall type or colour, no multiple goals, and no cue objects. An Arxitect hospital-floorplan converter exists.

### Cited Findings
**Schema.** `edges` with `from`/`to` points; `start_pos`, `goal_pos`; optional `spawn_regions`. [rooms/two_room.json](/Users/emers/projects/semproj-rl-comparison/rooms/two_room.json), [utils/geometry.py `walls_json_to_numpy`, `start_goal_pos_from_json`, `get_spawn_regions_json`, `region_to_box`](/Users/emers/projects/semproj-rl-comparison/utils/geometry.py)

**Room inventory** (edges; bbox in world units; start → goal; spawn regions):

| Room | Edges | Bbox | Start → goal | Spawn regions |
|---|---|---|---|---|
| one_room | 4 | 400×400 | (360,360) → (20,20) | 0 |
| two_room | 6 | 400×400 | (360,360) → (20,20) | 0 |
| I_maze | 7 | 400×400 | (180,280) → (220,280) | 0 |
| T_maze | 6 | 400×400 | (180,120) → (220,120) | 0 |
| four_room | 14 | 600×600 | (560,560) → (40,40) | 0 |
| cross_room | 20 | 800×800 | (320,20) → (320,520) | 5 |
| C_maze | 20 | 800×500 | (750,150) → (50,350) | 5 |
| medium_maze | 24 | 1200×400 | (1140,140) → (400,280) | 0 |
| hard_maze | 68 | 1200×540 | (60,40) → (460,340) | 0 |

`test_room` (100×100) and `grid_cell_training` (2500×2500) have no start or goal. Source: parsed from [rooms/*.json](/Users/emers/projects/semproj-rl-comparison/rooms/)

**Scale.** Rooms are about 100 world units per cell and the agent moves at most 13 units per step. [configs/head/auxiliary_head/base.yaml:12](/Users/emers/projects/semproj-rl-comparison/configs/head/auxiliary_head/base.yaml#L12)

**Spawn regions.** They were added by hand because room topology let agents spawn outside the room in C_maze and cross_room. [Weekly Progress PDF, Meeting 20.10](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf), [git log a427abf](/Users/emers/projects/semproj-rl-comparison/.git)

**Floorplan converter.** `convert_hospital_json_format` converts Arxitect `wallGraph` node/edge JSON into this schema. [utils/geometry.py `convert_hospital_json_format`](/Users/emers/projects/semproj-rl-comparison/utils/geometry.py)

**Room provenance.** The harder rooms (C-, medium- and hard-maze) were modelled on Mirowski et al., "Learning to Navigate in Complex Environments". [Weekly Progress PDF, 12.10](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)

### Inferences
- New rooms take minutes to write by hand or with a short generator script, since edges are plain line segments. A procedural generator (for example random grid mazes) would be a small script plus spawn-region boxes; none exists today.
- For a "similar rooms + cue" experiment the geometry is easy. The hard parts are in Q3:
  - per-episode goal choice between the rooms
  - removing the global Perlin place code, or giving the aliased rooms identical colours
  - adding a cue object type or colour to the ray observation
- Without these, identical-geometry rooms are not truly aliased, because their Perlin colours differ by global position.
- Bounding-box spawning is used for rooms without `spawn_regions`. In rooms with interior walls it can place agents inside closed-off pockets or right next to walls, and new rooms should define spawn regions explicitly.

### Gaps
- No room or maze generator exists in the repo. Whether the Arxitect hospital floorplans have been converted and tested is not recorded: there are no hospital JSONs in `rooms/`.
