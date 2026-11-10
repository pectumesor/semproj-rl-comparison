# Final paper idea and experiment plan

*Status: 2026-10-29. This plan supersedes the experiment list in [Workshop paper experiments RL navigation.md](Workshop%20paper%20experiments%20RL%20navigation.md). It keeps that report's code fixes, staged protocol and venue analysis, and adds the twin-room design, the supervisor's goal-ray idea and the hand-crafted grid code.*

## 1. The idea in one sentence

**A navigation agent can get the information it needs in three ways: it can *perceive* it, *remember* it, or *be told* it. The design choices in the project title (observation space, architecture, auxiliary losses, action space, reward) are mostly choices about which of these routes the agent uses. So whether a choice helps depends on what the other routes already provide.**

Working title: **"Perceive, Remember, or Be Told: How Design Choices Trade Off in RL Navigation Agents"**

This turns the project from "which configuration wins" (a leaderboard, which reviewers dislike and the current results don't support) into a conditional, testable claim. It also explains the current result instead of hiding it: the memoryless MLP wins because, in today's rooms, it can *perceive* everything it needs (the global Perlin colour works as a place code, and the goal is fixed).

## 2. The gaps it fills (framing for the introduction)

1. **Design choices are studied in isolation, in incompatible setups.** Mirowski et al. 2017, Banino et al. 2018, Wijmans et al. 2023 and Habitat PointNav each change one factor in a different environment. Nobody compares them under one controlled egocentric setup. (This is the proposal's original novelty claim, which stays.)
2. **"Memoryless wins" results are often confounded by information leaks.** Ni et al. 2023 show many "memory" benchmarks need little memory. We show the same thing happening by accident in a floorplan benchmark, and how to control it.
3. **Memory demand is rarely a controllable dial in navigation.** We make it one: cue-to-decision distance *L* in visually identical twin rooms.
4. **Goal and position information are usually given as a fixed modality.** We vary *how much* the agent is told (nothing / always / last seen / noisy) and in what format (vector, raw xy, Fourier features, hand-crafted grid code). This includes a cheap, training-free alternative to Banino's learned grid cells.
5. **Application gap:** occupant and wayfinding simulators in architecture are mostly heuristic (cogARCH, SimArch). No controlled study tells a practitioner which RL agent design to use.

**Contributions as they'd appear in the paper:**
- A controlled 2D ray-cast navigation testbed with independent dials for perception (colour, field of view), memory demand (*L*), goal information and position information. Released with configs.
- Evidence that memory, goal information and position codes act as **substitutes**: each helps only when the others are missing. Identical rooms show sharp failure points at each architecture's memory horizon.
- A staged, budget-aware ablation protocol (easy → hard rooms with explicit promotion rules) and a check of where cheap easy-room screening agrees with hard-room results.

## 3. Hypotheses (pre-register before running)

| ID | Hypothesis | Tested in |
|---|---|---|
| H1 | MLP's lead over LSTM/GRU shrinks or flips when the global colour place code is removed and goals are randomised | E1 |
| H2 | In twin rooms, each backbone fails once *L* exceeds its horizon: MLP at chance (0.5) for all *L*; frame stack *k* fails for *L* > *k*; LSTM fails once *L* goes well past the BPTT window; GRU lasts longest | E2 |
| H3 | Telling the agent the goal (oracle goal ray) or its position (xy / grid code) closes the MLP–recurrent gap in open layouts but not in dead-end layouts | E3 |
| H4 | "Last seen, frozen" goal and "noisy path-integrated" position bring back the recurrent advantage. "Last seen, updated by odometry" doesn't, because the environment does the remembering. | E3 |
| H5 | The hexagonal grid code is no better than random Fourier features of the same size on training rooms (consistent with Bartlett et al. 2022 after tuning). If it helps anywhere, it's on held-out layouts. | E3 (+ stretch) |
| H6 | Perception, capacity, action and reward choices rank consistently from easy to hard rooms; memory and aux-loss choices don't | E4 |
| H7 | Memory gain grows with a room's perceptual aliasing score and as the field of view narrows | E1a, E1b |

Negative outcomes are publishable too (ICBINB-style workshops reward them), as long as each test is clean.

## 4. The experiments

### Metrics and statistics (shared by all experiments)

- **Primary:** success AUC over env steps, **padded to the full budget**, so early-stopped and slow runs are scored on the same horizon.
- **Secondary**, converted so that higher is better and everything lies in [0, 1], which lets IQM average across rooms:
  - success rate;
  - **SPL** = success × shortest / max(actual, shortest) (Anderson et al. 2018), replacing normalised path length;
  - **nDTW** = exp(−DTW / (|R| · d_th)) (Ilharco et al. 2019, "General Evaluation for Instruction Conditioned Navigation"), with d_th the success radius and |R| the number of reference points. It is a published normalisation and needs no random-walk baseline.
- **Evaluation scheme:** keep your supervisor's 10 eval runs × 10 episodes per seed, and report per-seed means and standard errors as asked. In the paper, state that the **statistical unit is the training seed**, not the eval repeat.
- **Aggregation:** IQM with stratified-bootstrap 95% CIs, performance profiles and probability of improvement (rliable). Main-figure claims use Welch's t at α = 0.01 with Holm correction.
- **Recurrent backbone rule (pre-register it):** LSTM is the default recurrent backbone everywhere. GRU always runs in E2, the one task where memory is guaranteed to matter, as a head-to-head check. Elsewhere, GRU runs on a room or condition **only if** LSTM's IQM is below MLP's there, or below the leader in E3, with non-overlapping CIs. Every triggered case is reported, whatever the outcome, as a robustness check. Never report "recurrent = best of LSTM and GRU", because that gives recurrence a second chance only where it lost.
- **Seed sets:** tuning/screening seeds (0–2) never appear in reported results. E1, E1b, E2 and E3 therefore run on the stage-2 set 100–104 (the E0 pilot keeps seed 0). Comparisons that are still undecided after 10 seeds get extra seeds by a sequential rule (AdaStop, arXiv 2306.10882) instead of ad hoc reruns. The reasons are in "Why these seed sets" below.

#### Why these seed sets

There are three disjoint sets, each fixed before any run:

| Set | Seeds | Used for | Reported? |
|---|---|---|---|
| Tuning / screening | 0–2 (E0 pilot: 0) | learning-rate pilot, E4 Stage 1 | No. These choose configs; they never support claims. |
| Confirmation | 100–104 | E1, E1b, E2, E3, E4 Stage 2 | Yes, with CIs |
| Final claims | 1000–1009 | E4 Stage 3 and every main-figure claim that needs α = 0.01 | Yes |

1. **Seed variance alone can change conclusions.** Henderson et al. ("Deep Reinforcement Learning that Matters", AAAI 2018, arXiv 1709.06560) showed that two groups of 5 seeds of the *same* algorithm and hyperparameters can produce significantly different learning curves. A result therefore has to be a distribution over several seeds with uncertainty attached, and the training seed is the statistical unit. Agarwal et al. ("Deep RL at the Edge of the Statistical Precipice", NeurIPS 2021, arXiv 2108.13264) make the same point for few-run studies and motivate IQM with stratified-bootstrap CIs.
2. **Choosing and reporting on the same seeds inflates the winner.** A config picked because it scored highest on some seeds will, on average, score lower on new seeds: part of its lead was luck on those seeds. Patterson et al. ("Empirical Design in Reinforcement Learning", JMLR 2024, arXiv 2304.01315) call this the designer's curse. Their two-stage protocol (sweep, then re-run the selected config on fresh seeds) is "a reasonable place to start", though still slightly overconfident. Eimer, Lindauer & Raileanu ("Hyperparameters in Reinforcement Learning and How To Tune Them", ICML 2023, arXiv 2306.01324) find that RL hyperparameter landscapes depend strongly on the tuning seed, and recommend separate tuning and test seeds, as in AutoML. That is why seeds 0–2, which drive selection (learning-rate pilot, Stage-1 promotion), never appear in reported numbers, and why each later stage draws new seeds. E1, E1b, E2 and E3 are hypothesis tests whose outcome is reported, so they belong in the confirmation set, not the screening set.
3. **Why 3, then 5, then 10 seeds.** Colas, Sigaud & Oudeyer ("A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms", 2019, arXiv 1904.06979) find by power analysis that Welch's t-test at α = 0.01 needs about 5–10 seeds per condition for 80% power on a large effect, and about 20 for a medium one. So:
   - 3 seeds are enough only to *screen*, with a conservative promotion rule (keep everything whose CI overlaps the leader);
   - 5 seeds can confirm large effects such as memory versus no memory in the twin rooms;
   - claims at α = 0.01 use 10 seeds.
   
   If a comparison is still undecided, AdaStop (Mathieu et al., arXiv 2306.10882) adds seeds sequentially while controlling the error rate. Adding seeds by hand until p drops below the threshold is a form of p-hacking.
4. **The numbers themselves are arbitrary; fixing them in advance is what matters.** Any disjoint sets would do. Henderson et al. warn against reporting a "top-N" subset of seeds. Writing the sets into the plan and the sweep scripts before running (the pre-registration commit) rules out picking seeds after seeing results. The spacing (0…, 100…, 1000…) just makes a run's role readable from its folder name and W&B tags.
5. **What the seed controls here.** The RL seed sets network initialisation, spawn and goal sampling, and action sampling. The colour field has its own `env.colour_seed`, fixed across RL seeds, so seed-to-seed variance isn't mixed with a different colour map per seed (see E0).

All experiments share one **base config**: MLP encoder, Perlin colour off (or room-local, depending on E1), goal randomised per episode, padded AUC as the main metric, and the current PPO hyperparameters. Learning rate gets a 3-value pilot per backbone (Andrychowicz et al. tune learning rate in every theme).

### E0: Fixes and pilot (week 1, required)

Without these, later results aren't valid (details and line references in the earlier report):

- [ ] `colour_mode ∈ {global, off, room_local, per_wall}` plus a separate `colour_seed`, fixed across RL seeds
- [ ] Ray cast returns the index of the wall it hits; room JSON supports per-edge `colour`, so twin rooms can share the **same hard-coded colours**
- [ ] Per-episode goal sampling (goal tensor of shape (E, 2)); navmesh reference path for any start/goal pair
- [ ] AUC padded to the full step budget (carry the last value forward)
- [ ] Depth aux: remove the depth channel from the policy input when depth aux is on
- [ ] Separate `success_radius` from `goal_visual_radius`
- [ ] Fixed-size "information slots" in the proprio vector, zeroed when unused:
  - goal: `(d, cos θ, sin θ, seen_flag, age)`
  - position: `xy` / `RFF` / `grid`
- [ ] GRU backbone; spawn regions for medium_maze and hard_maze; delete the stale scripts
- [x] **Pilot:** one run per tier → real hours per run. **Every budget below scales with this.**
  - Run Nov 1 (`scripts/e0_pilot.sh`, W&B group `e0_pilot`, storage timestamp 20261101160538): two_room, four_room and medium_maze × MLP/LSTM × lr {1e-4, 3e-4, 1e-3}, seed 0, colour off, random goal, 1 CPU per job.
  - **Cost is set by iterations, not by the room:** 256 envs × 256 steps per iteration take 5–8 s (MLP) and 7–10 s (LSTM) in every room. A run that uses the full 1000 iterations costs ≈ 2.1 h (MLP) or ≈ 2.8 h (LSTM); early stopping is what makes easy rooms cheap.
  - two_room: every config reached 100% success and stopped early at 135–155 iterations, 15–19 min per run. four_room: same, at 115–220 iterations, 11–26 min.
  - medium_maze: 1.9–2.1 h (MLP) and 2.5–2.6 h (LSTM) for 1000 iterations; LSTM at lr 3e-4 stopped early at 605 iterations (1.7 h).
  - **Learning rate:** LSTM 3e-4 (medium_maze AUC 0.81 and 100% success, vs 0.38 at 1e-3 and 0.03 at 1e-4). MLP 1e-4 (AUC 0.57, vs 0.43 at 3e-4 and 0.21 at 1e-3). Its eval success swings between 0.3 and 0.8, so this pick is weak. Every config scored AUC 0.92–1.0 in two_room and four_room, so those rooms don't separate learning rates. One seed (screening set), not reportable.
  - **Ceiling:** with colour off and a random goal, two_room and four_room saturate (padded AUC ≥ 0.92 for all 12 configs). Tier-1 screening (E4 Stage 1) needs rooms that don't saturate or a shorter budget, otherwise the promotion rule keeps everything.
- [x] **Room probe** (Nov 1–2, `scripts/e0_room_probe.sh`, W&B group `e0_room_probe`): the pilot's best MLP (lr 1e-4) and LSTM (lr 3e-4), seed 0, on the remaining rooms.

  | Room | Episode | MLP AUC | LSTM AUC | Goal in view at start | Median path to goal | Time per run |
  |---|---|---|---|---|---|---|
  | two_room / T_maze / I_maze | 512 | 0.96–0.99 | 0.99–1.00 | 53% (two_room) | 265 | 15–19 min |
  | four_room | 512 | 0.99 | 1.00 | 52% | 328 | 11–26 min |
  | cross_room | 4096 | 1.00 | 1.00 | 51% | 328 | 17–19 min |
  | C_maze | 4096 | 0.93 | 0.96 | 16% | 721 | 26–27 min |
  | medium_maze | 4096 | 0.57 | 0.81 | 12% | 1124 | 1.9–2.6 h |
  | hard_maze | 4096 | 0.18 | 0.00 | 7% | 685 | 2.8–3.2 h |

  "Goal in view" is the share of random start/goal pairs with a free line of sight between them (default `min_goal_distance` = 100); "path" is the navmesh shortest path. Rooms where the goal is visible about half the time saturate: the task is mostly "turn and walk". Raising `min_goal_distance` doesn't change that (two_room still 42% at 300 and can't fit 500), so it is not a fix. C_maze has far, hidden goals and still saturates, likely because its single winding corridor is solved by following a wall.
- [x] **Episode-length probe** (`scripts/e0_episode_length.sh`, group `e0_episode_length`): 8192 instead of 4096 steps made things worse, C_maze AUC 0.91 (MLP) / 0.79 (LSTM) and hard_maze 0.02 / 0.05. All hard-tier rooms keep 4096 steps.
- **Consequence for E4:** every stage now runs on the hard tier (C_maze, medium_maze, hard_maze), see the E4 stage table. The easy rooms stay in E1, where the colour/goal conditions are what change the difficulty.

### E1: Information-leak audit ("perceive") — 90–120 runs

| Factor | Levels |
|---|---|
| Colour | global Perlin (current) / off |
| Goal | fixed (current) / random per episode, only with colour off |
| Backbone | MLP, LSTM (GRU only by the trigger rule) |
| Rooms | two_room, four_room, C_maze, medium_maze |
| Seeds | 5 (seeds 100–104) |

Use a reduced design: global+fixed, off+fixed, off+random = 3 conditions × 2 backbones × 4 rooms × 5 seeds = 120 runs. If the budget is tight, drop medium_maze (90).
**Output:** Figure 2 (left), "the MLP win comes from the place code". It also sets the base colour mode for everything after.

**BPTT windows at equal updates (Nov 2026).** RecurrentPPO's update count fell with the window (128, 32, 8 updates per iteration at windows 16, 64, 256, against 512 for the MLP), so a window comparison also compared training budgets. From now on `algorithm.minibatch_size` is 1024 / window (64, 16, 4), which gives every window 128 updates of 1024 samples (`reports/Recurrent PPO BPTT implementations.md`). E2 uses these pairs. `scripts/e4_lstm_updates.sh` runs a learning-rate pilot on them and an MLP arm at 128 updates of 1024 samples. Stage 2's windows 64 and 256 stay in the paper as a labelled fixed-minibatch arm. The MLP keeps its tuned 512 updates of 256 samples as the reference.

**Pilot result (37 runs, Nov 8; mean AUC, the pilot on screening seeds 0–2 and medium_maze only).** 1e-4 is the worst rate at every window (0.01–0.02). 3e-4 leads at window 16 (0.51, against 0.16 at 1e-3). At windows 64 and 256, 3e-4 and 1e-3 are within one seed of each other: 0.26 vs 0.35 and 0.12 vs 0.10, and window 64's 0.35 at 1e-3 rests on a single seed at 0.90. **Every window keeps the LSTM's 3e-4**, so E1–E3 and E2 run unchanged, and Stage 2's window 16 runs (64 envs × 16 = 1024 samples already) count as that window's equal-update arm. At equal updates the longer windows still trail window 16 here (0.51 / 0.26 / 0.12), and the confirmation runs (windows 64 and 256 at 3e-4, seeds 100–104, both rooms) test that on the Stage 2 seeds. MLP arm, 128 updates of 1024 samples at 2e-4 against the tuned MLP on the same seeds 100–104: hard_maze 0.60 vs 0.27 (better on all 5 seeds), medium_maze 0.42 vs 0.55 (worse on 4 of 5). The MLP's AUC depends on its update scheme in opposite directions per room, so any LSTM vs MLP claim names the MLP's update scheme.

**Confirmation result (`scripts/e4_bptt_confirm.sh`, 20 runs, Nov 9, seeds 100–104, mean AUC; window 16 is Stage 2's own run, already at 1024 samples).** Most of Stage 2's window deficit came from the update count. On medium_maze, equal updates bring window 64 from 0.06 to 0.29 and window 256 from 0.07 to 0.26, level with window 16's 0.28. On hard_maze, window 64 rises from 0.01 to 0.11 and window 256 stays at 0.04, against 0.03 at window 16, and every MLP-encoder LSTM sits near the floor there. So at equal updates no window is clearly better on the Stage 2 seeds. The pilot's ordering (0.51 / 0.26 / 0.12 on seeds 0–2) didn't replicate. The seeds vary a lot (medium window 64: 0.07–0.75), so the fixed-minibatch arm's "longer windows are worse" is reported as an update-count effect.

**Learning rates (E1–E3, Nov 2026).** The sweep scripts left `algorithm.lr` at its default 3e-4, so the MLP would have trained at the rate the E0 pilot rejected. They now use the pilot's rates, the same as E4: MLP 1e-4, LSTM 3e-4. The GRU and the frame stacks had no pilot: the GRU takes the LSTM's and the frame stacks the MLP's.

**E1a: Perceptual aliasing score (no training, about 1 day of scripting).** For each room and colour mode, sample many positions and headings and record the ray observation at each. The aliasing score is the fraction of observation pairs that are nearly identical (within distance ε) but come from positions far apart (more than *D*, e.g. one room width). This measures how much the observation alone leaves the agent unsure where it is. Plot **memory gain (best recurrent − MLP, from E1/E4) against aliasing score** for every room × colour mode. If the points line up, it's the most direct evidence for the thesis, and it costs no training runs.

**E1b: Field of view as a perception dial (about 40 runs).** FOV {90°, 180°, 270°, 360°} × {MLP, LSTM} × C_maze × 5 seeds. LSTM is used because it's the existing, tested backbone. Swap in GRU only if E2 shows it clearly better. Narrowing the view should increase the memory gain. The result is a single crossover plot, and it reuses E4 Theme A's FOV levels.

### E2: Twin rooms with a cue ("remember") — about 175 runs. Headline experiment.

**Layout:** a start alcove with a **cue patch** (one of two colours, always at the same spot) → a corridor of length *L* → a junction → **two geometrically identical rooms with identical hard-coded wall colours**. The goal is placed in one room at random each episode, and the cue colour says which.
- Entering the wrong room ends the episode (otherwise an MLP just checks both rooms).
- The goal isn't visible from the junction.
- The cue isn't visible past the start either. The alcove sits above a short shaft that bends 90° into the corridor. In the first layout the corridor was straight, and the edge ray of a 180° FOV saw the cue's back wall from the junction, so a memoryless agent could re-read it. `tests/test_experiment_features.py::test_twin_rooms_cue_hidden_past_the_bend` checks this with 360° rays.
- Shaping toward the junction (`env.junction_shaping=1.0`, needed since the pilot, see below): F = γΦ(s′) − Φ(s) with Φ = −d_J / d_J(start), d_J the path distance to the junction along the room's `junction_path`, and Φ = 0 from the first step in the junction on. Both rooms lie past the junction, so it carries nothing about which room is right; being potential-based it leaves the optimal policy and the memoryless 0.5 ceiling unchanged (Ng et al. 1999), and it keeps credit assignment short (Ni et al. 2023). In the straight corridor d_J ignores the lateral offset. Standing still earns (1 − γ)|Φ| ≤ 0.001 per step, below the 1/(2L + 256) step penalty at every L. AUC and learning speed then describe the shaped task, and the paper says so.
- Generate one JSON per *L*; the geometry is otherwise fixed.

| Factor | Levels |
|---|---|
| *L* (corridor length, in agent steps at nominal speed) | 8, 32, 64, 128, 256 |
| Backbone | MLP, FS4, FS16, LSTM-BPTT16, LSTM-BPTT64, LSTM-BPTT256, GRU (7) |
| Seeds | 5 (seeds 100–104) |

7 × 5 × 5 = **175 runs** (small rooms, so cheap). Optional: K = 4 rooms at *L* ∈ {32, 128} for the 3 best backbones (+30).

**Checks that memory is actually used** (cheap, run on saved checkpoints):
1. **Reset at the junction:** zero the hidden state when the agent enters the junction. Success should drop to 0.5.
2. **Linear probe:** decode the cue colour from the hidden state as a function of steps since the cue.
3. **Confusion matrix:** room chosen vs cue shown.
4. **Reset sweep:** the same reset at 0, 25, 50 and 75% of the corridor. Where it stops hurting, the choice is committed.
5. **Mirrored pose** (memoryless and recurrent, not frame stacks): halfway down the corridor, reflect position, heading and last turn about the centre line and keep the recurrent state (Déletang et al. 2021, T-maze push). Success falls below chance if the cue is kept in the body.
6. **Swap:** at the same point, take the recurrent state, the pose, or both from an episode of the other cue; the room follows whichever carries the cue. A swap with an episode of the same cue is the control for the patch itself.

All six are implemented in `scripts/e2_diagnostics.py` (1000 episodes per condition and seed in the sweep), which `scripts/e2_twin_rooms.sh` runs after training (or alone: `bash scripts/e2_twin_rooms.sh <timestamp>`); `table_e2.csv` gets a column per check. The probe alone isn't evidence of memory: once the cue shapes the trajectory, the current observation carries it too. Seed-level tests for every figure come from `scripts/seed_stats.py`.

**Pilot result (L = 32, seeds 100–104, 35 runs, Nov 10; diagnostics on 200 deterministic episodes per seed).** Outcomes are all or nothing per seed, and the gate is reaching the junction, not memory or the baffle search:

| Backbone | Seeds reaching the junction | ...of which solve the cue (success ≥ 0.98) | ...at chance (one fixed room) |
|---|---|---|---|
| MLP | 0 (one seed 34% of episodes) | 0 | 0 |
| FS4 / FS16 | 1 / 1 | 0 / 0 | 1 / 1 |
| LSTM-BPTT16 | 5 | 5 | 0 |
| LSTM-BPTT64 | 2 | 1 | 1 |
| LSTM-BPTT256 | 4 | 3 | 1 |
| GRU | 1 | 1 | 0 |

- Every seed that doesn't reach the junction times out on all 200 episodes, and every seed that reaches it also finds the goal behind the baffle. So the alcove, shaft and corridor are the exploration bottleneck, and success vs backbone currently measures which seeds get out of the start, not memory. The memoryless seeds that get through sit at chance (0.47–0.51), as they should, and no memoryless seed beats it, so there's no sign of a cue leak.
- The reset at the junction comes too late to show where the cue is kept. Of the five LSTM-BPTT16 seeds, two are unaffected by it (success stays 1.00), one falls to chance (0.54) and two fall part way (0.69, 0.72). The choice is often already committed on entering the junction (pose or heading), so the reset has to move earlier, into the corridor right after the cue goes out of view, before it can tell memory from body-as-memory.
- Fix before the full sweep: the junction shaping above, so every backbone reaches the choice. Acceptance test on a new shaped L = 32 pilot: every seed of every backbone reaches the junction, and memoryless seeds land at 0.43–0.57, none above. The unshaped pilot stays as the reference row. Literature and alternatives: `reports/Twin room exploration failure fixes.md` and `reports/Twin room memory fix and tests.md`.
- Checks 4–6 on two of the unshaped pilot's LSTM-BPTT16 checkpoints (200 episodes each): seed 101, which the junction reset didn't affect, and seed 102, which it did. Both give the same answer: a reset anywhere in the corridor drops success to chance (0.54), the mirrored pose leaves it at 1.00, the other cue's recurrent state sends the agent to the wrong room (success 0.00), the other cue's pose doesn't (1.00), and the same-cue swap does no harm (1.00). The cue is carried in the recurrent state, not the body. The junction reset most likely missed it on seed 101 because the turn into the arm is already under way when the agent enters the junction box.

**Output:** Figure 3, success vs *L* (log axis) per backbone, a 0.5 chance line, and vertical marks at 4 and 16 frames. Inset: BPTT window × *L* heatmap. Diagnostics go in a small table.

### E3: Information sources ("be told") — about 180 runs

Goal randomised per episode in all conditions. The input size is the same in every condition (unused slots are zeros).

| Group | Condition | Agent receives |
|---|---|---|
| — | I0 | Nothing extra (= base) |
| Goal | G1 | Oracle goal ray every step: distance and bearing through walls (supervisor's idea; PointGoal-style) |
| Goal | G2a | Goal last seen, **updated by odometry** into the current frame, plus age |
| Goal | G2b | Goal last seen, **frozen** at the moment of sighting, plus age |
| Position | P1 | Raw normalised (x, y) |
| Position | P2 | Random Fourier features of (x, y), same size as P3 |
| Position | P3 | Hand-crafted hexagonal grid code: 3 cosines at 60°, 3–4 modules with spacing ×1.4 each, 16 phases (Solstad et al. 2006) |
| Position | P4 | P3 computed from **noisy path-integrated** position |
| Both | P5 | P3 for the agent + P3 for the goal (Banino-style "vector navigation", hand-made) |

9 conditions × {MLP, LSTM} × {four_room (open), C_maze (dead ends)} × 5 seeds (100–104) = **180 runs**. GRU is added only under the trigger rule (at most ~30 extra runs).

**Output:** Figure 4, a heatmap of conditions × backbones of IQM padded AUC, one panel per layout, plus the "memory gain" row (best recurrent − MLP) for each condition. This is the direct test of "information substitutes for memory".

**Why this is cheap:** grid and Fourier features are a few cosines per step, vectorised in torch. No supervised grid network has to be trained, which avoids the runtime cost of reproducing Banino's.

### E4: Staged ablation of the remaining design choices ("perceive & act") — about 440 runs

This covers the rest of the project title (observation, action, reward, aux, capacity) without the full factorial.

**Themes** (each varied around the base config; the others stay fixed):

| Theme | Levels | # |
|---|---|---|
| A. Observation | encoder {MLP, CNN} × FOV {90°, 180°, 270°} | 6 |
| B. Action space | continuous (current) / discrete 3 actions / discrete 5 actions | 3 |
| C. Reward | time penalty only (current) / + collision penalty / γ 0.99 vs 0.999 | 3 |
| D. Aux losses | none / depth / loop closure / both, on the LSTM with distances masked; from Stage 2 also the unmasked LSTM ± loop closure | 4 (+2) |
| E. Capacity | width {16, 64, 256}; depth aux (masked) at {16, 64} = matched-parameter comparison; from Stage 2 also the masked MLP without aux at {16, 64}; added after Stage 1: hidden layers per module {1, 3} and 2 with a ReLU between modules | 5 (+2) (+3) |

Stage 1 ran 18 distinct configs (the baseline once); Stage 2 runs 23 (see "Stage 1 result"). Memory isn't a theme here: E2 and E3 decide it.

**Theme D is a replication of Mirowski et al. 2017**, which used the same depth and loop-closure auxiliary tasks in 3D DMLab mazes. Say so explicitly: "does Mirowski's aux-loss benefit transfer to 2D ray-cast agents, and only when depth is not already in the input?" That's an easy hook for reviewers. Theme E's matched-parameter points answer your 20.10 question of when an aux loss beats simply adding width.

**Extra protocol rules** (from the earlier report):
- **Probe panel:** 4 fixed configs (e.g. the best and worst encoders, and aux on/off) run on T_maze, C_maze and medium_maze no matter what is eliminated. Compute Kendall τ on these, not only on survivors. Survivors alone restrict the range and inflate τ.
- **Wildcards:** pick them with a reported RNG seed. If a wildcard beats the leader at the next tier, that theme's screening is reported as unreliable.
- **Early kill:** stop a run at 50% of its budget only if it has zero success in every room while the theme leader is above 0.5.
- **Re-baseline:** after Stage 1, each theme's leader becomes the default for the other themes (exceptions in "Stage 1 result").

**Stages** (moved to the hard tier after the E0 room probe: every easier room saturates under the base config):

| Stage | Rooms | Configs | Seeds | Runs | Rule |
|---|---|---|---|---|---|
| 1 Screen | C_maze, medium_maze | 18 (baseline once + 17 levels that differ from it; aux theme also runs depth + loop closure together) | 3 (0–2) | 108 | Keep the theme leader + every config whose IQM CI overlaps it, at most ⌈n/3⌉; + 1 random wildcard per theme; the baseline is always kept |
| 2 Confirm | medium_maze, hard_maze | 23 (11 around the new base, 11 aux / memory, CNN + LSTM, see "Stage 1 result") | 5 (100–104) | 230 | Same rule; check Kendall τ between stage-1 and stage-2 rankings, and reopen a theme if τ < 0.4 (heuristic) |
| 3 Final | medium_maze, hard_maze | best combined + baseline + leave-one-theme-out (~4) | 10 (1000–1009) | ~80 | Welch's t, α = 0.01, Holm correction; IQM + bootstrap CI + probability of improvement (rliable) |

**Stage 1 result (108 runs, seeds 0–2, screening only, AUC IQM [95% CI], wildcard RNG seed 0).** Baseline 0.73 [0.65, 0.75].

| Theme | Leader | Also kept | Wildcard | Dropped |
|---|---|---|---|---|
| Observation | CNN FOV 180°, 0.87 [0.80, 0.92] | CNN FOV 270°, 0.85 | MLP FOV 90°, 0.62 | CNN FOV 90° 0.84, MLP FOV 270° 0.83 (both over the cap, CIs overlap) |
| Action | continuous (baseline) | | discrete 3, 0.56 | discrete 5, 0.65 (over the cap, CI overlaps) |
| Reward | collision penalty 0.01, 0.86 [0.71, 0.90] | | γ 0.99, 0.72 | |
| Capacity | width 256, 0.81 [0.78, 0.83] | | width 16, 0.47 | |
| Aux (vs masked LSTM 0.12 [0.04, 0.17]) | depth + loop closure, 0.18 [0.17, 0.21] | | loop closure, 0.07 | depth, 0.11 |

The masked MLP points (depth + loop closure, width 64: 0.23, width 16: 0.18) had no matched reference in Stage 1, so their "capacity unmatched" ranking mixes the aux effect with the masking and isn't used: both widths stay, with their references, because they answer the aux-vs-width question.

**Depth levels (added after Stage 1).** Every earlier run has 2 hidden layers per module, so an MLP agent has 6 ReLU layers from input to action, and each module ends in a plain linear layer, so consecutive modules join with no nonlinearity. Stage 1's baseline also ran with 1 hidden layer per module (3 ReLU layers) and with 3 hidden layers per module plus a ReLU between modules (`module_output_activation=true`, 11 ReLU layers), on the same rooms and seeds (`scripts/e4_depth_screen.sh`, 12 runs). The 3-layer level changes depth and the junction activation together, so a control runs the usual 2 layers per module with that activation (`scripts/e4_depth_control.sh`, 6 runs, 8 ReLU layers). A level enters Stage 2's capacity theme only if its CI clears the baseline's.

**Depth result (126 runs, seeds 0–2, screening only, against the baseline's 0.73 [0.65, 0.75]).** 1 layer per module: 0.87 [0.79, 0.89], it clears the baseline and leads the capacity theme ahead of width 256; most of the gain is on medium_maze (0.74 vs 0.48). 2 layers with the junction ReLU: 0.77 [0.73, 0.80], overlaps the baseline. 3 layers with the junction ReLU: 0.68 [0.60, 0.72], overlaps the baseline but sits below the 2-layer control with no CI overlap, so the extra depth hurts and the activation doesn't explain it. Only the 1-layer level enters Stage 2, on the CNN base, where it has never run. The wildcards stay those of the 108-run analysis (`promotion/20261102095921_108runs.json`): bootstrap and wildcard draws share one RNG, so adding levels changes the draws of the 126-run analysis, and those aren't used.

Stage 2 base: **CNN encoder, FOV 180°, continuous, collision penalty 0.01, width 64**. Two departures from the re-baseline rule, both for cost. Width 256 stays a level of the capacity theme: on the CNN encoder it costs 1.8× per iteration (30 vs 17 s/it, Nov 4 timing jobs), and the two leaders never ran together, so this level also tests the combination. The aux, memory and BPTT configs stay on Stage 1's MLP encoder and reward: aux on/off are ranked against each other anyway, CNN + LSTM + aux takes twice as long (32 s/it, 13.4 s/it on the MLP encoder), and it keeps them comparable with Stage 1. Stage 1's baseline also stays, as the link between the stages and the MLP side of MLP vs LSTM. Kendall τ between stages can therefore only be computed per level name, not on identical configs, except for the aux, memory and Stage 1 baseline configs. Pass `+promotion.base.encoder=cnn +promotion.base.cpen=0.01` to `e4_promotion.py`.

**CNN + LSTM (added to Stage 2).** Stage 2's aux and memory configs stay on Stage 1's MLP encoder, so without this config the leading encoder and the recurrent backbone never run together, and Stage 3's "best combined" could only pair them untested. It runs at Stage 2's base settings with the LSTM's learning rate 3e-4 and BPTT 16 (`scripts/e4_cnn_lstm.sh`, 10 runs, same seeds, storage and timestamp). `e4_promotion.py` ranks it against the base in a "memory" block. It's a single config, so it tests the pairing but not the levels around it.

**CNN + LSTM + loop closure (added to Stage 2).** The same argument for aux: this config keeps Stage 3 from pairing the CNN + LSTM with loop closure untested, and at 32 s/it it's about 90 hours for 10 runs. Depth aux stays out, as its target is in the input when unmasked and the masked agent can't see the walls. CNN + LSTM is its matched reference (`scripts/e4_cnn_lstm_aux.sh`, 10 runs, same seeds, storage and timestamp), and it brings Stage 2 to 24 configs and 240 runs.

**Matched references (added after Stage 1).** Depth aux needs distances hidden from the policy, and in Stage 1 hiding them alone crippled the LSTM (interim AUC ≈ 0.11 on C_maze and medium_maze, against 0.96 / 0.81 unmasked in the E0 probe). So every aux-on level is ranked against the same config with the aux head off and the same masking, not against the unmasked baseline (`scripts/e4_promotion.py`). From Stage 2 the sweep adds those references: the masked MLP without aux at widths 16 and 64, and the unmasked LSTM with and without loop closure (loop closure needs no hidden target, so this tests it on an agent that can see distances). Keep a reference whenever one of its aux levels survives. Stage 1's capacity aux points ran depth + loop closure (loop_closure_coeff left at its default); from Stage 2 they are depth only, so they don't enter the Stage 1 → 2 Kendall τ.

**BPTT window (Stage 2 on).** The unmasked LSTM also runs at BPTT 64 and 256 next to the default 16, ranked against BPTT 16 (the "bptt" block of `e4_promotion.py`): 16 steps of gradient are short next to 4096-step episodes, and the LSTM scored 0.000 on hard_maze in the E0 probe. The only earlier BPTT sweep (W&B group `lstm_tbptt`, Sep 2026, 5 seeds per window) ran four_room with 512-step episodes, a fixed goal and the global colour place code, so memory wasn't needed there and it doesn't replace this ablation. There every window converged (early stop after 175–310 iterations); 128 and 256 reached a completion rate of 100% vs 92% for 16 and 64 (one seed below 65%) with shorter paths (NPL 1.50 vs 1.64–1.70), but learned slower (mean eval return over training 0.24 at 256 vs 0.60 at 16). E2 runs the same windows (16/64/256) on the twin rooms, where the memory horizon is set by *L*.

**Stage 2 result (240 runs, seeds 100–104, AUC IQM [95% CI], wildcard RNG seed 0).** Baseline (CNN + MLP) 0.84 [0.78, 0.87]. Kendall τ between the stages on medium_maze is 1.0 over the 6 configs that ran unchanged in both, so no theme reopens. Levels are matched by config, because labels are relative to each stage's base.

| Theme | Leader | Also kept | Wildcard | Dropped |
|---|---|---|---|---|
| Observation | baseline (CNN FOV 180°) | CNN FOV 270°, 0.80 | MLP encoder without collision penalty, 0.43 | MLP encoder 0.66, MLP FOV 90° 0.14 |
| Action | discrete 3, 0.94 [0.90, 0.96] | continuous (baseline) | | |
| Reward | collision penalty 0.01 (baseline) | | no penalty, 0.74 | γ 0.99, 0.83 (CI overlaps, over the cap) |
| Capacity | width 64 (baseline) | width 256, 0.81 | 1 layer per module, 0.77 | width 16, 0.41 |
| Memory (vs base) | baseline | | CNN + LSTM, 0.62 [0.44, 0.77] | |
| BPTT (fixed 64 envs per minibatch) | window 16, 0.09 | | window 64, 0.02 | window 256, 0.01 |
| Aux (vs matched reference) | no aux gain: every CI overlaps its reference on the MLP encoder | | | |

**CNN + LSTM + loop closure collapse.** It scored 0.01 [0.00, 0.09] against 0.62 for the same agent without the aux loss. This is a behaviour failure and shouldn't be read as "loop closure hurts memory". 9 of 10 runs stopped moving: mean speed command below 0.06, mean speed action around -2 and so clipped to 0, and already at iteration 100 in the seeds tracked over training (`hard_maze` 101). A stopped agent collects no reward and no loop labels, so the value and loop closure losses both go to about 0 and the policy can't recover. The stopped state is cheap: a collision costs 0.01, which is 41 steps of the 1/4096 time penalty. All 30 loop closure runs at collision penalty 0 sit on the MLP encoder and at most one per config ends with a dead value loss, but encoder and penalty change together there, so this doesn't separate the two. Gradient clipping isn't the cause: on a fresh agent the total norm is 0.30 against the limit of 0.5. The aux gradient is 7–9× the task gradient in the LSTM and encoder, though, so the shared features mostly follow the aux loss. Loop closure coefficient 1.0 was not tuned for this reward scale.

**Collapse test (`scripts/e4_aux_collapse.sh`, 6 runs, medium_maze, seeds 100–102, mean AUC).** Each change alone removes the collapse. Stage 2 at penalty 0.01 and coefficient 1.0 scores 0.15 (0.44 / 0.00 / 0.00). Penalty 0 at coefficient 1.0 scores 0.52 (0.72 / 0.56 / 0.28). Coefficient 0.1 at penalty 0.01 scores 0.60 (0.77 / 0.28 / 0.76), level with CNN + LSTM without aux on the same seeds (0.59). Every run of both arms reaches 0.90–1.00 eval success. So the stopping needs both the collision penalty and an aux loss that dominates the shared features. The paper reports it as that interaction, not as an effect of loop closure. At a coefficient that doesn't swamp the task loss, loop closure neither helps nor hurts the CNN + LSTM here. That's 3 seeds on one room, a diagnostic and not a result.

All stages use 4096-step episodes and gamma 0.999. Stages 1 and 2 share medium_maze, so the Kendall τ between their rankings measures seed stability there plus the change of the second room (C_maze → hard_maze). The "by tier" result (H6) compares rankings across the three rooms C_maze < medium_maze < hard_maze.

**Output:** Figure 2 (right), effect of each theme with confidence intervals by tier, and the τ between tiers (H6). The "how few parameters" and "aux vs width" results come from Theme E.

### Stretch (only if slots are idle)

- **Held-out layouts:** train on 4 or 16 procedurally varied four-room layouts and test on unseen ones, comparing I0 / P1 / P3. This is the cleanest test for H5 and the most relevant for Arxitect.
- **One converted hospital floor plan** with the final config as a demo figure (for eCAADe).

### Architecture-side twin paper (eCAADe 2027, abstract ~mid-January, estimate)

Reuse Figures 2 and 4 and add **trajectory statistics** that matter for occupant simulation. These need no human data and are computed from evaluation rollouts you already save:
- path ratio vs the navmesh/A* shortest path (= 1/SPL on successes);
- distance-to-wall histogram (wall hugging vs corridor centring);
- revisit rate (fraction of cells visited more than once) and dead-end entries;
- occupancy heatmaps per design choice;
- DTW to the funnel path (already implemented).

The story for architects is that design choices change *how* agents move, not only whether they arrive. A memoryless agent and a recurrent agent with the same success rate can produce very different circulation heatmaps, and that matters for using them in Arxitect.

### Cut (and say so in the paper's "limitations")

VLA/LLM baseline, SAC, transformer/SSM backbones, learned grid cells (Banino), curriculum, multi-goal routes. Each is either expensive, off-thesis, or not yet implemented.

### Related work to position against

- **Navigation RL:** Mirowski 2017 (aux losses), Banino 2018 (grid cells), Wijmans 2020/2023 (PointGoal, emergent maps), EDEN (arXiv 2506.03046, grid cells + PPO).
- **Memory benchmarks:** Ni et al. 2023 (memory length vs credit-assignment length, Passive T-maze), Memory Gym, POPGym, MiniGrid-Memory.
- **Grid codes as features:** Solstad 2006, Bartlett et al. 2022/2023, GridPE (arXiv 2406.07049), Schaeffer et al. 2022.
- **Empirical RL methodology:** Andrychowicz 2021, Obando-Ceron & Castro "Revisiting Rainbow" (small environments give valid insight), Agarwal 2021, Patterson 2024, Jordan et al. 2024 (hypothesis-driven testing over leaderboards).
- **Architecture/occupant simulation:** cogARCH and SimArch (heuristic wayfinding), SpaceLayoutGym (Kakooee & Dillenburger, ETH; RL that designs layouts rather than simulating occupants, worth contacting), TravelAgent (arXiv 2412.18985, LLM agents).
- **Positioning:** "a controlled study of RL design choices for egocentric occupant-like wayfinding". Avoid claiming "first" unless a CumInCAD search confirms it.

## 5. Budget

Hours per run are **measured** in the E0 pilot (see E0): a full 1000-iteration run costs ≈ 2.1 h (MLP) or ≈ 2.8 h (LSTM) in any room, and a room the agent solves stops early after 0.2–0.5 h. The column below is the **worst case**, every run going to 1000 iterations (≈ 2.45 h on average for a mix of MLP and recurrent runs); solved easy rooms make the real cost lower. Twin rooms aren't measured yet; per-iteration cost doesn't depend on the room, so the same cap applies.

| Block | Runs | Slot-hours (worst case) |
|---|---|---|
| E0 pilot + learning-rate pilot + room and episode-length probes | 32 (done) | 34 (measured) |
| E1 audit (+ E1a aliasing script, + E1b FOV × backbone) | 130–160 | ~320–390 |
| E2 twin rooms | 175 (+30) | ~500 |
| E3 information sources (+ triggered GRU runs) | 180–210 | ~440–515 |
| E4 staged ablation (Stage 1 measured 390 with the depth levels; Stage 2 ~1,200 for 230 runs at MLP 2.6 h, LSTM + aux 6 h, CNN 7 h, CNN + LSTM 9 h, CNN + width 256 12.5 h per run; Stage 3 ~600 if the final config uses the CNN) | ~440 | ~2,200 |
| **Total** | **~960–1,050** | **~3,500–3,650 ≈ 9.1–9.5 days on 16 Euler slots (less with early stopping)** |

If the pilot comes in slower, cut in this order: E4 Stage 3 to medium_maze only → E3 without P2 and P5 → E1 without medium_maze. **Never cut seeds below 5 for anything in a main figure.**

## 6. Paper layout (4 pages + appendix)

| | Content |
|---|---|
| Fig. 1 | Testbed: ray-cast agent, room tiers, twin-room task, the three routes (perceive / remember / be told) |
| Fig. 2 | Left: E1 place-code audit, with memory gain vs aliasing score (E1a) as an inset. Right: E4 theme effects by tier, with τ |
| Fig. 3 | **E2 headline:** success vs *L* per backbone, with BPTT inset |
| Fig. 4 | E3 information × backbone heatmap, open vs dead-end layout |

`scripts/paper_figures.py` draws every data figure from the saved seed folders (`+figure.name=e1|e1b|e2|e3|e4`), as .pdf, .png and a .csv of the plotted numbers under `<storage>/figures/<timestamp>/`. The E1–E3 sweep scripts call it at the end; for E4, run `scripts/e4_promotion.py` first. Fig. 2's panels are separate files (`fig2_left`, `fig2_right`) because they come from different sweeps. Fig. 1 is a hand-made diagram. Two choices made there: the memory gain is always LSTM − MLP, not "best recurrent", to follow the GRU trigger rule, and Fig. 3's lines show the LSTM and GRU at BPTT 64 (the window they share) while the side heatmap shows every LSTM window.
| Table 1 | Memory diagnostics (reset at junction, probe accuracy) |
| Appendix | Full configs, per-room numbers, capacity curve, protocol rules, compute |

**Draft abstract.** Reinforcement learning agents for indoor wayfinding are built from many design choices: observation space, memory architecture, auxiliary losses, action space, reward. Each has usually been evaluated in a different environment. In a controlled 2D ray-cast testbed, we find that these choices are not independently good or bad. They trade off according to how the agent obtains the information it needs: by perceiving it, remembering it, or being told it. A seemingly robust result, that memoryless agents outperform recurrent ones, disappears once an unintended place code is removed from the observation. In visually identical twin rooms, each architecture fails at a predictable memory horizon. Providing goal or position information, including a training-free hand-crafted grid-cell code, substitutes for memory in open layouts but not in dead-end mazes. We release the testbed and a staged, budget-aware ablation protocol.

*(The abstract states the hypothesised results. Rewrite it after the data is in.)*

## 7. Timeline

| Week | Dates | Work | Exit criterion |
|---|---|---|---|
| 1 | Oct 29 – Nov 4 | E0 fixes, pilot timings, **pre-registration commit** (hypotheses, metrics, promotion rules, seed sets) | Pilot hours known |
| 2 | Nov 5 – 11 | Launch E1 + E4 Stage 1; build twin-room JSONs and cue; E2 pilot at *L* = 32 | LSTM beats 0.5 at *L* = 32, or fix the task design |
| 3 | Nov 12 – 18 | E2 full sweep; E4 Stage 1 analysis → Stage 2 | E1 result known (decides framing of Fig. 2) |
| 4 | Nov 19 – 25 | E3 sweep; E2 diagnostics | — |
| 5 | Nov 26 – Dec 2 | E4 Stage 3 (long jobs); **ICLR 2027 workshop list out Nov 29** → pick a target | Target workshop chosen |
| 6 | Dec 3 – 9 | Fill gaps, extra seeds where CIs overlap, stretch if idle | All figure data in hand |
| 7 | Dec 10 – 16 | rliable analysis, figures frozen | Figures 1–4 done |
| 8–9 | Dec 17 – Jan 10 | Semester report + 4-page workshop version; supervisor review | Draft complete |
| — | ~mid-Jan / ~Feb 1 2027 | eCAADe abstract (estimate) / ICLR workshop deadline (confirmed suggested date) | Submitted |

AAAI-27 workshops (Nov 20) are skipped: the main results won't exist by then. Backups if the ICLR workshops don't fit: CVPR 2027 embodied-AI workshops (papers ~Mar–Apr, estimate) and RLC 2027 (~early March, estimate) for an extended full paper. Prefer non-archival workshops so RLC and eCAADe stay open.

## 8. Risks and fallbacks

| Risk | Fallback framing |
|---|---|
| E1: MLP still wins with colour off | Fine. It means the floorplans are reactive even without the leak. E2 and E3 carry the paper, and Fig. 2 becomes "standard floorplans don't test memory". |
| E2: LSTM/GRU fail even at small *L* | Check recurrent PPO details (stored hidden states, masking, burn-in). If they still fail, that's a finding about recurrent PPO optimisation in continuous control, and reporting it is valid. |
| E3: the grid code doesn't beat xy / Fourier features | Expected under H5 and consistent with Bartlett et al. 2022. It's a clean negative result that supports a cheap baseline over learned grid cells. |
| Stage-1 rankings don't transfer (τ < 0.4) | Report it. It answers "is easy-room screening valid?", which is itself a methodological contribution. |
| Compute overrun | Cut order in §5. |

## Key references

- Mirowski et al. 2017, Learning to Navigate in Complex Environments — arXiv 1611.03673
- Banino et al. 2018, Vector-based navigation using grid-like representations — Nature 557
- Wijmans et al. 2023, Emergence of Maps in the Memories of Blind Navigation Agents — arXiv 2301.13261
- Wijmans et al. 2020, DD-PPO — arXiv 1911.00357
- Ni et al. 2023, When Do Transformers Shine in RL? — arXiv 2307.03864
- Pleines et al. 2023, Memory Gym — arXiv 2309.17207
- Morad et al. 2023, POPGym — arXiv 2303.01859
- Solstad, Moser & Einevoll 2006, From grid cells to place cells — Hippocampus 16, doi 10.1002/hipo.20244
- Bartlett, Stewart & Orchard 2022/2023, grid-cell and SSP representations for RL — ICCM
- Schaeffer, Khona & Fiete 2022, No Free Lunch from Deep Learning in Neuroscience — NeurIPS
- Andrychowicz et al. 2021, What Matters in On-Policy RL? — arXiv 2006.05990
- Agarwal et al. 2021, rliable — arXiv 2108.13264
- Patterson et al. 2024, Empirical Design in RL — arXiv 2304.01315
- Anderson et al. 2018, On Evaluation of Embodied Navigation Agents (SPL) — arXiv 1807.06757
