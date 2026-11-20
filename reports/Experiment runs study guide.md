# Experiment runs: study guide

*Status: 2026-11-20. Covers every training run of the project up to the Nov 19 supervisor meeting: what each experiment and stage was for, how it was built, what ran, what came out and how to read it. The numbers are copied from [Final paper and experiment plan.md](Final%20paper%20and%20experiment%20plan.md) (the source of truth) and [E1 E3 results analysis.md](E1%20E3%20results%20analysis.md). When the two disagree with this guide, they win.*

## 0. The story in five lines

1. The original result was "a memoryless MLP beats the LSTM in every floorplan". The project set out to explain it rather than report a leaderboard.
2. The thesis: an agent gets the information it needs by **perceiving** it, **remembering** it or **being told** it. A design choice helps only if the other routes don't already provide that information.
3. **E1 / E1b / E3:** standard floorplans don't need memory. Removing the suspected colour leak doesn't change the result, and telling the agent the goal or position has no memory gap to close.
4. **E2 / E3T:** in twin rooms built so that memory is required, each backbone fails where its memory horizon ends. Being told the cue replaces memory only while the gap between tellings fits that horizon.
5. **E4:** a staged ablation of the other design choices (observation, action, reward, capacity, aux losses) finds three that survive and combine: a CNN encoder, 3 discrete actions and a collision penalty.

| Block | Runs | Role in the paper |
|---|---|---|
| E0 pilot and probes | 32 (+37 LSTM learning-rate pilot) | Set costs, learning rates and room tiers |
| E1 + E1b | 120 + 40 = 160 | Negative result: floorplans are reactive |
| E2 (+ layer check, + K = 4) | 175 + 10 + 30 = 215 | **Headline:** memory horizon per backbone |
| E3 + E3T (+ C64r, + L = 64) | 180 + 105 + 15 + 60 = 360 | Told vs remembered |
| E4 Stages 1–3 (+ side checks) | 126 + 266 + 100 = 492 | The method: staged ablation |
| **Total** | **≈ 1,260 training runs** | |

---

## 1. Shared setup (read this first)

### 1.1 Agent and environment

- **Environment:** 2D floorplans read from `rooms/*.json`, vectorised in torch (`envs/navigation_env.py`), 256 parallel envs for training.
- **Observation:** ray casts over the field of view (default 180°, 0.2 rays per degree, so 37 rays).
  - Each ray has 7 channels: hit class (no hit, goal, wall: channels 0–2), normalised distance (channel 3) and RGB colour (channels 4–6).
  - A small proprio vector, plus optional "information slots" for goal, position or cue.
- **Action:** continuous turn and speed by default (max speed 13 units per step, max turn 90° per step). E4 also tests 3 or 5 discrete actions.
- **Reward:**
  - −1/max_steps every step and +1 on success.
  - Optional collision penalty (E4: 0.01).
  - Optional potential-based shaping toward the junction, in the twin rooms only.
- **Models:** encoder → backbone → policy and value heads, built by `utils/factory.py`.
  - Encoder: MLP or 1D CNN over the rays.
  - Backbone: MLP, frame stack (FS4, FS16), LSTM or GRU (2 layers of 64 units).
  - Heads: Gaussian policy (categorical for discrete actions), value, and optional aux heads for depth and loop closure.
- **PPO:**
  - Each iteration is 256 envs × 256 steps; a run is at most 1000 iterations.
  - 2 epochs, clip 0.2, entropy 0.005, GAE λ 0.95, target KL 0.04, gradient clip 0.5, annealed learning rate.
  - The recurrent PPO trains on truncated BPTT windows of 16, 64 or 256 steps.
- **Evaluation:** every 5 iterations, 50 episodes.
  - Early stopping once the eval success (EMA) converges at ≥ 0.95.
- **Learning rates (from E0):** MLP 1e-4 and LSTM 3e-4. The GRU uses the LSTM's rate and the frame stacks use the MLP's.

### 1.2 Metrics

| Metric | Definition | Why |
|---|---|---|
| **Success AUC** (primary) | Area under the eval-success curve, **padded to the full budget** (the last value carried forward to 200 evals) | Rewards learning *and* learning speed. Padding scores early-stopped and slow runs on the same horizon. |
| Success rate / CR | Share of eval episodes reaching the goal, 10 runs × 10 episodes per seed ("mean of means", the supervisor's scheme) | The final ability |
| SPL | success × shortest / max(actual, shortest) (Anderson 2018) | Path efficiency, in [0, 1] |
| nDTW | exp(−DTW / (\|R\| · d_th)) (Ilharco 2019) | Path shape vs the navmesh shortest path. **Caveat:** degenerate for long fixed-goal paths, so compare it only within one goal mode. |
| Diagnostics success (E2/E3T) | Success over 1000 deterministic episodes from `best.pt` | "Solved" means ≥ 0.9 |

### 1.3 Seeds and statistics

| Seed set | Seeds | Used for | Reported? |
|---|---|---|---|
| Screening | 0–2 (E0: 0) | Learning-rate pilot, E4 Stage 1 | **Never.** These choose configs. |
| Confirmation | 100–104 | E1, E1b, E2, E3, E3T, E4 Stage 2 | Yes, with CIs |
| Final | 1000–1009 | E4 Stage 3, claims at α = 0.01 | Yes |

- **Why separate sets:**
  - Seed variance alone can flip conclusions (Henderson 2018).
  - Choosing and reporting on the same seeds inflates the winner, the "designer's curse" (Patterson 2024; Eimer 2023).
  - Power analysis says 3 seeds can only screen, 5 confirm large effects, and α = 0.01 needs about 10 (Colas 2019).
- **The statistical unit is the training seed,** not the eval repeat.
- **Aggregation:** IQM with stratified-bootstrap 95% CIs and probability of improvement (rliable, Agarwal 2021).
- **Claims:** Welch's t at α = 0.01 with Holm correction.
- **No seeds added after seeing p-values.** Adding seeds until p drops below the threshold is p-hacking.
- **GRU trigger rule:**
  - The LSTM is the default recurrent backbone.
  - The GRU runs in E2 always; elsewhere only if the LSTM's IQM is below the MLP's with non-overlapping CIs.
  - "Best of LSTM and GRU" is never reported.

### 1.4 Hypotheses and where they stand

| ID | Hypothesis | Experiment | Status |
|---|---|---|---|
| H1 | The MLP's lead shrinks or flips once the colour place code is removed and goals are random | E1 | **Partly.** The gap shrinks, mostly through random goals, never flips, and nothing is significant. |
| H2 | In the twin rooms each backbone fails past its memory horizon | E2 | **Supported**, with two twists: FS16 bridges L = 32 through its body, and nothing solves L = 256. |
| H3 | Telling the goal or position closes the MLP–recurrent gap in open layouts, not in dead-end ones | E3 | **Untestable as run:** there was no gap to close. |
| H4 | A frozen last-seen goal and a noisy position bring back the recurrent advantage | E3 | **Not supported** |
| H5 | The hand-made grid code is no better than Fourier features of the same size | E3 | **Supported** (null result) |
| H6 | Perception, capacity, action and reward rank consistently from easy to hard rooms; memory and aux don't | E4 | **Partly.** Best is first and baseline last in both rooms; the middle reorders. |
| H7 | Memory gain grows with aliasing and with a narrower FOV | E1a, E1b | **Not supported** |
| H8 | Being told the cue replaces memory only while the gap between tellings fits the agent's memory | E3T | **Holds**, cleanly at L = 64 |

---

## 2. E0: fixes, pilot and room probes

**Goal:** make later results valid and measure what a run costs before planning the budget.

**Fixes that every later run depends on:**
- Colour modes with their own colour seed, fixed across RL seeds.
- Per-episode goal sampling.
- AUC padded to the full budget.
- Depth hidden from the policy when depth aux is on.
- `success_radius` separate from the visual goal radius.
- Fixed-size information slots.
- The GRU backbone, and spawn regions for the mazes.

**Runs:**

| Probe | Runs | Seeds | What it varied | Result |
|---|---|---|---|---|
| Pilot (`e0_pilot`, 20261101160538) | 18 | 0 | two_room, four_room, medium_maze × MLP/LSTM × lr {1e-4, 3e-4, 1e-3} | **The cost is set by iterations, not by the room:** a full run takes ≈ 2.1 h (MLP) or 2.8 h (LSTM). **Rates:** MLP 1e-4, LSTM 3e-4. The small rooms saturate at AUC ≥ 0.92. |
| Room probe (`e0_room_probe`) | 10 | 0 | Best MLP and LSTM on the five remaining rooms | Tiers (below) |
| Episode-length probe (`e0_episode_length`) | 4 | 0 | 8192 vs 4096 steps, C_maze and hard_maze | Longer was worse, so 4096 is kept |

| Room | Episode | MLP AUC | LSTM AUC | Goal in view at start | Tier |
|---|---|---|---|---|---|
| two_room / T_maze / I_maze | 512 | 0.96–0.99 | 0.99–1.00 | 53% | easy, saturates |
| four_room | 512 | 0.99 | 1.00 | 52% | easy |
| cross_room | 4096 | 1.00 | 1.00 | 51% | easy |
| C_maze | 4096 | 0.93 | 0.96 | 16% | hard |
| medium_maze | 4096 | 0.57 | 0.81 | 12% | hard |
| hard_maze | 4096 | 0.18 | 0.00 | 7% | hard |

**What it decided:**
- Rooms where the goal is visible half the time are "turn and walk" tasks, which is why every config saturates there.
- **E4 therefore moved to the hard tier** (C_maze, medium_maze, hard_maze). The easy rooms stay only in E1, where colour and goal mode change the difficulty.

**LSTM learning-rate pilot at equal updates (`e4_lstm_updates`, 37 runs, Nov 8, seeds 0–2, medium_maze):**
- **Why it was needed:** RecurrentPPO's update count fell with the window (128, 32 and 8 updates per iteration at windows 16, 64 and 256), so a window comparison also compared training budgets.
- **The fix:** `minibatch_size = 1024 / window` (64, 16, 4 envs), so every window gets 128 updates of 1024 samples.
- **Result:** 3e-4 stays the rate at every window.
- **MLP side result:** the MLP at 128 updates is better on hard_maze and worse on medium_maze than at its tuned 512. So any LSTM vs MLP claim has to name the MLP's update scheme.

---

## 3. E1: information-leak audit ("perceive")

**Question:** does the MLP win because the global Perlin colour works as a place code, and because a fixed goal turns every room into a lookup?

**Why:** colour is sampled at the hit point's *world* coordinates, so a memoryless agent can read its position from it. A fixed goal makes memory unnecessary.

**Design (120 runs, `e1_place_code_audit`, 20261108114624):**
- 3 conditions: colour global + fixed goal; colour off + fixed goal; colour off + random goal.
- 2 backbones (MLP, LSTM) × 4 rooms (two_room, four_room, C_maze, medium_maze) × 5 seeds (100–104).

**Result (AUC IQM [95% CI]):**

| Room | Condition | MLP | LSTM |
|---|---|---|---|
| two_room, four_room | all | 0.98–0.99 | 0.98–0.99 |
| C_maze | global, fixed | 0.79 [0.62, 0.86] | 0.57 [0.09, 0.85] |
| C_maze | off, fixed | 0.79 [0.64, 0.93] | 0.68 [0.39, 0.92] |
| C_maze | off, random | 0.96 [0.95, 0.97] | 0.87 [0.72, 0.96] |
| medium_maze | global, fixed | 0.67 [0.43, 0.86] | 0.12 [0.05, 0.19] |
| medium_maze | off, fixed | 0.88 [0.57, 0.94] | 0.15 [0.08, 0.62] |
| medium_maze | off, random | 0.56 [0.47, 0.62] | 0.23 [0.09, 0.55] |

**How to read it:**
- **Colour isn't the place code.** Switching it off never hurts the MLP (+0.03 in C_maze, +0.14 in medium_maze).
- **The gap shrinks but never flips** (C_maze −0.24 → −0.10, medium_maze −0.54 → −0.27). The shrink comes mostly from random goals making the MLP worse.
- **No LSTM − MLP difference survives Holm at α = 0.01.** The closest is medium_maze, global colour and fixed goal: Holm p 0.046.
- **C_maze:** equal final success. The gap is learning speed, since the LSTM needs more evals before it early-stops.
- **medium_maze:** a real final-success gap within the budget. 13 of 15 LSTM seeds end at ≤ 0.62 success, while 8 of 15 MLP seeds reach ≥ 0.94. The failures are graded, not all-or-nothing.
- **E1a aliasing score:** ≈ 0 in every room (≤ 0.005), so it can't explain anything. **H7 is not supported.**
- **Decision:** colour stays off for everything after E1.
- **Paper role:** "standard floorplans don't test memory" (Fig. 2 left).

## 4. E1b: field of view as a perception dial

**Question:** does narrowing the view create a memory gain (H7)?

**Design (40 runs, `e1b_fov`, 20261108114624):** FOV {90°, 180°, 270°, 360°} × MLP/LSTM × C_maze × seeds 100–104, with colour off and random goals.

| FOV | MLP | LSTM |
|---|---|---|
| 90° | 0.81 [0.73, 0.89] | 0.47 [0.08, 0.84] |
| 180° | 0.96 [0.95, 0.97] | 0.87 [0.74, 0.96] |
| 270° | 0.92 [0.76, 0.96] | 0.94 [0.89, 0.95] |
| 360° | 0.83 [0.59, 0.95] | 0.46 [0.10, 0.84] |

**Reading:**
- A narrower view hurts the LSTM *more*: there is no memory gain.
- The 180° row reproduces E1's C_maze cell exactly. Same seeds, and CPU training is deterministic, so this is a free reproducibility check.

---

## 5. E2: twin rooms ("remember"), the headline

**Question:** when memory is *required*, where does each backbone fail?

**Room design** (`scripts/generate_twin_rooms.py`, `rooms/twin_rooms_L*.json`):
- **Layout:** a start alcove with a **cue** (one of two colours) → a short shaft that bends 90° → a corridor of length **L** → a junction → **two geometrically identical rooms** with identical wall colours.
- **Task:** the cue says which room holds the goal, and entering the wrong room ends the episode.
- **Memoryless ceiling:** 0.5.
- **The bend hides the cue.** In the first design the corridor was straight, and a 180° FOV's edge ray saw the cue wall from the junction, a leak. A test checks with 360° rays from every reachable point that the cue is hidden.
- **Junction shaping** (`env.junction_shaping=1.0`), added after the pilot:
  - Potential-based, F = γΦ(s′) − Φ(s) with Φ = −(path distance to the junction) / (that distance at the start), and Φ = 0 from the junction on.
  - It carries no information about which room is right, and it leaves the optimal policy and the 0.5 ceiling unchanged (Ng 1999).
- **Episode length:** 2L + 256 steps, γ = 0.999. **Encoder:** MLP (decided Nov 10; a CNN costs about 3×).

**Design (175 runs, `e2_twin_rooms`, 20261110145407):**
- L ∈ {8, 32, 64, 128, 256}.
- 7 backbones: MLP, FS4, FS16, LSTM with BPTT 16 / 64 / 256, and GRU at BPTT 64.
- Seeds 100–104.

### 5.1 The pilot that changed the design (unshaped, L = 32, 35 runs)

- **Seeds were all or nothing, and the gate was getting out of the start.** Every seed that reached the junction also found the goal, so the success differences across backbones measured exploration, not memory.
- **The fix:** junction shaping.
- **Acceptance test for the shaped task:** every seed reaches the junction, and memoryless seeds sit at 0.43–0.57.
- **Shaped L = 32 pilot:**
  - Every seed of every backbone reaches the junction (0.96–1.00).
  - MLP and FS4 sit at chance, and every recurrent seed solves.
  - **FS16 solved 2/5 at L = 32 "through the body"** (5.3).

### 5.2 Main result (diagnostics on 1000 deterministic episodes per seed; solved = success ≥ 0.9)

| Backbone | L = 8 | L = 32 | L = 64 | L = 128 | L = 256 |
|---|---|---|---|---|---|
| MLP | 3/5 (body) | 0/5 | 0/5 | 0/5 | 0/5 |
| FS4 | 5/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| FS16 | 5/5 | 2/5 (body) | 0/5 | 0/5 | 0/5 |
| LSTM-BPTT16 | 5/5 | 5/5 | 3/5 | 0/5 | 0/5 |
| LSTM-BPTT64 | 5/5 | 5/5 | 3/5 | 1/5 | 0/5 |
| LSTM-BPTT256 | 5/5 | 5/5 | 5/5 | 4/5 | 0/5 |
| GRU (BPTT64) | 5/5 | 5/5 | 4/5 | 4/5 | 0/5 |

**How to read it:**
- **L = 8 needs almost no memory:** the cue is close enough to steer by.
- **From L = 32 the memoryless backbones sit at chance.**
- **Past L = 32 the BPTT window sets the horizon:** BPTT16 drops at 64, BPTT256 holds to 128, and GRU at window 64 beats LSTM at window 64 at L = 128 (4/5 vs 1/5).
- **L = 256 is solved by no backbone,** including window 256 (the cue is about 257 steps old at the junction).
- **Every seed reaches the junction** (≥ 0.90), so every failure is a memory failure, not an exploration failure.

### 5.3 Diagnostics: proving where the cue is kept (`scripts/e2_diagnostics.py`)

The checks, and what they found:

| Check | What it does | Finding |
|---|---|---|
| **Swap** (the decisive one) | At mid-corridor, take the recurrent state, the pose, or both from an episode of the other cue | All 64 solved recurrent seeds follow the swapped *recurrent state* (success 0.00–0.07) and ignore the swapped pose (0.93–1.00) |
| Reset sweep | Zero the hidden state at 0, 25, 50 or 75% of the corridor | Drops solved seeds to chance (median 0.50) |
| Reset at the junction | Zero it on entering the junction | Costs much less (median 0.65–0.93): the choice is already committed in the heading |
| Mirror | Reflect the pose about the corridor's centre line | Mostly no effect (0.72–1.00) |
| Pose probe | Decode the cue from the lateral offset, heading and last action at mid-corridor | Shows the body *carries* the cue, not that the choice *uses* it |
| Confusion matrix / linear probe | Room chosen vs cue; cue decoded from the state | Supporting evidence |

- **FS16 body memory at L = 32:**
  - The cue was last seen 33 steps before the junction, twice FS16's window.
  - Seeds 100 and 103 steer to one side of the corridor depending on the cue (lateral offset +11.6 vs −13.4), and keeping to that side is reactive.
  - At L ≥ 64 this disappears.
- **The memoryless "body" route is a finding,** not a leak to remove: the body is a memory too.

**Bug found on the way (Nov 13):**
- The junction box was only as tall as the corridor, so corner-cutting agents were not counted as "at the junction".
- Fixed in the diagnostics (`at_junction()` also counts any point past the corridor's end), and every diagnostic was rerun.
- The training shaping latch had the same issue. It is fixed for E3T at L = 64; earlier runs keep the old latch, which leaves the optimal policy unchanged.

### 5.4 Layer check (10 runs, 20261112142837)

- **Question:** does a deeper LSTM fix BPTT16's failure at L = 64?
- **Design:** LSTM-BPTT16 with 1 and 3 layers at L = 64, seeds 100–104. The 2-layer seeds from E2 are the reference.
- **Result:** solved 4/5 (1 layer), 3/5 (2 layers) and 2/5 (3 layers), within 5-seed noise.
- **Why it can't help:** layers stack per step and don't lengthen the gradient path in time. The 2 layers stay.
- **Status:** exploratory.

### 5.5 K = 4 rooms (30 runs, `e2_twin_rooms_k4`, 20261118095532), exploratory

- **Design:** four identical rooms (cue red / blue / green / yellow) at L = 32 and 128.
  - E2's three best backbones: GRU-BPTT64, LSTM-BPTT256, LSTM-BPTT64.
  - Seeds 100–104.
- **What it asks:** memoryless chance is 0.25, and the state must hold 2 bits instead of 1.

| Backbone | L = 32, K = 4 | K = 2 | L = 128, K = 4 | K = 2 |
|---|---|---|---|---|
| GRU-BPTT64 | 4/5 | 5/5 | 0/5 | 4/5 |
| LSTM-BPTT256 | 3/5 | 5/5 | 0/5 | 4/5 |
| LSTM-BPTT64 | 0/5 | 5/5 | 0/5 | 1/5 |

**Reading:**
- 7 of 30 seeds solve, against 24 of 30 with two rooms.
- **The failures sit at chance** (they pick one or two fixed rooms) rather than "one bit right".
- **That is a learning failure, not a capacity limit:** a 25% reward signal for a 4-way choice. This design can't separate the two.
- **The partial seeds were still improving at the budget.**
- **Proposal:** appendix, exploratory.

---

## 6. E3: information sources ("be told")

**Question:** does telling the agent the goal or its position substitute for memory, in an open layout but not in a dead-end one?

**Design (180 runs, `e3_information`, 20261110205630):**
- 9 conditions × MLP/LSTM × {four_room (open), C_maze (dead ends)} × seeds 100–104.
- Random goal in every condition, and the same input size everywhere (unused slots are zeros).

| Group | Condition | Agent is told |
|---|---|---|
| — | I0 | nothing |
| Goal | G1 | oracle goal ray (distance and bearing through walls) every step |
| Goal | G2a / G2b | goal last seen, updated by odometry / frozen, plus its age |
| Position | P1 | raw (x, y) |
| Position | P2 | random Fourier features of (x, y) |
| Position | P3 | hand-made hexagonal grid code (Solstad 2006) |
| Position | P4 | P3 from a noisy path-integrated position |
| Both | P5 | grid code of the agent and of the goal |

**Result (C_maze; four_room saturates at 0.98–1.00 everywhere):**

| Condition | MLP | LSTM |
|---|---|---|
| I0 | 0.93 | 0.94 |
| G1 oracle goal | 0.94 | **0.52** (seeds 102 and 103 fail) |
| G2a / G2b | 0.94 / 0.95 | 0.97 / 0.83 |
| P1 / P2 / P3 / P4 / P5 | 0.92 / 0.88 / 0.88 / 0.89 / 0.91 | 0.91 / 0.80 / 0.86 / 0.73 / 0.79 |

**Reading:**
- **With nothing extra (I0) there is no memory gap,** so H3 can't be tested in these rooms.
- **H4 is not supported:** the frozen goal and the noisy grid code are where the LSTM trails most.
- **H5 holds as a null result:** the grid code is no better than Fourier features.
- **Puzzle:** the oracle goal *hurts* the LSTM, while the MLP with the oracle paths best (SPL 0.61–0.76).
  - Input scale is ruled out: the goal slot is normalised.
  - Untested explanation: the through-wall bearing traps it in the C's dead end.
- **No LSTM − MLP difference survives Holm.**
- **Consequence:** the information question moved to the twin rooms (E3T).

## 7. E3T: told or remembered, in the twin rooms

**Question (H8):** how often must the agent be *told* the cue so that it no longer needs to *remember* it?

**Mechanism:**
- `env.information.cue_info=every` with `cue_interval=N` puts a one-hot cue in a reserved slot every N-th step and zeros otherwise.
- The conditions are C1, C4, C16 and C64 (told every 1, 4, 16 or 64 steps), plus I0 (never told), G1 (told the goal) and P1 (told the position).

### 7.1 E3T at L = 32 (105 runs, 20261112133519)

- **Why L = 32:** a pre-registered rule. E2's LSTM-BPTT16 solved fewer than 4/5 seeds at L = 64.
- **Design:** 7 conditions × MLP / FS16 / LSTM-BPTT16 × seeds 100–104.

| Backbone | C1 | C4 | C16 | C64 | I0 | G1 | P1 |
|---|---|---|---|---|---|---|---|
| MLP | 5/5 | 5/5 | 0/5 | 0/5 | 0/5 | 5/5 | 0/5 |
| FS16 | 5/5 | 5/5 | 4/5 | 4/5 | 0/5 | 5/5 | 2/5 |
| LSTM-BPTT16 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 3/5 |

**Reading:**
- **MLP:** solves C4, which H8 predicted it wouldn't. It steers to the cued side between tellings, so the bit survives the 3 blank steps in its pose.
- **LSTM:** solves every interval. The swap shows it holds the cue in its state at C4 and C16. At C1 and G1 it ignores the swap: it reads the slot instead of remembering.
- **C64 is confounded:**
  - At L = 32 the agents reach the junction 50–79 steps into the episode, so **the step-64 telling lands at the decision.**
  - The timing check (`c64_timing.py`, which blinds the slot from the bend) shows FS16's and 3 of 5 LSTM seeds' C64 success is that late telling.

### 7.2 C64 with a random telling phase (15 runs, 20261115173643)

- **Fix:** `cue_phase=random`, so the first telling falls at a uniform step in 0…N−1 and the last telling before the choice is 0–63 steps old.
- **LSTM:** 5/5. It remembers: its success doesn't depend on a telling inside the corridor, and the swap flips every seed.
- **MLP:** 0/5, at chance.
- **FS16:** 5/5 by body memory. It succeeds even when blinded from the bend: the random tellings taught it to turn any cue into a lateral offset. This blurs "told vs remembered" again, so L = 64 was run next.

### 7.3 E3T at L = 64 (60 runs, 20261117110027), the clean version

- **Why L = 64:** at this length E2's FS16 never solves and LSTM-BPTT256 solves 5/5.
- **Design:** C4r, C16r, C64r (random phase) and I0 × MLP / FS16 / LSTM-BPTT256 × seeds 100–104.
- **The junction latch was fixed for these runs.**

| Backbone | C4r | C16r | C64r | I0 |
|---|---|---|---|---|
| MLP | 2/5 | 0/5 | 0/5 | 0/5 |
| FS16 | 5/5 | 5/5 | 1/5 | 0/5 |
| LSTM-BPTT256 | 5/5 | 5/5 | 5/5 | 5/5 |

**Reading: H8 holds cleanly.**
- **MLP:** at chance from C16r on. It partly solves C4r by steering between tellings.
- **FS16:** solves exactly while the last telling fits its 16 frames. Blinded from the bend it falls to 0.21–0.49, and it fails past its window.
- **LSTM:** solves everything, and the mid-corridor swap sends all 20 seeds to the wrong room. The cue is in the state.
- **Significant (Welch + Holm, α = 0.01):**
  - LSTM vs MLP at C16r, C64r and I0.
  - LSTM vs FS16 at C64r and I0.
- **Deck message:** being told replaces memory only inside the window.

---

## 8. E4: staged ablation ("perceive & act"), the method

**Question:** which of the remaining design choices (observation, action, reward, aux losses, capacity) help in hard mazes, and do cheap early stages pick the same winners as expensive late ones (H6)?

**Why staged:** a full factorial is unaffordable, so each theme is varied *around a base config* while the others stay fixed. The survivors go to the next stage on new seeds and harder rooms.

**Themes:**

| Theme | Levels |
|---|---|
| A. Observation | encoder {MLP, CNN} × FOV {90°, 180°, 270°} |
| B. Action | continuous / 3 discrete / 5 discrete |
| C. Reward | time penalty only / + collision penalty 0.01 / γ 0.99 vs 0.999 |
| D. Aux losses | none / depth / loop closure / both, on the LSTM with distances masked (a replication of Mirowski 2017) |
| E. Capacity | width {16, 64, 256}; depth aux at matched widths; layers per module {1, 3} |

**Promotion rule (pre-registered, `scripts/e4_promotion.py`):**
- Keep the theme leader, plus every config whose IQM CI overlaps it, up to ⌈n/3⌉.
- Add 1 random wildcard per theme (RNG seed 0, reported).
- The baseline is always kept.
- After Stage 1, each theme's leader becomes the new default ("re-baseline").
- Aux levels are ranked against a **matched reference**: the same config and masking with the aux head off.

**How to read the aux labels below:** `aux:a-b` means depth coefficient a and loop-closure coefficient b. `mask:3` means the distance channel is hidden from the policy.

### 8.1 Stage 1: screen (126 runs, `e4_stage1`, 20261102095921)

- **Goal:** cheaply drop clear losers. These are screening seeds, never reported as results.
- **Rooms:** C_maze + medium_maze. **Seeds:** 0–2. **Base:** MLP encoder, MLP backbone, continuous actions, no collision penalty, width 64, FOV 180°.
- **Runs:** 18 configs (108 runs), plus 18 runs of depth levels added afterwards.

| Theme | Leader (AUC IQM) | Kept / wildcard | Dropped |
|---|---|---|---|
| Observation | **CNN, FOV 180°: 0.87** | CNN 270° (0.85); wildcard MLP 90° | CNN 90° (0.84), MLP 270° (0.83), over the cap |
| Action | continuous (baseline 0.73) | wildcard discrete 3 (0.56) | discrete 5 (0.65) |
| Reward | **collision penalty 0.01: 0.86** | wildcard γ 0.99 (0.72) | |
| Capacity | width 256: 0.81; later **1 layer per module: 0.87** | wildcard width 16 | 3 layers (0.68, worse than its control) |
| Aux (vs masked LSTM 0.12) | depth + loop closure: 0.18 | wildcard loop closure | depth alone |

- **Stage 2 base after re-baselining:** CNN encoder, FOV 180°, continuous actions, collision penalty 0.01, width 64.
- **Two departures from the rule, both for cost:**
  - Width 256 stays a level instead of joining the base: it is 1.8× slower on the CNN.
  - The aux, memory and BPTT configs stay on the MLP encoder.
- **Worth noticing:** discrete 3 was a *wildcard* in Stage 1 (0.56, below the baseline) and became the Stage 2 leader. Wildcards exist to catch exactly this.

### 8.2 Stage 2: confirm (240 runs + 26 side runs, `e4_stage2`, 20261104133601)

- **Goal:** confirm the survivors on fresh seeds and in a harder room, and check that the ranking is stable (Kendall τ).
- **Rooms:** medium_maze + hard_maze. **Seeds:** 100–104. **Configs:** 24, including two added by hand:
  - **CNN + LSTM**, so that the best encoder and the recurrent backbone are tested together.
  - **CNN + LSTM + loop closure**, so that Stage 3 can't pair them untested.

Per-config AUC IQM over both rooms (from `promotion/20261104133601.json`):

| Block | Config | IQM | Promotion |
|---|---|---|---|
| Base | CNN + MLP, continuous, penalty 0.01 | **0.84** | leader of most themes |
| Action | **discrete 3** | **0.94** | **leader** |
| Observation | FOV 270° / MLP encoder / MLP encoder without penalty / MLP 90° | 0.80 / 0.66 / 0.43 / 0.14 | kept / wildcard / – / – |
| Reward | γ 0.99 / no collision penalty | 0.83 / 0.74 | wildcard / – |
| Capacity | width 256 / 1 layer per module / width 16 | 0.81 / 0.77 / 0.41 | kept / – / wildcard |
| Memory | **CNN + LSTM** | **0.62** | wildcard, loses to the base |
| Memory + aux | CNN + LSTM + loop closure | 0.01 | collapse, see below |
| BPTT (MLP-encoder LSTM) | window 16 (ref) / 64 / 64 + mb 16 / 256 / 256 + mb 4 | 0.09 / 0.02 / 0.13 / 0.01 / 0.11 | all near the floor |
| Aux (vs matched reference) | loop closure on the LSTM 0.12 vs 0.09; depth + loop 0.08 vs 0.06 (masked); depth 0.12 vs 0.12 (masked MLP) | | **no aux gain**: every CI overlaps its reference |

**How to read it:**
- **Only the action theme changes the base:** discrete 3 leads.
- **The CNN, penalty, width 64 and FOV 180° leaders are already the base.**
- **Memory loses:** CNN + LSTM scores 0.62 vs 0.84.
- **The MLP-encoder LSTMs stay near zero in hard_maze** at every window, even at equal updates (the BPTT confirmation, 20 runs).
- **Kendall τ between the stages = 1.0** on medium_maze over the 6 configs that ran in both, so no theme reopens.

**Side checks inside Stage 2:**
- **BPTT confirmation (`e4_bptt_confirm.sh`, 20 runs):** at equal updates no window is clearly better. Stage 2's "longer windows are worse" was an update-count effect.
- **CNN + LSTM + loop-closure collapse (6 runs, `e4_aux_collapse.sh`):**
  - 9 of 10 runs simply stopped moving: a stopped agent pays no collision penalty and gets no loop labels, so both losses go to 0.
  - Removing the penalty *or* lowering the aux coefficient to 0.1 removes the collapse (0.52 and 0.60 vs 0.15).
  - So it is an *interaction* (the collision penalty plus an aux loss that swamps the shared features), not "loop closure hurts". It is a diagnostic on 3 seeds, not a result.

### 8.3 Stage 3: final (100 runs, `e4_stage3`, 20261116094617)

- **Goal:** the claim test.
  - The best combination against each "leave one theme out" config and against Stage 1's baseline.
  - Fresh seeds 1000–1009, Welch + Holm at α = 0.01 over 4 comparisons × 2 rooms.
- **Frozen before launch.** Only three themes have something to leave out (encoder, action, reward); the others' leaders are the base.

| Config | Encoder | Action | Penalty | medium_maze IQM | hard_maze IQM |
|---|---|---|---|---|---|
| **best** | CNN | discrete 3 | 0.01 | **0.96** [0.93, 0.98] | **0.90** [0.87, 0.93] |
| −action | CNN | continuous | 0.01 | 0.88 (Holm p 0.054) | 0.71 (**p 0.0085**) |
| −encoder | MLP | discrete 3 | 0.01 | 0.83 (**p 0.0007**) | 0.80 (p 0.13) |
| −reward | CNN | discrete 3 | 0 | 0.89 (p 0.13) | 0.76 (p 0.13) |
| baseline | MLP | continuous | 0 | 0.60 (**p < 0.001**) | 0.38 (**p < 0.001**) |

**How to read it:**
- **Best wins every comparison on the mean, with the tightest seeds** (0.83–0.98). 4 of 8 tests pass at α = 0.01.
- **The non-significant ones come from single failing seeds** in the leave-one-out configs: −reward 0.12 / 0.10 on hard_maze, −encoder 0.09, −action 0.31.
- **So the three choices mostly make training reliable across seeds.** How much each helps a typical seed depends on the room.
- **Final success:** best 98.4% / 96.1% vs baseline 80.2% / 68.2%. **SPL:** 0.56 / 0.40 vs 0.31 / 0.22.
- **H6:** best is first and baseline last in both rooms; the three leave-one-out configs swap order between the rooms.
- **No seeds added to rescue the borderline tests.**

### 8.4 Rollout videos in the deck (hard_maze, median seed per config)

| Config | Stage, seeds | AUC | Episodes solved |
|---|---|---|---|
| CNN + MLP, discrete 3, penalty (best) | 3, 1000–1009 | 0.90 | 3/3 |
| MLP + MLP, discrete 3, penalty (best with the MLP encoder) | 3 | 0.79 | 3/3 |
| MLP + MLP, continuous, no penalty (baseline) | 3 | 0.30 | 2/3 |
| CNN + LSTM, BPTT 16, continuous, penalty (best LSTM) | 2, 100–104 | 0.62 | 3/3 |
| MLP + LSTM, BPTT 64, minibatch 16 (best MLP-encoder LSTM) | 2 | 0.11 | 1/3 |

- **The LSTM videos come from Stage 2,** because memory never reached Stage 3.
- **Compare within a row of the slide, not across rows:** the seeds and the action space differ.
- **All five are in the W&B run "E4 staged ablation"** (group `report_nov19`, panels "Rollouts/Stage 3 hard_maze" and "Rollouts/Stage 2 hard_maze").

---

## 9. Lessons that cut across experiments

1. **Check the task needs what you test.** E1 and E3 tested memory and information in rooms that need neither. The twin rooms were built to make memory required, and then *verified* (cue hidden by test, memoryless at chance, junction always reached).
2. **"Solved" is not "solved by memory".** The swap test is the decisive check. Pose probes only show that the body *carries* the cue.
3. **The body is a memory.** The MLP at L = 8 and C4, and FS16 at L = 32, bridge gaps by steering to one side.
4. **Timing confounds telling.** A fixed telling schedule plus a deterministic walk let agents time a late telling. The fix was a random phase.
5. **Training budgets hide in implementation details.** Update counts changed with the BPTT window, and the MLP's update scheme flips its room ranking.
6. **Auxiliary losses can break behaviour.** An aux loss that swamps the task gradient plus a cheap "stand still" option produced agents that stop moving.
7. **Wildcards and fresh seeds matter.** Discrete 3 was a Stage 1 wildcard and became the Stage 2 leader.

## 10. Where everything is

| Sweep | Euler folder (`/cluster/scratch/eazevedo/semproj/`) | Timestamp | Script |
|---|---|---|---|
| E0 pilot / probes | `e0_pilot`, `e0_room_probe`, `e0_episode_length` | 20261101160538 (pilot) | `scripts/e0_*.sh` |
| E1 | `e1_place_code_audit` | 20261108114624 | `scripts/e1_place_code_audit.sh` |
| E1b | `e1b_fov` | 20261108114624 | `scripts/e1b_fov.sh` |
| E2 | `e2_twin_rooms` | 20261110145407 | `scripts/e2_twin_rooms.sh` |
| E2 layers | `e2_twin_rooms` | 20261112142837 | `scripts/e2_lstm_layers.sh` |
| E2 K = 4 | `e2_twin_rooms_k4` | 20261118095532 | `scripts/e2_k4.sh` |
| E3 | `e3_information` | 20261110205630 | `scripts/e3_information.sh` |
| E3T L = 32 | `e3_twin_rooms` | 20261112133519 | `scripts/e3_twin_rooms.sh` |
| E3T C64r | `e3_twin_rooms` | 20261115173643 | `scripts/e3t_random_phase.sh` |
| E3T L = 64 | `e3_twin_rooms` | 20261117110027 | `scripts/e3t_L64.sh` |
| E4 Stage 1 | `e4_stage1` | 20261102095921 | `scripts/e4_staged_ablation.sh`, `e4_depth_screen.sh`, `e4_depth_control.sh` |
| E4 Stage 2 | `e4_stage2` | 20261104133601 | `scripts/e4_staged_ablation.sh`, `e4_cnn_lstm.sh`, `e4_cnn_lstm_aux.sh`, `e4_bptt_confirm.sh` |
| E4 Stage 3 | `e4_stage3` | 20261116094617 | `scripts/e4_stage3.sh` |
| E4 side checks | `e4_lstm_updates`, `e4_aux_collapse` | | `scripts/e4_lstm_updates.sh`, `e4_aux_collapse.sh` |

- **Layout of every sweep folder:**
  - `models/<timestamp>/<room>/<storage name>/seed_<s>/best.pt`
  - `eval_metrics/.../auc.npy`, `success_curve.npy` and the per-metric arrays
  - `stats/<timestamp>.json` from `seed_stats.py`
- **W&B:** project `eazevedo-eth-z-rich/semproj`. Each sweep has its own group, and group `report_nov19` holds one run per report section (curves, tables, rollout videos).
- **Report and deck:**
  - `reports/meeting_pdf/build_pdf.py` builds "Perceive Remember or Be Told - direction and results.pdf".
  - `reports/meeting_pdf/deck/build_deck.js` builds the meeting deck.

## 11. Questions to be ready for

- **"Why AUC and not final success?"** It captures learning speed and reliability. It is padded so that early stopping doesn't shorten the horizon. Final success and SPL are reported alongside.
- **"Why do the LSTMs lose in mazes?"**
  - The mazes are reactive (E1, E1b, E3), so memory has nothing to add and the LSTM is slower to train.
  - In medium_maze and hard_maze it also often fails to converge within 1000 iterations, even at equal updates and with tuned learning rates.
- **"Is the twin-room result just exploration?"** No. Shaping makes every seed reach the junction, and failures sit at chance with a fixed room choice.
- **"How do you know the LSTM uses memory?"** The swap: the other cue's recurrent state flips the room choice, and the other cue's pose doesn't.
- **"Why does nothing solve L = 256?"** The cue is about 257 steps old at the junction, just past the longest BPTT window. Gradient can't reach it.
- **"Isn't K = 4 a capacity result?"** Not with this design: the unsolved seeds learned no bit at all. A longer budget or a shaped second junction would be needed to separate the two.
- **"Why are only 4 of 8 Stage 3 tests significant?"** Single failing seeds in the leave-one-out configs inflate their variance. This is reported as tested, without rescue seeds.
- **"What does E4 say about memory and aux?"**
  - Memory: CNN + LSTM lost to CNN + MLP (0.62 vs 0.84) in mazes that E1 shows are reactive.
  - Aux: no gain against matched references. Loop closure at coefficient 1.0 collapsed the CNN + LSTM through an interaction with the collision penalty.
- **"Limitations?"**
  - The memory result was not checked on the CNN encoder.
  - E3's rooms couldn't test information vs memory.
  - nDTW is not comparable across goal modes.
  - K = 4 is exploratory.
  - The E2 and early E3T shaping used the box-only junction latch.
