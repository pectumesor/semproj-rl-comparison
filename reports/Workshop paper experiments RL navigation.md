# Make memory necessary, then measure it

The strongest workshop paper you can finish by early January is not a bigger ablation. It is a **conditional claim: "memory, auxiliary losses and goal information pay off only when the task's memory length exceeds what the observation already gives away"**, backed by three targeted experiments and a lean staged ablation. The experiments, in priority order, are: **(1) a cheap control that turns off the global Perlin colour field**, which currently acts as a position code and plausibly explains why the memoryless MLP+MLP wins; **(2) a K-identical-rooms cue task with a memory-horizon sweep L**, the headline crossover figure; **(3) a four-level goal-information factor** (none / per-step egocentric vector / start-only / noisy), crossed with backbones on an open layout and a dead-end layout. The full factorial (hundreds of thousands of combinations) should become a **themed, difficulty-as-fidelity successive-halving protocol of about 800 floorplan runs**, plus about 440 runs for the two new tasks. That is roughly 1,250 runs, or an estimated 9–11 days of wall-clock time on 16 Euler CPU slots; the per-run hours are unmeasured. Four code issues invalidate parts of the plan unless fixed in week 1:
- the global place code;
- a goal that is fixed per room (and evaluated from a single JSON start);
- an AUC that is not padded to the training budget, so early-stopped runs are scored on a different horizon;
- a depth auxiliary loss that regresses its own input channel.

The best confirmed-fit venue is an **ICLR 2027 workshop** (workshop list announced Nov 29, 2026; suggested paper deadline Feb 1, 2027, both CONFIRMED). **AAAI-27 workshops (Nov 20, 2026, CONFIRMED)** are a stretch that would only take current results. **eCAADe 2027** (abstract around mid-January, ESTIMATE) is the architecture-side twin.

## The MLP win is probably a place-code artifact, and four bugs must go first

Your current headline, "MLP+MLP seems to work the best", and "LSTM Backbone struggles to learn loop closure aux loss on complicated environments", come from the 20.10 meeting ([Weekly Progress PDF](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)). Read against the code, this result is unsurprising.

Each observation has 37 rays × 7 channels. Channels 4–6 are **Perlin RGB sampled at the hit point from a colour field over global (x, y)**, seeded by `cfg.seed` ([envs/navigation_env.py:68, 228-232](../envs/navigation_env.py#L228); [envs/env_utils.py:93-178](../envs/env_utils.py#L93)). A wall's colour therefore tells a reactive network where it is in world coordinates. On top of that, the goal is a single `(2,)` tensor read from the room JSON, broadcast to all envs, and **never resampled in `reset`** ([envs/navigation_env.py:61-66, 139-158, 213-219](../envs/navigation_env.py#L139)). A fixed goal plus a position-coding observation turns every room into a lookup from "colour pattern" to "heading". That is a task with memory length of a few steps.

The literature predicts exactly this. In Mirowski et al.'s static mazes the feed-forward agent was competitive, and memory and auxiliary losses only pulled ahead once goals were randomized ([arXiv 1611.03673](https://ar5iv.arxiv.org/html/1611.03673)). Ni et al. show that many nominal "memory" benchmarks need only short memory ([arXiv 2307.03864](https://arxiv.org/abs/2307.03864)). Wijmans et al. found that richer perception suppressed map-building in navigation agents ([arXiv 2301.13261](https://arxiv.org/abs/2301.13261)). The same mechanism has a second consequence: **two geometrically identical rooms are not visually aliased** while the global Perlin field is on, because their wall colours differ by position. Your "similar rooms" idea cannot work until colour is controllable.

The table lists what must change, ordered by how badly each issue damages the planned experiments. The P0 rows invalidate results if left alone. P1 rows are needed for a specific experiment or for clean reporting.

| Pri | Finding (source) | Why it matters for validity | Fix (est. effort) |
|---|---|---|---|
| P0 | Global Perlin colour as place code, seeded by RL seed ([navigation_env.py:68](../envs/navigation_env.py#L68)) | Confounds memory-vs-MLP. Breaks visual aliasing. Each training seed sees a different colour map, so RL-seed variance mixes with colour-map variance. | Add `colour_mode ∈ {global, off, room_local/periodic, per_wall}` and a separate `colour_seed` held fixed across RL seeds (½ day) |
| P0 | Fixed goal per room, broadcast to all envs; final eval uses the fixed JSON start ([navigation_env.py:139-158](../envs/navigation_env.py#L139), [train.py:103-111](../scripts/train.py#L103)) | Fixed-goal tasks need no memory (Mirowski). Final CR/DTW/NPL metrics come from one start–goal pair. | Make `goal_pos` shape (E, 2) and resample in `reset(done)` with a minimum distance. Evaluate over sampled pairs. Generalize `generate_reference_trajectory` to arbitrary start/goal ([navigation_mesh.py:282-297](../utils/navigation_mesh.py#L282), [metrics.py:243-286](../utils/metrics.py#L243)) (1–2 days) |
| P0 | AUC = `np.trapezoid(success)/(n_evals-1)` over the run's own evals, not padded to `n_iterations` ([ppo.py:399-416](../algorithms/ppo.py#L399)) | Equals the mean success over each run's own curve. A fast learner that early-stops and a slow learner that crawls to 95% can tie or swap ranks. It is unusable as a sample-efficiency metric and as the promotion metric. | Restore the 10-09 version: carry the last eval forward to the full budget and integrate over env steps (2 h) |
| P0 | Depth aux regresses `rays[..., 3, :]`, which is still a policy input ([ppo.py:222-236, 545-566](../algorithms/ppo.py#L222), [simple.py:18-22](../models/embeddings/simple.py#L18)) | The head reconstructs its own input, which your 12.10 notes already observed ("almost no learning signal"). Any "aux helps/doesn't help" claim is void. | When depth aux is on, mask channel 3 (and the no-hit channel 0) from the policy input, or predict depth for unobserved rays or the next step. Save aux heads in checkpoints (½ day) |
| P1 | `RayCast.intersect` discards which wall was hit ([env_utils.py:64-70](../envs/env_utils.py#L64)) | No per-wall cue colour or object type is possible | Return the argmin wall index. Add per-edge `color`/`type` to the room JSON (½ day) |
| P1 | `goal_radius` is both the success threshold and the visible goal disk (default 4, sweeps 16) ([base_env.yaml:23](../configs/env/base_env.yaml#L23)) | Goal-information levels need a goal whose visibility is controlled separately from success | Split into `success_radius` and `goal_visual_radius` (1 h) |
| P1 | `hard_rooms_v2` launcher was killed after ~23 min ([slurm-17612235.out](../slurm-17612235.out)) | No hard-room results exist yet | Re-run under the staged protocol with `timeout_min=720`; run the launcher as a long job or a SLURM array |
| P1 | Stale scripts override the removed `env.goal_pos.x/y` ([one_room_run.sh:22-23](../scripts/one_room_run.sh#L22); also `two_room_run.sh`, `euler_multirun_gpu_test.sh`) | They will likely fail under Hydra strict mode or silently confuse sweeps | Delete or update them, and keep one launcher per stage |
| P1 | medium_maze and hard_maze have **no spawn regions**, so they use bounding-box spawning ([rooms/](../rooms/), [utils/geometry.py](../utils/geometry.py)) | The same failure mode that put agents outside C_maze and cross_room (fixed in a427abf) can recur | Add hand-made spawn regions, as for C_maze (1 h) |
| P1 | Factory builds only `SimpleLSTM`; the NavA3C class is unused; grid cells not wired into PPO; `envs/curriculum.py` is empty ([factory.py:60-70](../utils/factory.py#L60), [train.py:40-47](../scripts/train.py#L40)) | Grid cells and curriculum are in the project title but cannot be ablated today | Add GRU (1 h). Implement curriculum radius growth only if time allows. State grid cells as future work rather than half-run them |

The supervisor asked for 10 evaluation runs × 10 episodes per seed, means of means, and standard errors ([Weekly Progress PDF](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)). Keep that evaluation scheme: it reduces evaluation noise. Note in the paper, though, that **the statistical unit is the training seed, not the evaluation repeat**. Report the per-seed values and SE as asked, and add IQM with stratified-bootstrap CIs, which reviewers now expect ([Agarwal et al., arXiv 2108.13264](https://arxiv.org/abs/2108.13264); [Patterson et al., arXiv 2304.01315](https://arxiv.org/html/2304.01315v2)).

## Eight candidate experiments ranked by insight per CPU-hour

The ranking puts first the experiments that turn the negative MLP result into a mechanism, then those that reuse machinery you already need. Run counts are this report's estimates using 5 seeds unless stated. "Eng" is estimated implementation effort after the P0 fixes.

| Rank | Experiment | Question it answers | Runs | Eng | Verdict |
|---|---|---|---|---|---|
| 1 | **E1 Place-code control**: colour_mode {global, off} × backbone {MLP, LSTM} on two_room, T_maze, four_room, C_maze, medium_maze | Does the MLP advantage come from Perlin localization? | 100 | ½ day | Must do; doubles as Stage-0 timing |
| 2 | **E2 K identical rooms + cue, memory-horizon sweep L** | When does each memory architecture stop working? | ~280 | 3–4 days | Headline figure |
| 3 | **E3 Goal-information levels** G0–G3 × {MLP, FS8, LSTM, GRU} × {four_room, C_maze} | Does a goal vector make memory redundant, and where does that break? | 160 | 2 days (shares E2 machinery) | Second headline |
| 4 | **E4 Staged ablation** of encoder, memory, aux, capacity on the 9 existing rooms | The "comparative study" spine of the project | ~800 | 1 day (launchers) | Required, but lean |
| 5 | **GRU drop-in** inside E2–E4 | Is the LSTM simply the wrong RNN? POPGym found GRU the best general memory model ([arXiv 2303.01859](https://arxiv.org/abs/2303.01859)) | included | 1 h | Do |
| 6 | **Capacity curve + aux at matched parameters** (widths {16…256}, aux on/off) | Your 20.10 idea: how few parameters suffice, and when aux beats width | inside E4 Theme D (+~30) | ½ day | Do, after the depth fix |
| 7 | **FOV/ray-density sweep** | A controllable partial-observability axis; memory should help more as FOV shrinks (hypothesis) | ~90 | none | Do if slots are idle |
| 8 | **Held-out layouts** (train on N ∈ {1, 4, 16} generated mazes, test on unseen) | Generalization, which matters for Arxitect; memorizing one layout favours the MLP ([Cobbe et al., arXiv 1812.02341](https://arxiv.org/abs/1812.02341)) | ~120 | 2–3 days (generator + spawn regions) | Stretch; strongest for eCAADe |

Several ideas should be deferred:
- **Transformer or SSM backbones.** Memory Gym found PPO+TrXL lost to GRU on its endless tasks, and its full study took about 25,000 GPU-hours ([Pleines et al., arXiv 2309.17207](https://arxiv.org/html/2309.17207v1)).
- **A small VLA or LLM baseline.** It is costly, prompt-confounded, and ray vectors are an unnatural VLA input. Cite TravelAgent's LLM occupant agents instead ([arXiv 2412.18985](https://arxiv.org/pdf/2412.18985)).
- **Sequential multi-goal routes**, until E3 exists.
- **Real hospital plans**, until E8 exists.
- **Grid cells and SAC.**

The cheap stand-in for the grid-cell arm is to add velocity and previous action as inputs under the start-only goal condition (G2). Mirowski used exactly these inputs ([arXiv 1611.03673](https://ar5iv.arxiv.org/html/1611.03673)), and Banino et al. showed that path-integration codes carry the goal vector ([Nature 2018](https://www.its.caltech.edu/~jkenny/nb250c/papers/Banino-2018.pdf)).

### E1 decides the story in a week

Hold everything at the current best config and flip only `colour_mode`. Use a fixed `colour_seed` so all RL seeds see the same map. The outcome determines the paper's framing either way:
- **If the MLP's lead shrinks or flips with colour off**, the paper has its mechanism: the original rooms were near-Markov because of a place code.
- **If the lead persists**, the explanation shifts to the fixed goal plus the visible goal beacon. E3's G0-random condition then tests that directly.

This is the "mechanistic explanation of the negative result" that ICBINB-style workshops reward ([ICBINB 2026](https://sites.google.com/view/icbinb-2026/submit)). Log wall-clock per iteration per room here. Those numbers replace every compute estimate below.

### E2 is a K-way passive T-maze in continuous 2D

The template is Ni et al.'s Passive T-Maze, with these properties ([arXiv 2307.03864](https://arxiv.org/abs/2307.03864)):
- the cue is shown only at the start;
- a corridor of tunable length follows;
- dense per-step reward for progress keeps credit-assignment length near 1 while memory length grows;
- a memoryless policy can at best guess.

MiniGrid-Memory is the closest existing analog. Its cue sits in the start room, the hallway sets the gap, and the wrong object ends the episode with zero reward ([MiniGrid docs](https://minigrid.farama.org/environments/minigrid/MemoryEnv/)).

The concrete design for your env:
- **Layout.** A start alcove holds a cue panel whose wall colour is one of K colours, sampled per episode. A corridor runs from the alcove to a hub with **K geometrically identical rooms**. Colour is set to `off` or `per_wall`, with identical colours inside the K rooms.
- **Goal.** It sits behind a turn inside the room matching the cue, so no ray can see it from the hub.
- **Commit rule.** Entering a wrong room terminates with reward 0. This is essential: without it, an MLP can visit all K rooms in sequence and still "succeed".
- **Reward.** Add a small dense progress term along the corridor, as in the Passive T-Maze.
- **Memory horizon L.** L is the number of steps between the last cue sighting and the commit. Set it by corridor length; at full speed the agent covers 13 world units per step ([base_env.yaml](../configs/env/base_env.yaml)), so L steps ≈ 13·L units. Alternatively, lower the agent speed, as Memory Gym does.

The sweep design:
- **L values:** L ∈ {8, 16, 32, 64, 128, 256}, chosen to bracket frame-stack sizes and BPTT windows.
- **Architectures:** MLP, frame stacks 4 and 16, LSTM with BPTT {16, 64, 256}, and GRU-64.
- **Main grid:** K = 2 at 5 seeds, ≈ 210 runs.
- **K = 4 check:** at L ∈ {16, 128}, ≈ 70 runs.
- **Pilot first:** 1 seed at L = 32 and L = 128.

The predicted pattern, which is a hypothesis, has four parts:
- the MLP sits at 1/K for every L;
- frame stack k collapses once L > k;
- LSTM/GRU hold until L exceeds roughly the BPTT window;
- the BPTT window × L heatmap shows a diagonal boundary.

That last panel makes your existing BPTT ablation meaningful for the first time. For calibration, Ni's LSTM agents began to fail at memory length 250 ([arXiv 2307.03864](https://arxiv.org/abs/2307.03864)). Memory Gym used GRU with sequence length 512 under PPO ([arXiv 2309.17207](https://arxiv.org/html/2309.17207v1)).

Validate memory use at evaluation time with three checks:
- **Reset-at-hub.** Zero the hidden state or frame stack at hub entry. Success should fall to 1/K; if it does not, geometry is leaking the answer.
- **Linear cue probe on h_t versus time since cue.** This follows Wijmans et al., whose probes decoded collision state at 98% accuracy, including untrained-network controls ([arXiv 2301.13261](https://arxiv.org/abs/2301.13261)).
- **Choice-versus-cue confusion matrix.**

All three are eval-only and cost no training runs.

### E3 crosses goal information with topology

Use four goal-information levels, with goals randomized per episode:
- **G0:** no goal input (current undirected search).
- **G1:** an egocentric (d/d_max, cos θ, sin θ) vector every step, the DD-PPO encoding ([arXiv 1911.00357](https://ar5iv.labs.arxiv.org/html/1911.00357)).
- **G2:** the same vector shown for the first 5 steps, then zeroed.
- **G3:** the vector visible with p = 0.1 per step, or with 15–30° bearing noise.

Keep the input width constant at 4 dims (vector + visibility flag), use the same sparse reward everywhere, and add no distance shaping to G1. The goal must be relative to the agent. In a 2025 single-room PPO test, an absolute goal was "equivalent to removing the goal input" ([arXiv 2503.15693](https://arxiv.org/html/2503.15693v1)).

Run four_room (open) and C_maze (dead ends) with MLP, FS8, LSTM and GRU at 5 seeds, for 160 runs. The expected interaction, again a hypothesis:
- **G1:** closes the MLP–LSTM gap on four_room but not on C_maze, where greedy steering into dead ends punishes reactive policies.
- **G2:** restores the LSTM advantage, since memory must carry the goal vector.
- **G3:** separates graceful (recurrent) from sharp (MLP) degradation.

Prior work brackets this. With perfect GPS+compass, DD-PPO's blind agent reached 0.973 success but 0.729 SPL, and its SPL fell to about 0.3 on long routes ([arXiv 1911.00357](https://ar5iv.labs.arxiv.org/html/1911.00357)). Removing per-step localization under noise collapses success to about 0.3% ([arXiv 2108.11550](https://arxiv.org/pdf/2108.11550v1.pdf)). Report success, SPL, and SPL binned by geodesic distance. SPL is the community standard ([Anderson et al., arXiv 1807.06757](https://arxiv.org/abs/1807.06757)). For a cheap generalization add-on, hold out one goal region at evaluation.

## A staged protocol of about 800 floorplan runs replaces the full factorial

Two lessons from the literature shape the protocol. First, Andrychowicz et al. found that sampling all choices jointly produces mostly failed agents, and that varying one choice while holding interacting choices fixed "may be misleading". Their answer was **thematic groups around a competitive base config**, scored by an AUC-like mean of evaluations with the median over 3 seeds ([arXiv 2006.05990](https://arxiv.org/abs/2006.05990)). Second, cheap-fidelity screening has a known bias:
- DyHPO reports that strong final configs looked weak at low budget in up to 40–80% of cases ([arXiv 2202.09774](https://arxiv.org/pdf/2202.09774)).
- In RL, hyperparameter optima transfer worst across environments ([Obando-Ceron et al., arXiv 2406.17523](https://arxiv.org/html/2406.17523v3)).
- RL hyperparameter landscapes also shift over training time ([Mohan et al., arXiv 2304.02396](https://arxiv.org/abs/2304.02396)).

Easy rooms are therefore a legitimate but biased proxy, and the protocol has to measure that bias rather than assume it away. No published study uses environment difficulty as the fidelity axis, so present it as your own protocol, justified by these sources.

The key adaptation for your project is that **the memory theme must not be eliminated on easy rooms**. On fixed-goal rooms with a place code, memory configs pay only their optimization cost. Treat the memory theme as follows:
- screen it on the floorplans only for "does it hurt?";
- carry all of its configs to Tier 2;
- decide it on E2 and E3, where L and goal information are its real fidelity axes.

Perception, capacity and aux choices can follow the room ladder.

### Themes, tiers and decision rules

Run E1 first, then set the base config: E1's winner, with `colour_mode` fixed for the rest of the study and padded AUC as the metric. With the stale and unwired switches removed, the remaining themes are:

| Theme | Configs |
|---|---|
| A. Encoder × FOV | {MLP, CNN} × {90°, 180°, 270°} = 6 |
| B. Memory | MLP, FS4, FS8, FS16, LSTM-BPTT{16, 64, 256}, GRU = 8 |
| C. Aux | none, depth (masked), loop closure, both = 4 |
| D. Capacity | width {16, 32, 64, 128, 256} = 5, plus aux-on at widths {16, 64} as the matched-parameter comparison = 2 |
| E. Curriculum (optional) | {none, radius growth}, Tier 2 only, if implemented |

That gives about 25 configs instead of the full cross product.

The room tiers are taken from your inventory and should be confirmed by Stage-0 baseline AUC:

| Tier | Rooms | Episode length |
|---|---|---|
| Smoke test | one_room | 512 |
| 1 (easy) | two_room, T_maze, I_maze | 512 |
| 2 (medium) | four_room, cross_room, C_maze | 4096 for cross_room and C_maze, as in `hard_rooms.sh` |
| 3 (hard) | medium_maze, hard_maze | 4096 |

Cross_room and C_maze use the longer horizon because cross_room failed at the default length ([Weekly Progress PDF](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf); [hard_rooms.sh:40-76](../scripts/hard_rooms.sh#L40)). Episode length and gamma are tier properties: fix them across configs and document them.

Pre-register these rules in a dated commit before Stage 1:
- **Primary metric:** budget-padded success AUC over env steps.
- **Promotion:** keep the theme leader, plus every config whose IQM bootstrap CI overlaps the leader's (equivalently, P(leader > config) < 0.75). Cap survivors at k = max(2, ⌈n/3⌉), i.e. η ≈ 3 as in Hyperband/ASHA. Break ties toward fewer parameters or a shorter BPTT window.
- **Wildcards:** add one uniformly random eliminated config per theme, using a reported RNG seed. This breaks the range restriction that would otherwise inflate the rank-correlation check.
- **Baseline:** always carry the baseline value of each theme forward.
- **Early kill:** stop a run at 50% of its budget only if it has zero success on every room while the theme leader exceeds 0.5.
- **Fidelity check:** a probe panel of 4 fixed configs runs on T_maze, C_maze and medium_maze. If the probe-panel Kendall τ between adjacent tiers falls below 0.4, or its bootstrap CI includes 0, re-open that theme at the next tier with 2k survivors. The 0.4 threshold is a heuristic and must be labelled as such; report a THC consistency score per theme alongside τ ([Obando-Ceron et al., arXiv 2406.17523](https://arxiv.org/html/2406.17523v3)).
- **Re-baseline:** after Stage 1, set each theme's leader as the new default for the other themes.
- **Final stage:** compose the best config and run a Rainbow-style leave-one-out ablation on fresh seeds.

### Seeds, statistics and run counts mapped to your rooms

Screening is selection, not inference, so screening stages report no p-values. Use separate seed sets so the final numbers are not inflated by selection, which Eimer et al. and Patterson et al. both warn about ([arXiv 2306.01324](https://arxiv.org/abs/2306.01324); [arXiv 2304.01315](https://arxiv.org/html/2304.01315v2)):

| Stage | Seeds |
|---|---|
| Screening | 0–2 |
| Stage 2 | 100–104 |
| Final | 1000–1009 |

Final claims use 10 seeds per room. By Colas et al.'s power analysis, that detects large effects (ε ≈ 2 needs N ≈ 5–10), while medium effects need N ≈ 20 ([arXiv 1904.06979](https://arxiv.org/html/1904.06979v2)). Test with Welch's t at α = 0.01 and Holm correction across the leave-one-out comparisons. Report IQM with stratified-bootstrap CIs and probability of improvement from `rliable` ([arXiv 2108.13264](https://arxiv.org/abs/2108.13264)). AdaStop can add seeds only to comparisons that remain undecided ([arXiv 2306.10882](https://arxiv.org/abs/2306.10882v2)). Patterson et al. are blunt that "5 runs is insufficient to make strong claims" ([arXiv 2304.01315](https://arxiv.org/html/2304.01315v2)).

The cost column below is this report's estimate. It rests on three assumptions:
- every PPO iteration is 65,536 env steps (256 envs × 256 steps) ([ppo.yaml](../configs/algorithm/ppo.yaml));
- 4096-step-episode runs take about 30 s per iteration, up to about 8.3 h for 1000 iterations ([hard_rooms.sh:48-65](../scripts/hard_rooms.sh#L48));
- early-stopped runs take roughly 1 h on Tier 1, 3 h on Tier 2 and 8 h on Tier 3, across 16 concurrent CPU jobs ([Weekly Progress PDF](/Users/emers/Downloads/Semester%20Project%20-%20Weekly%20Progress.pdf)).

Replace these with E1's measured timings.

| Block | Configs × rooms × seeds | Runs | Est. slot-hours |
|---|---|---|---|
| E1 place-code control (= Stage 0 timing) | 4 × 5 × 5 | 100 | ~300 |
| Probe panel (T_maze, C_maze, medium_maze) | 4 × 3 × 3 | 36 | ~150 |
| Stage 1 screen, Tier 1 | ~25 × 3 × 3 | ~225 | ~225 |
| Stage 2, Tier 2 (survivors + wildcards + all of memory theme) | ~12 × 3 × 5 | ~180 | ~540 |
| Stage 3 leave-one-out, Tier 3 (best, baseline, 4 LOO, 2 runner-ups) | 8 × 2 × 10 | 160 | ~1,280 |
| Stage 4 final, one room per tier (best, baseline, best-without-memory) | 3 × 4 × 10 | 120 | ~420 |
| E2 aliased cue rooms | 7 × 6 L (+K=4) × 5 | ~280 | ~420 |
| E3 goal-information levels | 4 × 4 × 2 × 5 | 160 | ~400 |
| **Total** | | **~1,260** | **~3,700 ≈ 9–11 days at 16 slots** |

The hard-room leave-one-out dominates the cost even though it is only an eighth of the runs. If Stage 0 timings come in higher than assumed, cut Stage 3 to medium_maze only before cutting seeds.

## One storyline and four figures fit a four-page workshop paper

A working title is "When does memory pay off? Design choices for ray-cast indoor navigation agents". Its arc is a negative result explained, then reversed under controlled conditions. That shape suits Jordan et al.'s call for hypothesis-driven "scientific testing" over leaderboards ([ICML 2024](https://proceedings.mlr.press/v235/jordan24a.html)), and Revisiting Rainbow's case that small environments yield valid insight cheaply ([arXiv 2011.14826](https://arxiv.org/abs/2011.14826)).

**Figure 1** has the complexity axis on x (Tier 1 → Tier 3). It shows IQM padded-AUC per design choice from the staged ablation, with the E1 colour-on/colour-off pair overlaid: "on standard floorplans memory buys nothing, and here is why." **Figure 2** is the headline: E2 success versus L on a log axis, per architecture, with a 1/K chance line and vertical lines at the frame-stack sizes, plus an inset BPTT-window × L heatmap. **Figure 3** is the E3 interaction plot of goal level × backbone, faceted by four_room versus C_maze. **Figure 4**, or an appendix figure, is the capacity curve with aux-at-matched-parameters points. It answers your "how few parameters" and "when does an aux loss beat width" ideas from 20.10.

The reset-at-hub and probe results go in one small table. Release the env with configs, which reviewers credit cheaply.

The architecture-side version of the paper reuses Figures 1 and 3. It adds trajectory statistics (path ratio versus A*, wall-distance histograms) and, if E8 happens, one converted Arxitect plan. It is positioned against heuristic occupant simulators such as cogARCH and SimArch ([cogARCH](https://researchportal.northumbria.ac.uk/en/publications/cogarch-a-cognitive-agent-framework-to-simulate-wayfinding-by-arc/); [arXiv 1807.03760](https://arxiv.org/abs/1807.03760)).

## Target ICLR 2027 workshops and treat November 20 as optional

| Venue | Key dates | Status | Fit |
|---|---|---|---|
| **ICLR 2027 workshops** | Workshop list to organizers **Nov 29, 2026**; suggested paper deadline **Feb 1, 2027**; author notification by Feb 26; workshops ~Apr 28–30 | CONFIRMED timeline ([ICLR CfW](https://iclr.cc/Conferences/2027/CallForWorkshops)); individual workshops and exact days unknown | Best. Short-paper tracks are mandatory at ICLR workshops ([ICBINB](https://sites.google.com/view/icbinb-2026/submit)). Watch for memory, world-model, agents, NeuroAI or negative-results workshops |
| **AAAI-27 workshops** | Papers **Nov 20, 2026**; notification Dec 2; Montréal Feb 22–23, 2027 | CONFIRMED general date ([AAAI-27](https://aaai.org/conference/aaai/aaai-27/aaai-27-workshop-program/)); individual workshops may differ | Weak to moderate: Human-Centric Agentic Mobility (4 pp, non-archival, Nov 20 AoE, [site](https://agentic-mobility-workshop.github.io)); NeuroAI 2.0 (4-pp non-archival short track, deadline unstated, [site](https://neuroai-multimodal-workshop.github.io)) |
| **eCAADe 2027** | Abstract ~mid-Jan 2027; full paper ~Apr 2027 | ESTIMATE from 2024/2025 editions ([eCAADe 2025](https://ecaade2025.metu.edu.tr/submission)) | Architecture twin (occupant wayfinding); archival |
| **CVPR 2027 embodied workshops** | Proposals Nov 4 and decisions Dec 22, 2026 (CONFIRMED, [CVPR CfW](https://cvpr.thecvf.com/Conferences/2027/CallForWorkshops)); papers ~Mar–Apr 2027 | ESTIMATE from 2026 calls ([WDFM-EAI](https://wdfm-eai.github.io/CVPR26/call_for_papers.html)) | Backup |
| **RLC 2027** | Call ~Jan, deadline ~early Mar 2027 | ESTIMATE from 2026 ([Amii](https://www.amii.ca/events/reinforcement-learning-conference)) | Full-paper follow-up if extended |
| Closed | NeurIPS 2026 workshops (suggested Aug 29) ([NeurIPS CfW](https://neurips.cc/Conferences/2026/CallForWorkshops)); CoRL 2026; CAADRIA 2027 (Jul 27, 2026) ([aggregator](https://www.myhuiban.com/conference/5719?lang=en-us)) | CONFIRMED / aggregator | — |

ICML, ICRA and IROS 2027 workshop timing is unverified and falls after the project ends. Prefer non-archival ML workshops: they keep RLC 2027 and eCAADe open. The archival options (NeuroAI full papers, W3PHIAI) do not.

### Prioritized timeline

| Week | Dates | Work | Exit criterion |
|---|---|---|---|
| 1 | Oct 29 – Nov 4 | P0 fixes (colour_mode + colour_seed, per-episode goals + navmesh for arbitrary pairs, padded AUC, depth masking), GRU, spawn regions, delete stale scripts; launch E1; pre-register commit | E1 running; timings logged |
| 2 | Nov 5 – 11 | Wall index + per-wall colour; cue-room generator; E2 pilot (L = 32, 128); goal-vector obs; launch Stage 1 | LSTM solves L = 32 above 1/K, or the design is fixed |
| 3 | Nov 12 – 18 | E2 main sweep; E3 sweep; Stage 1 analysis and promotion | AAAI go/no-go on Nov 16: submit only if E1 alone makes a clean 4-page story |
| 4 | Nov 19 – 25 | Stage 2 (Tier 2); E2 K = 4 + reset/probe diagnostics | Rank-consistency check passed or themes re-opened |
| 5 | Nov 26 – Dec 2 | Stage 3 leave-one-out on hard rooms (long jobs); read ICLR workshop list (Nov 29) | Target workshop chosen |
| 6 | Dec 3 – 9 | Stage 4; adaptive extra seeds; capacity/aux points | All figure data in hand |
| 7 | Dec 10 – 16 | rliable analysis, Figures 1–4; stretch E7/E8 if slots are idle | Figures frozen |
| 8–9 | Dec 17 – Jan 10 | Write the 4-page paper + appendix; supervisor review; eCAADe abstract draft | Draft complete before the project ends |
| — | ~mid-Jan / Feb 1 | eCAADe abstract (ESTIMATE); ICLR workshop submission (CONFIRMED suggested date; check the chosen workshop) | Submitted |

## Conclusion

The current results are not a failed comparison. They are evidence that the benchmark, as built, does not need memory. A global colour field plus a fixed goal leaves the observation nearly Markov, and the MLP wins because nothing else is required. That reframing changes what the remaining weeks are for: instead of more of the factorial, build tasks where the needed memory length is a dial you control (L in E2, goal visibility in E3), and show each architecture failing exactly where its horizon ends. The staged protocol then plays a supporting role. It stops being the paper and becomes a defensible, budget-aware check that the floorplan conclusions survive harder rooms and fresh seeds.

The biggest risk is not compute; it is validity. A comparison of aux losses where the depth head reconstructs its own input, an AUC scored over each run's own horizon, and "aliased" rooms that a colour map can tell apart would each sink a review on its own. Fixing them in week 1 costs about three days and makes every later run count. If E1 shows the MLP's lead disappearing once colour is off, you have a sharp, surprising result before mid-November. If it does not, E3's random-goal condition takes over the same explanatory role.
