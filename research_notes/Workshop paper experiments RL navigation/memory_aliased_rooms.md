# Memory-dependent navigation tasks: aliased rooms, cue recall, and when memory architectures pay off

Scope: design menu for a memory-requiring 2D ray-cast navigation task (PPO, continuous actions, rays with one-hot {empty, wall, door, goal(, color)} + distance), the known results on which memory architectures solve which task types, and how to verify that memory is actually used. Sources checked in this session are linked. Items recalled from background knowledge but not re-verified are marked "(not re-verified)".

---

## 1. Classic and recent memory benchmarks: what they look like and what they actually test

### Takeaway
The cleanest template for "memory is provably necessary" is the cue-then-decide T-maze family (Bakker 2001, then Ni et al. 2023 Passive T-Maze, MiniGrid-Memory, Passive Visual Match). A cue is shown only at the start, there is a corridor or distractor phase of tunable length, and the episode ends with a binary or K-way choice that a memoryless policy can only guess (expected success 1/K). Ni et al. 2023 also show that many "memory" benchmarks mix memory with long credit assignment. Your task should keep credit assignment short (dense shaping to the decision point) so it isolates memory.

### Cited Findings
**Formal framing (most useful for the paper's framing)**
- Ni et al. 2023 (NeurIPS oral) define the *memory length* of a POMDP and policy: the smallest n such that reward, transition, policy or value depends only on the last n observations. They define *credit assignment length* separately: how many steps until the greedy action's advantage shows up in n-step return. They prove that the optimal policy's memory length is bounded by max(reward memory length, transition memory length). — [Ni et al. 2023, arXiv 2307.03864](https://arxiv.org/abs/2307.03864)
- Ni et al.'s Table 1 manually estimates these lengths for prior tasks. T-Maze (Bakker 2001): T=70, memory=T, credit=T. MiniGrid-Memory 17x17: T=1445, policy memory ≤51, credit ≤51. Memory Maze 15x15: T=4000, credit ≤225. Passive Visual Match: T=600, memory T, credit ≤18. Mortar Mayhem: T=135, credit ≤25. PsychLab (Fortunato 2019): T=600. Key-to-Door and Active Visual Match are classed as credit-assignment tasks (credit=T). — [Ni et al. 2023, Table 1](https://arxiv.org/abs/2307.03864)
- Ni et al. argue that several works meant to test memory (Bakker 2001; Lambrechts 2021; Esslinger 2022; Hung 2018) "actually entangle memory and credit assignment together". They also show some POMDP benchmarks (PyBullet-P/V) need only short-term memory. — [Ni et al. 2023](https://arxiv.org/abs/2307.03864)

**T-maze family**
- **Passive T-Maze** (Ni 2023): a corridor of length L. The oracle state O at the start reveals goal G ∈ {G1, G2} and nothing else is informative until the junction J. Dense per-step reward rewards moving right, with a terminal +1 or −1 for the correct or wrong arm. Optimal Markovian return is 0.5 (it guesses); optimal return is 1.0. Memory length = T, credit assignment length = 1. — [Ni et al. 2023 §3.2](https://arxiv.org/abs/2307.03864)
- **Active T-Maze**: the oracle is one step *left* of the start, so the agent must take an unrewarded detour to see the cue. Credit assignment length becomes T as well. — [Ni et al. 2023 §3.2](https://arxiv.org/abs/2307.03864)
- Bakker 2001's T-maze is the original LSTM-RL cue-recall corridor (T=70 per Ni's Table 1). Lambrechts et al. use a parameterized T-maze (corridor length plus a stochasticity rate) derived from Bakker 2001, with γ=0.98. — [Lambrechts et al. TMLR 2022, arXiv 2208.03520](https://arxiv.org/abs/2208.03520); Bakker, "Reinforcement Learning with Long Short-Term Memory", NeurIPS 2001 (not re-verified: https://papers.nips.cc/paper/1953-reinforcement-learning-with-long-short-term-memory)
- Code for Passive/Active T-Maze, Passive Visual Match and Key-to-Door (DQN / SAC-Discrete, LSTM vs GPT) — [github.com/twni2016/Memory-RL](https://github.com/twni2016/Memory-RL)

**MiniGrid-Memory (closest existing analog to "identical rooms + cue")**
- Layout: a small start room containing a cue object, a narrow hallway, then a split with two objects, one matching the cue. The agent must go to the matching object. Success reward is 1 − 0.9·(steps/max_steps); the wrong object or a timeout gives 0 and ends the episode. Sizes are S11, S13, S13Random and S17Random. The observation is an egocentric 7x7 partial view encoding (object, color, state) per tile, so a color channel is already part of the observation. — [MiniGrid MemoryEnv docs](https://minigrid.farama.org/environments/minigrid/MemoryEnv/)

**Passive Visual Match / Key-to-Door (Hung et al. 2019, TVT)**
- Three phases. In P1 the agent sees a random target color. In P2 (distractor) it collects apples for immediate reward. In P3 it walks to the square with the matching color among several. Only the P2 length is varied. Key-to-Door is the credit-assignment counterpart: pick up a key in P1, apples in P2, open the door in P3. — [Ni et al. 2023 Fig. 2 description](https://arxiv.org/abs/2307.03864); original: [Hung et al. 2019, Nature Comms, arXiv 1810.06721](https://arxiv.org/abs/1810.06721)
- Example concrete Key-to-Door timing from a later paper: key phase 15 steps, apples 90 steps, door 15 steps, door reward 10. — [Zheng et al., Adaptive Pairwise Weights, arXiv 2102.04999, App. C](https://arxiv.org/pdf/2102.04999)

**Memory Maze (Pasukonis, Lillicrap, Hafner; ICLR 2023)**
- In randomized 3D mazes the agent must repeatedly navigate to one of several colored objects. Doing this quickly requires remembering object locations, wall layout and its own position. The paper provides an online RL benchmark, an offline dataset and an offline *probing* evaluation. A human baseline's reward rises within the episode as memory builds up. Agents trained with truncated BPTT succeed on small mazes but fall short of humans on large ones. — [arXiv 2210.13383](https://arxiv.org/abs/2210.13383)

**Memory Gym (Pleines et al.; ICLR 2023, endless version arXiv 2309.17207)**
- *Mortar Mayhem*: memorize a sequence of 10 commands shown at the start (clue task), then execute them in order. Failing a command ends the episode; each success gives +0.1. — [arXiv 2309.17207](https://arxiv.org/html/2309.17207v1)
- *Mystery Path*: an invisible path with pits. Falling resets the agent to the origin and a red cross marks the fall, so the agent must remember the path and its failure locations. Time limit 512 steps; difficulty is set by path size, agent speed and scale, and whether origin and goal are visible. — [arXiv 2309.17207](https://arxiv.org/html/2309.17207v1)
- *Searing Spotlights*: a dark arena revealed only by roaming spotlights. The agent must remember its own past actions and positions, plus coin and exit locations. — [arXiv 2309.17207](https://arxiv.org/html/2309.17207v1)
- *Endless* versions run until failure, so performance measures memory effectiveness rather than sample efficiency. Example: Endless Mortar Mayhem shows commands one at a time and the agent must re-execute all commands shown so far. — [arXiv 2309.17207](https://arxiv.org/html/2309.17207v1)

**POPGym (Morad et al., ICLR 2023)**
- 15 lightweight POMDPs (e.g., Repeat First/Previous, Autoencode, Count Recall, Battleship, Minesweeper, noisy CartPole/Pendulum) and 13 memory models. Morad et al. conclude the GRU is the best general-purpose memory model, with Elman RNNs as the best performance/efficiency tradeoff. They report "a notable disconnect" in that old RNNs beat linear transformers and newer models in RL. — [arXiv 2303.01859](https://arxiv.org/abs/2303.01859)

**Other suites (secondary)**
- Fortunato et al. 2019 Memory Tasks Suite (DMLab/Unity-style tasks: Spot the Difference, Goal Navigation, PsychLab-like tasks) tests memory generalization along several axes. Code: dm_memorytasks. — [NeurIPS 2019, arXiv 1910.13406](https://arxiv.org/abs/1910.13406)
- MIKASA (Cherepanov et al. 2025; listed as ICLR 2026) offers a classification framework for memory-intensive tasks in four categories, MIKASA-Base, and 32 memory-intensive robot manipulation tasks. — [arXiv 2502.10550](https://arxiv.org/abs/2502.10550)
- POPGym Arcade (2025): pixel POMDPs in JAX with memory analysis tooling. — [arXiv 2503.01450](https://ar5iv.labs.arxiv.org/html/2503.01450) (only the title and existence were confirmed via search)

### Inferences
- Your "K identical rooms, one has the goal, color cue earlier" task is a K-way, continuous-control, ray-cast generalization of MiniGrid-Memory and the Passive T-Maze. Framing it explicitly with Ni et al.'s memory-length vs credit-length definitions gives the paper a principled hook: "we vary memory length while keeping credit assignment length short".
- Your current rooms (find a visible goal in a floorplan) likely have policy memory length ≈ 1–few steps, which is consistent with the MLP winning. Ni's Table 1 makes the same point about many benchmarks.

### Gaps
- KeyCorridor (MiniGrid), DMLab `explore_goal_locations` / natlab, Psychlab and the Morris water maze for agents were not re-checked in this session. Their exact parameters are not given here. (A water-maze-like analog is the "multi-trial, fixed hidden goal" design in §3.)
- The robotics perceptual-aliasing literature (e.g., Whitehead & Ballard 1991; McCallum 1996 utile distinctions; Chrisman 1992) was not searched. Cite only after checking.
- I could not fetch the full Hung et al. 2019 text (PMC blocked by a CAPTCHA). Exact P1/P2/P3 durations for the original Passive Visual Match come only from Ni's reproduction (memory length 60–1000 by varying P2).

---

## 2. Which memory architectures win on which task types; cheap CPU-trainable options

### Takeaway
On pure cue-recall with short credit assignment, transformers (GPT/TrXL) scale to much longer gaps than LSTM in Ni et al. (1500 vs about 250 steps). On long, endless or continual memory tasks trained with PPO, and on POPGym, GRU is the robust default and often beats TrXL. Frame stacking solves anything whose memory length fits inside the stack (Memory Gym: 16 frames suffice for visible-goal Mystery Path), so the memory horizon must be swept well past the largest stack. Cheap additions: GRU (a must), Elman RNN, S5/linear-recurrent models, and a small 1–2 layer transformer with a sliding window. All are CPU-feasible at the hidden sizes these tasks need.

### Cited Findings
- **Transformer vs LSTM, pure memory**: on the Passive T-Maze, GPT-2-style agents solve memory lengths from 50 up to 1500. LSTM agents "start to falter at a memory length of 250" and fall below the optimal Markovian return of 0.5. Both used hidden size 128 and 1 layer (best), with DDQN for the T-mazes and SAC-Discrete for pixel tasks. — [Ni et al. 2023 §5.1, Fig. 3–4](https://arxiv.org/abs/2307.03864)
- On Passive Visual Match, both transformer and LSTM fall close to the memoryless optimum (≈1/3 success) beyond memory length ≈120. Ni attributes this to the final bonus being diluted by distractor-phase apple rewards. In ~30% of LSTM runs at length 1000, NaN gradient issues occurred. — [Ni et al. 2023 §5.1](https://arxiv.org/abs/2307.03864)
- **Credit assignment**: on the Active T-Maze (cue requires a 1-step left detour), both GPT and LSTM fail at length 250. Transformers do not help long-term credit assignment. — [Ni et al. 2023 §5.2](https://arxiv.org/abs/2307.03864)
- **Recurrent model-free RL as a strong baseline**: a well-tuned recurrent (LSTM/GRU) model-free agent can be a strong baseline across many POMDPs (Ni et al. 2022, ICML). — [arXiv 2110.05038](https://arxiv.org/abs/2110.05038) (not re-verified in this session beyond Ni 2023's citation of it)
- **Memory Gym, PPO + GRU vs PPO + TrXL**:
  - On finite tasks TrXL is more sample-efficient on Mortar Mayhem (TrXL + observation reconstruction solves all 10 commands; GRU ≈6; TrXL without reconstruction ≈2) and on Mystery Path. GRU is better on Searing Spotlights.
  - On endless tasks GRU wins everywhere. Endless Mortar Mayhem: GRU IQM ≈120 commands vs TrXL 18. Endless Searing Spotlights: ≈20 vs ≈6 coins.
  - Bigger TrXL did not close the gap. The authors suspect stale off-policy episodic memory in TrXL (untested).
  - — [Pleines et al., arXiv 2309.17207](https://arxiv.org/html/2309.17207v1)
- **Frame stacking baseline in Memory Gym**: memoryless PPO fails Mortar Mayhem and hidden-goal Mystery Path (≈11% success). On visible-goal Mystery Path, 4 stacked frames reach ≈40% and 16 stacked frames reach effective policies. A relative-position vector input alone nearly solves the grid-Act variant (≈9.5 commands), showing that a hand-crafted state feature can replace memory. — [arXiv 2309.17207](https://arxiv.org/html/2309.17207v1)
- **POPGym**: GRU is the best general memory model, Elman the best efficiency tradeoff, and older RNNs outperform linear transformers and newer models in RL. — [arXiv 2303.01859](https://arxiv.org/abs/2303.01859). Caveat: a later paper's reproduced POPGym table ranks FFM (0.400 MMER) above GRU (0.349), and POPGym's results use RLlib PPO with nonstandard choices. — see [FFM, arXiv 2310.04128](https://arxiv.org/abs/2310.04128) and [Stable Hadamard Memory, arXiv 2410.10132](https://arxiv.org/abs/2410.10132) as found via search (exact table attribution not verified).
- **S5 / state-space models**: a modified S5 with resettable hidden state (Lu et al., NeurIPS 2023) outperforms RNNs on the tested POPGym tasks (CartPole, Pendulum, Repeat) and on bsuite memory-length, while running >5x faster than RNNs. The 30x speedup figure comes from the pure-JAX POPGym reimplementation, not from the model itself. — [arXiv 2303.03982](https://arxiv.org/abs/2303.03982); code [github.com/luchris429/popjaxrl](https://github.com/luchris429/popjaxrl)
- **Fast and Forgetful Memory (FFM)** (Morad et al., NeurIPS 2023) is a lightweight RL-specific memory model, an alternative to the GRU with a reported advantage on POPGym. — [arXiv 2310.04128](https://arxiv.org/abs/2310.04128) (found via search; details not read)
- **GTrXL** (Parisotto et al. 2020): a gated Transformer-XL that stabilizes transformers for RL. — arXiv 1910.06764 (not re-verified in this session)
- **Recurrent PPO implementation details that matter** (Pleines et al.):
  - Initialize the hidden state to zeros at episode start and store the hidden state of every step during rollout.
  - At optimization, split into fixed-length sequences, start each from its *stored* hidden state (truncated BPTT), zero-pad short sequences and mask padded steps in the loss.
  - Their settings: GRU sequence length 512, hidden size 512, 1 layer, 3 epochs, 8 minibatches, γ=0.995, λ=0.95. They chose GRU over LSTM because LSTM underperformed in tuning and is costlier.
  - — [arXiv 2309.17207](https://arxiv.org/html/2309.17207v1); earlier treatment in [Pleines et al. 2022, arXiv 2205.11104](https://arxiv.org/abs/2205.11104)
- **Stale hidden states**: R2D2 (Kapturowski et al., ICLR 2019) identifies "representational drift and recurrent state staleness" when stored recurrent states come from an older network. The mitigations are stored-state initialization plus a burn-in prefix that is unrolled but not trained on. Reportedly, burn-in alone mainly fixes the early part of each sequence. — [DI-engine R2D2 docs](https://di-engine-docs.readthedocs.io/en/latest/12_policies/r2d2.html); [UT Austin slide summary](https://www.cs.utexas.edu/~pingali/CS395TFall25/presentations/cao_recurrentexperiencereplay.pdf). Sources disagree on R2D2's sequence length (80 vs 120); check the original before citing a number.
- **Compute warning**: Memory Gym's full study used ≈25,000 GPU-hours, with single runs on a 32-core CPU plus an A100 taking up to ≈110 h for TrXL. Endless tasks were not converged after 819M steps. — [arXiv 2309.17207](https://arxiv.org/html/2309.17207v1). In contrast, T-maze-scale tasks use 1-layer, 128-unit models trained for ≤10M steps. — [Ni et al. 2023 §5](https://arxiv.org/abs/2307.03864)

### Inferences
- **Expected ranking on your aliased-rooms task** (hypothesis to test, not established):
  - MLP stays at 1/K success for every gap length L.
  - Frame stack k succeeds only while L (in env steps from the last cue sighting to the commit point) ≤ k, then collapses to 1/K.
  - LSTM/GRU succeed up to a horizon limited by the BPTT window and gradient issues. Ni's LSTM limit was ≈250 steps (DQN, 1 layer, 128 units); your PPO setup may differ.
  - A windowed transformer succeeds while L ≤ its context.
  - This gives a clean crossover figure. Reaching the regime where LSTM degrades probably needs L in the hundreds of steps, which is feasible on CPU for a 2D ray-cast env.
- **Truncated BPTT window vs gap**: if the TBPTT window is shorter than the gap between cue and decision, the gradient that teaches "write the cue into memory" cannot reach the cue step within one sequence. The only path is through the stored (detached) hidden state. Expect a sharp drop when window < L. This is an inference from TBPTT mechanics, consistent with Memory Maze's note that agents "benefit from training with truncated BPTT" and with Pleines' long (512) sequence lengths. Your existing TBPTT-window ablation becomes meaningful on this task (likely the most interesting single curve).
- **PPO-specific staleness**: hidden states stored during rollout come from the pre-update policy. Across 3–10 PPO epochs they drift, mildly (same mechanism as R2D2, smaller magnitude). Either recompute hidden states at the start of each epoch (cheap on CPU for small nets) or accept the drift and note it.
- **Cheap architectures to add**: GRU (essential, and the literature's default winner), Elman RNN (POPGym efficiency pick), and S5 or another linear recurrence. A 1–2 layer causal transformer with a context window equal to the episode length (or a sliding window) only if time permits. All are CPU-trainable at hidden size 64–256 for episodes of a few hundred steps. GTrXL is the transformer variant usually used with PPO, but Memory Gym shows TrXL+PPO can underperform the GRU, so a transformer is a nice-to-have, not a must.

### Gaps
- No source found comparing Mamba/S5 vs GRU specifically on navigation cue-recall tasks with PPO on CPU.
- Ni et al. 2022 (arXiv 2110.05038) specifics were not re-read here.
- No source quantified PPO hidden-state staleness or the benefit of re-computing hidden states per epoch.

---

## 3. How to parametrize the task to sweep memory demand, and what figures to make

### Takeaway
Make memory length L (steps between the last cue visibility and the decision point) the primary x-axis. Keep credit assignment short with dense progress shaping, as in the Passive T-Maze. Use the aliasing degree K (identical rooms) as a second axis that sets chance level 1/K. Add distractors as an optional third axis. The headline figure is success vs L per architecture (MLP, frame-stack k ∈ {4,16}, LSTM/GRU with TBPTT windows, optional transformer/S5), with a horizontal line at the memoryless optimum 1/K and vertical lines at each frame-stack size. The template is Ni's Fig. 3 ("Optimal agent w/o memory" line, memory length 50–1500).

### Cited Findings
- Ni's figure template: return/success vs memory length (50→1500) per architecture, plus a horizontal "optimal agent w/o memory" line, with mean and 95% CI over 10 seeds. — [Ni et al. 2023 Fig. 3](https://arxiv.org/abs/2307.03864)
- Passive vs Active cue placement isolates memory from credit assignment. A cue at the start position means credit length 1; a cue requiring a detour means credit length T, and both architectures fail at L=250. — [Ni et al. 2023 §3.2, §5.2](https://arxiv.org/abs/2307.03864)
- Distractor phases with their own rewards (apples in Passive Visual Match) dilute the final cue-dependent reward. Both LSTM and transformer then drop to chance beyond L≈120. — [Ni et al. 2023 §5.1](https://arxiv.org/abs/2307.03864)
- MiniGrid-Memory's design: the cue sits in the start room, the hallway sets the gap, the wrong choice terminates the episode with 0 reward, and success reward decays with steps. — [MiniGrid docs](https://minigrid.farama.org/environments/minigrid/MemoryEnv/)
- Memory Gym's difficulty knobs: show duration and delay, number of items to remember, path length, agent speed (slower agents make horizons longer), visibility of origin/goal. Endless variants let the memory demand grow until failure, so "items recalled before failure" becomes the metric. — [arXiv 2309.17207](https://arxiv.org/html/2309.17207v1)
- Wijmans et al.'s memory-budget curve: success/SPL vs memory length k on a log scale (1 → 1000). The hidden state is rebuilt from only the last k observations, (h,c) = LSTM over o_{t−k+1..t} starting from zeros. Performance did not saturate until about 1000 steps, and memoryless (k=1) gives ≈0% success. — [Wijmans et al. ICLR 2023, arXiv 2301.13261, §3 Fig. 2](https://arxiv.org/abs/2301.13261)
- Memory Maze's human baseline shows reward rising within an episode as the map is memorized, which suggests a "multi-trial within one episode" design. — [arXiv 2210.13383](https://arxiv.org/abs/2210.13383)

### Inferences: concrete design menu for the ray-cast env
**Task A, "Aliased Rooms / Cue-Corridor" (primary; a K-way Passive T-Maze in continuous 2D)**
- **Layout**: a start alcove with a colored cue panel (new ray object type `cue_color` one-hot with C colors, or keep a single `cue` type plus a color one-hot appended per ray). Then a corridor of length L, then a hub with K perceptually identical rooms or doorways (identical geometry; the ray encoding must make them indistinguishable).
- **Mapping**: cue color c → room index (a fixed mapping, randomized per episode which color appears). Alternatively, the cue is a spatial landmark such as a colored panel on the left or right wall.
- **Cue visibility**: only in the start alcove, so the cue leaves the rays after leaving the alcove. Optionally hide it after N steps.
- **Decision and reward**:
  - Entering any room commits. A wrong room terminates with 0 (or −1), so a memoryless policy cannot try rooms sequentially. Without termination, an MLP can explore all K rooms and still "succeed", which is likely why your current results favor the MLP.
  - Keep the goal invisible from the hub (goal object only inside the room, behind a turn) so the rays cannot reveal it.
  - Dense shaping toward the hub (as in the Passive T-Maze) keeps credit assignment short, plus a terminal +1 for the correct room.
- **Sweep axes**:
  - L: the corridor length in agent steps (e.g., 8, 16, 32, 64, 128, 256, 512). Bracket the frame-stack sizes 2–16 and the TBPTT windows.
  - K ∈ {2, 3, 4} identical rooms (chance 1/K).
  - Distractors: decoy colored panels along the corridor that are irrelevant (tests selective storage) or "last-cue-counts" variants.
  - Agent speed: a slower agent gives a longer L in steps for the same geometry, as in Memory Gym.
- **Active variant**: the cue sits in a side alcove off the start, so the agent must detour to see it. This tests credit assignment and probably fails for all architectures at large L (Ni). It is a good contrast panel but optional.

**Task B, "Hidden-goal homing / water-maze-like" (secondary)**
- A fixed hidden goal location per episode in a symmetric (aliased) room. On reaching it, the agent is teleported to a random start and must return several times per episode.
- **Metric**: time-to-goal on trial 2..N relative to trial 1. Memoryless agents show no improvement within the episode; memory agents improve, as with the Memory Maze human baseline.
- This is related to Morris water maze designs (not re-verified in this session).

**Figures that would make a strong workshop paper**
1. Success vs L (log x) per architecture, with a 1/K chance line and vertical lines at frame-stack sizes and TBPTT windows. Mean ±95% CI over ≥5 seeds.
2. A heatmap of success over (TBPTT window × L) for LSTM/GRU, which should show the diagonal "window ≥ L" boundary.
3. Success vs K at fixed L.
4. Probe-accuracy-vs-time-since-cue curves (§4).
5. Optionally, return on the original rooms vs the aliased task, showing MLP best on the former and worst on the latter (the "when memory pays off" story).

### Gaps
- No published results exist specifically for PPO + continuous actions + ray-cast observations on a K-way aliased-room task. The expected crossover points are hypotheses.
- Sample budgets needed per L on CPU are unknown. A pilot at L=32 and L=128 is recommended.

---

## 4. How to verify that memory is actually used

### Takeaway
Use four complementary diagnostics: (1) memory ablation at eval: zero or reset the hidden state at the hub, or apply Wijmans' k-step memory budget, which should drop success to 1/K; (2) linear probes decoding cue identity from h_t as a function of time since the cue; (3) counterfactual memory transplant or cue swap, where the hidden state from an episode with cue c′ is injected and the choice should follow c′; (4) behavioral checks that the policy is not exploiting a non-memory shortcut (e.g., a positional feature or a sequential search).

### Cited Findings
- **Memory budget ablation**: rebuild (h,c) from only the last k observations, then measure performance vs k. A k=1 (memoryless) agent nearly completely fails, while performance keeps rising up to k≈1000. — [Wijmans et al. 2023 §3](https://arxiv.org/abs/2301.13261)
- **Linear probes on frozen hidden states**: a linear classifier on (h,c) predicts whether the last action caused a collision with 98% held-out accuracy (50% chance). An L1-sparse probe found 10 of 3072 neurons give 96%. — [Wijmans et al. 2023 §2](https://arxiv.org/abs/2301.13261)
- **Probe agent / memory transplant**: a second agent ("probe") whose LSTM is initialized with the trained agent's final memory (h_T, c_T), with no gradients flowing back to the agent. Probe SPL is 85.0% vs the agent's 62.9%. Controls: all-zero memory and an untrained agent's memory along the same path (both ≈71–72% SPL). — [Wijmans et al. 2023 §4, Fig. 3](https://arxiv.org/abs/2301.13261)
- **Map decoding**: a separate decoder predicts an occupancy grid from (h_T, c_T). Trained memory reaches 32.5% IoU vs 12.5% for untrained-agent memory. — [Wijmans et al. 2023 §5](https://arxiv.org/abs/2301.13261)
- **Forgetting curves**: decoders predict the agent position k steps in the past (k ∈ [1, 256]). Relative error grows with k, and loops ("excursions") are decoded worse, so memory is task-selective. — [Wijmans et al. 2023 §6, Fig. 5](https://arxiv.org/abs/2301.13261)
- **Belief / mutual information analysis**: in T-maze and Mountain Hike, the mutual information between RNN hidden states and the exact Bayesian belief over the relevant state increases during training and correlates with return. MI with irrelevant variables decreases. — [Lambrechts et al. TMLR 2022, arXiv 2208.03520](https://arxiv.org/abs/2208.03520)
- **Offline probing as a benchmark component**: Memory Maze ships an offline probing evaluation, decoding maze layout and object locations from agent representations. — [arXiv 2210.13383](https://arxiv.org/abs/2210.13383)
- **Shortcut-feature check**: in Memory Gym, adding a relative-position vector nearly solves a variant without memory, and memoryless PPO solving the Act Grid variant with memory-free inputs shows the Clue phase is what needs memory. Ablate inputs to confirm the task is not solvable from current observations. — [arXiv 2309.17207](https://arxiv.org/html/2309.17207v1)

### Inferences: concrete protocol for the aliased-rooms task
- **Reset-at-hub ablation**: at eval, zero the recurrent state (or clear the frame stack) when the agent enters the hub. Success should drop to ≈1/K for a memory-using agent. If it does not drop, the agent is using a non-memory cue (a geometry leak) and the task design must be fixed.
- **Cue-identity probe**:
  - Collect h_t from eval rollouts and train a logistic-regression probe for cue color at each time-since-cue τ.
  - Plot probe accuracy vs τ per architecture and TBPTT window. Expect accuracy to stay high for LSTM/GRU up to the horizon, and for frame-stack to fall to chance at τ > k.
  - Add an untrained-network control and a shuffled-label control, following Wijmans' controls.
- **Counterfactual transplant**: run episode A (cue c) to the hub and episode B (cue c′) to the hub, then swap hidden states. The fraction of choices that follow the transplanted cue measures causal use. Alternatively, swap the cue color mid-corridor (after it would normally be invisible) and verify no effect, versus a swap before leaving the alcove.
- **Memory-budget curve** (Wijmans-style): success vs k, where the hidden state is reconstructed from only the last k observations. This also gives an empirical "effective memory length" per architecture to compare with the designed L.
- **Behavioral sanity**: log first-room-entered vs cue (a confusion matrix). Confirm the MLP's choices are independent of cue (≈uniform or position-biased).

### Gaps
- No source found applying the exact "cue-swap" counterfactual in navigation RL. Wijmans' transplant is the closest published analog.
- Probe methodology details for Memory Maze (exact decoder architecture) were not read.

---

## 5. Relation to auxiliary losses

### Takeaway
Auxiliary losses mainly improve sample efficiency and representation learning; they do not substitute for memory. Depth and loop-closure aux tasks (Mirowski 2017) improved navigation data efficiency. Observation reconstruction helps memory agents in Memory Gym (it was essential for TrXL to solve Mortar Mayhem and gave large sample-efficiency gains in Searing Spotlights). A privileged "predict-the-cue / ground-truth" head can boost learning but is a sanity check, not a fair method. For the paper, aux losses are a secondary ablation on the aliased task. A cue-prediction or reconstruction loss is the most targeted candidate.

### Cited Findings
- Mirowski et al. (ICLR 2017) jointly train an RL navigation agent with auxiliary depth prediction and loop-closure classification. These "substantially improve data efficiency and task performance" in 3D mazes, including when goal locations change frequently. The paper also analyzes localization from network activity. — [arXiv 1611.03673](https://arxiv.org/abs/1611.03673)
- Memory Gym: TrXL with observation reconstruction solved all 10 Mortar Mayhem commands vs ≈2 without it. Reconstruction "significantly improves sample efficiency" on Searing Spotlights, where rare events make visual learning hard. On endless Mortar Mayhem, a ground-truth auxiliary head (described as a sanity check) raised TrXL to ≈50 commands but was unstable. Reconstruction slightly *lowered* endless TrXL performance (10 vs 18 commands). — [arXiv 2309.17207](https://arxiv.org/html/2309.17207v1)
- Hung et al.'s memory agent (RMA) combines reconstruction-based representation learning with external memory storage and solved Passive Visual Match. TVT then addresses credit assignment for the Active variant. — [arXiv 1810.06721](https://arxiv.org/abs/1810.06721) (details from search snippets; full text not read)
- Wijmans: the sighted (depth+egomotion) agents did *not* show decodable metric maps, while blind agents did. The authors conjecture that richer perception permits "trivial" solutions and suppresses map-building. — [arXiv 2301.13261 §5](https://arxiv.org/abs/2301.13261)

### Inferences
- **Aux losses to test on the aliased task** (cheap on CPU):
  - (a) Cue reconstruction / prediction: predict the cue color (seen earlier) from h_t at every step. This is privileged but standard as an "aux upper bound" condition and should be labeled as such.
  - (b) Your existing depth aux: likely little help, since depth is already in the rays.
  - (c) Loop closure: likely irrelevant for a linear corridor, more relevant for Task B.
  - (d) A CPC-style future-observation prediction loss: a plausible but unverified benefit.
- **Expected finding**: aux losses cannot rescue the MLP or frame-stack beyond their horizon (no storage capacity). They may shift the recurrent agents' success-vs-L curve rightward or speed learning. That is a clean, falsifiable hypothesis for the paper.
- Wijmans' observation suggests that observations which make the task "too easy" suppress memory use. This supports deliberately aliasing the rooms in the ray encoding.

### Gaps
- No source found evaluating contrastive predictive coding (CPC|A, Guo et al. 2018) or loop-closure aux specifically on cue-recall tasks; not searched in this session.
- No direct evidence on whether grid-cell representations (Banino et al. 2018) help cue-recall, as opposed to path-integration, tasks; not searched.
