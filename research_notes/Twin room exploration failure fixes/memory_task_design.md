# How established RL memory benchmarks design cue-recall (T-maze style) tasks so that success measures memory, not exploration

Our task for comparison: 2D continuous control, PPO, 180 deg egocentric ray casts. A cue alcove (red/blue) leads through a bent shaft and a corridor of L steps to a T-junction. Two identical rooms sit off the junction, each with a baffle hiding the goal point (success radius 16, about 130 units behind the door). Entering the wrong room terminates the episode. Reward is -1/max_steps per step and +1 on success. Horizon is 2L+256. Pilot at L=32 gave all-or-nothing seeds, and MLP got 0/5 even though memoryless chance should be 0.5.

---

## Q1. Passive vs Active T-maze (Bakker 2002; Ni et al. 2023): layout, termination, junction reward, per-step penalties, length scaling, exploration control, LSTM/GRU findings

### Takeaway
In both canonical T-mazes the reward is decided by a single choice next to the junction: the goal cell is adjacent to it and there is no search after the choice. The corridor is a 1-cell-wide line, and the corridor phase is made nearly exploration-free, by directed exploration (Bakker) or by a dense penalty for falling behind the optimal rightward pace (Ni). Ni et al. go further and fix the horizon at exactly corridor length + 1, so there is no slack at all. Under these conditions LSTM solves memory lengths of about 50-70 (Bakker) to about 200 (Ni) before falling to or below the 0.5 chance level.

### Cited Findings
**Bakker 2002 (NIPS 14, "Reinforcement Learning with Long Short-Term Memory")**
- Bakker says the task was "designed to test RL-LSTM's capability to bridge long time lags, without confounding the results by making the control task difficult in other ways." It is a grid T-maze with 4 actions (N/E/S/W). The agent goes from the start at the beginning of the corridor to the T-junction and then moves N or S "to a changing goal position, which it cannot see". The goal side is given by a "road sign" seen only at the start — [Bakker 2002 PDF](https://proceedings.neurips.cc/paper/2001/file/a38b16173474ba8b1a95bcbc30d3b8a5-Paper.pdf)
- Reward: the correct action at the T-junction gives **+4** and the wrong action gives **-0.1**. "In both cases, the episode ends", and a new random goal side is drawn. During the episode the agent gets **-0.1 when it stands still** — [Bakker 2002](https://proceedings.neurips.cc/paper/2001/file/a38b16173474ba8b1a95bcbc30d3b8a5-Paper.pdf)
- Observations are 3 bits: start = 011 or 110 (the cue), corridor = 101, junction = 010. So the junction is perfectly signalled, and the only hidden variable is the cue — [Bakker 2002](https://proceedings.neurips.cc/paper/2001/file/a38b16173474ba8b1a95bcbc30d3b8a5-Paper.pdf)
- Corridor length N was varied from 5 to 70, with 10 runs per condition — [Bakker 2002](https://proceedings.neurips.cc/paper/2001/file/a38b16173474ba8b1a95bcbc30d3b8a5-Paper.pdf)
- Bakker states the exploration confound explicitly: "The agent must first learn to reliably move to the T-junction. Once this is accomplished, the agent will begin to experience more or less consistent and shortest possible histories... The directed exploration mechanism is crucial in this regard: it learns to set exploration low in the corridor and high at the T-junction." The directed exploration is a separate feedforward net that predicts the magnitude of the TD error and sets the Boltzmann temperature — [Bakker 2002](https://proceedings.neurips.cc/paper/2001/file/a38b16173474ba8b1a95bcbc30d3b8a5-Paper.pdf)
- Success criterion per run: "A run was considered a success if the agent learned to take the correct action at the T-junction in over 80% of cases, using its stochastic action selection mechanism." This corresponds to 100% greedy correctness at the junction — [Bakker 2002](https://proceedings.neurips.cc/paper/2001/file/a38b16173474ba8b1a95bcbc30d3b8a5-Paper.pdf)
- Results are reported as the **number of successful runs out of 10** as a function of N (Fig. 3). RL-LSTM "has perfect performance up to N = 50, after which performance gradually decreases". An Elman-BPTT net (16 hidden units) and a table-based system with one memory bit do worse. In a noisy variant (corridor obs a0b with uniform random a,b) the RNNs are "hardly affected" — [Bakker 2002](https://proceedings.neurips.cc/paper/2001/file/a38b16173474ba8b1a95bcbc30d3b8a5-Paper.pdf)
- In a second, harder continuous task in the same paper, RL-LSTM "learned optimal performance in 2 out of 10 runs", and Elman-BPTT never succeeded in 10 runs — [Bakker 2002](https://proceedings.neurips.cc/paper/2001/file/a38b16173474ba8b1a95bcbc30d3b8a5-Paper.pdf)

**Ni et al. 2023 ("When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment", NeurIPS 2023)**
- Common layout: actions {L,R,U,D}, deterministic, and walls block movement without ending the episode. A corridor of length L runs from the oracle O to the junction J, and J connects to the goal candidates G1/G2. The observation is null except at J, O, G1 and G2 — [Ni et al. 2023, arXiv HTML](https://arxiv.org/html/2307.03864v4)
- **Passive T-Maze**: O = S (the cue is seen at step 0) and L = T-1. Per-step reward is R_t = (1(x_{t+1} >= t) - 1)/(T-1), which is a -1/(T-1) penalty on every step the agent falls behind the optimal rightward pace. Terminal reward is R_T = 1(o_{T+1} = G). Optimal return is 1.0, the best Markovian (guessing) policy gets 0.5, and the worst gets -1.0. The paper defines memory length = T and credit assignment length = 1, "since immediate penalties punish suboptimal actions" — [Ni et al. 2023](https://arxiv.org/html/2307.03864v4)
- **Active T-Maze**: O is one step left of S, so the agent must step left to see the cue, and L = T-2. R_1 = 0, and R_t = (1(x_{t+1} >= t-1) - 1)/(T-2) for 2 <= t <= T-1. The credit assignment length is T, because the initial left step only pays off at the end — [Ni et al. 2023](https://arxiv.org/html/2307.03864v4)
- The authors say both T-mazes "need little exploration because rewards are dense before the goal". They frame them as configurable unit tests in the bsuite style: passive = memory only, active = memory + credit assignment — [Ni et al. 2023](https://arxiv.org/html/2307.03864v4)
- Official env code ([twni2016/Memory-RL envs/tmaze.py](https://github.com/twni2016/Memory-RL/blob/main/envs/tmaze.py), [raw](https://raw.githubusercontent.com/twni2016/Memory-RL/main/envs/tmaze.py)):
  - `done = self.time_step >= self.episode_length`. **Termination is by time limit only**, and reaching a goal does not end the episode.
  - Terminal reward is `float(y == goal_y) * self.goal_reward` (default 1.0), so the wrong goal and staying in the corridor both give 0. There is no separate wrong-choice penalty.
  - `TMazeClassicPassive`: `episode_length = corridor_length + 1`, which leaves **zero slack steps**: the agent must move right every step and then up or down on the final step. `TMazeClassicActive`: `episode_length = corridor_length + 2*oracle_length + 1`.
  - The cue (`exposure = goal_y`) is shown only on the **first** visit to the oracle.
  - Optional knobs: `penalty` (<= 0, default 0 in the classic classes, applied when x < t - oracle_length), `distract_reward`, `ambiguous_position`, `expose_goal` (to test Markov policies), and `add_timestep`.
- Lengths tested: Passive memory length 50-1500, Active credit length 20-500 — [Ni et al. 2023](https://arxiv.org/html/2307.03864v4)
- Findings: Transformers (GPT-2) solve Passive T-Maze up to 1500. LSTMs "start to falter at a memory length of 250" and fall below the Markovian 0.5. In Active T-Maze both fail at 250, with transformers better at medium lengths (about 100). Transformers show "much greater stability and sample efficiency across seeds". A single layer was best for the memory tasks. Setup: hidden size 128, DDQN + epsilon-greedy for the T-mazes, up to 10M steps, **10 seeds**, mean and 95% CI — [Ni et al. 2023](https://arxiv.org/html/2307.03864v4)
- GRU is not part of Ni et al.'s main T-maze comparison (only LSTM vs GPT-2). I found no GRU-vs-length T-maze curve in this paper — [Ni et al. 2023](https://arxiv.org/html/2307.03864v4)

### Inferences
- Both canonical T-mazes put the reward on a **single binary choice right at the junction**, with the goal one cell away. Our design adds a second sub-task after the choice: finding a hidden point about 130 units behind a baffle. Neither Bakker nor Ni has anything like it. With that sub-task in place, success needs (a) corridor navigation, (b) the memory-dependent choice and (c) post-choice search. A failure at (c) gives exactly what we see: 0 success and every episode timing out, which is not chance level.
- Our MLP baseline scoring 0/5 rather than about 0.5 is the strongest diagnostic. In the Ni/Bakker designs a memoryless agent reliably reaches about 0.5 (Ni's "best Markovian policy"), because reaching a goal is trivial. Our memoryless agent cannot reach the goal even in the correct room half the time, so the bottleneck is reaching and finding the goal, not recall.
- Ni's per-step shaping (-1/(T-1) whenever the agent is behind optimal pace) makes the credit length for corridor traversal 1 step. Our uniform -1/max_steps penalty gives no directional signal: it is the same wherever the agent is, so it only adds a weak urgency term.
- Bakker shows that even in a 3-bit grid T-maze, learning to reach the junction reliably comes first. He needed a special directed-exploration mechanism to get consistent histories. Our 2D continuous version with a bent shaft and baffles makes this prerequisite much harder.

### Gaps
- The exact shape of Bakker's success-count curve beyond N = 50 (how many of 10 runs at N = 60 and 70) could not be read from the PDF text: the figure values are not in the extracted text.
- I did not find a published GRU-vs-length curve on the canonical T-maze in these two primary sources.

---

## Q2. How other memory benchmarks define success, whether wrong choices terminate, and how goal visibility is handled

### Takeaway
Across MiniGrid MemoryEnv, DeepMind's TVT tasks (Passive/Active Visual Match, Key-to-Door), Memory Gym and POPGym, the recall phase is always a **short, visually obvious choice among visible options**. Typical forms are a visible object or pad in a small room, or one action per step. A wrong choice usually ends the episode immediately, and the TVT tasks even give a small positive reward for a wrong pick. The cue and distractor phases have fixed durations (often via teleport), so memory length does not depend on the agent's navigation. None of them hide the goal after the choice. Memory Maze is the exception that hides goals, and it is explicitly a navigation-memory benchmark with fixed-duration episodes and repeated targets, not a one-shot cue-recall task.

### Cited Findings
**MiniGrid MemoryEnv** ([source memory.py](https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/envs/memory.py), [raw](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/envs/memory.py))
- Layout: a start room with the cue object (`_rand_elem([Key, Ball])`, green) at (1, h//2-1), a 1-cell-wide hallway, and a split at `hallway_end` with one Ball and one Key, one above and one below the split. With `random_length=True`, `hallway_end = _rand_int(4, width-2)` sets the hallway length.
- Agent start: `agent_pos = (_rand_int(1, hallway_end+1), h//2)`, a random column in the start room or hallway, facing east. So the cue-to-choice distance varies per episode.
- Success tile = the cell next to the matching object. Reaching it gives `reward = self._reward()` = 1 - 0.9*(step_count/max_steps) and `terminated = True`. **Failure tile (next to the wrong object): `reward = 0`, `terminated = True`**, so a wrong choice ends the episode at once.
- `max_steps = 5 * size**2` (320 for 8x8), handled by truncation.
- Both choice objects are visible at the split. The "goal" is the tile adjacent to the junction objects, with no hidden search.

**Hung et al. 2019 TVT ("Optimizing agent behavior over long time scales by transporting value", Nat. Comms.; arXiv 1810.06721)** — [arXiv PDF](https://arxiv.org/pdf/1810.06721)
- Three-phase structure: P1 "without any immediate reward, explore an environment to acquire information", P2 "an unrelated distractor task over a long time period with numerous incidental rewards", P3 "exploit the information acquired in P1". "Phases do not repeat within an episode." — [arXiv 1810.06721](https://arxiv.org/pdf/1810.06721)
- Timing: agents act at 15 frames per second (action repeat 4 on 60 fps DMLab). So 5 s = 75 agent steps and 30 s = 450 agent steps — [arXiv 1810.06721](https://arxiv.org/pdf/1810.06721)
- **Passive Visual Match**: P1 is a 1x3 corridor with the target colour square at one end, and the agent spawns facing it. There is no reward and the agent "remains in this phase for 5 seconds". P2 is the standard distractor: an 11x11 room, apples spawned with p = 0.3 per cell, r_apple = 5, lasting 30 s ("This length was varied in some experiments"). P3 is a 4x7 room with 4 colour squares on one wall and a groundpad in front of each, with the agent spawned facing them. "When the agent touches one of these pads, a reward of **10 points** is given if it is the pad in front of the target painting and a reward of **1** is given for any other pad. The episode then ends. If the agent does not touch a pad within 5 seconds then no reward is given for this phase and the episode ends." — [arXiv 1810.06721](https://arxiv.org/pdf/1810.06721)
- **Active Visual Match**: the same, except P1 is two 3x3 rooms joined by a 1x1 corridor and the target colour is on a random wall unit. The agent must find the cue, which is the information-seeking part — [arXiv 1810.06721](https://arxiv.org/pdf/1810.06721)
- **Key-to-Door**: P1 (5 s, no reward) has a key placed randomly, and touching it shows a black cue image. P2 is the standard distractor. P3 is a 1x3 corridor with a locked door and a goal object behind it. If the agent holds the key, walking into the door opens it, and walking into the goal gives 10 points; "Otherwise, no reward is given." — [arXiv 1810.06721](https://arxiv.org/pdf/1810.06721)
- Distractor length was varied over 0, 15, 30, 45 and 60 s, and apple reward over 0, 1 and 5. With r_apple = 5 (P2 reward variance 630), "all comparison models (the LSTM agent, LSTM+Mem agent, and RMA) failed to learn P1 behavior". On Passive Visual Match, the plain LSTM agent achieved "only slightly better than chance performance in P3", while RMA (memory + reconstruction) solved it — [arXiv 1810.06721](https://arxiv.org/pdf/1810.06721)
- Ni et al. note that in Passive Visual Match "the bonus-to-total-reward ratio shrinks as episodes lengthen", and both LSTM and transformers stay near the Markovian about 1/3 at long lengths. LSTMs hit NaN gradients in about 30% of runs at length 1000, and those runs are excluded from the plots — [Ni et al. 2023](https://arxiv.org/html/2307.03864v4)

**Memory Gym (Pleines et al., "Memory Gym: Towards Endless Tasks...", arXiv 2309.17207)** — [arXiv HTML v6](https://arxiv.org/html/2309.17207v6)
- Mortar Mayhem: in the finite version the agent is **immobile** while the 10 commands are shown, then must execute them. "Failure to execute a command results in episode termination." Each successful command gives +0.1, and the command-failure reward defaults to 0. A finite episode is 274 steps for 10 commands.
- Mystery Path: +0.1 per newly visited tile on an invisible path. Falling off sends the agent **back to the origin** rather than ending the episode (finite). "Show Origin" and "Show Goal" default to False. In the endless version leftward movement is removed.
- Searing Spotlights: health points, coins +0.25, exit +1 (finite).
- Recommendations from the paper: run naive baselines (no memory, frame stacking, positional encoding) to confirm memory is needed. Mystery Path Grid with visible origin/goal turned out to be solvable by a 16-frame stack. Also provide commands as a vector observation (Mortar Mayhem Act Grid) to remove cue decoding, and test with full observability to separate perception or optimisation issues from memory.
- Results: finite tasks, TrXL + reconstruction reaches 99% aggregate vs 83% for GRU. Endless tasks, GRU beats TrXL by a wide margin (normalized aggregate 0.98 vs 0.39). 5 seeds per experiment.

**POPGym (Morad et al., ICLR 2023)**
- RepeatPrevious (env code): the observation is the current card suit and the action is a suit guess. Reward is +/- 1/(num_cards - k) per step for correct or incorrect recall of the suit from k steps ago, and 0 for the first k steps. `terminated = len(deck) == 1`, so episode length is fixed at num_cards - 1. Easy: 1 deck, k = 4. Medium: 2 decks, k = 32. Hard: 3 decks, k = 64 — [popgym repeat_previous.py](https://raw.githubusercontent.com/proroklab/popgym/master/popgym/envs/repeat_previous.py)
- These tasks have no navigation at all: memory is tested by a per-step classification action with dense, normalized reward (max return about 1) — [popgym repeat_previous.py](https://raw.githubusercontent.com/proroklab/popgym/master/popgym/envs/repeat_previous.py)
- POPGym reports max-mean episodic reward (MMER), and GRU had the highest overall MMER (0.349) among the 13 memory models. This number comes via an aggregator review, so check it against the paper — [liner review](https://liner.com/review/popgym-benchmarking-partially-observable-reinforcement-learning); paper: [arXiv 2303.01859](https://arxiv.org/abs/2303.01859)

**Memory Maze (Pasukonis et al. 2022)** — [repo README](https://github.com/jurgisp/memory-maze); [arXiv 2210.13383](https://arxiv.org/abs/2210.13383)
- +1 when the agent touches the object whose colour is shown by the **border colour of the observation image**. Touching a wrong-colour object has no effect, so **wrong choices do not terminate**. The episode "continues for a fixed amount of time" and a new target is chosen after each success.
- Fixed episode lengths: 9x9 = 1000 steps (3 objects), 11x11 = 2000 (4), 13x13 = 3000 (5), 15x15 = 4000 (6). Human mean scores are 26.4/44.3/55.5/67.7.
- Design rationale: layout and object positions are fixed within an episode, so "an agent with memory can explore once and then take shortest paths to targets, while a memoryless agent must wander randomly". Memory is measured as the **within-episode improvement over many repeated targets**, not as a one-shot recall. The paper claims it "measures long-term memory separate from confounding agent abilities". It also provides an offline probing benchmark with a scripted-policy dataset and ExtraObs ground-truth (agent_pos, target_pos, maze_layout) for diagnostics. TBPTT training helps, 9x9 is solved by current RL, and 15x15 is "out of reach".

### Inferences
- **Summary table (success definition / wrong choice / goal visibility / memory-length control):**

| Benchmark | Success = | Wrong choice | Goal at choice time | Memory length control |
|---|---|---|---|---|
| Bakker T-maze | correct N/S move at junction | ends episode, -0.1 (vs +4) | adjacent, junction obs unique | corridor N 5-70 |
| Ni Passive/Active T-maze | at goal cell at final step T | 0 reward, fixed horizon T | adjacent cell | corridor L = T-1; horizon = L+1, no slack; pace penalty |
| MiniGrid MemoryEnv | step onto tile next to matching object | ends episode, 0 | both objects visible | hallway length (random_length) + random start |
| TVT Passive/Active Visual Match | touch pad in front of matching colour | ends episode, **+1** (vs +10); 5 s timeout = 0 | all 4 colours visible, agent spawned facing them | fixed-duration distractor phase (teleport), 0-60 s |
| TVT Key-to-Door | reach goal behind door in 1x3 corridor | no reward | goal visible, trivially reachable | fixed P2 duration |
| Memory Gym Mortar Mayhem | execute each command | ends episode | tiles visible; agent immobile during cue | number of commands |
| POPGym RepeatPrevious | per-step classification | -1/(N-k) per error, no termination | n/a | k = 4/32/64 |
| Memory Maze | touch target-colour object (repeat) | no effect, fixed duration | target colour shown on border; location must be remembered | maze size / episode length |

- The TVT reward for a wrong pick (+1 vs +10) is important for us. A memoryless policy learns quickly to go and touch some pad, so the "reach the choice point" skill is bootstrapped by reward on every episode. Memory then only has to raise the payoff from about 1/4*10 + 3/4*1 to 10. In our design a wrong room gives 0 and the right room needs a hidden-goal search, so a memoryless agent gets no gradient towards the choice at all.
- Every benchmark that uses navigation in the recall phase keeps that phase tiny: a 4x7 room with spawn facing the options (TVT), a 1x3 corridor (Key-to-Door), or an adjacent tile (MiniGrid, Bakker, Ni). Our post-door search (a hidden point about 130 units behind a baffle in a 200x200 room) is far larger than any of these.
- TVT and Memory Gym fix the cue and distractor durations (the agent is immobile, or teleported after N seconds). This makes the memory length a controlled experimental variable rather than a function of how fast the agent walks the corridor. Ours ties memory length to navigation competence: an agent that dawdles has a longer memory requirement and less time to search.

### Gaps
- DMLab-30 tasks (explore_goal_locations, the watermaze-style "rooms_watermaze") and the GTrXL (Parisotto 2020), R2D2 and Esslinger 2022 DTQN memory tasks were not fetched in this session, so their exact reward and termination details are not recorded here.
- BabyAI was not covered. Its memory-related levels are not standard cue-recall tasks, and I did not verify their details.
- I did not verify the exact Memory Maze IMPALA/Dreamer numbers. The README gives no numeric agent scores.

---

## Q3. Do benchmarks report bimodal (all-or-nothing) seed outcomes, and how do they handle them?

### Takeaway
Yes, implicitly. The oldest T-maze work (Bakker) reports results as **number of successful runs out of 10** with a binary per-run success criterion, which is effectively an admission that runs either solve the task or do not. Later papers mostly report mean +/- CI or SE over 5-10 seeds and mention outliers or failed runs only in passing. Ni et al. exclude NaN runs, and Pleines et al. note outliers and that 5 seeds "may still be insufficient". No source I read uses the word "bimodal".

### Cited Findings
- Bakker: per-run success = more than 80% correct junction choices under stochastic selection. Fig. 3/4 plot "Number of successful runs (out of 10) as a function of N", plus "Average number of timesteps until success". On the harder task, LSTM succeeded in "2 out of 10 runs" — [Bakker 2002](https://proceedings.neurips.cc/paper/2001/file/a38b16173474ba8b1a95bcbc30d3b8a5-Paper.pdf)
- Ni et al.: mean and 95% CI over 10 seeds. They note transformers' "much greater stability and sample efficiency across seeds" and that LSTMs hit NaN gradients in about 30% of runs at length 1000 in Passive Visual Match, "excluded from the plots" — [Ni et al. 2023](https://arxiv.org/html/2307.03864v4)
- Pleines et al.: 5 seeds per experiment (15 for aggregates), with 50 evaluation episode seeds x 3 repetitions per data point. They state 5 samples "may still be insufficient for drawing strong conclusions". For the relative-positional-encoding TrXL on endless Mortar Mayhem they report two outlier runs out of 5 (averaging 16 and 56 commands) — [Memory Gym arXiv HTML](https://arxiv.org/html/2309.17207v6)
- Hung et al.: learning curves show "standard errors about the mean, computed over 5 independent runs" — [arXiv 1810.06721](https://arxiv.org/pdf/1810.06721)

### Inferences
- Our pattern (each seed either about 1.0 or exactly 0 for 1000 iterations) matches Bakker's "successful runs out of N" picture. That metric is a defensible and literature-grounded way to report it: success-count vs L per backbone, plus time-to-success for the successful runs.
- In our case, though, the zero-seeds sit **below** memoryless chance (0, not 0.5), and every eval episode times out. That means the failed seeds never learned to reach any goal. They are exploration failures, not memory failures. A "solved/not solved" count would therefore mostly measure exploration luck, unless the task is changed so that a memoryless agent reliably reaches about 0.5.
- If the bimodality persists after redesign, report the per-seed outcome distribution (success counts, or every seed's final success as dots) next to means. Agarwal et al. 2021 ("Deep RL at the Edge of the Statistical Precipice", arXiv 2108.13264) is the standard reference for performance profiles, IQM and bootstrap CIs with few seeds. I did not fetch it in this session, so verify before citing.

### Gaps
- I found no memory-RL paper that explicitly analyses or names bimodal seed outcomes on T-maze tasks.
- POPGym's seed-handling details (number of seeds, any failed-run treatment) were not checked in the primary paper.

---

## Q4. Recommendations from the literature for isolating memory from exploration and credit assignment, and implications for our redesign

### Takeaway
The consistent recipe is:
1. Make the choice point trivial to reach and the choice outcome immediate. The goal should be adjacent or visible, with no post-choice search.
2. Fix the horizon or phase durations so memory length is a controlled variable.
3. Give dense shaping or a reward for any choice, so a memoryless agent reliably reaches chance level (0.5 for two options).
4. Run memoryless and frame-stack controls to confirm memory is needed.
5. Vary only the cue-to-choice delay.

Ni et al. formalise this as keeping credit-assignment length at about 1 (passive) while scaling memory length.

### Cited Findings
- Ni et al. propose configurable tasks with **known memory length and credit-assignment length**, scaling one at a time. The passive T-maze uses dense pace penalties so that "immediate penalties punish suboptimal actions" (credit length 1) and "little exploration" is needed. They warn that many existing benchmarks "entangle memory and credit" — [Ni et al. 2023](https://arxiv.org/html/2307.03864v4)
- In Ni's code the horizon equals corridor length + 1 in the passive maze, so there is no room for exploration. The cue is shown only on the first oracle visit, and the terminal reward depends only on the final position — [Memory-RL tmaze.py](https://raw.githubusercontent.com/twni2016/Memory-RL/main/envs/tmaze.py)
- Bakker designed the T-maze "without confounding the results by making the control task difficult in other ways". He stresses that the agent "must first learn to reliably move to the T-junction" and needed exploration tuned "low in the corridor and high at the T-junction" — [Bakker 2002](https://proceedings.neurips.cc/paper/2001/file/a38b16173474ba8b1a95bcbc30d3b8a5-Paper.pdf)
- TVT fixes phase durations (5 s cue, 30 s distractor, 5 s choice) via teleport between maps. It places the agent facing the visible options in the recall room, gives +1 for any wrong option vs +10 for the correct one, and ends the episode on the first pad touch. It also shows that distractor-reward variance (apples worth 5) can block learning of the memory behaviour entirely for LSTM agents — [arXiv 1810.06721](https://arxiv.org/pdf/1810.06721)
- Memory Gym: keep the agent immobile during the cue phase (Mortar Mayhem), return the agent to the origin rather than terminate (Mystery Path), and validate with no-memory and frame-stack baselines. A 16-frame stack solved one variant that was thought to need memory — [Memory Gym](https://arxiv.org/html/2309.17207v6)
- MiniGrid MemoryEnv randomises the start position along the hallway and the hallway length, so the delay varies. Both choice objects are visible at the split, and a wrong choice ends the episode with 0 — [MiniGrid memory.py](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/envs/memory.py)
- Memory Maze isolates memory by fixing the episode duration and repeating targets, so memory shows up as faster re-navigation. It supplies ground-truth probes (ExtraObs) for offline diagnostics — [memory-maze README](https://github.com/jurgisp/memory-maze)

### Inferences
These are concrete redesign options for our twin-room task, ranked by how directly the literature supports them.
1. **Make success region-based or remove the baffle.** Define success as entering the correct room's door region (Bakker, MiniGrid, Ni all reward the junction choice or an adjacent tile), or make the goal visible from the door. This removes the post-choice search. A memoryless MLP should then score about 0.5, which is the sanity check the literature implies (Ni's "best Markovian = 0.5").
2. **Reward any choice, more for the correct one (TVT style).** For example, +0.1 or +0.2 for entering the wrong room (terminating), and +1 for the correct one. This gives every seed a gradient to the junction and removes the all-or-nothing exploration lottery.
3. **Add Ni-style pace shaping in the corridor.** Penalise steps where progress along the corridor centreline falls behind the expected pace, or use potential-based progress shaping. Keep the credit length for the navigation part short, and set the horizon close to the shortest path plus a small slack rather than 2L+256.
4. **Decouple memory length from navigation.** Either fix the corridor phase duration (TVT teleport or a fixed delay) or report memory length as the actual steps between cue exposure and choice. Our current horizon of 2L+256 lets slow agents use extra memory time.
5. **Diagnostics to log now (cheap, no redesign):** junction-reached rate, room-chosen rate, and choice accuracy conditional on entering a room. If LSTM-w64 failed seeds never reach the junction or never enter a room, the failure is exploration. If they enter rooms at 50/50, it is memory. This mirrors Bakker's separate criterion for "correct action at the T-junction".
6. **Keep the memoryless and frame-stack controls** (Memory Gym practice). A frame stack of 16 at L = 32 cannot span the corridor, so FS16 reaching 0.5-0.6 is consistent with "reaches the goal at chance" once exploration is solved.

### Gaps
- No source tests a continuous-control, ray-cast T-maze with a hidden goal behind a baffle, so the size of the effect of each fix in our setting is untested.
- I did not verify PPO-specific T-maze results (Ni uses DDQN/SAC-D; Bakker uses Advantage(lambda)). PPO + LSTM T-maze baselines (for example in POPGym or Memory Gym, both PPO-based) were not checked in detail for corridor-length scaling.
