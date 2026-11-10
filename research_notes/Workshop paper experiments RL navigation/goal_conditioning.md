# Goal-conditioned (directed) vs undirected search in small-scale RL navigation

Scope: how to add a cheap, publishable goal-conditioning experiment to a PPO study with egocentric 2D ray-cast observations (one-hot {empty, wall, door, goal} + distance per ray), continuous actions, 2D floorplans/mazes of rising difficulty, comparing MLP / LSTM / frame-stack backbones, MLP vs CNN encoders, auxiliary losses (depth, loop closure), and grid-cell representations. Current setup is undirected search (no goal info).

Verification note: the sources marked "fetched" below were opened in this session and their numbers checked. The others (HER, UVFA, CMP, Active Neural SLAM, Habitat, target-driven ImageNav) are foundational papers cited by arXiv ID. Their details were not re-checked here, so the notes only make broad claims about them.

## Q1. Which goal encodings are standard, and which work with PPO plus ray-cast inputs?

### Takeaway
The standard encoding for a PointGoal is an egocentric polar vector to the goal: distance plus cos/sin of the bearing, recomputed every step from GPS+compass. It is the cheapest option, works with PPO, and fits a ray-cast setup with no changes. The goal must be given relative to the agent: an absolute goal coordinate without the agent's pose works about as badly as giving no goal at all. Image, object and map goals are mostly 3D or visual and would add cost without a clear benefit in a 2D ray world.

### Cited Findings
- DD-PPO (Wijmans et al., ICLR 2020) gives the PointGoal in egocentric polar coordinates (distance, angle), converted to (distance, cos θ, sin θ) "to avoid the discontinuity at the x-axis". The policy is a 2-layer LSTM (512-d hidden; 1024-d in the best model). There are 4 discrete actions (stop, forward 0.25 m, turn ±10°) — [arXiv 1911.00357 (fetched)](https://ar5iv.labs.arxiv.org/html/1911.00357)
- In Habitat PointNav the goal is given relative to the agent's current pose ("go 5 m north, 10 m east of you"). In a single-room PPO test, swapping the relative goal for an absolute one "was equivalent to removing the goal input" and cut success by 40%. Single-author preprint, March 2025 — [Song, arXiv 2503.15693 (fetched)](https://arxiv.org/html/2503.15693v1)
- "Blind" PointGoal agents (egomotion plus goal vector only, no vision) are trained with RL using fully connected and recurrent networks, with the goal given as a relative displacement (Δx, Δy) — [Wijmans et al., ICLR 2023, arXiv 2301.13261 (fetched)](https://arxiv.org/abs/2301.13261)
- Goal given only through reward (no goal input), as in this project's current setup: Mirowski et al. feed the agent RGB, agent-relative velocity, previous action and previous reward. The goal is found by exploring and must then be remembered within an episode — [arXiv 1611.03673 (fetched)](https://ar5iv.arxiv.org/html/1611.03673)
- Grid-cell vector navigation: Banino et al. train a recurrent path-integration network whose units become grid-like, then use it inside a deep-RL agent. Distance and direction to the goal can be decoded from the network. Silencing the grid units hurt navigation and made the goal-vector representation less accurate. Grid-like codes let the agent take shortcuts — [Banino et al., Nature 557:429–433 (2018)](https://www.its.caltech.edu/~jkenny/nb250c/papers/Banino-2018.pdf); [DeepMind blog](https://deepmind.google/discover/blog/navigating-with-grid-like-representations-in-artificial-agents/)
- Goal-conditioned RL survey: it frames GCRL as RL where the policy and value also condition on a goal, and organizes the field by goal representation (vectors/states, images, language) and by solution type (relabeling, sub-goal generation, representation learning) — [Liu, Zhu & Zhang, IJCAI 2022, arXiv 2201.08299](https://www.ijcai.org/proceedings/2022/770)
- UVFA: value functions V(s, g) that generalize across goals, the conceptual basis for feeding a goal vector to the critic — [Schaul et al., ICML 2015](https://proceedings.mlr.press/v37/schaul15.html)
- HER (hindsight experience replay): relabels failed trajectories with goals that were actually reached. It is off-policy, built for sparse rewards — [Andrychowicz et al., NeurIPS 2017, arXiv 1707.01495](https://arxiv.org/abs/1707.01495)
- Semantic or object goals in a multi-goal sequence (MultiON): the agent must visit an ordered set of objects, a generalization of ObjectGoal navigation — [Wani et al., NeurIPS 2020, arXiv 2012.03912](https://arxiv.org/abs/2012.03912)

### Inferences
- Encodings that fit this project, from cheapest to most expensive:
  1. **Egocentric polar goal vector** (d, cos θ, sin θ), appended to the ray vector. This is the DD-PPO standard, adds 3 input dimensions, and works with all backbones.
  2. **Goal "beacon" ray channel.** The project already has a goal one-hot per ray, so line-of-sight goal sensing exists. A "through-walls beacon" adds one extra per-ray channel that marks the ray closest to the goal bearing, even when the goal is occluded. This is the same information as option 1 in a spatial, egocentric form, which suits the CNN-encoder comparison. No prior paper using this exact encoding was found, so it is a novel but natural ablation.
  3. **Absolute goal (x, y) without agent pose** is a negative control: the source above suggests it acts like no goal at all.
  4. **Absolute goal plus absolute agent pose** tests whether the network can learn the coordinate transform itself. The MLP may manage it; this is a test of the encoder's inductive bias.
- HER is mostly used with off-policy learners (DDPG, SAC). With PPO it does not fit naturally and is not needed when the goal vector is always present and rewards are dense enough. Leave it out for a workshop paper.
- ImageNav and ObjectNav have no natural equivalent in a 2D ray world, beyond "goal type id" if several goal types existed. Skip them.

### Gaps
- No small-scale 2D ray-cast paper was found that directly compares goal encodings (vector vs beacon vs absolute) under PPO. The 40% result above comes from one single-author preprint in Habitat, not a 2D study.
- A search for a 2024–2026 paper that ablates LSTM memory with LiDAR/ray inputs and a relative goal under PPO found nothing direct. One 2024 benchmark of denied-sensor navigation reports that the authors "were unable to find an appropriate learning rate to get PPO-LSTM to converge", and that TD3 beat PPO there. This hints that LSTM-PPO on ray inputs can be fragile — [arXiv 2410.14616](https://arxiv.org/pdf/2410.14616) (only the search snippet was read, the paper was not fetched).

## Q2. Key results from prior work: what patterns did they find?

### Takeaway
With a perfect egocentric goal vector, PointNav is close to "solved" by memory-based PPO agents. Even blind agents (goal vector plus egomotion only) reach about 95–97% success, though with clearly worse path efficiency (SPL), and they build map-like memories. Remove or corrupt the goal and localization signal, and performance collapses unless odometry is learned back in. When the goal is fixed, memory matters less; when the goal is randomized per episode, memory plus auxiliary losses matter a lot (Mirowski).

### Cited Findings
- **DD-PPO, Gibson val, with GPS+compass:** RGB-D 0.997 success / 0.969 SPL; RGB 0.991 / 0.929; **Blind (GPS+compass only) 0.973 / 0.729**. Trained on 2.5B frames (about 6 months of GPU time, under 3 days on 64 GPUs). 90% of peak performance came early, at about 100M steps — [arXiv 1911.00357 (fetched)](https://ar5iv.labs.arxiv.org/html/1911.00357); [Meta AI blog](https://ai.meta.com/blog/near-perfect-point-goal-navigation-from-25-billion-frames-of-experience/)
- DD-PPO's blind agent does well on short routes, which the authors attribute to structural regularities in indoor layouts. On long routes (20–25 m) its SPL falls to about 0.3, against about 0.95 for RGB-D. Without GPS+compass, the RGB agent reaches 0 SPL at 100M steps and only 0.15 SPL after 2.5B steps, which the authors call an open frontier — [arXiv 1911.00357 (fetched)](https://ar5iv.labs.arxiv.org/html/1911.00357)
- **Emergence of maps (ICLR 2023 outstanding paper):** blind agents with egomotion only reach about 95% success in new environments. They use memory over about 1,000 steps, map-like representations and "collision detection" neurons emerge, and the maps are selective: the agent forgets exploratory detours — [arXiv 2301.13261 (fetched)](https://arxiv.org/abs/2301.13261); [ICLR 2023 oral](https://www.iclr.cc/virtual/2023/oral/12560)
- **Noise and localization:** an agent that is near-perfect with ideal GPS/compass drops to about 0.3% success under realistic actuation and sensor noise without GPS/compass. Adding visual odometry recovers 71.7% success (52.5% SPL) — [Zhao et al., ICCV 2021, arXiv 2108.11550](https://arxiv.org/pdf/2108.11550v1.pdf). With perfect GPS+compass, an agent still reaches 99.8% success despite sensor and motion noise. Scaled visual odometry lifts the Realistic PointNav challenge from 71% to 94% success and from 53% to 74% SPL, without an explicit map — [Partsey et al., CVPR 2022, arXiv 2206.00997](https://arxiv.org/pdf/2206.00997). Datta et al. separate a visual-odometry module from the navigation policy so it can be recalibrated without retraining the policy; this was runner-up in the 2020 Habitat PointNav challenge — [CoRL 2020, arXiv 2009.03231](https://arxiv.org/abs/2009.03231v1)
- **Mirowski et al. (ICLR 2017), undirected, reward-only goal:** in static mazes (fixed goal) the feed-forward A3C was competitive. Static 1 scores: FF 79, LSTM 98, Nav A3C+D2 119 (125% of human). In random-goal mazes the plain LSTM barely beat FF: Random Goal 1 FF 61 vs LSTM 65; Random Goal 2 FF 69 vs LSTM 57. Nav A3C+D2 (stacked LSTM, plus velocity/action/reward inputs, plus depth prediction from the LSTM) scored 96 and 103. Depth prediction from the LSTM (D2) beat depth from the conv layer (D1); reward prediction helped less. Adding D1+D2+L (loop closure) mainly sped up learning. Position decoding: Nav A3C+D2 85.5% vs about 50% for FF/LSTM (RG1). I-maze AUC: FF 75.5, LSTM 112.4, Nav A3C+D2 203.5 — [arXiv 1611.03673 (fetched)](https://ar5iv.arxiv.org/html/1611.03673)
- Mirowski "latency" (time to first goal find vs time between later finds within an episode): Nav A3C+D2 went from 14.0 s to 7.1 s on Random Goal 1, which shows within-episode memory of the goal. FF (11.0 to 9.9 s) and LSTM (11.1 to 9.2 s) improved less. On the larger Random Goal 2, no agent clearly reduced latency after the first find — [arXiv 1611.03673 (fetched)](https://ar5iv.arxiv.org/html/1611.03673)
- **Banino et al. 2018:** grid-like codes support goal-directed vector navigation and shortcuts; lesioning them hurts both navigation and goal-vector decoding — [Nature 2018](https://www.its.caltech.edu/~jkenny/nb250c/papers/Banino-2018.pdf)
- **MultiON:** for sequential multi-goal tasks, performance falls sharply as the number of goals rises. A simple semantic-map agent beats more complex neural feature-map agents and a NoMap RNN baseline, and even oracle-map agents score fairly low — [Wani et al., NeurIPS 2020, arXiv 2012.03912](https://arxiv.org/abs/2012.03912) (abstract/project-level claims only; tables not fetched)
- **Memory Maze:** a 3D benchmark built to isolate long-term memory (objects, walls, own position) in 9x9 to 15x15 mazes. Current agents benefit from truncated BPTT and succeed on small mazes but fall short of humans on large ones — [Pašukonis, Lillicrap & Hafner, NeurIPS 2022 DRL Workshop, arXiv 2210.13383](https://arxiv.org/abs/2210.13383)
- **PPO vs behaviour cloning in PointNav:** BC is better on seen start–goal pairs (about 96% SPL/success, 13 SPL points above PPO). PPO generalizes better to unseen pairs and unseen scenes (96.75% zero-shot success on unseen pairs). Both policies are recurrent — [arXiv 2503.15693 (fetched)](https://arxiv.org/html/2503.15693v1)
- Foundational, by reference only (not re-checked here): Habitat platform and PointNav benchmark — [Savva et al., ICCV 2019, arXiv 1904.01201](https://arxiv.org/abs/1904.01201); Cognitive Mapping and Planning (differentiable egocentric map plus planner for goal navigation) — [Gupta et al., CVPR 2017, arXiv 1702.03920](https://arxiv.org/abs/1702.03920); Active Neural SLAM (modular mapper + global policy + local policy for exploration, transferred to PointGoal) — [Chaplot et al., ICLR 2020, arXiv 2004.05155](https://arxiv.org/abs/2004.05155)

### Inferences
- Results in the literature fall on a spectrum of how much memory is needed:
  - Fixed goal with no goal input: a reactive policy can memorize the route, and FF is competitive (Mirowski static mazes).
  - Randomized goal with no goal input: the agent must explore and then remember, so memory plus auxiliary losses win (Mirowski random goal, I-maze).
  - Perfect egocentric goal vector every step: mostly reactive "greedy plus wall-following" behaviour suffices in simple layouts. Memory then helps mainly with dead ends and long routes (DD-PPO blind agent: high success but low SPL on long paths; Wijmans 2023 shows memory is still used over about 1,000 steps).
- The project's room curriculum (one room up to the hard maze) puts it in a good position to show where a reactive goal-vector policy stops being enough. In C-mazes and hard mazes, greedy steering toward the goal bearing leads into dead ends; this is an inference, not tested here.

### Gaps
- No exact SPL number was retrieved for the Wijmans 2023 blind agents (the abstract gives only about 95% success).
- MultiON numbers were not fetched.
- No paper was found that directly reports "MLP vs LSTM with and without a goal vector" in a 2D ray environment. That gap is the opening for the proposed experiment.

## Q3. How does goal conditioning interact with memory? Which variants force memory use?

### Takeaway
Expect a crossover. With a clean per-step egocentric goal vector, the advantage of memory should shrink in simple rooms (reactive MLP is roughly as good as LSTM), but should persist or grow in dead-end mazes. Removing the goal signal after the start (goal cue only at t=0), adding noise to it, or showing it only intermittently turns goal conditioning into a memory and path-integration task, where LSTM and grid-cell / path-integration features should clearly win. Prior work supports both ends, but the 2D, cheap version of the full sweep has not been done.

### Cited Findings
- Memory length and credit-assignment length can be separated. In a Passive T-maze, the cue that decides the correct final turn must be remembered across a corridor, and the authors configure how far back it must be remembered. Transformers scale memory to about 1,500 steps but do not improve long-term credit assignment, and they can be less sample-efficient than LSTMs on short-dependency tasks — [Ni et al., NeurIPS 2023, arXiv 2307.03864](https://arxiv.org/pdf/2307.03864); code with a configurable memory length: [github.com/twni2016/Memory-RL](https://github.com/twni2016/Memory-RL)
- Blind agents that have only egomotion and a goal must integrate their path and build maps internally to reach about 95% success, so memory is used heavily even when the goal is given — [arXiv 2301.13261 (fetched)](https://arxiv.org/abs/2301.13261)
- When GPS/compass (the per-step goal update) is removed under noise, success falls from about 99.6% to about 0.3%. Recovering it requires an egomotion estimate to be integrated over time — [arXiv 2108.11550](https://arxiv.org/pdf/2108.11550v1.pdf); [arXiv 2206.00997](https://arxiv.org/pdf/2206.00997)
- In the reward-only random-goal setting, plain LSTM gave little over FF, while stacked LSTM plus velocity/action/reward inputs plus depth auxiliary loss roughly doubled performance. The memory architecture alone was not enough; inputs and auxiliary losses mattered — [arXiv 1611.03673 (fetched)](https://ar5iv.arxiv.org/html/1611.03673)
- Grid-like path-integration codes carry the goal vector internally and are causally needed for vector navigation. This is the natural mechanism for a "goal shown at the start, then hidden" task — [Banino et al. 2018](https://www.its.caltech.edu/~jkenny/nb250c/papers/Banino-2018.pdf)
- Sequential multi-goal tasks become harder as the number of goals rises, and simple map memory helps — [arXiv 2012.03912](https://arxiv.org/abs/2012.03912)

### Inferences (testable hypotheses)
- **H1 (memory becomes redundant):** with a per-step egocentric goal vector, the gap between MLP and LSTM (and frame stacking) shrinks relative to undirected search in open and low-dead-end layouts (one room, two rooms, four rooms, cross room).
- **H2 (topology dependence):** in dead-end layouts (C-maze, I/T-maze variants, medium and hard mazes), the goal vector misleads greedy reactive policies. LSTM keeps or widens its lead, and goal conditioning may even hurt the MLP in sample efficiency because it is drawn into the "local minimum" of heading straight at the goal.
- **H3 (goal-at-start forces memory):** show the goal vector only for the first k steps (or only at t=0), then zero it out with a visibility flag bit. MLP and frame stacking should collapse to near undirected-search performance. LSTM should keep most of its performance only if it learns to integrate its own path. Explicit grid-cell / path-integration features (or the agent's velocity and previous action as inputs, as in Mirowski) should help most here. This is the clearest "memory is necessary" condition and links goal conditioning to the project's grid-cell arm.
- **H4 (noisy or intermittent goal):** Gaussian noise on the bearing (for example σ = 15–30°) or a per-step dropout probability (for example the goal is visible with p = 0.1). Memory lets the agent filter or average, so LSTM should degrade gracefully and MLP should degrade sharply. Frame stacking sits in between, which is an informative contrast for the frame-stack backbone.
- **H5 (auxiliary losses):** depth prediction should help in all goal conditions (obstacle avoidance). Loop closure should matter more in undirected search and in goal-hidden runs than with a clean goal vector, where revisiting places is less relevant.
- **H6 (fixed vs random goal):** with a fixed goal and no goal input, MLP can memorize the route and FF is competitive (Mirowski static). Randomizing the goal per episode without a goal input requires exploration and memory. Giving the goal vector to a randomized-goal task should turn it back into something close to the fixed-goal regime for reactive policies.

### Gaps
- No direct published 2D ray-cast result was found for H1–H4. These are hypotheses, not findings.
- The details of the Memory Maze task protocol (target cue and reward sequence) were not verified in this session.

## Q4. Which evaluation protocols are used?

### Takeaway
The standard is success rate plus SPL (Success weighted by Path Length) on held-out start–goal pairs and, ideally, held-out layouts. Report the learning curve (or AUC) for sample efficiency. Mirowski-style "first find vs later finds" latency is a cheap memory diagnostic for undirected and repeated-goal tasks.

### Cited Findings
- SPL = (1/N) Σ S_i · ℓ_i / max(p_i, ℓ_i), where ℓ_i is the shortest-path length, p_i the agent's path length, and S_i binary success. A 50% success rate with optimal paths gives SPL 0.5, and so does full success with paths twice as long. It is the consensus recommendation of the embodied-navigation working group — [Anderson et al., arXiv 1807.06757](https://arxiv.org/abs/1807.06757)
- Habitat PointNav success criterion (as reported in a 2025 replication): stop within 0.2 m of the goal; episodes end at 500 steps or 200 collision steps. Evaluation splits use seen vs unseen start–goal pairs and unseen scenes (4 training scenes, 14 validation scenes) — [arXiv 2503.15693 (fetched)](https://arxiv.org/html/2503.15693v1)
- DD-PPO reports SPL and success with standard deviations on the validation split, plus SPL as a function of geodesic distance, which is how the blind agent's long-route weakness shows up — [arXiv 1911.00357 (fetched)](https://ar5iv.labs.arxiv.org/html/1911.00357)
- Mirowski reports AUC of the learning curve, final score, % of human score, latency (first find vs later finds), and position decoding accuracy from hidden states as a representation probe — [arXiv 1611.03673 (fetched)](https://ar5iv.arxiv.org/html/1611.03673)
- Probing hidden states for maps or positions, and "forgetting" of detours, is used as evidence of memory use — [arXiv 2301.13261 (fetched)](https://arxiv.org/abs/2301.13261); Memory Maze also proposes offline probing — [arXiv 2210.13383](https://arxiv.org/abs/2210.13383)

### Inferences
- For this project:
  - Success, SPL (shortest path from BFS/A* on the floorplan grid), and steps-to-goal.
  - Held-out goal positions: for example train on goals in a subset of cells or rooms, test on the rest.
  - Held-out layouts: train on a subset of the room family, test on another. Optional; it is costlier.
  - The AUC the project already computes, for sample efficiency.
  - SPL binned by geodesic distance or by number of turns required, to expose the dead-end failure.
- A cheap memory probe: a linear regression from the LSTM hidden state to the agent's (x, y) and to the goal vector in the goal-hidden condition. If the hidden state decodes the goal vector after it is hidden, the agent is doing path integration (as in Banino and Wijmans).

### Gaps
- No standard SPL variant for continuous-action 2D agents was found; computing it on a discretized occupancy grid is the practical choice.

## Q5. Concrete minimal experiment designs (2–4 conditions, a few CPU-days)

### Takeaway
Run one 4-level "goal information" factor (none / per-step egocentric vector / start-only vector / noisy-intermittent vector), crossed with the existing backbones (MLP, LSTM, frame stack), on 2–3 representative layouts (open, rooms, dead-end maze), with goals randomized per episode, at 3–5 seeds. This gives a clean interaction plot ("which backbone wins depends on goal information × topology") that prior work motivates but has not reported in a cheap 2D setting.

### Cited Findings (motivation for the design choices)
- Egocentric polar (d, cos θ, sin θ) is the standard encoding — [arXiv 1911.00357 (fetched)](https://ar5iv.labs.arxiv.org/html/1911.00357)
- A relative goal is essential; an absolute goal behaves like no goal — [arXiv 2503.15693 (fetched)](https://arxiv.org/html/2503.15693v1)
- 90% of DD-PPO's peak performance came early (about 100M of 2.5B steps). Most of the gain arrives early, so short training budgets can still separate conditions, though the 2D task is far smaller — [arXiv 1911.00357 (fetched)](https://ar5iv.labs.arxiv.org/html/1911.00357)
- Fixed vs random goal flips whether a feed-forward agent is competitive — [arXiv 1611.03673 (fetched)](https://ar5iv.arxiv.org/html/1611.03673)
- A cue given only at the start, with a configurable delay, isolates memory — [arXiv 2307.03864](https://arxiv.org/pdf/2307.03864)

### Inferences (proposed designs)
**Design A: "Does goal information make memory unnecessary?" (core, cheapest)**
- Conditions, goal randomized per episode:
  - G0 none (current undirected search)
  - G1 egocentric (d, cos θ, sin θ) every step
  - G2 the same, shown only for the first ~5 steps, then zeroed, with a 1-bit "goal visible" flag
- Backbones: MLP, frame stack, LSTM.
- Layouts: four rooms (open/rooms) and C-maze or medium maze (dead ends).
- Size: 3 × 3 × 2 = 18 cells, times 3–5 seeds.
- Expected publishable pattern:
  - G1 closes the MLP–LSTM gap on open layouts but not on dead-end layouts.
  - G2 restores, or amplifies, the LSTM advantage.
  - Frame stacking helps in G1 and fails in G2 once the delay exceeds the stack length.

**Design B: "Noisy or intermittent goal signal" (robustness axis)**
- Fix the LSTM and MLP backbones on one maze.
- Sweep the probability that the goal is visible on a step, p ∈ {1.0, 0.3, 0.1, 0.0}, or bearing noise σ ∈ {0°, 15°, 45°}.
- Plot success/SPL vs p. Expected: the MLP curve drops steeply and the LSTM curve degrades gracefully. The crossing point is the headline number.

**Design C: "Goal encoding inductive bias" (encoder arm)**
- G1 vector vs beacon ray channel (goal bearing marked on the ray array, through walls) vs absolute goal + absolute pose vs absolute goal only (negative control).
- Cross with the MLP vs CNN encoder.
- Hypothesis: the beacon channel favours the CNN (spatially aligned with the rays), the vector favours the MLP, and absolute-only is about equal to G0.

**Design D: "Generalization to held-out goals" (cheap add-on to A)**
- Train goals sampled from a subset of rooms or cells; evaluate success/SPL on held-out goal regions, and optionally on a held-out layout from the curriculum.
- Expectation from the PointNav literature: relative goal vectors generalize well (PPO 96.75% zero-shot on unseen start–goal pairs, arXiv 2503.15693). Testing whether that holds for a 2D ray agent, and whether the LSTM generalizes better than the MLP, is cheap.

**Optional extensions, if time allows**
- Grid-cell arm under G2: compare LSTM vs LSTM + grid-cell features vs LSTM + velocity/previous-action inputs (Mirowski "Nav A3C" inputs). Hypothesis: path-integration features matter most when the goal is hidden after the start, mirroring Banino.
- Sequential multi-goal: 2–3 goals per episode, each revealed on reaching the previous one. Mirowski's first-find vs later-find latency is a free memory diagnostic.
- Auxiliary losses × goal condition: loop-closure benefit expected larger under G0 and G2 than G1.

**Practical notes**
- Use the same reward (sparse goal reward plus small step penalty) across all conditions. Avoid distance-based shaping in G1 only, or the comparison is confounded.
- Keep the goal input dimension constant across conditions (zeros plus a visibility flag in G0 and G2) so the network size is identical.
- Report success, SPL, AUC, and SPL binned by geodesic distance or number of turns, with confidence intervals over seeds.

### Gaps
- Exact CPU compute needs depend on the project's environment speed. No external benchmark of steps needed for 2D ray PPO PointNav was found, so check with a pilot run (for example 1 seed × all cells at a reduced budget).
- HER and UVFA-style relabeling with PPO is not standard and was not investigated further; it is recommended to leave it out.
