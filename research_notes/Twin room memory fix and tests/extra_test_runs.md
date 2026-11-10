# Extra test runs, controls and ablations a reviewer would expect for the E2 memory-architecture comparison (twin rooms, PPO, ray casts)

Scope: which controls and extra tests recent memory-RL and navigation-memory papers (2020–2026) include, what reviewers asked for, and a short, costed list of extra runs for E2. Run counts assume 5 seeds per config. The current E2 grid is 7 agents (MLP, FS4, FS16, LSTM-16/64/256, GRU) × 5 L values (8, 32, 64, 128, 256, matching `rooms/twin_rooms_L*.json`) × 5 seeds = 175 runs. With ~16 concurrent jobs, one "wave" is 16 runs, or about 1–5 h of wall time.

Overlap note: the sibling file `memory_diagnostics.md` covers eval-time causal interventions (hidden-state resets, swaps, probes) in depth. `exploration_fix_details.md` covers the shaping fix. This file references them only where a control needs them.

---

## Q1. Which controls do memory-RL papers include (tuning budget, LR sweeps, shared vs separate memory, forget-gate bias, burn-in/refresh, stochastic vs deterministic eval, seeds, parameter matching)?

### Takeaway
The controls reviewers and best-practice papers keep returning to are:
- **Equal tuning effort per architecture**, at least the same number of hyperparameter settings tried for each.
- **Calibration baselines**: an oracle/MDP version and a Markovian (memoryless) version.
- **Separate actor and critic memory**, or a justification for sharing.
- **Honest statistics** over 3–10 seeds.

Our two biggest exposures are the LR mismatch (3e-4 vs 1e-4, each picked by its own pilot) and the shared recurrent trunk. Both are documented failure modes in Ni et al. 2022 and Andrychowicz et al. 2021. Forget-gate bias and stored hidden states without refresh are second-order, but a reviewer can raise either cheaply.

### Cited Findings

**Tuning fairness and LR**
- Patterson et al. (JMLR 2024) say that "at a minimum, you should ensure all algorithms have the same number of hyperparameter settings that are tested". They also warn about untuned baselines with a worked example: DQN with default hyperparameters looks much worse than a new method until DQN is retuned. — [Patterson et al., Empirical Design in RL, arXiv 2304.01315 (JMLR 25, 2024)](https://arxiv.org/abs/2304.01315)
- Patterson et al. separate two claims: (1) "the baseline algorithm experiences failure" and (2) the proposed method fixes it. They say claim (1) is the hard one, because "we can always ask: would this happen for a different choice of hyperparameters?" A fair comparison therefore spends more effort understanding the baseline. — [Patterson et al. 2024](https://arxiv.org/abs/2304.01315)
- Patterson et al. criticise the common two-stage "sweep, then re-run the best config with new seeds" approach as overconfident (maximisation bias), but still call it "much better than one-stage tuning or 'tuning by hand'". They propose a bootstrapped two-stage alternative that resamples the H×N runs of a sweep. — [Patterson et al. 2024](https://arxiv.org/abs/2304.01315)
- Patterson et al.'s key insights:
  - "the seed is NOT a tuneable hyperparameter"
  - use random/oracle calibration baselines
  - use paired comparisons
  - set environment and agent seeds separately so that comparisons are repeated measures

  — [Patterson et al. 2024](https://arxiv.org/abs/2304.01315)
- Jordan et al. (ICML 2020) argue that a performance metric should include the time and effort spent tuning hyperparameters. — [Jordan et al., Evaluating the Performance of RL Algorithms, arXiv 2006.16958](https://arxiv.org/abs/2006.16958)
- Andrychowicz et al. (ICLR 2021, 250k PPO-style runs) found that the learning rate "influences the performance very strongly", with Adam 3e-4 performing well on all their tasks. They also note that LR interacts with minibatch size and the number of passes, so these "need to be tuned together". — [Andrychowicz et al., What Matters in On-Policy RL?, arXiv 2006.05990](https://arxiv.org/abs/2006.05990)
- POPGym (ICLR 2023) used one shared PPO hyperparameter set for all 13 memory models. It was found by trial and error on two environments and then made more conservative (larger batch, lower LR). The authors say structured per-model tuning was "not computationally feasible". — [Morad et al., POPGym, arXiv 2303.01859](https://arxiv.org/abs/2303.01859)
- Ni et al. (ICML 2022) used the same hyperparameters for every task within a benchmark. They trained oracle and Markovian reference policies "with the same hyperparameters as our recurrent model-free implementation", and admit this probably explains why the oracle sometimes underperformed. — [Ni et al., Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs, arXiv 2110.05038](https://arxiv.org/abs/2110.05038)
- GTrXL (ICML 2020) tested robustness by sampling 25 independent runs per model, each with hyperparameters drawn from a distribution as well as a fresh seed. It reported the ranked returns and the fraction of runs that diverged, a "hyperparameter-and-seed robustness" control. — [Parisotto et al., Stabilizing Transformers for RL, arXiv 1910.06764](https://arxiv.org/abs/1910.06764)

**Shared vs separate actor-critic memory**
- In Ni et al. 2022, a shared RNN encoder for actor and critic "failed to learn" two simple POMDPs, while separate encoders learned them. Gradient norms suggest "the gradient of critic loss may dominate the actor's". Summary finding 1: separate weights "can boost performance, likely because it avoids gradient explosion". (Off-policy setting.) — [Ni et al. 2022](https://arxiv.org/abs/2110.05038)
- Andrychowicz et al. 2021: separate value and policy networks performed better on 4 of 5 environments (MLP PPO), so the rest of their study used only separate networks. — [Andrychowicz et al. 2021](https://arxiv.org/abs/2006.05990)
- AMAGO (ICLR 2024): sharing one sequence model between actor and critic "has repeatedly been shown to be unstable". AMAGO only makes sharing work by putting each loss term on a predictable scale and carefully detaching the critic from the actor loss. — [Grigsby et al., AMAGO, arXiv 2310.09971](https://arxiv.org/abs/2310.09971)
- Counter-precedent: the Memory Gym baseline (JMLR 2025, PPO + GRU/TrXL) shares encoder and memory parameters "among several heads dedicated to the policy, state-value function, observation reconstruction, and ground truth estimation". — [Pleines et al., Memory Gym, arXiv 2309.17207](https://arxiv.org/abs/2309.17207)
- The CleanRL "37 implementation details" post notes that a shared network is "the default setting in PPO", but that separate networks "clearly outperform" in simpler MLP environments. — [Huang et al., The 37 Implementation Details of PPO, ICLR Blog Track 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- POPGym Arcade (ICML 2026 spotlight) uses PQN as its main algorithm specifically to eliminate "common confounding factors such as shared value/policy trunks, target networks, and replay buffers". It repeats most experiments with PPO and DQN in an appendix. — [Wang et al., Investigating Memory in Model-Free RL with POPGym Arcade, arXiv 2503.01450](https://arxiv.org/abs/2503.01450); acceptance: [ICML 2026 decision on OpenReview](https://openreview.net/forum?id=vBMdonzFEV)

**LSTM init (forget-gate bias)**
- Jozefowicz et al. (ICML 2015) searched over ten thousand RNN variants. They found that a forget-gate bias of 1 closes the LSTM–GRU gap, and recommend raising the forget-gate bias "before attempting to use more sophisticated" architectures. (Gers et al. 2000 originally advocated this.) — [Jozefowicz et al., An Empirical Exploration of Recurrent Network Architectures](https://proceedings.mlr.press/v37/jozefowicz15.html)
- The CleanRL PPO-LSTM reference initialises LSTM weights with std=1 and biases at 0, so there is no forget bias. It starts hidden states at zero, resets them on episode end, and "reconstructs LSTM states during training" from a saved `initial_lstm_state`. — [Huang et al. 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

**Stored states, burn-in, refresh**
- R2D2 (ICLR 2019) identified "recurrent state staleness" when stored hidden states are replayed after the parameters have changed. It combined stored states with a burn-in prefix that is unrolled without gradient. (Known through secondary summaries here; I did not read the original full text.) — [Kapturowski et al., R2D2, OpenReview](https://openreview.net/forum?id=r1lyTjAqYX); summary: [Acme paper, arXiv 2006.00979](https://arxiv.org/abs/2006.00979)
- Memory Gym raises the same concern for PPO. If an episode outlives the 512-step rollout, "the older hidden states from that episode become outdated over multiple iterations of PPO". The authors suggest recomputing or periodically refreshing cached states as future work. — [Pleines et al. 2025](https://arxiv.org/abs/2309.17207)
- Memory Maze found that truncated BPTT, which replays full trajectories sequentially and carries the state across batches, gave "a clear boost" over standard Dreamer at all maze sizes. — [Pasukonis et al., Evaluating Long-Term Memory in 3D Mazes, arXiv 2210.13383](https://arxiv.org/abs/2210.13383)

**Context length and BPTT window**
- Ni et al. 2022: there is no significant overall difference between LSTM and GRU. Context length 64 dominated the best variants, but "the choice of context length seems to be problem-specific and may require tuning". Shortening from 64 to 5 sometimes helped by 2.2×. — [Ni et al. 2022](https://arxiv.org/abs/2110.05038)
- Cherepanov et al. (ICLR 2026) define the agent context K (for an RNN this is the backprop window) and the correlation horizon ξ (event-to-recall delay). A task tests long-term memory (LTM) only if ξ > K. — [Cherepanov et al., Unraveling the Complexity of Memory in RL Agents, arXiv 2412.06531](https://arxiv.org/abs/2412.06531)

**Stochastic vs deterministic evaluation**
- Classic POMDP theory: optimal memoryless policies in POMDPs are in general stochastic. Montúfar et al. report that restricting the policy class appropriately gave better and faster policy-gradient convergence. — [Montúfar, Ghazi-Zahedi & Ay, arXiv 1503.07206](https://arxiv.org/abs/1503.07206)

**Seeds and statistics**
- rliable (NeurIPS 2021): "3-10 runs are prevalent in deep RL". The authors recommend stratified-bootstrap interval estimates, IQM, performance profiles, and reporting the "average probability of improvement". — [Agarwal et al., Deep RL at the Edge of the Statistical Precipice, arXiv 2108.13264](https://arxiv.org/abs/2108.13264)
- Seed counts in comparable papers:
  - Ni et al. 2023 (T-maze): 10 seeds, mean ± 95% CI
  - Memory Maze: 5 seeds
  - Memory Gym: 5 seeds per config, 15 for aggregated curves
  - POPGym: 3 trials per model–env–difficulty (>1700 runs)
  - MIKASA: 3 training runs × 100 eval episodes
  - POPGym Arcade: 5 seeds

  — [Ni et al. 2023](https://arxiv.org/abs/2307.03864); [Pasukonis et al. 2022](https://arxiv.org/abs/2210.13383); [Pleines et al. 2025](https://arxiv.org/abs/2309.17207); [Morad et al. 2023](https://arxiv.org/abs/2303.01859); [Cherepanov et al., MIKASA, arXiv 2502.10550](https://arxiv.org/abs/2502.10550); [Wang et al. 2025/26](https://arxiv.org/abs/2503.01450)

**Parameter or state-size matching**
- POPGym states "there is no clear axis for a truly fair comparison between memory models". It fixes recurrent state size at 256 for every model, above the common 64–128 (Ni et al. 2022), rather than matching parameter counts. — [Morad et al. 2023](https://arxiv.org/abs/2303.01859)
- POPGym Arcade likewise used recurrent state size as the comparison axis. The authors note that parameter count or wall-clock time "could change how the models compare to each other". — [Wang et al.](https://arxiv.org/abs/2503.01450)
- Ni et al. 2023 fixed hidden size 128 for both LSTM and GPT-2 and tuned only depth {1, 2, 4}. — [Ni et al. 2023](https://arxiv.org/abs/2307.03864)

### Inferences
- **LR confound.** "Each architecture got its own pilot" is defensible only if the pilots tried the same number of LR values per architecture (Patterson's minimum). If they did not, the cheapest fix is an LR swap: MLP/FS at 3e-4 and LSTM/GRU at 1e-4, on the diagnostic L values. The stronger fix is a 3-point LR grid {1e-4, 3e-4, 1e-3} per agent at one L.
- **Shared trunk.** This is the most likely reviewer attack on the recurrent rows, because the strongest evidence (Ni 2022) says shared recurrent actor-critic trunks can fail outright. Memory Gym gives a PPO precedent for sharing. An honest framing is "we follow the common PPO convention (CleanRL, Memory Gym) and test separate trunks as a control".
- **Forget-gate bias.** At L ≤ 256 with BPTT 16/64/256, the 1.0 init matters most for LSTM-16 at large L. It is cheap to add as part of a "best-practice LSTM" control rather than as a full factorial.
- **Deterministic mean-action eval.** For the MLP and frame-stack agents at the junction, the theoretically best memoryless policy is a 50/50 guess. A deterministic agent instead commits to one side, so per-seed eval success sits near 50% only if the cue sides are balanced. It can also show loop or stuck behaviour that a stochastic policy escapes. Reporting both eval modes costs zero training runs.
- **Per-seed binary outcomes.** With all-or-nothing results, 5 seeds give a very coarse estimate. A 95% Clopper–Pearson interval for 5/5 solved seeds is [0.48, 1.0], and for 3/5 it is [0.15, 0.95]. With 10 seeds, 10/10 gives [0.69, 1.0]. (My own binomial calculation.) IQM over 5 binary-ish seeds hides this, so also report "fraction of seeds solved" with exact binomial CIs.

### Gaps
- I found no paper that runs an explicit equal-budget per-architecture LR sweep for PPO + LSTM/GRU/frame stack in navigation. The standard practice in POPGym and Ni et al. is shared hyperparameters, which is itself a known limitation.
- I found no direct empirical study of forget-gate bias, or of deterministic vs stochastic eval, specifically in recurrent PPO navigation.
- I did not retrieve the R2D2 full text, so the claim about stored state vs burn-in effectiveness rests on secondary summaries.

---

## Q2. Which extra memory tests strengthen a claim (distractors, K>2 rooms, variable L, train-L/test-2L, transformer/SSM reference, oracle and no-cue bounds)?

### Takeaway
The most persuasive additions are, in order:
1. Calibration bounds: an oracle that is given the cue, and a cue-removed or Markovian floor. Reviewers and Patterson expect these.
2. A fixed-ξ vs variable-ξ design that respects ξ > K when claiming long-term memory (Cherepanov ICLR 2026).
3. A held-out-L generalisation test.

Richer cues (K>2 rooms) and distractors answer the common "it's only one bit" critique. A transformer or SSM reference is nice but not expected at workshop level.

### Cited Findings

**Oracle and Markovian bounds**
- Ni et al. 2022 plot an oracle policy (sees the hidden state: "an upper bound"), a Markovian policy ("lower bound") and a random policy on every learning curve. — [Ni et al. 2022](https://arxiv.org/abs/2110.05038)
- Ni et al. 2023 report the optimal Markovian return (0.5 on Passive T-Maze, about 1/3 success on Passive Visual Match) as a reference line. — [Ni et al. 2023](https://arxiv.org/abs/2307.03864)
- MIKASA tabulates "Oracle Info" per task and trains state-based (oracle) baselines next to RGB MLP/LSTM agents. — [Cherepanov et al., MIKASA (ICLR 2026), arXiv 2502.10550](https://arxiv.org/abs/2502.10550)
- POPGym Arcade gives every environment an MDP twin and a POMDP twin with identical pixel states. It uses the observability gap between them as a controlled measure. The authors find "the return is a confounded metric for benchmarking memory": memory models differ in "Memory Bias", and history can even hurt. — [Wang et al.](https://arxiv.org/abs/2503.01450)
- Patterson et al. recommend random and oracle calibration baselines to show "what does that line even mean". — [Patterson et al. 2024](https://arxiv.org/abs/2304.01315)
- POPGym: the plain MLP got "the highest score on almost all navigation tasks, beating all memory models". The authors say this "raises doubts concerning previous models evaluated solely on navigation tasks". They also recommend a positional MLP (an MLP given the timestep) as "a standard baseline in future memory comparisons", because it unexpectedly outperformed the MLP in many tasks. — [Morad et al. 2023](https://arxiv.org/abs/2303.01859)

**Fixed vs variable horizon, and K vs ξ**
- Cherepanov et al. (ICLR 2026) ran SAC-GPT-2 on Minigrid-Memory. With variable corridor length (ξ ∈ [7, 22]), success was high for both K=22 and K=14 ("implying good memory"). With fixed L=21 (ξ=22) and K=14, the agent failed. "Mixed-horizon tasks can hide LTM deficits - only fixed ξ > K setups expose them." — [Cherepanov et al., arXiv 2412.06531](https://arxiv.org/abs/2412.06531)
- The same paper's Table 2 lists the correlation horizons of popular tasks, e.g. Passive T-Maze {L+1} and Memory Maze 9x9 [28, 1000]. — [Cherepanov et al.](https://arxiv.org/abs/2412.06531)

**Long horizons and reference architectures**
- Ni et al. 2023 vary Passive T-Maze memory length from 50 up to 1500. Transformers solve all lengths, and "LSTM-based agents start to falter at a memory length of 250". They separate memory length from credit-assignment length (Active T-Maze, Key-to-Door), where both architectures fail at 250. — [Ni et al. 2023](https://arxiv.org/abs/2307.03864)
- Results across benchmarks are mixed on whether newer architectures beat RNNs in RL:
  - GTrXL surpassed LSTMs on memory-based DMLab-30 levels while matching them on reactive ones. — [Parisotto et al.](https://arxiv.org/abs/1910.06764)
  - POPGym found "classical RNNs outperformed modern memory methods". GRU was the best general-purpose model, S4D was unstable, and linear-attention models were among the worst. — [Morad et al. 2023](https://arxiv.org/abs/2303.01859)
  - Memory Gym found TrXL often superior on finite tasks, but GRU "marks a comeback" on endless tasks. — [Pleines et al. 2025](https://arxiv.org/abs/2309.17207)
  - S5 outperformed GRUs on POPGym and ran over six times faster. — [Lu et al., Structured State Space Models for In-Context RL, arXiv 2303.03982](https://arxiv.org/abs/2303.03982)
  - LRU had the best overall return in POPGym Arcade, "however this is unlikely to hold across different implementations or configurations". — [Wang et al.](https://arxiv.org/abs/2503.01450)

**Endless or harder variants and distractors**
- Memory Gym argues that "finite environments provide only a truncated view of the agents' capabilities". Its endless versions act as an automatic curriculum that separates GRU from TrXL. — [Pleines et al. 2025](https://arxiv.org/abs/2309.17207)
- One NeurIPS 2023 reviewer of Ni et al. 2023 raised distractors: with irrelevant background activity, "excelling at pure memory is actually disadvantageous". Another (confidence 5) said Passive T-Maze "only evaluates whether a small bit of information can be stored and carried across long timespans" and asked that the paper state it does not test "memory capacity, or robustness of memory to noise". — [OpenReview reviews of "When Do Transformers Shine in RL?", NeurIPS 2023](https://openreview.net/forum?id=APGXBNkt6h)

**Memory-length probes without retraining**
- Wijmans et al. (ICLR 2023) restrict the agent's memory budget k at evaluation: they rebuild the LSTM state from only the last k observations, starting from zeros. Performance as a function of k shows how far back memory is actually used. In their case, memoryless agents get near 0% success and performance keeps rising up to hundreds of steps. — [Wijmans et al., Emergence of Maps in the Memories of Blind Navigation Agents, arXiv 2301.13261](https://arxiv.org/abs/2301.13261)

### Inferences
- **Oracle agent.** An oracle MLP with the cue bit appended to the observation, run at every L, is the single most informative training control. If the oracle also fails at L=256, the large-L drop is navigation or exploration difficulty, not memory, and the memory claim must be restated.
- **No-cue control.** An agent whose cue is identical on both sides (or masked) should sit at about 50%. Under the new cue-independent junction shaping, this empirically verifies that neither shaping nor room geometry leaks the cue.
- **K vs ξ.** Cherepanov's framework fits E2 directly. LSTM-16 at L ≥ 32 is a genuine LTM test (ξ > K), whereas LSTM-256 at L ≤ 256 tests within-window (STM) memory. The paper should state ξ for each L, which depends on the steps from cue to decision, not only on the corridor length L. It should also note that frame stack 4/16 has a hard context K = 4/16.
- **Cue-delay generalisation.** Train at L and test at 2L (and L/2) on existing checkpoints. This costs no training runs and is the Cherepanov-recommended way to expose failures that within-distribution success hides. Training on a mix of L values and testing on held-out L is the run-costing version.
- **Transformer/SSM reference.** Given the mixed literature (GRU often wins under PPO; transformers win only at very long horizons), a transformer or SSM row is "upper reference" rather than "expected baseline" for L ≤ 256. For a workshop it can be deferred with a sentence citing POPGym, Memory Gym and Ni 2023.

### Gaps
- I did not find a navigation paper that runs exactly a K>2-room cue variant with PPO memory baselines. K-way cues appear in benchmarks (e.g. Memory Maze object colours, MIKASA categories) but not as a controlled K sweep.
- I found no reviewer statement that explicitly demands a transformer baseline for an RNN-only comparison at short horizons.

---

## Q3. What do recent navigation-memory papers (2022–2026) report, and which baselines or ablations did reviewers ask for?

### Takeaway
Navigation-memory papers rely heavily on **mechanistic evidence beyond success rate**: memory-budget truncation, probes and decoders, ground-truth heads, and human or oracle ceilings. They also use **5–10 seeds with CIs**. Reviewers ask for:
- significance tests and better-tuned LSTM baselines
- more than one algorithm or environment
- richer-than-one-bit memory demands
- justification that a new toy benchmark is needed

### Cited Findings

**What the papers report**

| Paper | What they report |
|---|---|
| Wijmans et al. 2023 (ICLR) | Blind LSTM agents. Memory-length truncation curve (k from 1 to 10³). A "probe agent" with an implanted memory takes shortcuts. Collision detection is linearly decodable (98% held-out accuracy; 96% from 10 neurons). Occupancy maps are decodable from memory. — [Wijmans et al.](https://arxiv.org/abs/2301.13261) |
| Memory Maze 2022 | Online RL benchmark plus an offline dataset, an offline probing benchmark (object locations, walls), human-player baseline and oracle maximum score. Baselines: IMPALA and Dreamer with and without TBTT, plus a "TBTT + Probe loss" auxiliary variant. 5 seeds. — [Pasukonis et al.](https://arxiv.org/abs/2210.13383) |
| Memory Gym (JMLR 2025) | PPO with GRU vs TrXL on finite and endless tasks. Ablations include observation-reconstruction and ground-truth-estimation auxiliary heads; TrXL improved only with the reconstruction loss. Diagnoses stale cached states. 5 seeds per config, 15 for aggregates. — [Pleines et al.](https://arxiv.org/abs/2309.17207) |
| MIKASA (ICLR 2026) | Classifies memory tasks into four categories. Online RL baselines with MLP and LSTM backbones, plus oracle-information baselines. 3 runs × 100 eval episodes. — [Cherepanov et al.](https://arxiv.org/abs/2502.10550) |
| POPGym (ICLR 2023) | 13 memory models (including MLP, positional MLP and frame stack) on 15 envs. Shared PPO hyperparameters, state size 256, 3 trials, max-mean episodic reward (MMER). Finds navigation tasks weak at discriminating memory. — [Morad et al.](https://arxiv.org/abs/2303.01859) |
| POPGym Arcade (ICML 2026) | MDP/POMDP twins, memory saliency and "recall density" tools, the observability gap and Memory Bias metrics. 7 architectures × 5 seeds, PQN with PPO and DQN replications. Out-of-distribution observations "contaminate memory". — [Wang et al.](https://arxiv.org/abs/2503.01450) |
| Ni et al. 2023 (NeurIPS oral) | Passive/Active T-Maze, Visual Match, Key-to-Door. LSTM vs GPT-2 at equal hidden size with depth tuned. 10 seeds, 95% CI. — [Ni et al. 2023](https://arxiv.org/abs/2307.03864) |

**What reviewers asked for**
- **Ni et al. 2023, NeurIPS reviews:**
  - "a significance test would benefit the interpretation of these results"
  - the reported LSTM failure at 250 conflicts with LSTM succeeding at 750 in Fig. 2, "suggests that some other effects might influenced LSTM training, such as hyperparameter tuning. Maybe the authors can improve the results of LSTM by further tuning"
  - Passive T-Maze is "very minimalistic" (one bit)
  - "experimental evaluation ... in additional environments other than the toy tasks"
  - a distractor/irrelevant-information scenario

  — [OpenReview forum APGXBNkt6h](https://openreview.net/forum?id=APGXBNkt6h)
- **POPGym Arcade, ICLR 2026 submission:**
  - "Only one algorithm (PQN) is used", asking for more memory-focused methods (R2I)
  - tasks "relatively simple", asking for harder ones such as Memory Maze
  - a rating-0 review contesting the need for a new set of "toy" environments when DMLab, Memory Gym and POPGym exist

  The authors answered by adding PPO and DQN experiments and a reward-predictor (γ=0) control. The revised paper was accepted as an ICML 2026 spotlight; the meta-review says reviewers praised the controlled "mechanistic" study. — [ICLR 2026 submission forum](https://openreview.net/forum?id=gwZ1DacxEk); [ICML 2026 forum](https://openreview.net/forum?id=vBMdonzFEV)
- **Navigation-specific ICLR 2025 reviews** (Memory Proxy Maps, a visual-nav memory paper): reviewers asked for "necessary ablation studies (especially ... quantitative metrics)" for each component, and for evidence that the method generalises to new environments. — [OpenReview, ICLR 2025 Submission 5038](https://openreview.net/forum?id=Zf7EFQt04n)

### Inferences
- For a small 2D ray-cast navigation paper, the reviewer template above translates to:
  - "is the LSTM well-tuned?" (the LR, shared-trunk and forget-bias controls)
  - "is this just one bit?" (K>2 rooms or a distractor phase)
  - "is success rate confounded?" (oracle/no-cue bounds, probes, memory-budget truncation)
  - "how significant?" (probability of improvement, exact binomial CIs over seeds)
- POPGym's finding that MLPs win on navigation tasks is a direct risk for us. The paper should show that the twin-room task provably defeats memoryless agents (≈50% at the junction) and report the measured MLP floor at every L.
- E4's collapse of CNN + LSTM + loop-closure aux, with the aux gradient dominating, echoes Ni 2022's critic-dominates-actor gradient finding and AMAGO's need for per-loss scaling. Logging per-loss gradient norms on the shared trunk is a cheap supporting diagnostic.

### Gaps
- OpenReview's forum API asked for a bot-verification challenge, so I did not read every review thread. I could retrieve reviews only through the search endpoint. I could not get the POPGym (ICLR 2023) or Wijmans (ICLR 2023) review texts, so I cannot say what their reviewers asked for.
- Ni et al. 2022 (ICML) reviews are not public on OpenReview.

---

## Q4. Given ~16 concurrent CPU jobs at 1–5 h per run, which 3–5 additional runs give the most evidence per compute?

### Takeaway
Do the zero-training eval battery first. Then, in priority order:
1. Oracle and no-cue bounds (~25–35 runs)
2. An LR-swap fairness control (~20–40 runs)
3. A "best-practice recurrent" control: separate trunks plus forget bias 1 (~20 runs)
4. 5 extra seeds at the decision-boundary cells (~20–30 runs)

That is about 85–125 runs, roughly 6–8 waves of 16, which neutralises the main reviewer attacks. K>2 rooms, a distractor phase, variable-L training, burn-in and a transformer/SSM row are second-tier.

### Cited Findings
- Each item's rationale is sourced in Q1–Q3. The key anchors:
  - oracle and Markovian bounds — [Ni et al. 2022](https://arxiv.org/abs/2110.05038), [Patterson et al. 2024](https://arxiv.org/abs/2304.01315)
  - equal tuning effort — [Patterson et al. 2024](https://arxiv.org/abs/2304.01315)
  - shared trunk failures — [Ni et al. 2022](https://arxiv.org/abs/2110.05038), [AMAGO](https://arxiv.org/abs/2310.09971)
  - forget bias — [Jozefowicz et al. 2015](https://proceedings.mlr.press/v37/jozefowicz15.html)
  - fixed ξ > K and held-out horizons — [Cherepanov et al.](https://arxiv.org/abs/2412.06531)
  - memory-budget truncation at eval — [Wijmans et al.](https://arxiv.org/abs/2301.13261)
  - seeds and statistics — [Agarwal et al.](https://arxiv.org/abs/2108.13264)

### Inferences

Prioritised list. Each entry gives what it rules out and the cost in training runs (5 seeds per config). Cell choice follows the rule "spend runs where E2 curves transition": the largest L where the best recurrent agent still succeeds, plus the next L up. Below this is written as L* and L*+.

**P0. Eval-only battery on existing E2 checkpoints. 0 training runs; eval jobs only (minutes to an hour each).**
- (a) Deterministic mean-action and stochastic sampling eval, both reported. Rules out an eval-mode artifact, especially for MLP/FS, where the best memoryless policy is a 50/50 guess. [Montúfar et al.](https://arxiv.org/abs/1503.07206)
- (b) Cue flip or mask at test. Success should invert or drop to ~50% for genuine memory users. Rules out geometry or shaping shortcuts.
- (c) Hidden-state reset right after the cue, or a Wijmans-style memory budget k. Rules out "success without using memory". [Wijmans et al.](https://arxiv.org/abs/2301.13261)
- (d) Train at L, test at L/2 and 2L. Rules out horizon overfitting, and reports whether LSTM-16 generalises beyond its BPTT window. [Cherepanov et al.](https://arxiv.org/abs/2412.06531)
- (e) Linear probe of the cue from the hidden state at the junction (details in `memory_diagnostics.md`).

**P1. Calibration bounds. 15–35 runs.**
- (a) Oracle MLP with the cue bit appended, at all 5 L = 25 runs, or at 3 L (8, L*, 256) = 15 runs. Rules out that large-L failure comes from navigation or exploration difficulty rather than memory. [Ni et al. 2022](https://arxiv.org/abs/2110.05038); [MIKASA](https://arxiv.org/abs/2502.10550)
- (b) No-cue control: LSTM-64 with the cue made identical, at 2 L = 10 runs. Rules out cue leakage through the new junction shaping or room geometry, and gives the empirical Markovian floor (~50%).
- Optional: a positional MLP (MLP + timestep) at 2 L = 10 runs, the POPGym-recommended baseline. It rules out timing shortcuts. [Morad et al. 2023](https://arxiv.org/abs/2303.01859)

**P2. LR fairness swap. 20–40 runs.**
- MLP and FS16 at 3e-4, and LSTM-64 and GRU at 1e-4, at L* and L*+ (4 agents × 2 L × 5 = 40 runs). The cheaper version at L* only is 20 runs.
- Rules out "the ranking is an LR artifact" and satisfies the same-number-of-settings minimum, since each agent then has 2 LR values. [Patterson et al. 2024](https://arxiv.org/abs/2304.01315); [Andrychowicz et al. 2021](https://arxiv.org/abs/2006.05990)
- The stronger version is a 3-point grid {1e-4, 3e-4, 1e-3} for MLP, FS16, LSTM-64 and GRU at L*: 40 new runs.

**P3. Best-practice recurrent control. 20 runs.**
- LSTM-64 with separate actor and critic trunks plus forget-gate bias 1.0, and GRU with separate trunks, at L* and L*+ (2 × 2 × 5 = 20 runs).
- Rules out "the recurrent agents are handicapped by implementation". If results match the shared-trunk runs, report the shared trunk as harmless. If they improve, they become the headline and the shared-trunk runs become an ablation. [Ni et al. 2022](https://arxiv.org/abs/2110.05038); [Jozefowicz et al. 2015](https://proceedings.mlr.press/v37/jozefowicz15.html); [AMAGO](https://arxiv.org/abs/2310.09971)
- A combined config is cheaper than a factorial. A 2×2 factorial (trunk × bias) would add 20 more runs.

**P4. Extra seeds at the decisive cells. 20–30 runs.**
- Seeds 105–109 for the comparisons that carry the claim, e.g. LSTM-16 vs LSTM-256 and FS16 vs LSTM-64 at L* and L*+ (2–3 pairs × 2 L × 5).
- Rules out a 5-seed fluke given all-or-nothing outcomes. With 10 seeds, a 10/10 vs 0/10 split has non-overlapping exact CIs ([0.69, 1] vs [0, 0.31]); with 5 seeds, 5/5 vs 0/5 gives [0.48, 1] vs [0, 0.52]. [Agarwal et al.](https://arxiv.org/abs/2108.13264)
- Keep environment and agent seeds paired across architectures. [Patterson et al. 2024](https://arxiv.org/abs/2304.01315)

**Second tier (run only if budget remains)**

| Test | What it rules out | Runs |
|---|---|---|
| K>2 rooms (4 cue values) for FS16, LSTM-64, GRU at 2 L | "only one bit" critique; raises chance from 50% to 25%, sharpening the memory signal ([Ni 2023 reviews](https://openreview.net/forum?id=APGXBNkt6h)) | 30 (needs env work) |
| Distractor phase (random irrelevant colour patches between cue and junction) for the same 3 agents at 2 L | memory robustness to noise ([Ni 2023 reviews](https://openreview.net/forum?id=APGXBNkt6h)) | 30 |
| Variable-L training (mix of L values), test on held-out L, for FS16, LSTM-64, GRU | horizon overfitting. Must be read alongside fixed-L results, since mixed horizons can hide LTM failures ([Cherepanov et al.](https://arxiv.org/abs/2412.06531)) | 15 |
| Hidden-state refresh or burn-in for LSTM-16 at L*+ | that the short-window failure is a stale-state artifact, not window length ([Pleines et al.](https://arxiv.org/abs/2309.17207); [R2D2](https://openreview.net/forum?id=r1lyTjAqYX)) | 5–10 |
| GTrXL, LRU or S5 at all 5 L | none required at L ≤ 256: an upper-reference row, defensible to omit given the mixed literature ([POPGym](https://arxiv.org/abs/2303.01859), [Memory Gym](https://arxiv.org/abs/2309.17207), [Ni 2023](https://arxiv.org/abs/2307.03864)) | 25 + implementation |

**Budget sketch**

| Plan | Runs | Waves of 16 | Wall time |
|---|---|---|---|
| P1 (25) + P2 (40) + P3 (20) + P4 (20) | 105 | ~7 | ~7–35 h at 1–5 h per run |
| Minimal: P1 at 3 L (15) + P1b (10) + P2 at L* only (20) + P3 (20) + P4 (20) | 85 | ~5–6 | — |

P0 (eval only) should precede everything, because its outcome decides L* and whether P1a is needed at all L.

**Reporting add-ons (no extra runs)**
- per-L "fraction of seeds solved" with exact binomial CIs next to the IQM
- probability of improvement for the key pairs
- the ξ vs K table (BPTT window, frame-stack size and correlation horizon per L)
- per-loss gradient norms on the shared trunk

### Gaps
- Run-time estimates assume the 1–5 h range given in the task. Oracle and MLP runs are likely at the short end, and LSTM-256 or large-L runs at the long end. I did not measure this.
- L* is not known yet: the pilot is all-or-nothing and the shaping fix is pending. Exact cell choices must wait for the post-fix E2 sweep.
- I found no source quantifying how much a forget-gate bias of 1 helps in recurrent PPO specifically, so P3's expected effect size is unknown.
