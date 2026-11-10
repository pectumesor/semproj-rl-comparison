# Diagnostics separating internal (recurrent) memory from embodied/external memory and observation leaks in cue-memory navigation (E2 twin rooms)

Context: PPO, 2D ray casts, cue in alcove, bent shaft, L-step corridor (60 units wide), T-junction to twin rooms. Pilot (L=32, LSTM-BPTT16, 5 seeds, all ~1.0 success): resetting h at the junction gives 1.00 / 1.00 / 0.54 / ~0.7 / ~0.7. A linear probe decodes the cue at ~0.99 in every distance bin. Working hypothesis: the choice is committed before the junction through pose or heading ("body as memory"). Under that hypothesis the junction reset comes too late, and the probe cannot tell memory from pose.

Labels used below: "Established" means a published procedure with a source. "Suggestion" means my own adaptation for E2, with no source.

## 1. Causal interventions on memory (resets, truncation, swaps, patching, unit ablation)

### Takeaway
Two causal memory tests have clean, published procedures:
- **Memory-budget truncation** (Wijmans et al. 2023): rebuild the hidden state from only the last k observations, starting from zero state, and sweep k.
- **Probe-guided hidden-state editing** (Taufeeque et al. 2024/25, Sokoban DRC): edit the hidden state along a probe direction and check that the actions follow.

I found no published RL paper that swaps whole hidden states between cue conditions. That test is still the natural counterfactual for E2, and it should be reported as a suggestion built on the standard activation-patching recipe.

### Cited Findings
- **Wijmans et al., "Emergence of Maps in the Memories of Blind Navigation Agents"** (ICLR 2023). Blind PointGoal agents (egomotion only) reach about 95% success. The abstract says they remember about 1,000 steps of past experience in an episode. — [arXiv abs](https://arxiv.org/abs/2301.13261)
  - **Memory-length procedure.** At each time t, the LSTM state is rebuilt from zero, (h,c)_{t−k} = (0,0), and run over only o_{t−k+1}..o_t. With k=1 (memoryless) success is near 0%. Performance does not saturate until about 1,000 steps (89.1 ± 0.66 m travelled). A separate training run at fixed budget k=256 (the largest they could train) did not match the unbounded agent. — [arXiv HTML](https://arxiv.org/html/2301.13261v1)
  - **Memory transplant and "probe agent".** A structurally identical agent is initialised with the trained agent's final memory and has to navigate back. Controls are AllZeroMemory and UntrainedAgentMemory, a random LSTM run along the same trajectory. The evaluation uses 1,008 episodes, with the ~5% of failed agent episodes dropped. Probe SPL is 85.0 ± 1.6 vs agent 62.9 ± 1.6. TrainedAgentMemory beats both controls by 13 SPL and 4 success points. Intervals are 95% CIs over 5 agent–probe pairs. — [arXiv HTML](https://arxiv.org/html/2301.13261v1)
  - **Statistics.** Map-decoder IoU is 32.5% (trained memory) vs 12.5% (untrained memory), tested with a Wilcoxon signed-rank test, p ≤ 1e-5. Path-difference metrics are reported with 99% bootstrap CIs. — [arXiv HTML](https://arxiv.org/html/2301.13261v1)
- **Taufeeque et al., "Planning in a recurrent neural network that plays Sokoban"** (arXiv 2407.15421; ICLR 2025 slides exist).
  - **Probes.** Linear probes read the per-square hidden state of a DRC(3,3) ConvLSTM, concatenated over 3 layers, and predict the agent's future movement direction on that square. — [arXiv](https://arxiv.org/abs/2407.15421v2)
  - **Intervention.** The hidden state is edited so the probe logit p·h is pushed to an adaptive value α. The agent's subsequent actions follow the edit, which makes the probe causal rather than only correlational. The authors say a simpler additive steering vector was too coarse on hard levels. — [arXiv HTML](https://arxiv.org/html/2407.15421v2)
- **Standard activation-patching recipe.** Run a clean input and a corrupted input, then copy a chosen hidden state from the clean run into the corrupted run. Patched states can be off-distribution, so a behaviour change from patching alone is not proof of mechanism. — summarised in search results citing [arXiv 2407.15421](https://arxiv.org/pdf/2407.15421) and related patching work; I did not verify the original patching-method source.
- **POPGym Arcade** (arXiv 2503.01450) tests "recurrent state contamination" in two ways:
  - Add noise to one frame, or shuffle the first 10 steps of a trajectory.
  - Then measure how advantages and selected actions change at later steps.
  - Perturbations still changed actions 50–100 steps later in CartPole with an LRU. The authors call this a proof of concept.
  - — [arXiv HTML](https://arxiv.org/html/2503.01450v6)
- **POPGym Arcade "memory saliency" / recall density.**
  - Take the gradient of π(a_n|ŝ_n), or of Q, with respect to each past observation o_t.
  - Reduce it to a per-step L1 norm z(t), normalise it to sum to 1, and re-index time as τ=t/n.
  - Average over trajectories and skip terminal states.
  - This shows which past observations the policy actually uses.
  - — [arXiv HTML](https://arxiv.org/html/2503.01450v6)

### Inferences
- **Suggestion: reset sweep instead of a single junction reset.** Zero (h,c) at several points: alcove exit, shaft exit, corridor 25/50/75/100%, junction, room entrance. Plot success against reset position for each seed. If memory is internal and the choice is made at the junction, success should drop to chance for any reset before the junction. If success recovers for late resets, the decision was already externalised into pose, or committed earlier. This is the same idea as Wijmans' k-truncation, indexed by position instead of lag.
- **Suggestion: Wijmans-style k-truncation.** Rebuild h_t from zero over the last k observations, with k ∈ {1, 4, 16, 32, 64, L+shaft}. Measure success when the policy acts with this truncated state, either from the junction onward or throughout. This gives an "effective memory horizon" curve per backbone and BPTT window, and it maps directly onto Cherepanov's K vs ξ framing (Q4).
- **Suggestion: hidden-state swap (counterfactual cue).** This is the decisive test against pose.
  - Run paired episodes from the same seed and spawn, one with cue A and one with cue B.
  - At step t (corridor midpoint, or just before the junction), copy (h,c) from the B run into the A run. Leave the agent's physical pose as it is in A.
  - The fraction of episodes whose room choice follows the patched cue measures causal memory use from t onward. Choices that follow the physical cue mean pose or observations carry it.
  - Do the converse as well: keep h and teleport the agent to the pose it had in the B run at step t (a "pose swap"). The 2×2 design (h from A/B × pose from A/B) separates the two channels.
- **Suggestion: unit ablation.** Fit an L1-penalised logistic probe for the cue, as Wijmans did for collision neurons (top 10 of 3,072 units). Zero, or mean-ablate across cue conditions, the top-k cue units, and measure the drop in success relative to ablating random k units.

### Gaps
- I found no published RL navigation paper that does whole-hidden-state swaps between cue conditions and reports the "follows-patched-cue" rate. Hidden-state swapping is standard in mechanistic-interpretability work on LLMs, but I could not source it in RL.
- I did not verify these from primary sources in this pass:
  - Banino et al. 2018 (grid cells; Nature): reportedly used dropout on the linear "grid" layer and ablations of grid-like units.
  - Paischer et al. 2022/2023 (HELM, SHELM).
  - Mini et al. 2023 (maze-solving policy, "cheese vector" activation addition).
- I did not find "memory probing" papers from 2025–2026 that combine probing with causal hidden-state interventions in navigation. POPGym Arcade (2025) and Taufeeque (2024/25) are the closest.

## 2. Pose / body-as-memory controls (perturbations, teleport, autopilot, pose decoders, stigmergy)

### Takeaway
The pose-perturbation test exists in the literature with an exact procedure and numbers: the "floor-memory" T-maze in Déletang et al. 2021 (DeepMind). The agent sees a cue, walks a wide corridor, and is pushed to the opposite wall near the end of it.
- An LSTM agent still chooses correctly (0.996).
- A memoryless CNN agent that off-loads the cue onto its lateral position fails (0.107 / 0.004).

This is almost exactly the E2 setup and should be adopted as the primary pose control. Related theory on agents off-loading memory onto the environment dates to Peshkin et al. 1999 and continues in "Artifacts as Memory" (2026).

### Cited Findings
- **Déletang et al. 2021, "Causal Analysis of Agent Behavior for AI Safety", Sec. 3.2 "Memory."**
  - **Problem framing.** "Does the agent use its internal memory … or does it off-load the memory onto the environment?" The paper names position-encoding as the example. It says the difference is "undetectable without intervening".
  - **Environment.** Floor-memory T-maze: the cue is a grass or sand tile at the corridor start, and the agent can see only one tile around it.
  - **Subjects.** Agent a is an LSTM. Agent b is a CNN with no memory.
  - **Procedure.** The agent observes the cue and acts freely. Near the end of the wide corridor it is pushed to the opposite wall, then continues freely.
  - **Causal model.** Variables F (cue), P (left/right half of the corridor at the intervention point) and T (terminal side). Whether to intervene is decided at random with p=0.5, and each case gets 1,000 Monte-Carlo rollouts.
  - **Observational results.**

    | Query | Agent a (LSTM) | Agent b (CNN) |
    |---|---|---|
    | P(T=l\|F=g) | 0.996 | 0.990 |
    | P(T=r\|F=s) | 0.996 | 0.977 |
    | P(P=l\|F=g) | 0.984 | 0.991 |
    | P(P=r\|F=s) | 0.996 | 0.985 |

  - **Interventional results.**

    | Query | Agent a (LSTM) | Agent b (CNN) |
    |---|---|---|
    | P(T=l\|do(P=r),F=g) | 0.996 | 0.107 |
    | P(T=r\|do(P=l),F=s) | 0.996 | 0.004 |

  - **Conclusion.** Agent b uses "an external memory mechanism that generalizes poorly". Both agents look the same until the intervention.
  - — [arXiv PDF 2103.03938](https://arxiv.org/pdf/2103.03938); [abs](https://arxiv.org/abs/2103.03938)
- **Déletang et al. general framework.** Agent memory m_t and world state w_t form a causal Bayesian network. The simulator is made deterministic by re-parameterising with a random seed ω, so a trace can be rewound to any time t and any component (seed, state, observation/action) can be edited before re-running. This is the formal basis for "same seed, intervene at t, compare" experiments. — [arXiv PDF](https://arxiv.org/pdf/2103.03938)
- **Peshkin, Meuleau & Kaelbling, "Learning Policies with External Memory"** (ICML 1999). Stigmergic setting: actions set or clear bits of an external memory, and the bits are fed back as input. VAPS(1) learns near-optimal reactive policies where sarsa(1) may fail. — [arXiv cs/0103003](https://arxiv.org/abs/cs/0103003); [ICML 1999 record](https://mlanthology.org/icml/1999/peshkin1999icml-learning)
- **Pajak, Saleh, Mince & Martin, "Artifacts as Memory Beyond the Agent Boundary"** (arXiv 2604.08756, Apr 2026).
  - They formalise how observations of environmental "artifacts" can reduce the history information an agent must store.
  - Experiments are spatial navigation tasks with linear Q-learning and DQN. Observing paths or landmarks gives higher performance with less internal capacity.
  - They report that this externalisation "emerges unintentionally through the agent's sensory stream".
  - — [arXiv 2604.08756](https://arxiv.org/pdf/2604.08756); summary at [arxivlens](https://arxivlens.com/PaperView/Details/artifacts-as-memory-beyond-the-agent-boundary-4142-7d2d2043)
- **Animal-behaviour precedent ("mediating behaviour").**
  - Rats can solve delayed alternation by positional mediation (Mogensen, Iversen & Divac 1987, as summarised).
  - In delayed non-matching-to-position, apparent memory deficits can reflect disrupted motor/positional strategies.
  - Futter & Aggleton 2006: egocentric/idiothetic memory in rats can fade with delays as short as 10 s.
  - — [Cardiff ORCA, Aggleton 1995](https://orca.cardiff.ac.uk/34952/1/Aggleton%201995.pdf); [Springer DNMP study](https://link.springer.com/doi/10.1007/s002130050427); [Futter & Aggleton 2006](https://orca.cardiff.ac.uk/id/eprint/11367/)
  - Only search-result summaries were read. Verify before citing.

### Inferences
- **Suggestion: port the Déletang push test to E2.** For the 60-unit corridor, do the following.
  - At corridor fraction f ∈ {0.25, 0.5, 0.75, 0.95}, teleport the agent to the mirrored lateral offset (y → −y about the corridor centreline). Optionally mirror the heading too (θ → −θ about the corridor axis). Keep (h,c) intact.
  - Decide intervene/no-intervene at random with p=0.5. Use about 1,000 rollouts per condition per seed, as Déletang did.
  - Report P(correct room | cue) and P(correct room | cue, do(mirrored pose)).
  - Interpretation: if the LSTM uses internal memory, success under do() should stay near the observational value. If the cue lives in pose, success should flip below chance, as for the CNN (0.1 / 0.004).
  - The MLP or frame-stack agent is the built-in "agent b" control. If the MLP solves L=32 at all, it must be using pose, so the push test should break it.
- **Suggestion: pose-only decoder.** Fit a logistic classifier from pose features alone (lateral offset y, heading θ, and possibly the last-k ray casts) at each corridor step to the cue. Plot accuracy against steps since the cue, alongside the hidden-state probe.
  - If pose decodes the cue near 1.0 throughout the corridor, the hidden-state probe at 0.99 is uninformative.
  - The quantity to report is hidden-state accuracy over and above a pose plus current-observation baseline (see Q3).
- **Suggestion: autopilot / forced centreline.** For the corridor segment, replace the policy with a scripted controller that drives along the centreline at the policy's mean speed, while still feeding the observations through the RNN so h keeps updating. Hand control back at the junction.
  - If success stays high, the cue survived in h. If it drops to chance, it lived in pose.
  - Compare with the "autopilot + h reset at handover" condition.
- **Suggestion: random cue-irrelevant perturbations.** Apply small, symmetric random lateral or heading kicks, with no mirroring, as a control for "any perturbation breaks the agent". The cue-relevant mirror push should hurt far more than an equal-magnitude random push.
- **Suggestion: randomise the start pose and the shaft exit.** The aim is to break any deterministic cue → trajectory mapping. Note that this changes training, so treat it as a fix rather than a diagnostic.
  - Option: randomise the entry lateral offset and heading at the corridor start.
  - Option: add a "lateral reset" corridor segment, such as a narrow neck or chicane that forces the agent to the centreline. This is the environment-level analogue of a pose lesion.
- **Explanation of the pilot pattern.** Mixed seeds (reset → 1.00 / 0.54 / 0.7) fit a mixture of strategies across seeds: some seeds commit via pose, one uses internal memory read at the junction, and some do both. Déletang's point is that observational success cannot distinguish these, so each seed should be classified separately by the push and swap tests.

### Gaps
- I found no published deep-RL paper that measures the agent's lateral offset as a decoder of the cue in a continuous-control T-maze. Déletang works in a discrete gridworld with left/right halves.
- I did not find literature on agents encoding information in heading specifically. The rat "mediating behaviour" literature is the closest. Only summaries were read.

## 3. Probing best practice (control tasks, selectivity, inputs vs hidden state, cell vs hidden, probe type, splits)

### Takeaway
High probe accuracy alone says little. Report three things:
- **Selectivity**: accuracy on the real task minus accuracy on a control task where labels are assigned at random per input type (Hewitt & Liang 2019).
- **Baselines**: probes trained on untrained-network states, on states from other trajectories, and on the raw inputs or pose.
- **Held-out data**: test episodes or scenes never seen by the probe.

Wijmans et al. 2023 give a concrete RL template (frozen h and c, logistic probe, untrained-agent and other-trajectory chance baselines, held-out scenes, Wilcoxon test).

### Cited Findings
- **Hewitt & Liang 2019, "Designing and Interpreting Probes with Control Tasks."** Control tasks assign random outputs to each word type, so they can be learned only by the probe memorising. A good probe is "selective": high on the linguistic task, low on the control task. Dropout did not improve MLP-probe selectivity, while other regularisation did. — [arXiv 1909.03368](https://arxiv.org/abs/1909.03368)
- **Wijmans et al. 2023, probe details.**
  - **Collision probe.** Logistic regression on the frozen (h_{t+1}, c_{t+1}), i.e. both the cell and hidden states, with per-dimension normalisation. Data: 1M of 10M samples for training. Adam, lr 5e-4, 20 epochs, batch 256. Result: 98% held-out accuracy vs 50% chance. An L1-penalised variant (penalty sweep 0–1000) keeps 96% accuracy with 10 of 3,072 units.
  - **Position decoder.** 3-layer MLP on (h_t, c_t) predicting displacement s_{t+k} − s_t for k ∈ [1,256], evaluated on held-out scenes with relative L2 error. The chance baseline is a decoder trained on states from *different trajectories*. At 100 steps back: relative error 0.55 / 1.0 m vs chance 1.0 / 3.2 m.
  - **Map decoder.** Trained-agent memory vs untrained-agent memory (random LSTM run along the same trajectory), evaluated on scenes unseen by both agent and decoder.
  - — [arXiv HTML](https://arxiv.org/html/2301.13261v1)
- **POPGym Arcade warning.** Comparing returns only indirectly measures the information in the latent Markov state, and few works examine how the memory model builds that state. Its recall-density saliency is a complementary "what is used" measure, while a probe answers "what is decodable". — [arXiv HTML 2503.01450](https://arxiv.org/html/2503.01450v6)

### Inferences
- **Suggestion: E2 probe protocol (adapted from the cited practice).**
  1. **Inputs.** Probe h, c, and [h;c] separately. The LSTM cell state is the more natural long-term store. h is gated output and mixes in current observations.
  2. **Baselines in the same bins.**
     - (a) Current observation o_t, or the last-k frame stack.
     - (b) Pose (y, θ, speed).
     - (c) [o_t; pose].
     - (d) Untrained-network h along the same trajectories.
     - (e) Shuffled-label control in Hewitt & Liang's spirit: labels assigned at random per episode, so the probe can succeed only by memorising episodes.
  3. **Headline number.** Report probe accuracy on h minus the best of baselines (a)–(c), per distance bin. Equivalently, compare a probe on [pose; o_t] against one on [pose; o_t; h], measuring the conditional information h adds beyond pose and observations.
  4. **Splits.** Split by episode, and preferably by seed of the env RNG, never by timestep, because adjacent timesteps are near-duplicates. Use grouped k-fold cross-validation (e.g. 5-fold) with the episode as the group.
  5. **Probe type.** L2-regularised logistic regression with regularisation chosen by nested cross-validation. Ridge regression is for continuous targets such as y or θ. Also report an L1 sparse probe to locate cue units for the ablation in Q1.
  6. **Condition on behaviour.** Fit the probe within strata of lateral offset. If cue accuracy collapses once y is held fixed (e.g. only episodes where the agent is within ±5 units of the centreline), the probe was reading pose.
  7. **Make the probe causal.** Use the probe direction for a Taufeeque-style edit: push the cue logit to the opposite sign at step t and check that the room choice flips (Q1).
- **Why ~0.99 everywhere is suspicious.** Once the agent's trajectory depends on the cue (it drifts toward one wall), every later observation is cue-dependent. The cue then becomes decodable from o_t alone, so the probe "measures" pose and observation leakage. The current-observation and pose baselines will show whether that is happening.

### Gaps
- I did not fetch these primary sources:
  - Belinkov 2021 ("Probing Classifiers: Promises, Shortcomings, and Advances").
  - Pimentel et al. 2020 (information-theoretic probing).
  - Elazar et al. 2021 (amnesic probing, i.e. causal removal of a probed property via INLP).
- These are the standard references for "probe accuracy ≠ use" and for the causal probe-removal procedure. Verify them before citing.
- I found no RL-specific paper that directly compares probes on the LSTM cell vs hidden state. Wijmans concatenates (h,c).

## 4. Measuring the effective memory horizon (definitions, benchmarks, seeds, statistics)

### Takeaway
Use Cherepanov et al.'s ξ vs K framing. Test long-term memory only with fixed-horizon tasks where the correlation horizon ξ exceeds the agent's context K. Mixed-horizon tasks hide long-term-memory failures.

Measure an agent's effective horizon by truncating or resetting its memory at varying lags (Wijmans), or by sweeping the task's corridor length (Ni et al.: T-maze up to 1,500 steps).

For reporting, use 95% CIs over about 5–10 seeds (Ni: 10 seeds; POPGym Arcade: 5; Wijmans: 5 agent–probe pairs). Use the rliable tooling (IQM with stratified-bootstrap CIs) for small seed counts.

### Cited Findings
- **Cherepanov et al., "Unraveling the Complexity of Memory in RL Agents"** (arXiv 2412.06531; listed as ICLR 2026).
  - **Definitions.**
    - Context length K: how many (o,a,r) steps the agent processes. An MLP has K=1. An RNN also has K=1 by this definition, but carries longer dependencies through its hidden state.
    - Memory mechanism μ(K)=K_eff ≥ K.
    - Correlation horizon ξ = t_r − t_e − Δt + 1.
    - Short-term memory (STM) when ξ ≤ K, long-term memory (LTM) when ξ > K.
    - Declarative vs procedural memory depends on whether n_envs × n_eps = 1 or > 1.
    - A memory-intensive environment has min ξ > 1.
  - **Algorithm 1.**
    1. Count event–recall pairs.
    2. Compute K̄ = min Ξ − 1.
    3. To test STM, set K > K̄. To test LTM, set K ≤ K̄ ≤ K_eff.
  - **Regimes.** K ∈ [1, K̄] validates LTM only. K ≥ max Ξ validates STM only.
  - **Choice of K.** "Choosing smaller K better reveals memory mechanism effects."
  - **Fixed vs mixed horizons.** Mixed-horizon tasks hide LTM deficits. Minigrid-Memory looked solved with variable horizons but failed with fixed L=21 (ξ=22) and K=14.
  - **Generalisation test.** Train on some lengths and evaluate on longer, unseen ones.
  - — [arXiv HTML](https://arxiv.org/html/2412.06531); [PDF](https://arxiv.org/pdf/2412.06531)
- **Ni et al. 2023, "When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment"** (NeurIPS 2023).
  - **Definitions.** Reward, transition, policy and value memory lengths. For example, policy memory length l_mem(π) is the smallest n such that the action is conditionally independent of earlier history given the last n steps. Theorem 1: l_mem(π*) ≤ l_value(π*) ≤ max(m_reward, m_transit).
  - **Passive T-Maze.** Memory length T, credit-assignment length 1. Swept from 50 to 1,500. Transformers succeed up to 1,500, while LSTMs falter around 250.
  - **Passive Visual Match.** Swept from 60 to 1,000.
  - **Credit assignment** is tested separately with Active T-Maze and Key-to-Door.
  - **Statistics.** Mean ± 95% CI over 10 seeds.
  - — [arXiv HTML 2307.03864](https://arxiv.org/html/2307.03864)
- **MIKASA** (Cherepanov et al., arXiv 2502.10550; ICLR 2026). Three parts: a classification framework, MIKASA-Base (Gymnasium memory tasks) and MIKASA-Robo (32 manipulation tasks in 12 groups). Evaluated with online RL, offline-RL datasets and VLAs. Seed protocol not retrieved. — [arXiv v3](https://arxiv.org/pdf/2502.10550v3); [arXiv v2 HTML](https://arxiv.org/html/2502.10550v2)
- **POPGym Arcade** (arXiv 2503.01450).
  - Paired MDP/POMDP twins that share observation space, action space and dynamics.
  - Observability Gap = J(MDP) − J(POMDP).
  - Memory Bias = J(memory policy, MDP) − J(memory-free policy, MDP).
  - 5 seeds, mean ± 95% CI, bootstrapped CIs for throughput.
  - — [arXiv HTML](https://arxiv.org/html/2503.01450v6)
- **Agarwal et al. 2021, "Deep RL at the Edge of the Statistical Precipice"** (NeurIPS 2021). With few runs, report interval estimates using stratified-bootstrap CIs. Prefer the interquartile mean (IQM), which is more robust than the mean and more efficient than the median. Implemented in the `rliable` library. — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf); [arXiv 2108.13264](https://arxiv.org/pdf/2108.13264); [Google Research blog](https://research.google/blog/rliable-towards-reliable-evaluation-reporting-in-reinforcement-learning/)

### Inferences
- **Mapping E2 onto Cherepanov.**
  - **Event and recall.** The event is the cue observation in the alcove (Δt = number of steps the cue is visible). The recall is the junction decision.
  - **Correlation horizon.** ξ ≈ shaft steps + L. The pose hypothesis makes this an *upper bound*: if the agent commits its heading in the shaft, the effective ξ the RNN must bridge is only a few steps.
  - **Context length.** Truncated BPTT window 16/64/256 is a training-time credit horizon, not K in Cherepanov's sense, because all RNNs have K=1 with K_eff = μ(K). The k-truncation test (Q1) estimates K_eff empirically for each backbone.
  - **Long-term memory claims.** Requires ξ > BPTT window (e.g. L ≥ 64 for BPTT16) on fixed-L runs, not mixed-L. The twin_rooms_L8…L256 room files already allow a fixed-L sweep.
- **Suggestion: report three horizon numbers per backbone.**
  - (i) Largest fixed L solved (Ni-style sweep).
  - (ii) Reset or truncation lag k at which success falls to chance (Wijmans-style).
  - (iii) Earliest corridor position at which a mirrored-pose push no longer hurts (Déletang-style), i.e. when the decision is committed.
- **Suggestion: statistics.**
  - Pilot: 5 seeds, matching POPGym Arcade and Wijmans' 5 pairs.
  - Headline claims: about 10 seeds, matching Ni et al.
  - Report per-seed points plus IQM with stratified-bootstrap 95% CIs (rliable).
  - Report diagnostics *per seed*, because the pilot shows a strategy mixture across seeds. Pooled means will hide it.
  - For intervention tests, use about 1,000 rollouts per condition per seed (Déletang), or a binomial CI on the per-seed success rate.

### Gaps
- I did not retrieve the seed counts or statistics used by Cherepanov 2024 or MIKASA. The appendices were truncated.
- I did not fetch the original Memory Gym (Pleines et al.) or POPGym (Morad et al. 2023) papers this pass, so I cannot give their exact protocols for horizon or seeds.
- No source I found defines an "effective memory horizon" for an RNN as a single agreed statistic. Wijmans' k-truncation curve is the most direct published operationalisation.
