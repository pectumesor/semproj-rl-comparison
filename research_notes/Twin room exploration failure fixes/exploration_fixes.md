# Twin-room PPO exploration failure: causes and non-leaking fixes

Context recap (from the assignment, not a source): 2D continuous-control PPO, 256 envs x 256 steps x 1000 iterations, gamma 0.999, Gaussian policy, egocentric 180-degree ray casts. Cue alcove -> corridor (L=32) -> T-junction -> two identical 200x200 rooms, baffle hides goal (r=16, ~130 units behind door). Wrong room terminates. Reward -1/max_steps per step, +1 on success. Episode length 320. Seeds are bimodal: success ~1.0 or exactly 0.00 with every eval episode timing out (no early wrong-room termination either). MLP 0/5, frame stack 1/5, GRU 1/5, LSTM 2-5/5.

## Why do some seeds never find any reward?

### Takeaway
The failing seeds are an exploration failure, not a memory failure: they time out without ever entering either room, so they never reach the junction decision at all. The likely causes are well documented in the literature. A sparse binary reward gives no learning signal until the first success. The Gaussian std shrinks under PPO, so exploration collapses. The value target is noisy because time-to-timeout is unobservable. Seed-to-seed variance in deep RL is large enough that 5 seeds can split into "learns" and "never learns".

### Cited Findings
- **Seed variance is a known, large effect.** Henderson et al. show that averaging runs over different random seeds can give misleading results. A secondary source describes their experiment: 10 runs of the same algorithm and hyperparameters, split into two groups of 5 seeds, produced learning curves that look like they come from different distributions (notably HalfCheetah). — [Henderson et al. 2018, arXiv 1709.06560](https://arxiv.org/abs/1709.06560); description of the 2x5-seed split from [Colas et al. 2018, "How Many Random Seeds?"](https://ar5iv.labs.arxiv.org/html/1806.08295)
- **Code-level details drive much of PPO's behaviour.** "Implementation Matters" attributes much of PPO's reported advantage over TRPO to code-level optimisations (value clipping, reward scaling, orthogonal init, learning-rate annealing, and so on), not to the clipped objective itself. — [Engstrom et al. 2020, arXiv 2005.12729](https://arxiv.org/abs/2005.12729)
- **Early goal-hit rate predicts success.** In minimum-time sparse-reward formulations (constant -1 per step, terminate at goal), the initial policy's goal-hit rate is a robust early indicator of whether learning succeeds. This matches the bimodal outcome: seeds that hit the goal early learn, the others do not. — [Vasan et al. 2024, "Revisiting Sparse Rewards for Goal-Reaching RL", RLJ](https://rlj.cs.umass.edu/2024/papers/Paper231.html); [arXiv html](https://arxiv.org/html/2407.00324v2)
- **Sparse reward with no signal for random behaviour is hard for typical agents.** A goal reward plus a small negative per-step reward and no other guidance is described as difficult "as there is no reward signal for the initial random behaviour". — [Aquatic Navigation benchmark, arXiv 2405.20534](https://arxiv.org/pdf/2405.20534)
- **Gaussian PPO tends to lose entropy.** The RPO authors note that the policy "often wants to reduce the variance in selected action as training progress, which might collapse entropy and lead to sub-optimal exploration". Their fix perturbs the Gaussian mean with uniform noise before sampling, which keeps entropy from collapsing. — [Rahman & Xue 2022, Robust Policy Optimization, arXiv 2212.07536](https://ar5iv.labs.arxiv.org/html/2212.07536)
- **An entropy bonus alone often does not stop the collapse.** A student project reports entropy collapsing to near 0 "inspite of entropy bonus". This is weak evidence (course project). — [Stanford CS224R project report](https://cs224r.stanford.edu/projects/pdfs/CS224R_Final_report%20(4)12.pdf)
- **Initial std and init scale matter.** Andrychowicz et al. (MuJoCo) found an initial action std of 0.5 best, recommend a softplus std with a negative offset, and recommend initialising the last policy layer with 100x smaller weights. They found no regulariser (including entropy) helped significantly. — [Andrychowicz et al. 2021, "What Matters in On-Policy RL", arXiv 2006.05990](https://ar5iv.labs.arxiv.org/html/2006.05990)
- **CleanRL continuous PPO defaults.** The state-independent log std is initialised to 0 (std = 1). The policy output layer is initialised with scale 0.01. ent_coef = 0.0 for MuJoCo. Observations and rewards are normalised and clipped. The post says "we did not implement the correct handling for truncated environments". — [Huang et al. 2022, "The 37 Implementation Details of PPO", ICLR blog track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- **Timeouts treated as terminal corrupt the value target.** If a time limit is only a training device, treating it as a true termination causes state aliasing and instability. The fix is partial-episode bootstrapping, which in PPO uses the value of the last state when no real termination occurred. If the time limit is part of the task, the agent should observe the remaining time. — [Pardo et al. 2018, "Time Limits in RL", ICML](https://proceedings.mlr.press/v80/pardo18a.html); [ar5iv](https://ar5iv.labs.arxiv.org/html/1712.00378)
- **A step penalty that is too large teaches the agent to give up.** One informal tutorial recommends keeping the step penalty to about 1-5% of the goal reward. This is a low-quality source and a heuristic only. — [rl-cycle-demo docs](https://www.mintlify.com/JhonZacipa/rl-cycle-demo/solutions/other-techniques)

### Inferences
- **The failing seeds never reach the junction.** Under -1/max_steps per step, a timeout returns -1. Entering the wrong room returns roughly -(t/320) > -1, so the reward actually favours ending the episode early by entering any room. Failing seeds do not even do that, which means they never reach either door. The bottleneck is getting through the L=32 corridor and the junction, not the hidden goal and not memory.
- **The MLP result points to exploration confounding the comparison.** A memoryless policy can reach 0.5 success by guessing a room: Ni et al. note that the optimal Markovian policy in a T-maze gets 0.5. An MLP stuck at exactly 0.00 has never explored to the goal. Today's success rate therefore mixes "can explore" with "can remember". A clean memory benchmark should let the MLP reach ~0.5.
- **The time penalty gives no direction.** The only reward seen before the first success is a constant -1/320 per step. After per-minibatch advantage normalisation (CleanRL default), the advantages are mostly value-estimation noise. That gives the mean and log-std gradients no consistent direction, and PPO's tendency to shrink std (RPO) can lock in a wall-hugging or spinning behaviour before any success.
- **Value noise from the timeout.** With gamma = 0.999 (effective horizon ~1000 > 320), V(s) in a timeout-terminated episode depends on steps remaining, which is unobservable. That adds value noise unless the code bootstraps at truncation or adds time to the observation. Check how `algorithms/ppo.py` handles `truncated` vs `terminated`.
- **Why recurrent agents may succeed more often.** Recurrent agents use lr 3e-4 vs 1e-4 for the MLP. Recurrent state also adds temporally correlated behaviour, which can explore further down a corridor. Part of the LSTM advantage may be exploration, not memory. This is speculation, but it is testable by matching learning rates.

### Gaps
- No source found that measures, for Gaussian PPO, how fast log-std shrinks under a reward that is only a constant penalty. The mechanism above is inferred.
- The exact 2x5-seed passage of Henderson 2018 was not retrieved from the primary text; only a secondary description of it.
- No primary source found specifically on a per-step time penalty causing degenerate behaviour before the first success in navigation.

## Which fixes keep the cue as the only source of the room choice (no goal-room leakage)?

### Takeaway
The literature's pure-memory benchmarks deliberately remove exploration as a bottleneck with cue-independent dense shaping: Ni et al.'s Passive T-Maze penalises falling behind a "move toward the junction" schedule. That is the cleanest precedent. The safe fixes are:
- reward terms that are functions only of position along the shared corridor
- a bigger or unbaffled goal, or success on room entry
- curricula over L or over start position that never condition on the cue
- std floors and initialisation fixes
- symmetric novelty bonuses, used with caution

Anything that differs between the left and right rooms as a function of the cue (distance-to-goal, room-specific potentials) leaks the answer.

### Cited Findings
- **Precedent: memory benchmarks shape toward the junction.** In Ni et al.'s Passive T-Maze, the per-step reward is 1(x_{t+1} >= t) - 1, divided by T-1. The agent pays a penalty only when it falls behind the "move right every step" schedule toward the junction J. The terminal reward is 1 for reaching the correct goal. The authors state the tasks "require little exploration due to the dense rewards before reaching the goal". They also note that the optimal Markovian policy can only guess, with expected return 0.5. The shaping is identical for both goal positions, so it does not leak. The slides summarise this as a small reward for advancing toward J. — [Ni et al. 2023, "When Do Transformers Shine in RL?", NeurIPS, arXiv 2307.03864](https://arxiv.org/abs/2307.03864); [code](https://github.com/twni2016/Memory-RL)
- **Ni et al. also isolate exploration from the algorithm.** They used DDQN with epsilon-greedy for T-Mazes "to better control the exploration strategy". They report mean and 95% CI over 10 seeds. They note that at memory length 1000, LSTM agents hit NaN gradients in about 30% of runs, which were not shown in the plots. — [Ni et al. 2023](https://arxiv.org/abs/2307.03864)
- **Potential-based shaping (Ng et al.).** Shaping F = gamma*Phi(s') - Phi(s) is necessary and sufficient to preserve the optimal policy. The paper gives distance- and subgoal-based potentials. So Phi = -(geodesic distance to the junction / door line) preserves the optimal policy and, being cue-independent, adds no room information. — [Ng, Harada & Russell 1999, ICML](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html); [mlanthology entry](https://mlanthology.org/icml/1999/ng1999icml-policy/)
- **Reverse curriculum (Florensa et al.).**
  - Assumptions: the env can reset the agent to arbitrary start states, and one goal state is known.
  - Method: start states come from short random-action ("Brownian") rollouts (T_B = 50) seeded from previous "good starts". Good starts are those where the policy's success rate is between R_min = 0.1 and R_max = 0.9.
  - Tasks: point-mass maze, ant maze, ring-on-peg, key insertion.
  - Results: it beats uniform start sampling, which on the maze had high variance with some policies succeeding only from one side.
  - [Florensa et al. 2017, "Reverse Curriculum Generation", arXiv 1707.05300](https://ar5iv.labs.arxiv.org/html/1707.05300)
- **Automatic curriculum in a memory benchmark.** Memory Gym turns its finite memory tasks into endless ones that serve as an automatic curriculum. It benchmarks PPO with GRU and Transformer-XL, and GRU beats TrXL by large margins in the endless versions. — [Pleines et al., Memory Gym, JMLR 2025, arXiv 2309.17207](https://arxiv.org/html/2309.17207v6)
- **Observation normalisation.** Andrychowicz et al.: "Always use observation normalization". CleanRL/SB3 VecNormalize clips normalised observations to about [-10, 10]. — [Andrychowicz et al. 2021](https://ar5iv.labs.arxiv.org/html/2006.05990); [Huang et al. 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- **Initial std.** Initial std around 0.5 (MuJoCo, actions in [-1, 1]) worked best. Use softplus with an offset, and a last policy layer with 100x smaller weights so initial actions are near zero-mean with controlled std. — [Andrychowicz et al. 2021](https://ar5iv.labs.arxiv.org/html/2006.05990)
- **Entropy regularisation is weak evidence.** Andrychowicz et al. found no investigated regulariser helped significantly in MuJoCo. RPO's mean perturbation is an alternative that maintains entropy in Gaussian PPO. — [Andrychowicz et al. 2021](https://ar5iv.labs.arxiv.org/html/2006.05990); [Rahman & Xue 2022](https://ar5iv.labs.arxiv.org/html/2212.07536)
- **More parallel environments.** RND found that mean return improves with more parallel envs on Montezuma's Revenge, leveling off sooner for CNN than for RNN policies. Andrychowicz et al. found that many envs can hurt sample complexity, likely because experience chunks get shorter. — [Burda et al. 2018, RND](https://ar5iv.labs.arxiv.org/html/1810.12894); [Andrychowicz et al. 2021](https://ar5iv.labs.arxiv.org/html/2006.05990)
- **Truncation handling.** Bootstrap at timeouts (partial-episode bootstrapping) rather than treating them as terminal, or give the agent remaining time as input. — [Pardo et al. 2018](https://proceedings.mlr.press/v80/pardo18a.html)
- **Count-based episodic terms are what make exploration bonuses work.** E3B found that RIDE, AGAC and NovelD depend on a count-based episodic term and fail when it is removed. E3B generalises this to continuous spaces via an elliptical bonus over inverse-dynamics embeddings, computed per episode. It was state of the art on 16 MiniHack tasks and matched baselines on sparse-reward VizDoom. The Habitat experiments used DD-PPO, so it is compatible with on-policy agents. — [Henaff et al. 2022, E3B, arXiv 2210.05805](https://arxiv.org/abs/2210.05805)
- **RND practical details.**
  - Intrinsic reward: prediction error against a fixed random target network.
  - Normalisation: observations are whitened and clipped to [-5, 5], and intrinsic rewards are scaled by a running std of intrinsic returns.
  - Separate value heads for intrinsic (non-episodic, gamma_I = 0.99) and extrinsic (gamma_E = 0.999) returns. Two heads gave no benefit in the episodic setting.
  - [Burda et al. 2018](https://ar5iv.labs.arxiv.org/html/1810.12894)
- **NGU** multiplies an episodic (within-episode, kNN-based) novelty by a life-long novelty modulator from RND, capped at L = 5. The design encourages revisiting states across episodes but not within the same episode. — [Badia et al. 2020, "Never Give Up", arXiv 2002.06038](https://arxiv.org/abs/2002.06038)

### Inferences (leakage audit of each candidate fix)

| Fix | Leaks goal room? | Notes |
|---|---|---|
| PBRS with Phi = -distance to junction / door line (corridor progress only), clipped at the junction | No | Cue-independent. Directly analogous to Ni et al.'s T-maze. Policy-invariant (Ng 1999). Strongest literature-backed fix. |
| Ni-style "behind-schedule" penalty in place of the flat -1/max_steps | No | Same as above, in a non-potential form. Not policy-invariant in general, but symmetric. |
| PBRS with Phi = -distance to the *correct* goal | **Yes** | Reveals the room at the junction. Memory becomes unnecessary. |
| PBRS with Phi = -min(distance to either goal) or distance to the nearest door | No (symmetric) | The rooms are mirror images, so it does not distinguish them. It does pull the agent into the wrong room, but entry there terminates, so the shaping only accelerates reaching a door. |
| Success on entering the correct room (or door-line crossing) | No | The reward still depends on the cue only through the correct room, as the +1 already does. Removes the hidden-goal search inside the room. Shortens the horizon from about L + 130 + baffle detour to about L + junction. |
| Bigger goal radius, or remove the baffle | No | Raises the chance a random policy succeeds. With the baffle removed, the ray casts may see the goal from the door. Then room entry, not the cue, guides the final approach, which is fine because the room choice is already made at the door. |
| Curriculum over L (8 -> 32 -> ...), using the existing `rooms/twin_rooms_L*.json` | No | Cue-independent. Matches Memory Gym's automatic-curriculum idea. Report the final-L performance. Caveat: the cue-memory span grows with L, so the curriculum also trains memory gradually. Apply the same schedule to all backbones. |
| Reverse curriculum / start near goal (Florensa) | Partly | Starting *inside* a room skips the cue, so for those episodes the cue gives no information. The agent learns "in-room goal finding" without memory, which is fine. Starts must include the cue alcove for the memory evaluation. Starting in the corridor after the cue skips the cue observation entirely, so the correct room is unobservable. Never start past the cue without having shown it. |
| Std floor (min log-std), RPO-style mean perturbation, larger init std, entropy bonus | No | Pure exploration knobs, cue-independent. |
| Obs normalisation of ray casts | No | Recommended default. |
| Truncation bootstrapping / time-in-obs | No | Fixes value noise from the 320-step timeout. |
| More envs / longer training | No | Expensive. Only makes it more likely a seed gets lucky; does not remove the bimodality. |
| Count/RND/E3B/NGU on egocentric observations or position | Usually no, with caveats | The two rooms are geometrically identical, so the expected bonus is symmetric. But a *life-long* bonus (RND, counts across episodes) makes the wrong room relatively more novel once the agent learns the right one. That rewards wrong-room entry and fights the memory policy. An *episodic* bonus (E3B, NGU episodic term) avoids that. |
| Bonus computed on (position, cue) or on observations that include the cue colour | Could leak indirectly | Not a direct leak, but it complicates the memory story. Prefer bonuses on corridor/position features shared by both cue conditions. |

### Gaps
- No paper found that applies count/RND/E3B to a T-maze memory task and audits the bonus for room symmetry. The symmetry argument above is my inference.
- No direct evidence found comparing Gaussian std floors vs. entropy bonuses in sparse-reward continuous navigation with PPO.
- Florensa's numbers were reported for TRPO-era policy-gradient agents. No PPO-specific replication was checked.

## Is there evidence that intrinsic rewards interact badly with memory tasks?

### Takeaway
Yes, indirectly. The documented failure modes are:
- Novelty bonuses become unreliable under partial observability, where aliased observations look alike.
- Prediction-error bonuses are attracted to stochastic observations (noisy TV).
- Life-long bonuses keep pulling the agent toward rare (for example, dangerous) states after it has learned the task.

In the twin-room task these predict that a life-long bonus would reward wrong-room entry and would compete with the cue-following policy.

### Cited Findings
- **Noisy TV.** Prediction-error curiosity is attracted to stochastic observations ("A TV showing white noise would be such an attractor"). RND avoids this with a deterministic target network. — [Burda et al. 2018](https://ar5iv.labs.arxiv.org/html/1810.12894)
- **The bonus keeps acting after the task is learned.** In RND's Montezuma runs, once the agent reliably collects the rewards it knows how to get, it "settles into a pattern of behavior where it keeps interacting with potentially dangerous objects". The authors attribute this to dangerous states staying rare, and therefore novel. — [Burda et al. 2018](https://ar5iv.labs.arxiv.org/html/1810.12894)
- **RNN vs CNN under RND.** RND's recurrent policies did worse than non-recurrent ones at gamma_E = 0.99 and better at 0.999. CNN runs were more stable than RNN runs, with many outliers across seeds. — [Burda et al. 2018](https://ar5iv.labs.arxiv.org/html/1810.12894)
- **Novelty bonuses under partial observability.** "In such partially observable settings, observations may look alike in different states, and the benefit of approximate state novelty bonuses as intrinsic rewards is unclear." RC-GVF proposes bonuses from temporally extended predictions. It reports gains on partially observable MiniGrid without ground-truth episodic counts. — [Ramesh et al. 2022, RC-GVF, arXiv 2211.10282](https://arxiv.org/pdf/2211.10282)
- **Surprise is confounded by noise in POMDPs.** "Beyond Surprise" proposes measuring surprise novelty through the retrieval errors of a memory network, which reduces attraction to unpredictable observations. — [arXiv 2308.04836](https://arxiv.org/pdf/2308.04836)
- **Episodic bonuses carry the benefit.** Removing the count-based episodic term makes RIDE/AGAC/NovelD fail. This suggests episodic (within-episode) novelty, rather than life-long novelty, is what helps in per-episode-varying environments. — [Henaff et al. 2022](https://arxiv.org/abs/2210.05805)
- **Episodic plus capped life-long novelty.** NGU's design intends exploration across episodes but not within the same episode, using episodic novelty modulated by a capped life-long RND term. — [Badia et al. 2020](https://arxiv.org/abs/2002.06038)

### Inferences
- **Life-long bonuses conflict with the memory policy.** In the twin-room task, the cue changes per episode, so this is a contextual MDP like E3B's setting. Once a seed learns the correct-room policy, the wrong room becomes a rarely visited region. A life-long bonus (RND/counts) then rewards entering it, which is exactly the behaviour memory should suppress. An exploration bonus could therefore *depress* the measured success rate of good memory agents, and depress it differently per backbone. If a bonus is used, prefer episodic (E3B/NGU-episodic), anneal it to 0, or keep it strictly below the +1 success reward, and report extrinsic success only.
- **The bonus could become a memory confound.** An episodic bonus computed from embeddings of the ray casts rewards novelty relative to the agent's own episode memory. Its quality depends on the embedding, not the backbone, so it should not confound the backbone comparison much. A bonus computed *from the backbone's hidden state* would confound it.
- **Noisy TV is probably not a risk here.** The environment is deterministic given actions, so it has no stochastic TV. The cue colour is random per episode, but it is seen only in the alcove.

### Gaps
- No paper found that directly measures intrinsic-reward effects on LSTM/GRU vs MLP memory comparisons in T-maze-like tasks.
- Savinov et al. 2018 (Episodic Curiosity through Reachability) is the classic "couch-potato" source, but it was not retrieved in this session.

## How do papers report results when some seeds never learn?

### Takeaway
Use interval estimates and distribution-level summaries rather than mean +/- std over 5 seeds:
- rliable's stratified-bootstrap CIs, IQM, performance profiles, probability of improvement and optimality gap
- for a bimodal outcome, report "fraction of seeds solving" (with a binomial CI) separately from success conditioned on solving

Ni et al. use 10 seeds with 95% CIs and explicitly note runs excluded for NaNs.

### Cited Findings
- **rliable: few runs make point estimates unreliable.** With only 3-10 runs, point estimates "are unlikely to provide definitive answers". In the Atari 100k case study, a different set of runs could reverse algorithm rankings. — [Agarwal et al. 2021, "Deep RL at the Edge of the Statistical Precipice", NeurIPS](https://ar5iv.labs.arxiv.org/html/2108.13264)
- **rliable: tools.**
  - Stratified percentile bootstrap CIs (good for as few as about 10 runs).
  - IQM, the mean of the middle 50% of runs, which gives smaller CIs than the median.
  - Performance profiles, the tail distribution of scores across runs, which show run-to-run variability better than aggregate intervals.
  - Optimality gap.
  - Probability of improvement P(X > Y).
  - They prefer CIs to significance tests. The library is `rliable`.
  - [Agarwal et al. 2021](https://ar5iv.labs.arxiv.org/html/2108.13264)
- **Non-standard protocols inflate results.** Reporting max-during-training scores makes results inflated and incomparable with end-of-training numbers. — [Agarwal et al. 2021](https://ar5iv.labs.arxiv.org/html/2108.13264)
- **Henderson et al.** recommend more seeds and significance testing, because seed splits alone can produce apparently different algorithms. — [Henderson et al. 2018](https://arxiv.org/abs/1709.06560); [Colas et al. 2018](https://ar5iv.labs.arxiv.org/html/1806.08295)
- **Ni et al.** report mean and 95% CI over 10 seeds, and state in a footnote that about 30% of LSTM runs at memory length 1000 hit NaN gradients and were not plotted. This is an example of disclosing excluded or failed runs. — [Ni et al. 2023](https://arxiv.org/abs/2307.03864)

### Inferences
- **Recommended reporting for bimodal outcomes.** Report:
  1. Fraction of seeds solving (success > threshold), with a Wilson/Clopper-Pearson CI. With 5 seeds, the 95% CI on 2/5 is roughly 0.05-0.85, so 5 seeds cannot separate backbones on this axis.
  2. Success, SPL and AUC conditioned on solved seeds.
  3. A performance profile over all seeds.
  4. IQM with bootstrap CIs.
- **Better: make the exploration bimodality disappear.** Use the fixes in the previous section, then verify that the MLP reaches ~0.5 (the guessing baseline) on every seed. That way the backbone gap above 0.5 measures memory. Conditioning on solved seeds is a fallback, since it selects on exploration luck and can bias in favour of backbones that rarely solve.
- **Seed count.** At least 10 seeds per backbone is the minimum rliable considers workable for percentile CIs.

### Gaps
- No standard metric named "fraction of seeds solving" was found in a primary source in this session; it is common practice but uncited here.
- No primary source found on how success-conditioned metrics bias comparisons. The selection-bias point is my inference.
