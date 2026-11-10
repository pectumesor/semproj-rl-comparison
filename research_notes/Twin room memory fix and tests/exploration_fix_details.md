# Fixing the pre-junction exploration bottleneck in the twin-room cue task: shaping details, alternatives, and how recent memory benchmarks get agents to the decision point

Scope recap (from the assignment, not a source). The setup is 2D continuous-control PPO with 256 envs × 256 steps × 1000 iterations, gamma 0.999, a Gaussian policy, and 180° grey ray casts. The agent spawns in a cue alcove (red or blue), leaves through a 60-unit shaft with a 90° bend, follows a corridor of L steps to a T-junction, and chooses between two identical rooms with a baffle hiding the goal. Entering the wrong room terminates the episode. Reward is −1/max_steps per step and +1 on success, with max_steps = 2L+256. In the L=32 pilot, failed seeds never reached the junction (100% timeouts). Every seed that reached the junction also found the goal, and memoryless agents sat at 0.5. Earlier notes in `research_notes/Twin room exploration failure fixes/` already cover the classic benchmark designs and the general list of fixes. This file covers the parameters of the fix and the newer work.

[repo] facts come from reading local code, not from the literature:
- [repo] `envs/navigation_env.py::NavigationEnvEasy.compute_reward` adds `(prev_dist − ‖agent_pos − goal_pos‖)/(max_range·max_steps)`. That is Euclidean progress toward the goal, and the goal sits in the cued room, so **this term leaks the answer for twin rooms**. Twin rooms must not use this class, and its shaping must not be reused. A junction-only potential needs new code.
- [repo] `NavigationEnv.in_junction()` (junction_box) and `utils/navigation_mesh.py` (Delaunay triangulation + funnel/string-pull shortest path, `generate_reference_trajectory`) already exist. A geodesic distance to the junction can therefore be precomputed from the room JSON.
- [repo] Per an earlier repo note, the agent's max speed is ≈13 units/step and the corridor is 60 units wide. The bend in the shaft is there to stop the 180° FOV from re-reading the cue at the junction.

---

## Q1. Potential-based shaping details: geodesic vs Euclidean, scale vs +1, gamma interaction, clipping at the junction, practical PPO speed-up, and effect on the memory comparison

### Takeaway
Use exact PBRS, F = γΦ(s′) − Φ(s) with the PPO γ = 0.999. Set Φ(s) = −k·d_J(s)/D_J, where d_J is the **geodesic** (path) distance to the junction entry, clipped to 0 at and past the junction, and k ≈ 1 (fall back to k = 0.5 if stalling appears at L = 256). Because Φ = 0 everywhere the episode can terminate, the episodic zero-terminal-potential condition holds automatically. Because Φ is cue-independent and constant after the junction, the room choice is still credited only by the terminal reward. In GAE-PPO, PBRS is exactly equivalent to adding Φ to the critic as a fixed baseline. Its benefit is therefore a better initial value landscape, with no change to what is optimal. The cost is that sample-efficiency metrics (AUC) now measure a shaped task and must be labelled as such.

### Cited Findings
**Theory: invariance, terminal states, gamma**
- Ng et al. 1999 showed that shaping F = γΦ(s′) − Φ(s) is necessary and sufficient for optimal-policy invariance, and gave distance- and subgoal-based potentials. Covered in the earlier notes; primary: [Ng, Harada & Russell 1999, ICML](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
- Wiewiora 2003 proved that a Q-learner given potential-based shaping rewards makes the same updates as an unshaped learner whose Q-table is initialised from Φ. The paper says the result extends to Sarsa and other TD methods. A consequence is that a poor Φ is as harmful as a poor initialisation — [Wiewiora 2003, JAIR 19:205–208](https://www.jair.org/index.php/jair/article/view/10338); [arXiv copy](https://arxiv.org/pdf/1106.5267)
- Berkeley CS294 lecture notes state that shaping leaves the policy gradient invariant and "just adds baseline to estimator", and that Φ = V^π turns the one-step shaped reward into the advantage — [Berkeley Deep RL Sp17, lec 6](https://rll.berkeley.edu/deeprlcoursesp17/docs/lec6.pdf)
- Grzes 2017 (AAMAS) is cited for this result: in episodic tasks, PBRS preserves optimality only if every terminal state has potential 0. Otherwise the shaped return differs by a term in Φ(s_T) and the optimal policy can change. I did not read the full text; the claim comes from the paper record and citing papers — [Grzes 2017, KAR record](https://kar.kent.ac.uk/60614); [PDF](https://www.cs.kent.ac.uk/people/staff/mg483/documents/grzes17goals-in-pbrs.pdf)
- Müller & Kudenko 2025 ("Improving the Effectiveness of PBRS in RL", AAMAS 2025 extended abstract plus arXiv) make three points. Policy invariance needs Φ = 0 in all terminal states, "including the last state before a fixed-step truncation". The constant shift Φ_b(s) = Φ(s) + b/(γ−1) adds b to every non-terminal shaped reward, with b = (1−γ)Q_init − r_∞. With **positive** potentials, small improvements δ < ((1−γ)/γ)Φ(s) receive **negative** shaping, and with **negative** potentials small *decreases* receive positive shaping. So no continuous potential gets every sign right, and they propose an exponential potential (base 32 in their experiments) — [Müller & Kudenko 2025, arXiv 2502.01307](https://arxiv.org/html/2502.01307v1); [AAMAS 2025 abstract](https://aamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2684.pdf)
- Their experiments are small, with no quantified speed-ups: a tabular 25×25 gridworld with γ = 0.95, plus DQN on CartPole and MountainCar. In CartPole (+1 per step reward) biases ≤ 0 learned faster than no shaping, positive biases stayed near zero, and negative biases "produced immediately terminating policies". In MountainCar (−1 per step) bias 1 was best — [Müller & Kudenko 2025](https://arxiv.org/html/2502.01307v1)
- Devlin & Kudenko 2012 proved that a time-varying potential Φ(s,t), with F = γΦ(s′,t′) − Φ(s,t), keeps policy invariance in the single-agent case. They also showed it is no longer equivalent to a Q-initialisation — [Devlin & Kudenko 2012, AAMAS, White Rose PDF](https://eprints.whiterose.ac.uk/75121/2/p433_devlin.pdf)
- Partial-episode bootstrapping treats a timeout as a non-terminal cut, bootstrapped with V(s_T), so the task becomes effectively infinite-horizon at truncation — [Pardo et al. 2018](https://arxiv.org/abs/1712.00378v2). This conflicts with Müller & Kudenko's finite-horizon requirement that Φ = 0 at the last state before truncation; see the inferences below.

**Newer PBRS variants (2024–2025)**
- Bootstrapped Reward Shaping (BSRS) uses the agent's current V estimate as Φ. It has tabular convergence proofs and is reported to speed up training on Atari — [Adamczyk, Makarenko, Tiomkin & Kulkarni, AAAI 2025, arXiv 2501.00989](https://arxiv.org/abs/2501.00989); [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/33679). Note: this is the 2025 "bootstrapped" PBRS paper. I found no "Forbes 2024 bootstrapped PBRS".
- "Forbes 2024" most plausibly refers to Forbes et al., "Potential-Based Reward Shaping for Intrinsic Motivation" (AAMAS 2024), and its follow-up on non-Markovian shaping. These convert intrinsic-motivation bonuses into a potential-based form that preserves optimality. I read only titles and snippets — [arXiv 2402.07411](https://arxiv.org/pdf/2402.07411); [arXiv 2410.12197](https://arxiv.org/pdf/2410.12197)

**Geodesic vs Euclidean, and practice with PPO**
- DD-PPO (Habitat PointNav) uses a terminal reward r_T = 2.5·SPL and a shaped per-step reward r_t = −Δgeo_dist − 0.01, where Δgeo_dist is "the change in geodesic distance to the goal" caused by the action. Their transfer tasks use r_t = 5(D_t − D_{t−1}) on normalised geodesic displacement. This is the de-facto standard for PPO navigation: geodesic progress shaping, no γ factor in the difference, and a total shaping budget larger than the success reward — [Wijmans et al. 2020, DD-PPO, ICLR](https://arxiv.org/html/1911.00357v2)
- The Habitat-Lab `DistanceToGoalReward` measure is −(new_distance − previous_distance) — [Habitat-Lab docs](https://aihabitat.org/docs/habitat-lab/habitat.tasks.nav.nav.DistanceToGoalReward.html)
- Trott et al. 2019 report that naive distance-to-goal shaping "tends to trap learning in local optima" in maze navigation, and propose Sibling Rivalry. Their shaping is a non-potential reward of −distance, not PBRS — [Trott et al. 2019, NeurIPS, arXiv 1911.01417](https://ar5iv.labs.arxiv.org/html/1911.01417)
- In minimum-time sparse goal-reaching, the initial policy's goal-hit rate predicts whether a run learns. Covered in the earlier notes — [Vasan et al. 2024, RLJ](https://arxiv.org/html/2407.00324v2)

**Pace shaping in memory tasks (for scale comparison)**
- Ni et al. 2023's Passive T-Maze uses R_t = (1(x_{t+1} ≥ t) − 1)/(T−1), a −1/(T−1) penalty for each step behind optimal pace, so the total magnitude is ≤ 1 and matches the +1 terminal reward. The horizon is T = L+1. The paper counts credit-assignment length as 1 "since immediate penalties punish suboptimal actions" — [Ni et al. 2023, NeurIPS](https://arxiv.org/html/2307.03864v4)

### Inferences
- **The recommended potential (concrete).** Let d_J(s) be the geodesic distance from s to the junction entry line along the alcove→shaft→bend→corridor path. Compute it from the nav-mesh funnel path, or as arc-length along the corridor centreline polyline, since the path is a single tube. Clip it to 0 inside `junction_box` and in both rooms. Let D_J = d_J(spawn), computed per room JSON so that the total shaping budget is k at every L. Then Φ(s) = −k·d_J(s)/D_J, F = γΦ(s′) − Φ(s) with γ = 0.999 (the same γ as GAE/PPO), and k = 1.
  - Φ = 0 at every state where the episode can terminate (success or wrong room, both past the junction). So the Grzes/Müller–Kudenko zero-terminal condition holds with no special terminal handling, and there is no −Φ "payback" when the episode ends.
  - After the junction F ≡ 0 exactly, so the room choice and the in-room goal search are shaped by nothing.
  - Because the potential is negative and zero at the junction, it also respects the sign result above: progress always gets positive shaping. Standing still gets a small *positive* shaping of (1−γ)k·d_J/D_J ≤ 0.001k per step (the Müller–Kudenko negative-potential case).
- **Gamma arithmetic.** Compare that stand-still bonus with the step penalty 1/(2L+256): 0.0037 (L=8), 0.0031 (L=32), 0.0026 (L=64), 0.0020 (L=128), 0.0013 (L=256). Stalling stays net-negative for k < 1/((1−γ)(2L+256)), i.e. k < 3.7 at L=8 and k < 1.3 at L=256. **k = 1 is safe at every L, but the margin at L=256 is thin (net −0.0003/step).** Use k = 0.5 there if seeds stall in the alcove. Alternatively drop γ from F (DD-PPO style F = Φ(s′) − Φ(s)). That adds a cue-independent per-step bias of −(1−γ)k·d_J/D_J ≤ 0.001k before the junction, which strengthens the hurry signal but technically breaks exact invariance. For the paper, keep γ in F so the invariance claim is clean.
- **Signal size.** Assuming ≈13 units/step max speed (repo note) and corridor length ≈13L units, D_J ≈ 13L + shaft + alcove. Full-speed progress then earns k/(L + ~5) per step: about 0.08 at L=8, 0.027 at L=32, and 0.004 at L=256. That is about 3–20× the step penalty, and the total of k over the traversal is comparable to the +1 goal. This mirrors Ni's design, where pace shaping is bounded by 1. Note the per-step signal shrinks ∝ 1/L, just like the step penalty, so long-L exploration stays harder per step. That is intended, because the budget relative to +1 stays fixed.
- **Geodesic over Euclidean.** Under Ng, invariance holds for any Φ, so Euclidean does not leak or bias the optimum. Speed-up, however, depends on Φ approximating V* (the Wiewiora equivalence). Euclidean distance to the junction from inside the alcove or the shaft before the 90° bend points through walls. That creates exactly the local optimum Trott et al. describe: hugging the wall nearest the junction. The geodesic version is cheap here (one tube) and is what Habitat and DD-PPO use.
- **Why it helps PPO, mechanically (my derivation; it agrees with the Berkeley "adds a baseline" statement).** With shaped rewards and critic V′, the TD residual is δ′_t = r_t + γΦ(s_{t+1}) − Φ(s_t) + γV′(s_{t+1}) − V′(s_t). If V′ = V − Φ, then δ′_t = δ_t for every t. So GAE advantages are identical for any λ, and shaped PPO equals unshaped PPO whose critic is offset by +Φ. The whole benefit is that early on, when V′ ≈ 0, the effective critic is already Φ, a correct-shaped "progress toward the junction" value. The sparse task's critic is instead ≈ constant, so advantages for corridor actions are pure noise until a first success. This is the actor-critic analogue of Wiewiora 2003. It also implies the benefit fades once the critic is learned, and that asymptotic performance should not change.
- **Truncation handling.** The repo bootstraps timeouts with V(s_T) (Pardo-style, infinite-horizon view). Under that view, use Φ(s_T) normally at the last pre-truncation step and do **not** zero it. Müller & Kudenko's "zero at the last state before truncation" applies to the finite-horizon framing and would inject a spurious −Φ(s_T) = +k·d_J/D_J at every timeout. That bonus is largest for agents that never left the alcove, which is the opposite of what we want.
- **Effect on what the memory comparison measures.**
  - (a) The asymptotic optimum and the memoryless ceiling (0.5) are unchanged, because Φ is cue-independent and F ≡ 0 past the junction.
  - (b) Credit for the room choice still comes only from the terminal reward. The cue→decision credit-assignment length is unchanged, while the navigation credit length becomes short (Ni's distinction).
  - (c) What changes is sample efficiency of reaching the junction, so AUC and learning-speed metrics belong to the shaped task. State this, and ideally include an unshaped L=8 or L=32 reference row.
  - (d) Faster, near-shortest-path traversal makes the cue→junction delay tight. The correlation horizon ξ (Cherepanov et al. 2024) then becomes close to a fixed function of L rather than of exploration luck, which makes "memory length" better defined. Log the realised steps from the last cue sighting to the junction.
  - (e) It removes the pilot confound in which MLP 0/5 and some LSTM failures were navigation failures. The comparison then becomes the intended one: success | reached junction.
- **Leak checks for the shaping.**
  - Φ must not depend on goal_pos or room identity. In particular, do not reuse `NavigationEnvEasy`.
  - Clip exactly at the junction entry so that neither door side is closer in Φ.
  - Rooms are mirror-identical and Φ is constant inside them, so this holds by construction.
  - Shaping is cue-independent, so even agents that see previous reward as input gain no information about the room.

### Gaps
- I found no paper that measures PBRS speed-ups with PPO+GAE specifically on memory or T-maze tasks. The practical evidence is navigation PPO (DD-PPO) plus theory and small DQN/tabular experiments (Müller & Kudenko).
- I did not read Grzes 2017's full text. The terminal-zero claim relies on the paper record plus citing papers and Müller & Kudenko.
- An appendix of an unrelated 2025 paper (arXiv 2511.22119, Prop. A.3) claims GAE-PPO invariance under PBRS with critic shift −Φ. I did not verify it and do not rely on it; the derivation above is mine.
- I could not find the Forbes et al. 2024 and 2410.12197 details (beyond titles). They matter only if an intrinsic-motivation bonus such as RND is added.

---

## Q2. Alternatives (curriculum over L or start position, shorter horizon, std floor, removing the bend or alcove), ranked by whether they keep the cue as the only room information and the memory length well defined

### Takeaway
Ranked: (1) junction-clipped geodesic PBRS (Q1); (2) Ni-style pace penalty, equivalent in spirit but not invariant; (3) curriculum over L ending at the target L, cue-safe but it changes the training distribution per backbone; (4) init-std/std-floor tweaks as a cheap complement; (5) shorter horizon, a minor help; (6) start-position curriculum along the path, risky because it removes the cue or mixes memory lengths; (7) removing the bend or alcove, rejected because it reintroduces the line-of-sight cue leak. Cherepanov et al. warn that mixing correlation horizons hides long-term-memory failures, which argues against any fix that makes the cue→decision delay variable at evaluation.

### Cited Findings
- **Memory-length definitions.** Cherepanov et al. 2024 define the correlation horizon ξ = t_r − t_e − Δt + 1 and the context memory border K̄ = min ξ − 1. A task tests long-term memory only if the agent's context K ≤ K̄. They recommend fixed ξ > K over mixed-horizon setups, "since mixed setups can hide LTM deficits". In their Passive T-Maze, T = L+1 and ξ = T. In MiniGrid-Memory, a variable setup has ξ ∈ [7, L+1] — [Cherepanov et al. 2024, "Unraveling the Complexity of Memory in RL Agents", arXiv 2412.06531](https://arxiv.org/html/2412.06531)
- **Endless tasks as automatic curriculum.** Memory Gym's endless variants grow the task as the agent succeeds: Endless Mortar Mayhem alternates showing and executing commands, with +0.1 per command. GRU outlasted TrXL (more than 3,000 vs more than 400 steps in Endless MM) — [Pleines et al., Memory Gym, arXiv 2309.17207](https://arxiv.org/html/2309.17207)
- **Reverse curriculum.** Florensa et al. 2017 start episodes near the goal and expand outward. Covered in the earlier notes — [Florensa et al. 2017](https://ar5iv.labs.arxiv.org/html/1707.05300)
- **Std and init.** Andrychowicz et al. found an initial action std of 0.5 best, recommend softplus std with an offset and a 100× smaller last-layer init, and found no regulariser (including entropy) helped significantly. Covered in the earlier notes — [Andrychowicz et al. 2021](https://ar5iv.labs.arxiv.org/html/2006.05990)
- **Tight horizon.** Ni et al. set the horizon to exactly corridor length + 1, so a T-maze episode has no slack — [Ni et al. 2023](https://arxiv.org/html/2307.03864v4)
- **MDP/POMDP twins.** POPGym Arcade gives every environment fully and partially observable "twins" with identical dynamics, so observability can be varied with everything else fixed. The authors report that controlling for partial observability is critical — [POPGym Arcade, arXiv 2503.01450](https://arxiv.org/html/2503.01450v5)

### Inferences
- **Ranking with leak and memory-length notes.**
  1. **Junction-clipped geodesic PBRS (k=1, γ=0.999).** No leak. Memory length gets *better* defined, because traversal approaches a fixed duration. Optimum unchanged. Cost: AUC now refers to the shaped task.
  2. **Ni-style "behind pace" penalty in the corridor**, e.g. −1/(T_J) for each step that progress along d_J lags 13 units/step. No leak and a tight ξ. It is not exactly policy-invariant: it penalises slow-but-legitimate behaviour, and with a turning-rate-limited continuous agent the "optimal pace" around the bend must be set conservatively. Prefer PBRS, which gives the same signal with an invariance guarantee.
  3. **Curriculum over L (8→32→…→target L)**, using the existing `rooms/twin_rooms_L*.json`, keyed on the success rate in reaching the junction, never on the cue. No leak. At evaluation, ξ is fixed at the target L. Confounds:
     - Backbones that learn faster advance sooner, so their training distributions differ.
     - The memory is trained at short delay and then stretched. That tests delay extrapolation, which is different from learning at L.
     - It needs a promotion rule and makes "1000 iterations" non-comparable across L.
     It is useful as a fallback if PBRS alone fails at L ≥ 128.
  4. **Initial std 0.5–1.0 plus a floor (e.g. log-std ≥ log 0.2), or RPO-style mean noise.** No leak, and cheap. It does not create a gradient where none exists. It helps only if failing seeds are collapsing std before the first success (check the logged std on failed seeds). It is a good complement to PBRS, not a replacement.
  5. **Shorter horizon (e.g. 1.5L + 128 instead of 2L + 256).** No leak. It raises the per-step penalty and gives more resets per rollout, so more cue samples per iteration. With 256-step rollouts and 320-step episodes, an env currently completes fewer than one episode per iteration. But it does not fix the absence of signal, and too tight a horizon times out competent policies around the bend and the baffle. It is a minor help.
  6. **Start-position curriculum along the path (Florensa-style).** Starts past the bend never see the cue: those episodes become memoryless coin flips (fine for learning navigation, but they dilute the memory signal). If the cue is also shown at those starts, the cue→decision delay becomes variable, which is exactly the mixed-ξ setup Cherepanov et al. warn hides LTM failures. If used at all, restrict it to a warm-up phase, keep the full-path start for at least 50% of episodes, and evaluate only from the alcove.
  7. **Removing the bend or alcove.** Not recommended: per the repo note, the bend exists to stop the 180° FOV from re-reading the cue down a straight corridor. Removing it reintroduces an observation leak, which makes memory unnecessary at short L.
- **A diagnostic control in the POPGym-Arcade "twin" spirit.** Run a cue-visible twin, for example cue colour also painted on the junction walls, at one L. This gives an upper bound that separates navigation or exploration from memory per backbone. Report "success | reached junction" both for it and for the POMDP version.

### Gaps
- I found no memory-RL paper that uses a start-position curriculum while also controlling ξ. The risk is inferred from Cherepanov's mixed-horizon argument.
- I found no benchmark that reports PBRS vs curriculum head-to-head on a memory task.

---

## Q3. How recent memory-RL benchmarks (2023–2026) make sure agents reach the decision point, and whether their dense rewards leak

### Takeaway
Recent benchmarks mostly remove navigation burden instead of solving it. Ni et al. use a 1-D corridor with an exact horizon and a pace penalty. Memory Gym freezes the agent during the clue phase, uses grid-like moves, and pays dense progress rewards. MIKASA-Robo shows the cue for a fixed number of steps and offers a dense reward. MIKASA-Robo's dense reward **does leak**: it rewards closeness to the *correct* cube. This is a concrete example of the failure mode our junction-clipped, cue-independent potential avoids.

### Cited Findings
- **MIKASA-Robo RememberColor dense reward (read from code).**
  - `reward = 1.0·(1 − tanh(10·‖tcp → true-colour cube‖)) + 0.5·(1 − tanh(5·‖qvel‖)) + 0.5·is_static·is_obj_placed`, with reward = 3.0 on success and normalised dense reward = reward/3.
  - `obj_to_goal_pos` is built by summing `(cube_k.pos − tcp.pos)·mask_k`, where `mask_k = (true_color_indices == k)`. So the distance term targets the cue-indicated cube, i.e. the dense reward encodes the answer.
  - Episodes have a fixed `max_episode_steps=60`.
  - [MIKASA-Robo `remember_color.py`, lines ~214–289](https://github.com/CognitiveAISystems/MIKASA-Robo/blob/main/mikasa_robo_suite/rl/memory_envs/remember_color.py)
- **MIKASA-Robo timing and results.**
  - Fixed timeouts: 60 steps for RememberColor/Shape, 90 for ShellGame/Intercept/Rotate, 120 for BunchOfColors/SeqOfColors, 180 for TakeItBack.
  - In RememberColor, the cube is shown for 5 steps, hidden for 5, and the choices appear at t=10.
  - Under dense reward, PPO-LSTM beats PPO-MLP at 3 colours, and both are near zero at 5 or 9 colours. Under sparse reward both fail even at 3 colours. In state mode, PPO-MLP reaches 100%.
  - [Cherepanov et al. 2025, MIKASA, arXiv 2502.10550](https://arxiv.org/html/2502.10550)
- **Memory Gym.**
  - Mortar Mayhem: the agent "starts immobile" while 10 commands are shown, then gets +0.1 per executed command, and a failure ends the episode.
  - Mystery Path: +0.1 per newly visited path tile, and falling off teleports the agent to the origin. Max 512 steps (128 for the grid variant).
  - Endless Mystery Path removes leftward movement and gives a 20-step budget per tile.
  - Grid variants simplify locomotion. The paper does not explicitly justify them as reducing exploration.
  - [Pleines et al., Memory Gym, arXiv 2309.17207](https://arxiv.org/html/2309.17207)
- **Ni et al. 2023.** A 1-D corridor, horizon T = L+1, and the pace penalty above. They describe the T-mazes as needing little exploration. Covered in the earlier notes — [Ni et al. 2023](https://arxiv.org/html/2307.03864v4)
- **POPGym Arcade.** Uses MDP/POMDP twins to isolate observability, rather than shaping — [arXiv 2503.01450](https://arxiv.org/html/2503.01450v5)

### Inferences
- None of the recent benchmarks I checked has a continuous-control agent navigating a long, bent path *before* a cue-dependent decision with only sparse reward. Ours is harder on exploration than all of them.
- Two families keep the decision reachable without leaking:
  - (i) Remove the navigation degrees of freedom: Ni's 1-D corridor, Memory Gym's frozen clue phase.
  - (ii) Dense, cue-independent progress reward: Ni's pace penalty, Memory Gym's per-tile reward. Memory Gym's tile reward is part of the memory task itself, not cue-independent shaping.
  Junction-clipped geodesic PBRS is the type-(ii) fix that also keeps continuous navigation.
- The MIKASA example shows why "add a dense reward" is dangerous in memory benchmarks. A reward that depends on the cued target lets the critic, and any agent that sees rewards, learn the answer from reward rather than memory. It also gives memoryless agents a shaped gradient toward the correct option within each episode. Our design must make shaping identically zero (or identical) in both rooms. Cite MIKASA as the contrast case in the paper.
- I did not check ELMUR, AMAGO, Stabilizing Transformers (GTrXL), Recurrent Linear Transformers or BabyAI for this question. To my knowledge they reuse existing tasks (Ni's T-maze, Memory Gym, DMLab, POPGym) and add no new reach-the-decision mechanism, but I have not verified this here.

### Gaps
- I did not read MIKASA's appendix K reward descriptions. The leak finding comes from the RememberColor source only; other MIKASA tasks (ShellGame, Intercept) may or may not target the correct object.
- I did not fetch the AMAGO, ELMUR or Recurrent Linear Transformer task details, so the inference about them is unverified.
- I found no source for "Ni 2023 continuous variants". Their T-mazes are discrete gridworlds as far as the sources checked show.
