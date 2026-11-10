# Reward-design flaws, memory leaks, and recurrent-PPO pitfalls in the twin-room cue task

Scope: (a) reward and termination design flaws for this setup, (b) ways a memoryless agent can cheat a cue-memory task, (c) recurrent-PPO training issues that could explain the all-or-nothing seeds and why GRU and longer BPTT windows fail more often.

Repo facts used below come from reading the local code (no literature source). They are labelled "[repo]":
- [repo] Twin rooms use base `NavigationEnv` (no distance shaping). Reward is `-1/max_steps` per step and `+1` on success (`envs/navigation_env.py::compute_reward`). The pilot uses `max_steps = 2L + 256` (= 320 at L=32) and gamma 0.999 (`scripts/e2_twin_rooms.sh`).
- [repo] `terminated = success | wrong-room`. Time-limit truncation is bootstrapped by adding `gamma * V(next_obs)` to the last reward (`algorithms/ppo.py` around lines 147-149 and 651-653). The bootstrap uses the pre-reset observation, which is correct. Remaining time is not in the observation.
- [repo] Recurrent rollouts store each step's incoming `(h, c)` state. Training slices each env's 256-step rollout into windows of `bptt_window_size`, and each window starts from the stored state (`sample_mini_batch`).
- [repo] One recurrent backbone feeds both actor and critic (`recurrent_agent.forward` is used by `predict_action` and `get_value`). `nn.LSTM`/`nn.GRU` use PyTorch default init, with no forget-gate bias override and no orthogonal init in the backbone.
- [repo] Eval uses `act_inference` (deterministic mean action) from zero initial state. A timeout counts as a failure.
- [repo] `generate_twin_rooms.py` already removed one observation leak. With a straight corridor, the edge ray of the 180° FOV could re-read the cue at the junction. The shaft now bends 90°, so the cue is out of line of sight from about 30 units (≈2 steps) past the bend. Ray range is the room diagonal (`range_type: diagonal`). The corridor is 60 units wide (≈4.6 steps at max speed 13).

---

## 1. Per-step time penalty + early termination: the "suicide" incentive, wrong-choice reward, and time limits

### Takeaway
A negative living reward makes any terminal state an escape hatch, and the sign of non-terminal rewards decides whether agents cut episodes short or drag them out. In our reward, a wrong-room termination (return ≈ −k/320 ≥ −1) strictly beats wandering until timeout (bootstrapped value ≈ −3.1). The reward-optimal memoryless policy is therefore "enter a room fast" with success ≈ 0.5. A seed that times out on every eval episode is not exploiting the reward. It is an exploration or optimisation failure, or a deterministic-eval artefact.

### Cited Findings
- Pardo et al. (ICML 2018) separate time-limit terminations from environmental ones. They recommend partial-episode bootstrapping, i.e. bootstrapping at states where termination comes from the time limit or any non-environmental cause. They also recommend time-awareness, which adds remaining time to the input so the Markov property holds. They report gains with tabular Q-learning and PPO — [Pardo et al. 2018, arXiv:1712.00378](https://arxiv.org/abs/1712.00378v2); [project page](https://sites.google.com/view/time-limits-in-rl)
- The ICLR "37 implementation details" blog notes that the reference PPO (openai/baselines) treats time-limit truncation as `done=True` and does not bootstrap it. The authors "intuitively" think it should, but kept the original behaviour for fidelity — [Huang et al. 2022, ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- In the DERAIL diagnostic environments, the sign of the reward in non-terminal states decides whether agents cut episodes short or prolong them. Fixing terminal states to zero value is a common source of this bias — [Freire et al., DERAIL, arXiv:2012.01365](https://arxiv.org/pdf/2012.01365)
- A per-step penalty pushes agents to finish fast. With several termination conditions, a short path that ends the episode can beat completing the task — [Reward shape exploration in AIL, arXiv:2104.06687](https://arxiv.org/pdf/2104.06687)
- "RL with a Terminator" uses a constant penalty on termination as a baseline for termination handling and compares it with other approaches — [arXiv:2205.15376](https://arxiv.org/pdf/2205.15376)
- Informal practitioner case: a SUMO traffic agent learned to crash early because termination cut off a stream of negative per-step rewards. The fix was to make per-step rewards non-negative — [7174andy SUMO write-up](https://7174andy.github.io/sumo-rl-mixed-traffic/latest/reward-redesign) (blog, not peer-reviewed)

### Inferences
- [Inference, arithmetic on repo values] At L=32 each step costs 1/320. A policy that never terminates has bootstrapped value ≈ −(1/320)/(1−0.999) ≈ −3.1, since partial-episode bootstrapping treats a timeout as continuing forever. A wrong-room entry after k steps returns −k/320 ∈ (−1, 0]. Correct-room success returns 1 − k/320. So:
  - The "suicide" direction is benign here: terminating in either room dominates wandering. The memoryless optimum is a fast coin-flip, about 0.5 success. This matches the frame-stack seeds at 0.5-0.6.
  - MLP 0/5 and failing recurrent seeds that time out on every eval episode sit far below the memoryless optimum. They are not reward-optimal for any memory capacity. The bottleneck in those seeds is reaching or entering a room or finding the goal behind the baffle (exploration or optimisation), or an eval-mode artefact. Memory is not the bottleneck.
- [Inference] Eval is deterministic (mean action) while training samples. A seed whose stochastic training policy reaches rooms by noise can still have a mean action that stalls, e.g. driving into a wall or spinning, and time out on every eval episode. Low-cost check: log training-episode outcomes per termination type (success / wrong room / timeout), and run eval with sampled actions as well.
- [Inference] Setting the wrong-room return equal to the timeout return (e.g. a −3.1 terminal penalty) would make the agent indifferent between guessing wrong and stalling. That removes the pull towards rooms that currently helps exploration. A strongly negative wrong-room penalty (worse than a timeout) creates the reverse incentive: "never enter a room" becomes a local optimum, which looks exactly like timing out on every episode. Keeping the wrong-room terminal at 0 (or a small penalty, less negative than the timeout value) is the safer choice for a memory probe. It keeps chance at 0.5 and keeps rooms attractive.
- [Inference] The implementation follows Pardo's partial-episode-bootstrapping recommendation but not time-awareness. Without remaining time in the input, a recurrent agent can partly infer elapsed time from its hidden state, and an MLP cannot. This is a small, systematic asymmetry between backbones. It probably matters little here because gamma 0.999 gives an effective horizon of 1000 steps, longer than the 320-step limit.
- [Inference] Gamma 0.999 with a timeout value of about −3 means one iteration's value targets span roughly [−3, 1]. Value-loss scale and normalisation can matter for PPO stability. Not verified for this run.

### Gaps
- I found no peer-reviewed paper that names and quantifies the "suicide" incentive for PPO specifically. The evidence is DERAIL's diagnostic result, the AIL study, and practitioner reports.
- Not verified this session: how the Ni et al. 2023 passive/active T-maze sets the wrong-choice and per-step rewards, which would be a good template for a memory probe. Check Sec. 5 / App. of [arXiv:2307.03864](https://arxiv.org/abs/2307.03864v4).

---

## 2. Embodied / external memory leaks: how a memoryless agent can cheat a cue task, and how designers prevent it

### Takeaway
"Writing" the cue into the world or into one's own body is a known memoryless strategy (stigmergy; Peshkin et al. 1999). Benchmark designers block it by removing the agent's ability to change position or observation during the delay ("limbo"), or by making memory requirements continuous and dynamic. The repo has already fixed the line-of-sight leak. The remaining open channels are lateral position and heading in the 60-unit corridor, which map directly to "turn up" vs "turn down" at a symmetric T-junction.

### Cited Findings
- Stigmergy: Peshkin, Meuleau & Kaelbling give an agent external memory bits it can set and clear, fed back as input. With enough bits, the optimal *memoryless* policy of the augmented agent is optimal in the POMDP. A memoryless agent can solve memory tasks if it can write state into its input — [Peshkin et al., ICML 1999, arXiv:cs/0103003](https://arxiv.org/abs/cs/0103003)
- In the T-maze of the Low-pass RNN paper, after the teleporter the agent sits in "limbo", a tiny space where it cannot move or change its observation, for a 280-step delay. The paper states the player is thereby prevented from using its position as a means of recording the cue, so internal memory is the only place the cue can be kept — [Stepleton et al., Low-pass RNNs, arXiv:1805.04955, Sec. 5.2.2 and App. B.2](https://ar5iv.labs.arxiv.org/html/1805.04955)
- In Memory Gym, Pleines et al. note that Ballet "can be made fully observable by encoding the full cue sequence", removing the need for policy memory. In Searing Spotlights the appendix lists behavioural strategies such as moving to boundaries so the agent's position becomes easier to estimate. These are environment-mediated substitutes for internal memory, though the paper does not report agents learning them — [Pleines et al. 2022, arXiv:2205.11104](https://ar5iv.labs.arxiv.org/html/2205.11104)
- Memory Maze randomises maze layouts and object positions per episode. Its stated aim is to isolate long-term memory from confounds such as exploration. It also releases an offline probing evaluation that decodes what agents remember — [Pasukonis, Lillicrap & Hafner 2022, arXiv:2210.13383](https://arxiv.org/abs/2210.13383)
- Some T-maze variants also give the agent explicit position encodings (e.g. an 8-bit Gray code of the current location). Here position is a given signal, not a leak — [AGaLiTe, arXiv:2310.15719](https://arxiv.org/pdf/2310.15719)

### Inferences
- [Inference, repo-specific] Line-of-sight leak: closed by the 90° bend (cue hidden about 2 steps after the bend). It is worth re-checking that no ray can reach a cue wall from the corridor at any heading. Rays have diagonal (room-wide) range, so a single grazing ray suffices. A quick test: assert that the cue-colour channel is zero for every ray at every state past the bend, across random poses.
- [Inference, speculative] Body-as-memory channel: the corridor is 60 units wide (about 4.6 full-speed steps), the T-junction is symmetric, and room A/B lie on opposite arms. A memoryless policy could see the cue in the alcove, steer to the upper or lower half of the corridor (or keep a slight heading bias), then follow that wall into the matching arm. Ray distances to both side walls reveal lateral offset, and heading is implicit in the ray geometry. This is a textbook wall-following strategy. MLP 0/5 and frame stacks ≈ 0.5 suggest memoryless agents have not found it so far. A recurrent agent can still use a hybrid: a weak hidden-state trace plus a pose bias.
- [Inference, important] LSTM with BPTT window 16 solves 5/5 seeds. The cue-to-decision gap is at least shaft (120/13 ≈ 9 steps) plus corridor (32 steps), so more than 40 steps, which exceeds the gradient horizon of 16. Three explanations:
  - (i) The cue is retained in the forward hidden-state dynamics across window boundaries even though no gradient spans the gap. TBPTT still carries state forward, so only the "write and retain" side lacks a direct gradient.
  - (ii) The cue is offloaded into pose (lateral offset or heading) for part of the corridor.
  - (iii) A residual observation leak.

  Proposed diagnostics:
  - Linear-probe cue identity from (a) the hidden state and (b) pose (lateral offset, heading) at several corridor depths.
  - Intervention "pose lesion": at mid-corridor, re-sample lateral position and heading, keeping the hidden state. If success drops, the body carries the cue.
  - Intervention "memory lesion": zero or shuffle the hidden state mid-corridor, keeping the pose. If success drops to ≈ 0.5, the cue lives in the hidden state.
  - Design fixes in the spirit of the "limbo" T-maze: narrow the corridor or add a mid-corridor reset of lateral pose and heading (teleport to the centre line, random heading, or a funnel or one-way door). Randomise spawn pose in the alcove. Mirror the junction so the correct arm is not always on the side matching a natural corridor bias.
- [Inference] Other possible implicit cue carriers: the `proprio` slots (last speed, last turning). A one-step action echo can carry a bit if the agent keeps turning slightly left vs right through the corridor. Turning bias is the same as a heading leak and shows up under the pose-lesion test, provided the last-action proprio is also reset.

### Gaps
- I found no paper that reports RL agents specifically "hugging one wall" to carry a T-maze cue. The mechanism follows from Peshkin's stigmergy result and the limbo design, but there is no documented case in a continuous-control T-maze.
- Not verified: whether DeepMind Lab / DMLab-30 or the MiniGrid `Memory` environments document pose-randomisation as an anti-leak measure.

---

## 3. Recurrent-PPO pitfalls: stale hidden states, BPTT window, burn-in, initialisation, LSTM vs GRU, implementation details

### Takeaway
The literature is split on stale stored hidden states. One informal study found refreshing critical on a memory-heavy task, while Memory Gym found it not worth the cost. The well-established findings are:
- Zero-state starts are worse than stored states, and burn-in helps (R2D2).
- A forget-gate bias of 1 closes the LSTM-GRU gap in supervised learning (Jozefowicz 2015, after Gers 2000).
- A shared actor-critic encoder can hinder recurrent RL (Ni 2022, off-policy).
- LSTM vs GRU usually differs little in RL (Ni 2022; GRU slightly better in Pleines 2022).

Our GRU at 1/5 and window 64 at 2/5 vs window 16 at 5/5 do not match published trends. With 5 seeds only the GRU-vs-LSTM-16 gap is borderline significant.

### Cited Findings
- **Stale hidden states (pro-refresh).** Stored hidden states go stale as parameters change within an update. The current policy then "falsely appear[s] more similar" to the stored one, which weakens PPO clipping and makes KL-based early stopping unreliable. On a memory-intensive gridworld (goal cycle), refreshing hidden states (borrowed from R2D2) was "critical to achieving good performance". Without it, PPO-Clip and early stopping failed to stop large policy changes, with KL divergence an order of magnitude larger. On a less memory-dependent environment, rewards were similar. Caveat: 3 seeds, 2 envs, blog — [Kandou, "Stale hidden states in PPO-LSTM"](https://kandouss.github.io/blog/ppo_stale_states)
- **Stale hidden states (contra).** In Memory Gym, recomputing hidden states and advantages every other epoch gave only slight gains and was judged not worth the extra forward pass. More epochs on stale data clearly hurt — [Pleines et al. 2022, arXiv:2205.11104](https://ar5iv.labs.arxiv.org/html/2205.11104)
- **Memory Gym implementation pitfalls** — [Pleines et al. 2022](https://ar5iv.labs.arxiv.org/html/2205.11104):
  - Passing `None` as the hidden state (PyTorch zero-init) caused exploding gradients.
  - Reshaping without `batch_first` caused a severe distribution shift.
  - An extra activation after the GRU/LSTM output may hurt.
  - Unmasked padding gives misleadingly good results.
  - Small-scale verification gives false confidence.
  - Settings: zero initial state; sequence length 128, longer was generally better in their grid search; GRU (hidden 512) was slightly better than LSTM.
- **Truncated BPTT bookkeeping.** In the MarcoMeter reference implementation, a truncated episode's previous hidden state must be saved for the next optimisation. All hidden states are stored during sampling, and the needed ones are selected during optimisation — [recurrent-ppo-truncated-bptt](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt); [issue #17](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt/issues/17)
- **Baselines / CleanRL LSTM details** — [Huang et al. 2022, ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/); [CleanRL ppo_atari_lstm docs](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy):
  - LSTM weights initialised with std=1 and bias 0.
  - Zero initial states.
  - States reset at episode end.
  - Sequential minibatches per sub-environment.
  - States *reconstructed during training* from a saved `initial_lstm_state` taken at rollout start, so the training distributions match the rollout's. Only the rollout-start state is stored, not per-step states.
- **SB3 RecurrentPPO defaults.** `lstm_hidden_size=256`, `n_lstm_layers=1`, `shared_lstm=False`, `enable_critic_lstm=True`, so the critic has its own LSTM by default. If the LSTM is shared, only the actor gradient flows through it — [SB3-contrib RecurrentPPO docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
- **Ni et al. 2022, recurrent model-free RL** (off-policy TD3/SAC-based; matched or beat specialised methods on 18/21 envs) — [Ni et al. 2022, arXiv:2110.05038](https://arxiv.org/html/2110.05038v3):
  - A shared actor-critic encoder "increases the gradient norm and hinders learning". In two simple POMDPs the shared version failed to learn; the authors suspect critic gradients dominate.
  - LSTM vs GRU: "generally no significant difference".
  - Optimal context length is task-specific; 64 was best in most benchmarks.
  - Adding past rewards to inputs can help substantially (1.3× on Walker-P).
- **R2D2 (recurrent state in replay).** Compared zero start state, stored state, burn-in, and stored state + burn-in:
  - Zero start states cause a "significantly more severe" staleness effect and prevent the agent from relying on long-term recurrent state.
  - Stored states mitigate staleness much better.
  - Burn-in (unrolling a prefix without gradients) prevents destructive updates from inaccurate early outputs after zero-init.
  - The final agent combines stored state and burn-in.

  — [Kapturowski et al., ICLR 2019, OpenReview r1lyTjAqYX](https://openreview.net/forum?id=r1lyTjAqYX). The primary PDF was not accessible this session. Details come from secondary summaries: [Sherwin Chen blog](https://xlnwel.github.io/blog/reinforcement%20learning/R2D2), [DI-engine R2D2 docs](https://di-engine-docs.readthedocs.io/en/latest/12_policies/r2d2.html).
- Memory Gym notes that learning initial hidden states is non-trivial under truncated gradients. It is also unclear how burn-in would initialise state at deployment without a replay buffer — [Pleines et al. 2022](https://ar5iv.labs.arxiv.org/html/2205.11104)
- **Forget-gate bias.** Adding a bias of 1 to the LSTM forget gate closes the gap between LSTM and GRU. The practice was advocated by Gers et al. (2000) but dropped in later papers. The authors recommend it before trying fancier architectures. The forget gate is "extremely significant" on all problems except language modelling — [Jozefowicz, Zaremba & Sutskever, ICML 2015](https://proceedings.mlr.press/v37/jozefowicz15.html)
- Memory Maze reports that truncated BPTT helped the Dreamer agent — [Pasukonis et al. 2022](https://arxiv.org/abs/2210.13383)

### Inferences
- [Inference, repo] Our scheme matches the MarcoMeter / Memory Gym "store every step's state, start each window from the stored state" design, not the baselines/CleanRL "store the rollout-start state and recompute" design. Consequences:
  - Window 256 = rollout length, so it is effectively the CleanRL scheme. It is fresh within the rollout, with only the rollout-start state one iteration stale.
  - Window 16 injects 16 stored (stale after epoch 1) states per rollout.

  So staleness is highest at window 16, which performed best. Staleness is therefore unlikely to explain why longer windows fail more. This holds if Pleines' "minor effect" result transfers. Kandou's result suggests checking KL and clip-fraction per window size anyway.
- [Inference, speculative] Why longer windows might fail more often:
  - (a) With equal gradient updates per window, a 256-step window gives fewer, longer, more correlated sequences per minibatch, i.e. fewer independent sequence starts. Gradient variance and norm grow with unroll length, which risks exploding gradients. Memory Gym hit exploding gradients from zero-init states.
  - (b) Longer unrolls put more steps under one PPO ratio, so the policy shifts more across a sequence for the same learning rate.
  - (c) Longer windows start more often from zero state at episode starts relative to carried state. Not directly applicable here, since episodes (320 steps) exceed the rollout length (256).

  Check gradient norm, approx-KL and clip-fraction curves per window size, plus failing vs solving seeds.
- [Inference] Default PyTorch init gives forget-gate and update-gate pre-activations near 0, so gates ≈ σ(0) = 0.5. A 0.5 retention per step erases a trace over 40+ steps (0.5^40 ≈ 1e-12). Neither LSTM nor GRU retains the cue "for free" at init; retention must be learned. A forget-gate bias of +1 (σ ≈ 0.73), or chrono initialisation, is a cheap intervention. For the GRU, the analogous move is biasing the update gate towards keeping the old state; mind PyTorch's sign convention, h' = (1−z)·n + z·h. This is the only literature-backed reason LSTM and GRU differ early in learning (Jozefowicz 2015). It is a supervised-learning finding, and its transfer to PPO is an untested hypothesis.
- [Inference, repo] Shared actor-critic backbone. Ni 2022 (off-policy) and SB3's default (separate critic LSTM) both point away from sharing. With gamma 0.999 and value targets in roughly [−3, 1], the value loss can dominate the shared recurrent gradient. That would shape the hidden state for value prediction, which is cue-independent here: both rooms have equal value before the decision, until memory is used. Testing a separate critic recurrent network, or a detached critic, is a cheap ablation that could reduce seed failures.
- [Inference] Seed-level significance (Fisher's exact, own calculation): LSTM-16 5/5 vs GRU 1/5 gives two-sided p ≈ 0.048. LSTM-16 5/5 vs LSTM-64 2/5 gives p ≈ 0.17. LSTM-16 vs LSTM-256 4/5 gives p = 1. Only the GRU gap is borderline. The window-size ordering (16 > 256 > 64) is not monotonic and is consistent with seed noise. More seeds are needed before interpreting it.
- [Inference] Eval starts from zero state, which matches training resets (zero state at `episode_starts`), so there is no train-eval mismatch from initialisation.

### Gaps
- R2D2 hyperparameters (sequence length, burn-in length) could not be verified from the primary PDF (OpenReview blocked). Secondary sources disagree: one cites sequence length 15 with burn-in 10, another cites 20-40-step burn-in being better. My recollection is length 80 / burn-in 40, unverified.
- POPGym (Morad et al., ICLR 2023, [arXiv:2303.01859](https://arxiv.org/abs/2303.01859)) is the largest memory-model comparison in RL. I could not confirm from search snippets whether GRU ranked best, as commonly cited.
- Not found: an RL study that varies LSTM forget-gate bias, or compares LSTM vs GRU specifically in early-learning PPO.
- Not checked: PyTorch's exact default init for `nn.LSTM`/`nn.GRU`. I believe it is U(−1/√H, 1/√H) for all weights and both bias vectors ([PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)), not re-fetched this session.

---

## 4. Credit assignment vs memory: does the hidden goal search lengthen the credit-assignment horizon?

### Takeaway
Ni et al. 2023 formally separate memory length (how far back an observation must be recalled) from credit-assignment length (how far an action's effect on reward is delayed). Transformers extend memory to about 1500 steps but do not improve long-term credit assignment, which degrades severely around 250 steps. In our task the memory length is cue → junction (more than 40 steps at L=32, growing with L). The baffle search adds a modest credit-assignment delay, junction → goal (roughly 15-30 steps at full speed). The two are separable factors and should be reported separately as L grows.

### Cited Findings
- Ni et al. define memory length and credit-assignment length formally and build configurable tasks that vary each separately. Transformers recall observations from about 1500 steps ago but do not improve long-term credit assignment. The slides report severe degradation once credit-assignment length reaches about 250 — [Ni, Ma, Eysenbach & Bacon, NeurIPS 2023 oral, arXiv:2307.03864](https://arxiv.org/abs/2307.03864v4); [NeurIPS slides](https://neurips.cc/media/neurips-2023/Slides/72542_gYjr3Ri.pdf); [code](https://github.com/twni2016/Memory-RL)
- Hung et al. (TVT) frame long-term credit assignment as linking actions to rewards after long delays, using a three-phase task: encode, distractor, exploit. Standard agents fail as the distractor delay grows. Temporal Value Transport, which uses memory reads to splice value back to distant steps, learned with a 60 s distractor — [Hung et al. 2019, Nature Communications, arXiv:1810.06721](https://arxiv.org/abs/1810.06721); [PMC full text](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6864102/)
- Memory Gym found recurrent PPO failed on full Searing Spotlights. It attributes this to noise and distraction that make credit assignment hard, not to memory alone — [Pleines et al. 2022](https://ar5iv.labs.arxiv.org/html/2205.11104)

### Inferences
- [Inference, repo geometry] Memory length ≈ (shaft 120 + corridor 13L)/13 ≈ L + 9 steps, plus alcove time. Credit-assignment length for the junction choice is the arm (60) plus baffle path to a goal 130 units into the room, roughly 15-30 steps if driven straight. During learning it is much longer, because the agent must first find the goal behind the baffle by exploration. Before a reliable search policy exists, the junction choice gets almost no differential signal: both rooms usually end in timeout or wrong-room termination. That is a credit-assignment and exploration bottleneck that precedes memory.
- [Inference] With truncated BPTT, the "memory length" that gradients can reach is the window length. The credit-assignment path is carried by the critic (GAE). Even if credit for the junction is assigned correctly, learning to *write* the cue requires the gradient to reach the cue observation, which is more than 40 steps back, beyond window 16. This links to the window-16 puzzle in Section 2 and argues for the probe and lesion diagnostics.
- [Inference] Scaling L from 8 to 256 increases memory length while the credit-assignment length of the junction decision stays roughly fixed, a clean "memory" sweep in Ni's sense. The credit assignment of the *cue-encoding* step, however, does grow with L: the reward that justifies attending to the cue arrives L+~30 steps later. With GAE λ (e.g. 0.95 → about 20-step effective trace), that signal reaches the cue step only through value bootstrapping. Value bootstrapping cannot differentiate the cue colour unless the hidden state already encodes it, a chicken-and-egg problem. Reporting results against both lengths would align with Ni 2023's framing.
- [Inference] Ni 2022 found past rewards as inputs help some POMDPs. Here the rewards are uninformative until success, so this is unlikely to help.

### Gaps
- Not extracted: the exact LSTM limits Ni et al. 2023 report for passive vs active T-mazes (e.g. at what memory length LSTM-based agents fail). That would calibrate the expected L range for LSTM/GRU.
- No literature found on goal search behind occluders (baffles) as a source of credit-assignment delay in navigation memory tasks. The point above is my own reasoning.
