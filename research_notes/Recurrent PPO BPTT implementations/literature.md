# BPTT length and sequence handling in recurrent PPO and recurrent model-free RL: literature

Evidence labels used below: **[on-policy]** PPO/A2C-family, **[off-policy]** replay-based, **[supervised/prediction]** no control loop. Figures labelled "read off plot" are approximate. Anything derived from the user's numbers is in Inferences, not Cited Findings.

## Which papers measure the effect of sequence length on recurrent PPO or A2C performance, and what did they find?

### Takeaway
Only one on-policy paper I found gives a clean table of performance by sequence length: Pleines et al. 2022, Appendix D. It shows a steep gain up to about the task's memory span, then a plateau, and almost no effect on a task where the cue only has to be maintained. Most other on-policy work fixes the length (usually long or full-episode), so the on-policy dose-response evidence is thin. Off-policy and prediction studies agree on the direction, but they also show the best length is task-specific.

### Cited Findings

**On-policy, with a measured sequence-length effect**

- [on-policy] Pleines et al. 2022 ran a grid search over sequence length {1, 8, 16, 32, 64, Max}, hidden size {64 to 1024} and layer type {GRU, LSTM} on Mortar Mayhem (200 seeds) and Ballet (500 seeds). Each permutation was trained 3 times for about 27M steps and evaluated on 5 training and 5 novel seeds. They conclude that longer sequences and larger hidden states help, and that GRU usually beats LSTM. — [Pleines 2022, App. D](https://arxiv.org/abs/2205.11104)
- [on-policy] Mortar Mayhem, training seeds, fraction succeeding on at least N commands (rows read from Table 2, which appear to be marginals over the other two factors; "Max" means full episode):

| Seq. length | at least 1 | at least 5 | at least 8 | at least 9 | novel seeds, at least 1 |
|---|---|---|---|---|---|
| 1 | 0.77 | 0.17 | 0.00 | 0.00 | 0.23 |
| 8 | 0.93 | 0.36 | 0.03 | 0.02 | 0.28 |
| 16 | 0.87 | 0.45 | 0.14 | 0.11 | 0.30 |
| 32 | 0.99 | 0.67 | 0.40 | 0.29 | 0.69 |
| 64 | 0.99 | 0.77 | 0.54 | 0.45 | 0.63 |
| Max | 0.99 | 0.76 | 0.55 | 0.49 | 0.65 |

  — [Pleines 2022, Table 2](https://arxiv.org/abs/2205.11104)
- [on-policy] Ballet, training seeds, success at 8 dances by sequence length 1/8/16/32/64/Max: 0.64, 0.72, 0.72, 0.77, 0.77, 0.80. Novel seeds at 8 dances are flat at 0.14 to 0.17. At 2 dances, training success is 0.93 to 0.97 at every length. — [Pleines 2022, Table 3](https://arxiv.org/abs/2205.11104)
- [on-policy] Same grid, layer type: Mortar Mayhem at least 5 commands was GRU 0.55 vs LSTM 0.51; Ballet 8 dances was GRU 0.81 vs LSTM 0.66. — [Pleines 2022, Tables 2 and 3](https://arxiv.org/abs/2205.11104)
- [on-policy] Final Pleines 2022 settings: Mortar Mayhem and Ballet use sequence length Max, Searing Spotlights uses 128, all with 512 worker steps, 8 minibatches, 16-32 workers, 3-4 epochs, GRU 512. — [Pleines 2022, Table 1](https://arxiv.org/abs/2205.11104)
- [on-policy] Memory Gym (follow-up, Optuna search) sets sequence length from the longest possible episode in finite environments and reports 512 as the best length so far for GRU in endless environments. Sequence length is not one of the searched dimensions in Table 4, so this is a design choice, not a dose-response result. — [Pleines et al. 2025, App. B](https://arxiv.org/abs/2309.17207)
- [on-policy] Elelimy et al. 2024 compare GRU and LRU under T-BPTT with PPO on masked MuJoCo (Brax), POPGym and a Reacher POMDP, with truncation values including 4 and 64, and state that 64 is more than these MuJoCo tasks need. They note that with non-overlapping sequences only the last sample in each sequence gets a full T-step gradient, and that all standard RL implementations use this cheap option. PPO settings: buffer 2048, 10 epochs, 32 minibatches, step size swept over 1e-5 to 1e-3. — [Elelimy 2024, main text, App. H and Table 1](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e616bde0438cb10cb6adf076ae7d336-Paper-Conference.pdf)

**On-policy, length fixed (useful as settings, not as evidence of effect)**

- [on-policy] POPGym: PPO with BPTT truncation listed as infinite, max episode 1024, batch 65536, minibatch 8192, 30 SGD iterations, lr 5e-5, recurrent state size 256, with over 1700 trials. Authors say they chose conservative settings by trial and error rather than structured tuning. Mean normalized MMER (first column all environments, second apparently excluding navigation, from a fragmentary table header): GRU 0.349/0.326, LSTM 0.255/0.294, frame stack 0.190/0.177, MLP 0.067/-0.010. — [Morad 2023, Fig. 3 and Table 2](https://arxiv.org/abs/2303.01859)
- [on-policy] POPGym found the MLP scored highest on almost all navigation tasks and argues that memory-free wall following explains this, which casts doubt on navigation as a memory benchmark. — [Morad 2023, Sec. 2.3 and 6](https://arxiv.org/abs/2303.01859)
- [on-policy] OpenAI Five used truncated BPTT over 16 timesteps with a 4096-unit LSTM and batches of 120 such samples per GPU. — [OpenAI 2019](https://arxiv.org/abs/1912.06680)
- [on-policy] MAPPO uses chunk length 10 and does not ablate it, so it reports no chunk-length effect. — [Yu 2022, Table 7 and App. C.1](https://arxiv.org/abs/2103.01955)
- [on-policy] DD-PPO PointGoal navigation: 2-layer LSTM with 512 hidden, up to 128 frames from 4 agents per worker, 2 epochs, 2 minibatches, lr 2.5e-4, no advantage normalization. — [Wijmans 2020](https://arxiv.org/abs/1911.00357)
- [on-policy] Sample Factory says rollout length T is generally wanted in the 32 to 128 range for BPTT with recurrent policies; its VizDoom table uses T=32, batch 2048, 1 epoch. — [Petrenko 2020](https://arxiv.org/abs/2006.11751)
- [on-policy] CleanRL's LSTM PPO for Atari trains on full 128-step rollouts with 8 envs, 4 minibatches, 4 epochs, lr 2.5e-4, so each minibatch holds 2 whole sequences. — [CleanRL ppo_atari_lstm.py](https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_atari_lstm.py)

**Off-policy and prediction, for direction**

- [off-policy] Ni et al. 2022 treat context length as a key choice, report that the best value is task-specific and that prior POMDP methods used 1 to 2048. A medium length of 64 dominates most best variants, but raising or lowering it can each help, and reducing 64 to 5 gave about a 2.2x gain on Walker-V. — [Ni 2022, Sec. 4 and 5.2](https://ar5iv.labs.arxiv.org/html/2110.05038)
- [off-policy] Ni et al. 2022 also note that the on-policy methods PPO-GRU and RL2 beat their off-policy implementation on Walker-V and Ant-Dir, while their implementation beats PPO-GRU and A2C-GRU on the plotted benchmarks in their Figure 1. — [Ni 2022, Sec. 5.2](https://ar5iv.labs.arxiv.org/html/2110.05038)
- [off-policy] Morad et al. 2024 compare segment-based batching at L=10, 20, 50, 100 with untruncated tape-based batching across POPGym tasks over ten seeds, with Double Dueling DQN named for the truncation experiment. Larger L always beat shorter L, and a significant gap to no truncation remained even at L=100. They conclude that gradient error matters more than the smaller effective batch from padding. — [Morad 2024](https://arxiv.org/abs/2402.09900)
- [off-policy] Hausknecht and Stone 2015 report only unroll 10 for performance. Their unroll 1/10/30 table is timing (backward pass 18.2/77.3/204.5 ms at 1 frame), so DRQN does not measure a length effect. They found sequential full-episode and random zero-start updates perform similarly across a set of games. — [DRQN](https://arxiv.org/abs/1507.06527)

### Inferences
- The Mortar Mayhem table lines up with memory span. Using the paper's default timing (2 steps per shown command, 5 per executed command), command k of 10 is executed about 20+3k steps after being shown, so 23 to 50 steps. The large jump is between 16 and 32 and saturation occurs by 64, which brackets that span. This reading is mine, not the authors', and assumes the grid used the default 10 commands.
- For the user's study: 16 vs 64 could behave very differently while 64 vs 256 may not, if the needed horizon sits below 64. Pleines is the closest analogue for the shape, but its tasks are memory-cued gridworlds with 16-32 workers.
- The Ballet row warns that a task can need memory yet be insensitive to window length when the cue only needs maintaining, not updating. Navigation that uses memory only to carry a goal or room identity may resemble Ballet more than Mortar Mayhem.
- POPGym's navigation result means an LSTM-vs-MLP gap in navigation should not be assumed, and any null window effect should be checked against the possibility that the task does not require memory at that scale.

### Gaps
- I found no on-policy paper with a dedicated sequence-length dose-response study beyond the Pleines appendix grid. That grid has 3 repeats per cell and no confidence intervals in the table.
- Elelimy's truncation sweeps are in figures; I did not extract numeric values from them.
- No study reports BPTT dose-response for PPO in continuous 2D ray-cast navigation.
- Andrychowicz 2021 does not address this: its text contains no mention of recurrent networks or LSTMs. Engstrom 2020 is also feedforward.

## What do R2D2 and later work say about stale hidden states and burn-in, and does it apply to on-policy methods?

### Takeaway
Stale state is clearly established off-policy, where stored state beats zero state and burn-in adds a further gain. On-policy evidence is mixed and small: one blog shows a large effect on a memory-heavy task with 3 seeds, and Pleines 2022 finds refreshing not worth its cost. No paper I found tests burn-in inside PPO. All of that on-policy evidence comes from low update counts per iteration, so it may not carry over to a much higher update count.

### Cited Findings
- [off-policy] R2D2 replays 80-step sequences overlapping by 40 and never crossing episode boundaries, with optional burn-in of 20 or 40, in batches of 64 sequences. It measures staleness as a normalized Q-value discrepancy between stored-state and recomputed unrolls. — [Kapturowski 2019, review draft](https://openreview.net/references/pdf?id=Hy7PKCFCQ)
- [off-policy] On DMLab tasks, zero state gives much worse staleness than stored state. The effect shrinks a lot by the last sequence states but remains worse for zero state. Burn-in partially helps on early states of a sequence, with no significant effect on later ones, yet improves performance. Stored state plus burn-in gives the smallest discrepancy and most robust gains, and they use burn-in 40. — [Kapturowski 2019, Sec. 3](https://openreview.net/references/pdf?id=Hy7PKCFCQ)
- [off-policy] The authors attribute burn-in's value to preventing destructive updates caused by badly wrong early outputs after a zero start. Longer zero-state unrolls did not reproduce the gain. — [Kapturowski 2019, Sec. 3 and 6](https://openreview.net/references/pdf?id=Hy7PKCFCQ)
- [on-policy] Pleines 2022 refreshed hidden states and advantages before each epoch or every other epoch, with 3 to 8 epochs. Every-other-epoch refresh gave a slight gain on Mortar Mayhem novel seeds and Ballet training seeds, judged not worth the extra forward pass. Using more epochs (8) severely degraded performance in both environments. Refreshing did not change gradient norm or KL in a way that explained anything. — [Pleines 2022, Sec. 5.3, 6.3 and App. E](https://arxiv.org/abs/2205.11104)
- [on-policy] Pleines 2022 suggest on-policy data may suffer less from staleness than off-policy data, and leave the question open. They also note that the discrepancy with Ndousse might relate to advantage normalization scope. — [Pleines 2022, Sec. 6.3](https://arxiv.org/abs/2205.11104)
- [on-policy] Ndousse 2020 refreshed LSTM states every few minibatches in PPO-Clip. On a cluttered environment, policy divergence grew without refresh but returns were similar. On a memory-heavy goal-cycle gridworld, refresh had a large effect, with the divergence scale about an order of magnitude higher. This is a blog post with 3 seeds and no numeric return values; BPTT length and batch settings are not stated. — [Ndousse 2020](https://kandouss.github.io/blog/ppo_stale_states)
- [on-policy] Memory Gym flags that with 512-step rollouts, cached states of episodes longer than a rollout become outdated over several PPO iterations, and asks for future work on this. This passage concerns transformer memory, not RNN state. — [Pleines 2025, Sec. 5.6.4](https://arxiv.org/abs/2309.17207)
- [on-policy] OpenAI Five found parameter staleness of about 8 versions caused significant slowdowns, and that reusing data 2 to 3 times could slow learning by a factor of two, with 8 times possibly preventing competent play. They targeted reuse near 1. This is asynchronous data lag, not hidden-state staleness. — [OpenAI 2019, data-quality section and App. M](https://arxiv.org/abs/1912.06680)
- [on-policy] Hilton et al. 2022 found even small artificial staleness hurt standard PPO on 16 Procgen environments, but decoupling the proximal policy from the behaviour policy kept performance robust until about 8 iterations of staleness. This work is feedforward. — [Hilton 2022, Sec. 5.1](https://arxiv.org/abs/2110.00641)
- [on-policy] Elelimy et al. 2024 report that in RTRL with PPO, stale gradient traces did better than recomputed ones on Ant-P at 4 and 8 epochs, and hypothesize that staleness helps PPO keep its trust region. Recomputing targets and advantages had minimal effect. This is stale RTRL gradient traces, not stale start states under T-BPTT. — [Elelimy 2024, App. G](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e616bde0438cb10cb6adf076ae7d336-Paper-Conference.pdf)
- [on-policy] Andrychowicz 2021 found stale advantages can hurt in feedforward MuJoCo PPO, and recomputing them once per data pass did best among minibatch variants. Pleines 2022 did not reproduce a benefit in recurrent PPO. — [Andrychowicz 2021, Sec. 3.5](https://arxiv.org/abs/2006.05990); [Pleines 2022, Sec. 6.3](https://arxiv.org/abs/2205.11104)
- [on-policy] Standard recurrent PPO recipes store the state at the start of each sequence and reuse it across all updates: CleanRL saves the initial LSTM state per rollout, and Pleines' buffer keeps the first step's hxs/cxs per sequence. — [37 details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/); [Pleines buffer.py](https://raw.githubusercontent.com/MarcoMeter/recurrent-ppo-truncated-bptt/main/buffer.py)
- Pleines 2022 note a practical obstacle to burn-in and learned initial states: gradients are truncated, so they cannot reach the state's origin, and they list a stochastic sliding window and overlapping sequences as future work. — [Pleines 2022, Sec. 6.4](https://arxiv.org/abs/2205.11104)

### Inferences
- R2D2's mechanism partly transfers. The replay age that drives R2D2's staleness is absent from PPO, where data is one iteration old, but drift within an iteration remains, and it scales with updates per iteration.
- Staleness is tied to window length, not independent of it. By R2D2's result the mismatch sits at the start of a sequence and decays along it, so at w=16 about 1 sample in 16 is a window start, versus 1 in 256 at w=256. Short windows expose more of the batch to stale starts.
- Pleines' null result comes from about 24 to 32 updates per iteration. Moving to 512 updates per iteration would be about 20x more drift per iteration, so that null may not hold. This applies most to w=256, which would go from 8 to 512 updates per iteration.
- The stored state from the collection policy is still usually the better default: zero start within an episode is the off-policy failure case, and the user's code already stores states.

### Gaps
- No on-policy paper tests burn-in. PPO loss masking over a burn-in prefix is untested in the literature I found.
- No study measures a Q-discrepancy analogue for PPO as a function of window length or update count.
- I could not get a numeric staleness result from R2D2's Figure 1, as it is given as curves. I read the anonymous ICLR review draft, not the final proceedings copy.

## What does work on minibatch size, epochs and gradient steps in PPO imply for comparing configs with very different update counts?

### Takeaway
Update count, minibatch size and learning rate are coupled, and studies warn against varying one while holding the others fixed. Learning rate is the strongest knob, and for Adam the batch-size compensation follows a square root. No feedforward study covers a 16x spread in samples per update paired with one learning rate, and none is recurrent. There is also real evidence that more updates are not simply better.

### Cited Findings
- [on-policy] Andrychowicz et al. 2021 warn that learning rate and minibatch size interact, so varying one with the other fixed may mislead, and they include learning rate in every experiment group. Study: 5 MuJoCo tasks, feedforward, about 2000 configurations per task. — [Andrychowicz 2021, Sec. 2](https://arxiv.org/abs/2006.05990)
- [on-policy] Same study, training-setup group over num_envs {64, 128, 256}, iteration size {512 to 4096}, epochs {1, 3, 10}, batch size {64, 128, 256}, Adam lr {3e-5 to 3e-3}. Multiple passes are crucial for sample complexity. More environments can sharply reduce performance on some tasks, likely from shortened experience chunks and earlier bootstrapping. Larger batch size did not hurt sample complexity in the tested range. Iteration size matters quite a lot. They recommend shuffling individual transitions. — [Andrychowicz 2021, Sec. 3.5 and App. H](https://arxiv.org/abs/2006.05990)
- [on-policy] Hilton et al. 2022: Adam is batch-size invariant, up to a critical batch size, if the step size is divided by the square root of c when batch size is divided by c. They support the square-root rule over the linear rule empirically. — [Hilton 2022, Sec. 3.2 and App. C](https://arxiv.org/abs/2110.00641)
- [on-policy] Hilton's invariance needs more than the learning rate: standard PPO is not batch-size invariant, because its proximal policy is tied to the iteration. It also requires 1 policy epoch, sufficiently small batches, a matched proximal EWMA and a rescaled advantage-normalization window. With these, PPG-EWMA on Procgen (hard mode, 100M steps, 256 down to 1 parallel environments) shows a final normalized return spread of 0.052, or 0.019 without the outlier Heist. — [Hilton 2022, Sec. 4 and 5.2](https://arxiv.org/abs/2110.00641)
- [on-policy] In Hilton's ablations, the Adam step adjustment matters most at every batch size and training is highly unstable at small batch sizes without it. Advantage normalization adjustment matters a lot at the smallest batch sizes on some environments. — [Hilton 2022, Sec. 5.2](https://arxiv.org/abs/2110.00641)
- [on-policy] Hilton's advice: learning rate is by far the most important setting, 1 epoch is a sensible start, clip fraction well under 1% suggests a larger iteration batch, and well over 10% (1 epoch) or 20% (multiple epochs) suggests the rate is too high. Their default rollout geometry is 256 envs by 256 steps (65,536 samples) with 8 minibatches per epoch and Adam 5e-4, on a CNN. — [Hilton 2022, practical advice section and App. B](https://arxiv.org/abs/2110.00641)
- [on-policy] Dossa et al. 2021 report that PPO performance is sensitive to the number of update iterations per epoch K, and that KL early stopping reduces this sensitivity by adjusting iterations dynamically. I could only read the abstract (via OpenAlex), and no K values are in it. — [Dossa 2021](https://doi.org/10.1109/ACCESS.2021.3106662)
- [on-policy] MAPPO (multi-agent, recurrent chunk length 10): sample reuse hurts, 15 epochs gives suboptimal policies on hard SMAC maps, one minibatch gives best performance on 22 of 23 maps, and with 4 minibatches it fails on the selected maps. Their guidance is to avoid splitting data into minibatches on hard problems. On simple MPE tasks settings barely matter. — [Yu 2022, Sec. 5 and App.](https://arxiv.org/abs/2103.01955)
- [on-policy] Engstrom et al. 2020 show code-level choices in PPO (including orthogonal init, LR annealing) drive much of its performance. Their ablation grid-searches the learning rate separately for each configuration. No recurrent models and no update-count factor. — [Engstrom 2020, Sec. 3](https://arxiv.org/abs/2005.12729)
- [on-policy] Memory Gym's final settings use 3 epochs and 8 minibatches (2 and 4 epochs were searched, and 4 minibatches), and it found turning off advantage normalization important in low-return tasks. DD-PPO also avoids normalization, citing instability. — [Pleines 2025, Sec. 5.7 and App. B](https://arxiv.org/abs/2309.17207); [Wijmans 2020](https://arxiv.org/abs/1911.00357)
- [off-policy] Luo et al. 2024 find that RNN outputs change more per parameter step as the sequence grows, because altered hidden states feed back at each step. Best encoder lr was 5e-6 to 1e-5 with 3e-4 elsewhere on WalkerBLT-V. Using 3e-4 for both hurt performance and produced NaN or infinite outputs in some AntBLT-V and HalfCheetahBLT-V runs. Tasks: 18 POMDP plus 5 MDP, full-length trajectories, 6 seeds. — [Luo 2024](https://arxiv.org/html/2405.15384v1)
- [off-policy] pomdp-baselines' config sets 64-step sequences and a batch of 64 with the comment "to tune based on sampled_seq_len". — [pomdp-baselines config](https://raw.githubusercontent.com/twni2016/pomdp-baselines/main/configs/pomdp/ant_blt/p/rnn.yml)

### Inferences
Reference points, computed from the settings above (lr×U per iteration is a crude bound on Adam path length, not a measured quantity):

| Config | Samples/iter | Updates/iter U | Samples/update | Sequences/update | lr | lr×U | lr×U per 1k samples |
|---|---|---|---|---|---|---|---|
| Pleines 2022 Mortar | 16,384 | 32 | 2,048 | 2,048/L | 3e-4 | 0.0096 | 0.00059 |
| Memory Gym | 16,384 | 24 | 2,048 | 4 (L=512) | 2.75e-4 | 0.0066 | 0.00040 |
| POPGym | 65,536 | 240 | 8,192 | not stated | 5e-5 | 0.012 | 0.00018 |
| CleanRL Atari LSTM | 1,024 | 16 | 256 | 2 (L=128) | 2.5e-4 | 0.004 | 0.0039 |
| User, w=16 now | 65,536 | 128 | 1,024 | 64 | 3e-4 | 0.0384 | 0.00059 |
| User, w=64 now | 65,536 | 32 | 4,096 | 64 | 3e-4 | 0.0096 | 0.00015 |
| User, w=256 now | 65,536 | 8 | 16,384 | 64 | 3e-4 | 0.0024 | 0.00004 |
| User, MLP now | 65,536 | 512 | 256 | shuffled | 1e-4 | 0.0512 | 0.00078 |
| User proposal, all w | 65,536 | 512 | 256 | 16 / 4 / 1 | 3e-4 | 0.1536 | 0.0023 |

(Updates/iter counts all epochs. The user's numbers imply 64 envs per minibatch: 2 × 65,536 / (64w) gives 128, 32, 8. CleanRL's rate anneals, so its figure is an upper value.)

- The present design gives a 16x spread in lr×U across windows. The current w=64 cell happens to match Pleines' Mortar value exactly.
- The proposal fixes the confound but moves all LSTM cells to about 13 to 23 times the per-iteration lr×U of POPGym, Pleines and Memory Gym, and 3 times the user's own MLP. Per environment sample it sits between those references and CleanRL, so it is not absurd, but it is outside every recurrent PPO config above on a per-iteration basis, and per-iteration is what bears on trust-region drift and stale state.
- Under Hilton's square-root rule, re-expressing the present cells at 256 samples and 512 updates gives equivalent Adam rates of 1.5e-4, 7.5e-5 and 3.75e-5 for w=16, 64, 256. If that extrapolation holds, the present design compares effective step sizes that differ by 4x. It also suggests that 3e-4 at 256 samples is 2x to 8x hotter than the current cells. This extrapolates beyond Hilton's conditions (1 epoch, EWMA proximal policy, feedforward), so treat it as an order-of-magnitude guide.
- Luo's amplification argument, if it carries to PPO, says a fixed learning rate moves the policy further per update at longer windows. So "same lr across windows" is not automatically neutral.
- MAPPO and Andrychowicz pull in opposite directions on large batches with few updates, so w=256 at 8 updates may be slower or may be fine. Literature does not settle which, and direction in this setup is an empirical question.

### Gaps
- No study varies updates per iteration by a factor of 64 in recurrent PPO.
- Dossa 2021 was read at abstract level only (IEEE full text would not load), so I cannot give K values or tasks.
- MAPPO, Luo 2024 and Ni 2022 passages were read through HTML extraction with quoted snippets, not full-text reading of every table. The MAPPO note that Table 17 uses 2 minibatches for MMM2 sits awkwardly with its general advice and is unexplained.
- Evidence on Adam's critical batch size for PPO at 16,384 samples is missing.

## Do any studies report holding update count constant when varying sequence length, or warn about this confound?

### Takeaway
I found no study that names this confound or reports holding update count constant as a deliberate control. One major on-policy reference holds it constant by construction: Pleines' code fixes the number of minibatches, so sequence length changes only how many sequences fit into each update. The proposal is therefore aligned with existing practice, though the precedent is implicit.

### Cited Findings
- [on-policy] Pleines' repository sets the number of minibatches from config, independent of sequence length, and gives each minibatch num_sequences // n_mini_batches sequences. Table 1 lists 8 minibatches and batch 16384. So the sequence-length grid ran at near-constant updates and near-constant samples per update (about 2048), modulo padding and any grid-specific settings not listed in the paper. — [buffer.py](https://raw.githubusercontent.com/MarcoMeter/recurrent-ppo-truncated-bptt/main/buffer.py); [Pleines 2022, Table 1](https://arxiv.org/abs/2205.11104)
- [on-policy] The paper itself does not discuss this as a control, and its appendix states only the factors in the grid. — [Pleines 2022, App. D](https://arxiv.org/abs/2205.11104)
- [on-policy] CleanRL also fixes num_minibatches and derives envs per minibatch from it. SB3-contrib's RecurrentPPO batches by a step count (batch_size=128 default). — [CleanRL](https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_atari_lstm.py); [SB3-contrib](https://raw.githubusercontent.com/Stable-Baselines-Team/stable-baselines3-contrib/master/sb3_contrib/ppo_recurrent/ppo_recurrent.py)
- [off-policy] Off-policy recurrent work fixes the count of sequences per batch (R2D2: 64 of length 80; pomdp-baselines: 64) with updates set by a replay ratio, so longer sequences add samples per update without reducing updates. — [Kapturowski 2019](https://openreview.net/references/pdf?id=Hy7PKCFCQ); [pomdp-baselines config](https://raw.githubusercontent.com/twni2016/pomdp-baselines/main/configs/pomdp/ant_blt/p/rnn.yml)
- [off-policy] Ni 2022 varied gradient steps by benchmark mainly to fit a 72-hour budget and did not tune them otherwise. — [Ni 2022, appendix](https://ar5iv.labs.arxiv.org/html/2110.05038)
- [off-policy] Morad 2024 name a related trade-off: longer segments give better gradients but smaller effective batches because of zero padding. — [Morad 2024, Sec. 4](https://arxiv.org/abs/2402.09900)
- Elelimy et al. 2024 discuss the compute side: under non-overlapping T-BPTT the per-sample update cost is independent of T. — [Elelimy 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e616bde0438cb10cb6adf076ae7d336-Paper-Conference.pdf)
- [on-policy] Andrychowicz's general warning about interacting choices is the nearest explicit statement. — [Andrychowicz 2021, Sec. 2](https://arxiv.org/abs/2006.05990)

### Inferences
- The user's 128/32/8 pattern is specific to combining the off-policy convention (fixed sequence count) with an on-policy data budget. Off-policy, a fixed count of sequences leaves updates alone. On-policy, with a fixed rollout, it forces updates to scale as 1/w.
- Confounds that a window sweep carries under fixed envs per minibatch: updates per iteration, samples per update, lr×U, advantage-normalization sample, gradient noise and intra-iteration drift. Holding updates and samples per update equal removes the first three, not the rest.
- Equal samples per update does not mean equal information. A 256-sample minibatch of shuffled MLP transitions differs from one 256-step trajectory. R2D2 flags highly correlated states in whole trajectories as a source of update variance, and Andrychowicz's transition shuffling works by increasing diversity.

### Gaps
- I searched for an explicit statement of this confound and found none. Absence from my search is not proof of absence, particularly in theses or code comments.
- The Pleines paper does not publish per-cell settings for the grid, so I infer constancy from Table 1 and the code.

## How does a truncation window relate to the needed memory horizon, and can an LSTM use information older than its BPTT window?

### Takeaway
Yes, an LSTM can carry and use information older than its window, because the forward state is not truncated, but learning to store a cue for a span longer than the window is unreliable. Evidence supports setting the window at or above the memory span, and a window shorter than the span may still work on an easy retention task. Two distinct questions should be kept apart: what the state carries forward, and what gradients can teach it to keep.

### Cited Findings
- [supervised/prediction] Truncation biases the gradient and favours short-term dependencies. Truncated BPTT carries hidden state between subsequences but cuts gradient flow, and can have trouble with dependencies past its range. On a synthetic balancing task it was unreliable and sometimes diverged even at large truncation lengths. — [Tallec and Ollivier 2017](https://arxiv.org/abs/1705.08209)
- [prediction] Rafiee et al. 2022 (online prediction, trace conditioning): with expected ISI 20, LSTM and GRU needed T at or above 20 to match RTRL. With ISI 10, all T values worked. With ISI 30, no T-BPTT variant did well even with T above the ISI. They state that most recurrent systems cannot learn relationships longer than T. — [Rafiee 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9814020/)
- [prediction/on-policy] Elelimy et al. 2024: GRU performance improved with T, and T-BPTT approached RTRL as T grew to cover the task's whole context. Their prediction work fixes T=45 as exceeding the maximum CS-US distance. — [Elelimy 2024, Sec. 3 and App.](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e616bde0438cb10cb6adf076ae7d336-Paper-Conference.pdf)
- [off-policy] Morad 2024 argue that dependencies longer than L are unlikely to be learned given zero gradient across segments. On Repeat Previous with required memory 10 and L=10, the value function nevertheless depended on observations far outside the segment: for FFM, about 90% of Q value came from outside it. They read this as truncation leaving models unable to ignore old input. — [Morad 2024, Sec. 4](https://arxiv.org/abs/2402.09900)
- [off-policy] R2D2's evaluation-time restriction zeroes the state k steps back. Stored-state agents lose performance faster than zero-state agents as k falls, and the authors read this as stored state enabling real memory use. Read off Figure 5 (approximate): Ms. Pac-Man stored state falls from about 21k at full history to about 15k at k=120 and 9k at k=80, while training used 80-step sequences with 40 burn-in. — [Kapturowski 2019, Sec. 5](https://openreview.net/references/pdf?id=Hy7PKCFCQ)
- [on-policy] Pleines' sequence length 1 row has no multi-step gradient at all, but stored state is still fed forward. It reaches 0.64 at Ballet 8 dances (training) and 0.17 at 5 or more Mortar Mayhem commands, against 0.80 and 0.76 for full episodes. — [Pleines 2022, App. D](https://arxiv.org/abs/2205.11104)
- [off-policy] Ni et al. 2023 separate memory length (how far back the optimal policy must look) from credit assignment length (how long after an action its reward shows). With full-episode context, Transformer agents solve Passive T-Maze at memory lengths 50 to 1500, while both Transformer and LSTM agents fail at credit assignment length 250 in Active T-Maze. LSTM had NaN gradients in about 30% of runs at memory length 1000. — [Ni 2023, Sec. 3 to 5](https://arxiv.org/abs/2307.03864)
- [on-policy] Memory Gym: GRU beat TrXL by large margins on endless tasks with sequence length 512, despite a 256-step TrXL memory window, finishing at 23.9 coins vs 6.7 on the task reported in Sec. 5.4. Long-range use of recurrent state is therefore compatible with a fixed 512 window. — [Pleines 2025, Sec. 5.4](https://arxiv.org/abs/2309.17207)

### Inferences
- With non-overlapping windows, a sample has on average (w-1)/2 earlier steps in its gradient path: 7.5, 31.5 and 127.5 for w=16, 64, 256. So "window 16" is closer to 8 in practice, in line with Elelimy's point about the last sample.
- For a cue at distance H from the decision and uniform placement, the chance that both lie in one window is max(0, 1-H/w). For H=16: 0%, 75%, 94%. For H=64: 0%, 0%, 75%. Random window offsets per epoch would not change this average, but would stop the same boundaries being excluded each epoch.
- The window cannot exceed the rollout. w=256 is a full rollout, and with episodes up to 4096 steps, dependencies longer than 256 are never directly trained for any setting in this study. It is worth reporting the study's actual horizon coverage against the task's.
- Pleines' sequence-length-1 result is striking but not proof of learned storage. Random recurrent dynamics may retain enough for a trained readout. Reading the row this way is my interpretation, and the row is a marginal.
- R2D2's plot reading suggests dependence beyond the training sequence, but zeroing state is itself an atypical start, so some of the fall at k=120 may be state mismatch rather than lost information.
- Memory horizon and credit assignment horizon should be separated. A window gives gradients only for memory; γ and GAE-λ govern credit. Windows may not help if the bottleneck is credit, as Ni 2023 suggests.

### Gaps
- I found no on-policy study with a controlled H-vs-w factorial design.
- Rafiee is online prediction with small networks, so mapping T≥H to PPO is by analogy.
- No published result tests whether PPO with short windows learns to retain cues via carried state, versus exploiting cue persistence or other strategies.

## Is there evidence on LSTM initialisation, forget-gate bias or layer norm for PPO that would matter here?

### Takeaway
There is no controlled PPO evidence on forget-gate bias, chrono initialisation or LSTM layer norm. Strong supervised evidence says forget-gate bias matters for long dependencies, and this interacts with a window comparison, because a naive bias may cap effective gradient reach regardless of window. PPO implementations commonly use zero bias, and layer norm's support is limited to feedforward PPO.

### Cited Findings
- [supervised] Jozefowicz et al. 2015: small random init gives forget gates near 0.5, causing a vanishing factor of 0.5 per step. Initialising the forget bias to 1 or 2 fixes this, and with bias 1 the LSTM nearly matched GRU, which otherwise beat it on nearly all tasks except language modelling. — [Jozefowicz 2015, Sec. 2.2](https://proceedings.mlr.press/v37/jozefowicz15.pdf)
- [supervised] Tallec et al. 2018: chrono init sets forget bias to log(U[1, Tmax-1]) and input bias to its negative, for expected longest range Tmax. It heavily beat standard init on synthetic long-range tasks, and matched or beat it on real-world tasks. No RL tests. — [Tallec 2018](https://arxiv.org/abs/1804.11188)
- [on-policy] CleanRL's LSTM uses orthogonal weights (gain 1.0) and all biases 0, so no forget bias. The 37-details post documents the same. — [CleanRL](https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_atari_lstm.py); [37 details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [on-policy] Pleines 2022 Appendix A warns against adding another activation on top of LSTM or GRU output, and describes bugs from passing None as hidden state and from sequence/batch dimension order. Appendix F finds last action usually useful as input, last reward not. — [Pleines 2022, App. A and F](https://arxiv.org/abs/2205.11104)
- [on-policy] GRU tends to beat LSTM in recurrent PPO: Pleines (above), POPGym's advice that there is little reason to use LSTM, and Memory Gym's note that LSTM mostly underperformed when applied directly without tuning. — [Pleines 2022](https://arxiv.org/abs/2205.11104); [Morad 2023, Sec. 6](https://arxiv.org/abs/2303.01859); [Pleines 2025, Sec. 5.1](https://arxiv.org/abs/2309.17207)
- [on-policy] Juliani and Ash 2024 (PPO, MLP and CNN only, no recurrent models): LayerNorm before ReLU resolves plasticity loss in training but acts inconsistently on generalization. In a gridworld permute condition it was much worse at test than the reset baseline (t(8)=6.959, p<0.001). — [Juliani 2024](https://arxiv.org/html/2405.19153v2)
- [on-policy] Memory Gym found pre-layer-norm was needed for TrXL, as post-norm caused vanishing gradients. This is transformer evidence. — [Pleines 2025, Sec. 4.3](https://arxiv.org/abs/2309.17207)
- [on-policy] Andrychowicz 2021 recommends a 100x smaller last policy layer, for feedforward PPO. — [Andrychowicz 2021, Sec. 3](https://arxiv.org/abs/2006.05990)

### Inferences
- Back-of-envelope gate-only decay across a window (ignoring weights and input-dependent gates): bias 0 gives 0.5^16, about 1.5e-5. Bias 1 gives about 0.0066 at 16 steps. Bias 2 gives about 0.13 at 16 and 3e-4 at 64. So with the usual zero bias, early training might see w=64 and w=256 add little reach, and a null result could reflect init rather than task. Gates learn, so this is mainly an early-training effect.
- In the repo, a grep of `lstm_backbone.py`, `recurrent_agent.py` and `ppo.py` for "forget" and "orthogonal" found nothing. Init could live elsewhere, so this needs checking, not assuming.
- If bias changes, it should change for all windows. Chrono with Tmax near the task horizon is principled, but is untested in PPO, so it would count as a separate ablation, not a default.
- Layer norm inside the LSTM would be an unvalidated change to the study. There is no evidence that justifies adding it to a controlled comparison.

### Gaps
- No PPO study of forget-gate bias, chrono, or LSTM layer norm was found, and no navigation-specific study.
- I saw Lyle et al. 2023 on plasticity and layer norm only as a search summary, so it is not cited as read.
- A CleanRL issue argues PyTorch's fused LSTM weights need per-gate orthogonal init. I saw this via a search summary only, and I found no measured effect.

## What is good practice for a controlled comparison of BPTT windows?

### Takeaway
Hold updates per iteration and samples per update equal, as Pleines' code does, but prefer a moderate value where w=256 still gets several sequences per minibatch. Sweep learning rate at least coarsely, and report drift diagnostics plus history-restricted evaluation. Literature supports the parts of this design; nobody has validated the design as a whole.

### Cited Findings
- [on-policy] Interacting choices should be varied together, with learning rate in each group. — [Andrychowicz 2021, Sec. 2](https://arxiv.org/abs/2006.05990)
- [on-policy] Engstrom's PPO ablation searches the learning rate separately per configuration, a precedent for not sharing one rate across variants. — [Engstrom 2020, Sec. 3](https://arxiv.org/abs/2005.12729)
- [on-policy] The smallest established sequence count per minibatch I found is 2 (CleanRL, L=128) and 4 (Memory Gym, L=512). — [CleanRL](https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_atari_lstm.py); [Pleines 2025, App. B](https://arxiv.org/abs/2309.17207)
- [on-policy] Clip fraction, approximate KL and gradient norm are the usual monitors; refreshing costs one forward pass of the data. — [Hilton 2022, practical advice section](https://arxiv.org/abs/2110.00641); [Pleines 2022, App. E](https://arxiv.org/abs/2205.11104)
- [on-policy] Pleines reports IQM with 95% CIs from rliable, with 5 agents. — [Pleines 2022, Sec. 5](https://arxiv.org/abs/2205.11104)
- [off-policy] History-restricted evaluation (zero state k steps back) and value memory length measure how much history a trained model really uses. — [Kapturowski 2019, Sec. 5](https://openreview.net/references/pdf?id=Hy7PKCFCQ); [Morad 2024, Sec. 4](https://arxiv.org/abs/2402.09900)
- [off-policy] Padding reduces effective batch size and biases normalization, so padded steps should be excluded from sample counts and masked from loss. — [Morad 2024, Fig. 1](https://arxiv.org/abs/2402.09900); [Pleines 2022, Sec. 3.1](https://arxiv.org/abs/2205.11104)

### Inferences
Assessment of the user's proposal, envs per minibatch = 256/w:

- It correctly removes the update and sample-count confound, and it matches Pleines' structure.
- Its weak point is w=256: one whole trajectory per minibatch, from one env. I found no recurrent PPO precedent for 1 sequence (2 and 4 exist). Per-minibatch advantage normalization then runs on a single correlated trajectory, which Hilton's and Memoroids' notes suggest is a risk.
- It raises w=256 from 8 to 512 updates per iteration at 3e-4, a regime where Pleines' stale-state null is unproven.
- It also discards the existing runs in all three cells.

A variant that keeps the same control (minibatches per epoch M fixed, so envs per minibatch = 65,536 / (M × w)):

| M per epoch | U/iter | Samples/update | Sequences for w=16 / 64 / 256 | lr×U at 3e-4 | Note |
|---|---|---|---|---|---|
| 16 | 32 | 4,096 | 256 / 64 / 16 | 0.0096 | w=64 cell equals the current config, so those runs may be reusable if nothing else changed |
| 64 | 128 | 1,024 | 64 / 16 / 4 | 0.0384 | w=16 cell equals the current config; 4 sequences at w=256 matches Memory Gym's count |
| 256 | 512 | 256 | 16 / 4 / 1 | 0.1536 | The user's proposal |

- Suggested primary arm: M=16 or M=64, chosen by a short pilot, with a coarse learning-rate sweep (for example three values spanning 10x) at each window on a reduced seed count, before spending the main budget. M=256 is worth keeping as a sensitivity arm if budget allows.
- Keep the existing fixed-envs runs as a labelled sensitivity arm. Agreement between arms is a useful robustness statement; disagreement is itself a finding about the confound.
- For the MLP reference, one option is to report it twice: at its tuned 512 updates, and at the LSTM arm's U and samples per update. An LSTM-vs-MLP comparison at unequal budgets answers a different question from the window comparison, and equal samples still do not mean equal decorrelation.
- Log approximate KL, clip fraction, gradient norm and advantage std per window, using Hilton's clip-fraction bands as a guide. If KL rises with w at fixed lr, that would support the Luo amplification argument in PPO.
- Add one cheap staleness probe: compare outputs from the stored start state and a freshly recomputed state on a held-out batch, as a PPO analogue of R2D2's discrepancy. This metric is my proposal and has no published PPO counterpart.
- Add history-restricted evaluation per window (zero state k steps back, k from 0 to beyond 256). This directly tests whether w=16 agents rely on information older than 16.
- State window starts, forget-bias choice, advantage-normalization scope and padding handling explicitly in the paper, as they are the details Pleines says are rarely documented.

### Gaps
- The design has no published validation for 2D navigation.
- I did not verify in the user's code how windows interact with episode resets and padding, or which window 3e-4 was tuned on. Both affect which existing runs remain reusable.
- Effective sample size of a trajectory minibatch relative to shuffled transitions is not measured in any paper I read. Its practical size in this task is unknown.
- How I read sources: Pleines 2022, Memory Gym, POPGym, DRQN, What Matters, Ni 2023, Hilton, Memoroids, Elelimy, OpenAI Five, DD-PPO, Sample Factory, Tallec, Jozefowicz, chrono and Engstrom were read from extracted PDF text, by section and keyword rather than cover to cover. Ni 2022, MAPPO, Luo, Juliani and Rafiee were read through an HTML extraction tool returning quoted passages. Repository files were read the same way. Dossa was abstract only. Figure values were taken from a plot only where marked.
