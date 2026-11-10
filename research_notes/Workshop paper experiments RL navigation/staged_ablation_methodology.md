# Staged ablation methodology for a compute-limited RL design-choice study (PPO navigation, Euler CPU cluster)

Note on sourcing: items marked [verified] were read in this session (abstract, full text or official page). Items marked [ref] are standard references whose arXiv IDs and core claims are well established, but whose text was not re-fetched in this session. The report writer should treat [ref] details (beyond the title and main idea) as needing a quick check before quoting exact numbers.

## Q1. Multi-fidelity / early-elimination methods, and "difficulty as fidelity" (is low-fidelity bias a known risk, and how to check it?)

### Takeaway
Successive halving, Hyperband, ASHA, BOHB and racing (F-Race/irace) all follow one pattern: test many configurations cheaply, cut the bottom fraction, and give survivors more budget. All of them assume that low-fidelity rankings predict high-fidelity rankings. The AutoML literature reports that this assumption often fails, and RL-specific work finds that hyperparameter optima do *not* transfer reliably across environments or data regimes. Using easy rooms as a low-fidelity proxy is therefore legitimate, but it carries a known risk. Measure the risk explicitly with a fixed probe/wildcard set run at every stage and a rank-correlation statistic (Kendall tau / Spearman rho with bootstrap CIs).

### Cited Findings
- **Successive halving (SH):** start n configs on a small budget, keep the top 1/eta, multiply the budget by eta, and repeat. Jamieson & Talwalkar, "Non-stochastic Best Arm Identification and Hyperparameter Optimization", AISTATS 2016, arXiv:1502.07943 [ref]. The procedure itself comes from Karnin, Koren & Somekh, ICML 2013 ("Almost optimal exploration in multi-armed bandits") [ref].
- **Hyperband** runs several SH "brackets" with different n-vs-budget trade-offs. It hedges against not knowing how informative low budgets are. Li et al., JMLR 18(185), 2018, arXiv:1603.06560 [ref].
- **ASHA (asynchronous SH)** promotes a config as soon as it is in the top 1/eta of its rung, with no synchronization barrier. This fits a SLURM cluster with a fixed number of parallel slots. Li et al., MLSys 2020, arXiv:1810.05934 [ref].
- **BOHB** replaces Hyperband's random sampling with a TPE-like Bayesian model. Falkner, Klein & Hutter, ICML 2018, arXiv:1807.01774 [ref].
- **Racing / irace:** "evaluates [candidates] on instances, and discards statistically worse ones until a termination criterion is met". It uses a Friedman test or a paired t-test. The t-test is recommended when performance values across instances are commensurable and the objective is the mean. Each candidate is first evaluated on `firstTest` instances before the first elimination test, then tested again every `eachTest` instances. López-Ibáñez et al., "The irace package: Iterated racing for automatic algorithm configuration", Operations Research Perspectives 3:43-58, 2016, doi:10.1016/j.orp.2016.09.002 — [RePEc](https://econpapers.repec.org/RePEc:eee:oprepe:v:3:y:2016:i:c:p:43-58); procedure summary from [Bischl et al. HPO survey, arXiv:2107.05847](https://arxiv.org/pdf/2107.05847) [verified via search summary]. The original F-Race is Birattari et al., GECCO 2002 [ref].
  - Relevance: in racing, the "instances" can be *environments × seeds*. Configs are dropped once a paired test shows they are worse on the instances seen so far. This maps directly onto "rooms as instances".
- **Low-fidelity rank correlation is a known weakness of multi-fidelity HPO.** DyHPO (Wistuba, Kadra & Grabocka, NeurIPS 2022) lists "poor rank correlations among the validation performance of candidates at different budgets" as a property of multi-fidelity HPO. It reports that configurations which ended strong had looked weak at lower budgets in up to 40% of cases (LCBench) and up to 80% (NAS-Bench-201) — [arXiv:2202.09774](https://arxiv.org/pdf/2202.09774) [verified via search summary].
- MFES-HB notes that low-fidelity measurements are "a biased surrogate" of high-fidelity quality and can mislead the search "towards a wrong objective function" — [arXiv:2012.03011](https://arxiv.org/pdf/2012.03011.pdf) [verified via search summary].
- Egele et al. 2023, "Is One Epoch All You Need for Multi-Fidelity HPO?", found that a trivial 1-epoch baseline matched multi-fidelity methods at an order of magnitude less compute. They attribute this to a few dominant learning curves in the benchmarks: the low-fidelity proxy *can* work well when the landscape has dominant configurations — [arXiv:2307.15422](https://arxiv.deeppaper.ai/papers/2307.15422v2) [verified via search summary].
- **RL-specific evidence that rankings do not transfer across environments:** Obando-Ceron, Araújo, Courville & Castro, "On the consistency of hyper-parameter selection in value-based deep RL", RLC 2024 / RLJ vol. 3 — [arXiv:2406.17523](https://arxiv.org/html/2406.17523v3) [verified]:
  - They define the **THC (Tuning Hyperparameter Consistency) score**. For each hyperparameter value they rank by IQM, with overlapping CIs giving averaged ranks. They take the peak-to-peak range of that value's rank across settings, normalized by |H|−1, then average over values. THC = 0 means perfectly consistent rankings.
  - Across agents (DER vs DrQ(ε)), optima mostly transfer (all THC < 0.5).
  - Across data regimes (100k vs 40M frames), optima mostly do *not* transfer (e.g., Adam ε optimum <1.5e-4 at 100k vs ~1.5e-2 at 40M).
  - Across environments (Atari games), THC is highest. For example, batch size 4 is best on Asterix, Breakout, Gopher and Seaquest but among the worst on most other games.
- **Hyperparameter landscapes change over training time** for DQN, PPO and SAC (Cartpole, Bipedal Walker, Hopper): "hyperparameter landscapes strongly vary over time". This undercuts early-checkpoint (time-budget) fidelity as well. Mohan, Benjamins, Wienecke, Immer & Lindauer, "AutoRL Hyperparameter Landscapes", [arXiv:2304.02396](https://arxiv.org/abs/2304.02396) [verified; published at AutoML-Conf 2023 per my recollection, not confirmed on the page].
- **Hyperparameter landscapes depend strongly on the tuning seed** ("can strongly depend on the tuning seed"), which leads to overfitting. The authors recommend separate tuning and test seeds, as in AutoML. HPO tools often beat hand-tuning at lower compute. Eimer, Lindauer & Raileanu, "Hyperparameters in RL and How To Tune Them", [arXiv:2306.01324](https://arxiv.org/abs/2306.01324) [verified abstract; ICML 2023 per recollection].

### Inferences
- **Difficulty as fidelity is defensible but biased by theme.** Perception and capacity choices (encoder, FOV, ray density, width/depth) are plausibly ranked consistently from easy to hard rooms. Choices whose benefit only appears under partial observability or long horizons will likely be *under-rated* on easy rooms: memory (LSTM, BPTT, frame stacking), loop-closure aux loss, grid cells, curriculum. On one_room an MLP is near-optimal, so memory configs pay only their optimization cost. That is a textbook low-fidelity bias. Remedy: pick the screening tier per theme. Screen memory and aux-loss themes on rooms that already need memory or exploration (T-maze, I-maze, two_room), not only on one_room.
- **How to check:**
  1. Choose a **probe panel** of 6–8 configs before Stage 1: baseline, one deliberately "weak" config, and one or two a-priori favorites per theme. Run it on *every* tier (cheap if seeds = 3). Compute Kendall tau between tiers with a bootstrap over seeds.
  2. At each promotion, also promote 1 **wildcard** per theme, chosen uniformly at random from the non-promoted configs. Rank correlation computed only on promoted configs is attenuated by range restriction (selection on the predictor). Wildcards give an approximately unbiased check of whether eliminated configs would have stayed bad.
  3. Report Kendall tau (robust for small n, handles ties) with a 95% bootstrap CI. Report the THC score per theme between tiers ([arXiv:2406.17523](https://arxiv.org/html/2406.17523v3)) as a second, RL-native consistency measure.
  4. Pre-register a fallback rule. If tau between tier k and tier k+1 is below a threshold (e.g., 0.4, or its CI includes 0) for a theme, re-open that theme at tier k+1 with more survivors (Hyperband's "hedge": run a second bracket with less aggressive elimination).
- **Early stopping as within-stage fidelity:** the project already stops on eval success rate. That is a time-based fidelity. Mohan et al. show time-based fidelity is also non-stationary, so the elimination rate inside a stage should stay conservative (eta = 2–3, not 5+).
- **Racing fits the setting best:** the number of configs is small (tens, not thousands), evaluations are noisy, and the user has natural "instances" (rooms × seeds). A manual F-Race/irace-style procedure (add seeds or rooms in rounds; drop configs significantly worse than the leader by a paired Friedman or t-test) is simple to implement in Python with a SLURM array, and citable.

### Gaps
- I found no paper that explicitly uses *environment difficulty* (easy→hard maze) as the fidelity dimension in Hyperband/SH for RL and measures its rank correlation. The idea is related to curriculum and transfer-HPO, but I found no direct citable precedent. The workshop paper can present it as its own protocol and justify it with the THC and DyHPO evidence above.
- The exact venue of Mohan et al. 2023 and Eimer et al. 2023 was not confirmed on the fetched pages.

## Q2. How large-scale RL design-choice studies structured their ablations

### Takeaway
The best template is Andrychowicz et al. 2021. They group interacting choices into themed experiments, sample choices randomly within a group while the rest stay at a competitive baseline, score configs by an AUC-like metric (median over 3 seeds), and analyze with conditional 95th-percentile plots and "distribution among top 5%". For the final story, add a Rainbow-style leave-one-out ablation from the best composite config, and follow Revisiting Rainbow's argument that small environments give valid, cheap insights.

### Cited Findings
- **Andrychowicz et al., "What Matters in On-Policy RL? A Large-Scale Empirical Study"** (ICLR 2021), [arXiv:2006.05990](https://arxiv.org/abs/2006.05990) [verified full text]:
  - Over 250,000 agents, 50+ choices, 5 MuJoCo environments.
  - They state two problems. Sampling *all* choices randomly gives very bad performance ("the probability of such failure increases exponentially"). Choices interact, so "experiments where only a single choice is varied but interacting choices are kept fixed may be misleading."
  - Design: "We create groups of choices around thematic groups where we suspect interactions." Adam learning rate (C24) was included in *every* group because it may interact with many choices. Within an experiment, choices are sampled randomly. All other choices stay at "a competitive base configuration" (close to PPOv2 defaults, 256 parallel envs). This ensures good models are present and combinations of interacting choices are covered.
  - 8 groups: Policy Losses, Network architecture, Normalization & clipping, Advantage estimation, Training setup, Timesteps handling, Optimizers, Regularization. Appendix C notes that 2000 configs were sampled per environment for one group, for example.
  - Performance: 3 seeds per config; evaluate every 100k steps (100 episodes); average over evaluations, which is "proportional to the area under the learning curve" and rewards fast learners. Config score = **median over the 3 seeds**.
  - Analysis 1, **conditional 95th percentile**: for each value of a choice, take the 95th percentile of the score distribution of sampled configs with that value, with a binomial CI. This is roughly "the performance one can expect if all other choices in the group were tuned with random search and a limited budget of roughly 20 hyperparameter configurations."
  - Analysis 2, **distribution of each choice among the top 5% configs**: values are sampled uniformly, so over-representation signals importance.
- **Engstrom et al., "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO"**, ICLR 2020, arXiv:2005.12729 [ref]. Code-level optimizations (value clipping, reward scaling, orthogonal init, learning-rate annealing, etc.) explain much of PPO's advantage over TRPO. Relevance: ablate "implementation detail" choices separately from the "research" choices, or hold them fixed and document them.
- **Huang et al., "The 37 Implementation Details of PPO"**, ICLR Blog Track 2022 — https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ [ref]. Includes LSTM-specific details: reset hidden state at episode boundaries, initialize hidden states to zero, and the recurrent minibatch handling relevant to truncated BPTT. Use it as a checklist to freeze the baseline.
- **Hessel et al., "Rainbow"**, AAAI 2018, arXiv:1710.02298 [ref]. Combine all components, then remove one at a time from the full agent (leave-one-out ablation) to measure each component's contribution in the context of the others.
- **Obando-Ceron & Castro, "Revisiting Rainbow: Promoting more insightful and inclusive deep RL research"**, ICML 2021, [arXiv:2011.14826](https://arxiv.org/abs/2011.14826) [verified abstract]. Argues that "the traditional small-scale environments can still yield valuable scientific insights" and that small-scale work can "reduce the barriers to entry for underprivileged communities". Revisits the Rainbow ablation on small and mid-scale environments. Good to cite as justification for a CPU-only study on gridworld-like rooms.
- **Henderson et al., "Deep RL that Matters"**, AAAI 2018, arXiv:1709.06560 [ref]. Shows that seeds, hyperparameters, codebases and environment choice change conclusions. Two groups of 5 seeds of the *same* algorithm can produce significantly different learning curves.
- **Jordan et al., "Evaluating the Performance of RL Algorithms"**, ICML 2020, arXiv:2006.16958 [ref]. Argues that evaluation should account for hyperparameter-tuning effort and performance across environments (normalization by score distributions), not only the best-tuned result.
- **Jordan, White, da Silva, White & Thomas, "Position: Benchmarking is Limited in RL Research"**, ICML 2024 (PMLR 235:22551–22569), [arXiv:2406.16241](https://proceedings.mlr.press/v235/jordan24a.html) [verified via search summary]. Rigorous benchmarking "will likely have computational costs that are often prohibitive". They advocate an additional "scientific testing" paradigm: targeted experiments that test specific hypotheses about algorithm behavior. This supports framing a workshop paper around hypotheses per theme (e.g., "memory helps only when the maze has aliased states") rather than a leaderboard.
- **Patterson, Neumann, White & White, "Empirical Design in RL"**, JMLR 2024, [arXiv:2304.01315](https://arxiv.org/html/2304.01315v2) [verified, partial text]. Lists as common errors: "Untuned agents in ablations", "Reusing code from another source (including hyperparameters) as is", and "Not comparing against stupid baselines". Includes sections on "Designer's curse" and "Ranking algorithms (maybe don't)".

### Inferences
- For this project, Andrychowicz's design scales down well. Themed groups, with all other choices held at a strong baseline, avoid the "all-random → nothing learns" failure. Including a key global hyperparameter (learning rate, 2–3 values) in each theme protects against the "untuned agents in ablations" critique, at 2–3× cost. With limited budget, include LR only in the capacity and memory themes, where interactions with LR are most plausible.
- Andrychowicz's score (mean of periodic evals ≈ AUC; median over 3 seeds) is almost exactly the project's normalized success-AUC metric. Cite it as precedent for using AUC as the primary ranking metric.
- Rainbow leave-one-out from the best composite is the right *final* stage: it answers "does each choice still matter given the others?" Revisiting Rainbow justifies the small-environment setting.

### Gaps
- I did not re-verify Engstrom/Huang/Henderson/Jordan 2020 text in this session. Their main claims above are well known, but exact numbers (e.g., Henderson's 5-seed split result) should be checked before quoting.

## Q3. Statistics: rliable, seed counts, multiple comparisons, aggregation across environments

### Takeaway
Use rliable-style reporting throughout: IQM across (runs × environments) of normalized scores, stratified-bootstrap 95% CIs, performance profiles, and probability of improvement. In screening stages, 3 seeds are acceptable *only* as a selection heuristic with conservative promotion (keep everything whose CI overlaps the leader). Claims in the paper need about 10 seeds per environment. Use Welch's t-test at alpha = 0.01 (or probability of improvement with CIs) with Holm correction across the leave-one-out comparisons. A sequential test (AdaStop) can add seeds only where needed.

### Cited Findings
- **Agarwal et al., "Deep RL at the Edge of the Statistical Precipice"** (NeurIPS 2021 Outstanding Paper), [arXiv:2108.13264](https://arxiv.org/abs/2108.13264) [verified abstract]. Few runs per task exacerbate "the statistical uncertainty in point estimates". Point estimates led to substantial discrepancies in an Atari 100k case study. Recommends interval estimates of aggregate performance, performance profiles, and **IQM** as a robust and efficient aggregate. Library: `rliable` — https://github.com/google-research/rliable [ref: provides IQM, optimality gap, stratified bootstrap CIs, performance profiles, probability of improvement].
- **Colas, Sigaud & Oudeyer, "How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments"**, 2018, [arXiv:1806.08295](https://arxiv.org/abs/1806.08295) [verified abstract]. A tutorial on t-test and bootstrap tests, and on choosing seed counts via power analysis.
- **Colas, Sigaud & Oudeyer, "A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms"**, 2019, [arXiv:1904.06979](https://arxiv.org/html/1904.06979v2) [verified full text]:
  - Recommends **Welch's t-test** with a low significance level, alpha = 0.01. With skewed distributions the false positive rate is ~0.1 at nominal 0.05.
  - For 80% power: large effect (standardized ε=2) needs N≈5–10; medium (ε=1) N≈20; small (ε=0.5) N≈100. SAC vs TD3 needed N≈10–15 for means, plus ~5 for medians.
  - N ≤ 5 gives high false positive rates. Avoid bootstrap tests for N < 50 and permutation tests for N < 10. Use Mann-Whitney or ranked t-test only with checked assumptions. Use Bonferroni (alpha/n_c) for multiple comparisons.
- **Patterson et al. 2024** [arXiv:2304.01315](https://arxiv.org/html/2304.01315v2) [verified, partial]:
  - "5 runs is insufficient to make strong claims"; "even 30 runs can be insufficient if the distribution is heavily skewed".
  - Recommend bootstrap CIs by default, and not reporting standard errors. Use **tolerance intervals** if per-agent behavior matters (they do not shrink with more runs).
  - The **two-stage approach** (sweep, then re-run the best config with fresh seeds) is "a reasonable place to start" and far better than hand-tuning, but "is likely to underestimate the maximum performance and is overconfident". Bootstrapped two-stage tuning allows N≈10 rather than 30 per config.
  - Fairness: "ensure all algorithms have the same number of hyperparameter settings that are tested". IQM "is most usefully applied across collections of environments", and on a single environment can hide failure modes.
- **AdaStop** (Mathieu et al., TMLR 2024), [arXiv:2306.10882](https://arxiv.org/abs/2306.10882v2) [verified via search summary]. Argues that "<5 independent executions" is generally not enough. It is a multiple group-sequential test that "adapts the number of executions to stop as early as possible while ensuring that enough information has been collected", with a proven low family-wise error. It is evaluated on MuJoCo. Python package available (adastop).
- **Eimer et al. 2023** [arXiv:2306.01324](https://arxiv.org/abs/2306.01324) [verified abstract]: separate tuning seeds from test seeds.

### Inferences
- **Normalization for aggregation across rooms:**
  - Success rate and success-AUC are already in [0,1].
  - Path length: use shortest/actual (SPL-like, in [0,1], higher better) rather than actual/shortest, so it can be IQM-aggregated with the others.
  - DTW: normalize per environment by a reference, e.g., DTW of a random-walk policy, giving 1 − DTW/DTW_random clipped to [0,1].
  - Choose one **primary** metric for promotion decisions, pre-registered (normalized success-AUC). Treat the others as secondary or descriptive, to avoid the garden of forking paths.
- **Early stopping and AUC:** if a run stops early after reaching the success threshold, the AUC must be computed over the *full* nominal budget with the last eval value carried forward. Otherwise early stopping penalizes fast learners or truncates comparisons. Verify this in metrics.py.
- **Treat screening as selection, not inference:** with 3 seeds × 3 rooms = 9 runs per config, stratified bootstrap over seeds within rooms gives usable but wide CIs. Promotion rule: keep the leader plus all configs whose IQM CI overlaps the leader's, capped at k. The quantity of interest for promotion is P(config is not worse), not significance. Report no p-values from screening stages.
- **Inference stage:** effect sizes between design choices in navigation are often large (memory vs no memory in T-maze is close to binary success). By Colas 2019, N=10 per room has ~80% power for ε≈1.3–2 effects, so 10 seeds suffices for large effects. Medium effects need ~20; AdaStop can add 5–10 seeds only to comparisons that stay undecided.
- **Multiple comparisons:** in the final leave-one-out stage (~6–8 comparisons vs the best config), use Holm–Bonferroni (uniformly more powerful than Bonferroni, same FWER control). Standard references are Holm 1979 and Benjamini–Hochberg 1995 [ref]; I did not fetch them.
- **Fresh seeds:** final-stage seeds must differ from all screening seeds. For example, use seeds 0–2 for screening, 100–104 for stage 2, and 1000–1009 for the final stage. This addresses maximization bias / the designer's curse (Patterson; Eimer).

### Gaps
- I did not verify the exact list of rliable functions from the repo in this session, but these features are widely documented.
- I found no source on how many seeds are needed specifically for binary-ish success-rate metrics in navigation. Do a power analysis from the project's own Stage-0 variance estimates.

## Q4. Design-of-experiments: fractional factorial / LHS / OFAT-from-baseline vs random search + marginal analysis (fANOVA)

### Takeaway
Use the Andrychowicz hybrid for this project: an exhaustive or random design *within* each theme, with OFAT *between* themes from a strong baseline. Within a theme, exhaustive grids are small enough (8–20 configs) that fANOVA or simple main-effect/interaction analysis is possible. Between themes, interactions are handled by a compose-then-leave-one-out stage. Random search over the full joint space plus fANOVA is the "purer" alternative, but with ~600k combinations and ~1000 affordable runs, almost all samples would be poor configs.

### Cited Findings
- Andrychowicz et al. explicitly reject both pure OFAT ("may be misleading" under interactions) and pure all-choices random sampling (very bad performance). They adopt within-theme random sampling around a competitive base config — [arXiv:2006.05990](https://arxiv.org/abs/2006.05990) [verified].
- **fANOVA** (Hutter, Hoos & Leyton-Brown, "An Efficient Approach for Assessing Hyperparameter Importance", ICML 2014, PMLR v32) [ref] fits a random-forest surrogate to (config, performance) data. It decomposes performance variance into main effects and pairwise interactions per hyperparameter, so it needs a reasonably spread sample of configs. Implementation: `fanova` / DeepCAVE / `pyrfr` [ref].
- **van Rijn & Hutter, "Hyperparameter Importance Across Datasets"**, KDD 2018, arXiv:1710.04725 [ref]. Runs fANOVA per dataset and aggregates importance across many datasets. This is the template for "importance across rooms": run fANOVA per room and report the distribution of importance across rooms.
- **Random search** beats grid search when only a few hyperparameters matter (low effective dimensionality). Bergstra & Bengio, JMLR 13, 2012 [ref].
- Patterson et al.: check sensitivity curves. If the best value sits at the edge of the swept range, extend the range — [arXiv:2304.01315](https://arxiv.org/html/2304.01315v2) [verified].

### Inferences
- **Fractional factorial (2^(k−p)) designs** are useful only for many binary factors with mostly main effects. The binary toggles (depth aux, loop-closure aux, grid cells, curriculum, collision penalty, discrete/continuous) could form a 2^(6−1) resolution-VI half-fraction (32 runs instead of 64), which estimates all main effects and two-factor interactions. This is an efficient alternative to theme-by-theme screening for the binary toggles. Standard DoE references: Box, Hunter & Hunter, *Statistics for Experimenters*; Montgomery, *Design and Analysis of Experiments* [ref, not fetched].
- **Latin hypercube** sampling matters only for continuous or ordinal factors (LSTM hidden size, BPTT window, width/depth, FOV, ray count, penalty magnitudes). Within a theme, an exhaustive small grid is simpler and easier to plot.
- **Recommended analysis outputs per theme** (Andrychowicz-style):
  1. A per-choice-value plot of IQM success-AUC with bootstrap CI (the analogue of the conditional 95th percentile when each value has few configs).
  2. For themes with ≥ ~15 configs (memory theme: backbone × hidden × BPTT), a fANOVA main-effect and interaction plot, e.g., hidden size × BPTT window.
  3. A heatmap of IQM per (config, room) to show where low-fidelity bias appears.

### Gaps
- I did not verify that the `fanova` package works with modern Python. DeepCAVE (AutoML Freiburg) is the maintained alternative [ref].

## Q5. Concrete staged protocol (rooms → stages, configs/seeds per stage, promotion rules, wildcards, run counts) and theme grouping

### Takeaway
A 4-stage "difficulty-as-fidelity" successive-halving/racing protocol:
- **Stage 0** calibrates.
- **Stage 1** screens every within-theme config on easy rooms with 3 seeds.
- **Stage 2** promotes about one third (eta ≈ 3) plus wildcards to medium rooms with 5 seeds.
- **Stage 3** composes the best config and runs a leave-one-out ablation on hard rooms with 10 fresh seeds.
- **Stage 4** evaluates finalists on all rooms with fresh seeds.

The total is about 1,500 runs. Rank correlation between stages is checked with a probe panel and wildcards. Statistical claims are made only from Stages 3–4.

### Cited Findings
(The protocol is a synthesis. The design elements are sourced as follows.)
- Themed groups + competitive baseline + AUC-like score — [Andrychowicz et al., arXiv:2006.05990](https://arxiv.org/abs/2006.05990) [verified].
- Keep top 1/eta per rung, eta = 3 typical — Hyperband arXiv:1603.06560 / ASHA arXiv:1810.05934 [ref].
- Statistical elimination over instances (rooms × seeds) — [irace, doi:10.1016/j.orp.2016.09.002](https://econpapers.repec.org/RePEc:eee:oprepe:v:3:y:2016:i:c:p:43-58) [verified via summary].
- Leave-one-out from the best — Rainbow, arXiv:1710.02298 [ref].
- Fresh test seeds / two-stage tuning — [Patterson et al.](https://arxiv.org/html/2304.01315v2); [Eimer et al.](https://arxiv.org/abs/2306.01324) [verified].
- IQM + stratified bootstrap + probability of improvement — [Agarwal et al.](https://arxiv.org/abs/2108.13264) [verified].
- Welch alpha=0.01, N≈10 for large effects — [Colas et al. 2019](https://arxiv.org/html/1904.06979v2) [verified].
- Cross-setting consistency (THC) — [Obando-Ceron et al. 2024](https://arxiv.org/html/2406.17523v3) [verified].
- Adaptive seed allocation — [AdaStop](https://arxiv.org/abs/2306.10882v2) [verified via summary].

### Inferences

**Theme grouping** (interacting choices go in the same theme, as in Andrychowicz). Example level counts are illustrative; adjust to the actual sweep values.

| Theme | Choices (levels) | Within-theme design | # configs |
|---|---|---|---|
| A. Perception | encoder {MLP, CNN} × FOV {3 levels} × ray density {3 levels} | full factorial | 18 |
| B. Memory | backbone {MLP, FS2, FS4, FS8, FS16, LSTM}; LSTM: hidden {64,128,256} × BPTT {16,32,64,128,256} | MLP(1) + FS(4) + LSTM grid(15) | 20 |
| C. Auxiliary / representation | depth-aux {0,1} × loop-closure {0,1} × grid cells {0,1} | full factorial | 8 |
| D. Reward & curriculum | time penalty {0, low, high} × collision penalty {0,1} × curriculum radius growth {0,1} | full factorial | 12 |
| E. Capacity | depth {1,2,3} × width {64,128,256} | full factorial | 9 |
| F. Action space (+ goal conditioning, optional) | discrete vs continuous (× goal-cond {0,1}) | full | 2–4 |
| **Total** | | | **~69–71** (vs full factorial 18·20·8·12·9·2 ≈ 622k) |

- Capacity (E) and memory (B) interact (hidden size vs width). Either merge LSTM hidden size into E, or run E after B is fixed. Running B first is recommended: the memory story is likely the paper's headline, and on POMDP mazes it is the theme most prone to low-fidelity bias.
- Optionally add learning rate {3e-4, 1e-3} to themes B and E only (Andrychowicz included LR in every group). This doubles those themes (+29 configs).

**Room tiers** (verify the difficulty ordering with Stage 0):
- Tier 0 (smoke test only): one_room. All reasonable configs probably saturate it, so it discriminates little except through AUC speed.
- Tier 1 / easy (screening fidelity): two_room, I-maze, T-maze. Include T-maze and I-maze on purpose: aliased corridors need memory, so themes B and C are not judged in rooms where memory cannot help.
- Tier 2 / medium: four_room, cross_room, C_maze.
- Tier 3 / hard: medium_maze, hard_maze.

**Stage 0: calibration (≈ 90–100 runs).**
- Run the baseline on all 9 rooms × 10 seeds. Outputs: per-room training budget (steps at which the baseline plateaus), normalization references (random policy, shortest path), and per-room seed variance for a power analysis (Colas). Also confirm the tier ordering by baseline success-AUC.
- Fix a **probe panel** of ~6 configs: baseline, a "weak" config (e.g., MLP backbone + no aux), and the a-priori favorite of themes A, B, C and D. Run it at 3 seeds on all tiers: 6 × 3 × 8 rooms ≈ 144 runs. This yields an *early* cross-tier Kendall tau estimate. If tau(Tier1→Tier3) is poor for theme B or C, screen that theme on Tier 2 directly.
- Stage 0 total ≈ 90 + 144 ≈ 234 runs (cheap rooms dominate).

**Stage 1: screening on Tier 1 (≈ 69 × 3 seeds × 3 rooms = 621 runs).**
- Every within-theme config, other themes at baseline, seeds {0,1,2}.
- Score: IQM of normalized success-AUC across the 9 runs (rooms × seeds), stratified bootstrap 95% CI (rliable).
- Optional within-stage early elimination (ASHA-like): at 50% of the budget, kill a run only if its config's eval success is 0 on all rooms while the theme leader is above 0.5. Be conservative, given the time-varying landscapes of Mohan et al.
- **Promotion rule (per theme, pre-registered):**
  1. Keep the leader.
  2. Keep every config whose IQM CI overlaps the leader's CI, or equivalently whose probability of improvement P(leader > config) < 0.75.
  3. Cap at k = max(2, ceil(n_theme/3)) (eta ≈ 3). Break ties by lower compute (fewer params, shorter BPTT).
  4. Add 1 **wildcard**, drawn uniformly at random from the eliminated configs of that theme (fixed RNG seed, reported).
  5. Always carry the baseline value of each theme forward as the reference.
- Expected survivors ≈ A:6+1, B:7+1, C:3+1, D:4+1, E:3+1, F:2 → ~30 configs (fewer if CIs separate well).

**Stage 2: medium rooms, Tier 2 (≈ 25–30 configs × 5 seeds × 3 rooms ≈ 375–450 runs).**
- Re-baseline first: set each theme's *Stage-1 leader* as the new default for the other themes (one round of coordinate descent), so the Stage-2 configs are evaluated in a stronger context. Use seeds {100..104}, not reused from Stage 1.
- Check rank consistency: Kendall tau between Stage-1 and Stage-2 IQM over all promoted + wildcard configs, per theme and pooled, with bootstrap CI; plus a THC-style score per theme.
  - Decision rule: if a wildcard beats the Stage-2 leader's CI lower bound, or tau's 95% CI includes 0 for a theme, flag that theme as "fidelity-inconsistent". Re-promote its Stage-1 configs ranked k+1..2k to Stage 2 (one extra bracket, Hyperband-style hedge).
- Promotion: per theme, keep the top 1–2 by the same CI-overlap rule (eta ≈ 3 again), then compose the **best composite** config (each theme at its Stage-2 leader).

**Stage 3: hard rooms + leave-one-out ablation, Tier 3 (≈ 10–12 configs × 10 seeds × 2 rooms ≈ 200–240 runs).**
- Configs:
  - the best composite;
  - the original baseline;
  - for each theme (6), the best composite with that theme reverted to its baseline value (Rainbow leave-one-out);
  - for themes where a runner-up survived Stage 2, the best composite with the runner-up swapped in (2–3 configs).
- Seeds {1000..1009}, fresh. If undecided comparisons remain, use AdaStop or simply add 5 seeds to only those comparisons.
- Statistics: per room and aggregated over the 2 rooms, IQM + stratified bootstrap CI and probability of improvement (best composite vs each ablation) with CI. Use Welch t-tests at alpha = 0.01 on per-seed success-AUC with Holm correction across the ~8 comparisons. Report effect sizes, not only p-values.

**Stage 4: final full evaluation (≈ 4 configs × 10 seeds × 9 rooms = 360 runs).**
- Configs: best composite, baseline, and the 1–2 most informative alternatives, e.g., best-without-memory and best-without-aux.
- Fresh seeds {2000..2009}, all rooms, all metrics: success, AUC, normalized path length (SPL-style), normalized DTW.
- Figures: rliable performance profiles across all 9 rooms, a per-tier IQM bar chart with CIs, and a probability-of-improvement matrix. Optionally add tolerance intervals for per-seed reliability (Patterson).
- Optionally add 1–2 hard-room runs of the best Stage-1 *non-promoted* config of the memory theme, to show directly whether easy-room elimination was wrong.

**Run-count summary (illustrative):**

| Stage | Configs | Seeds | Rooms | Runs |
|---|---|---|---|---|
| 0 calibration + probe panel | 1 + 6 | 10 / 3 | 9 / 8 | ~234 |
| 1 screening (easy) | ~69 | 3 | 3 | ~621 |
| 2 medium | ~25–30 | 5 | 3 | ~375–450 |
| 3 hard LOO | ~10–12 | 10 | 2 | ~200–240 |
| 4 final | ~4 | 10 | 9 | ~360 |
| **Total** | | | | **~1,800–1,900** |

- Wall-clock ≈ Σ(runs × hours-per-run for that room) / 16 slots. Measure hours-per-run per room in Stage 0. Hard-room runs dominate cost even though they are a small share of runs.
- If the budget is tighter:
  - drop Stage-4 rooms to one per tier (4 configs × 10 seeds × 4 rooms = 160);
  - drop the probe panel to 4 configs;
  - screen Stage 1 on 2 rooms (T-maze + two_room): 69 × 3 × 2 = 414.
  These cuts give a total of ~1,200 runs.

**Why this is defensible for a workshop paper (claims to make, with citations):**
1. Screening stages are *selection*, done with conservative CI-overlap promotion and random wildcards (racing / successive-halving lineage).
2. All reported claims come from fresh-seed stages with ≥10 seeds per room, rliable intervals, and multiple-comparison control.
3. Low-fidelity bias is measured, not assumed: probe-panel and wildcard Kendall tau, plus per-theme THC.
4. Interactions are handled within themes by factorial grids, and across themes by re-baselining plus leave-one-out (Andrychowicz + Rainbow).
5. Pre-register the primary metric (normalized success-AUC), the promotion rules and the seed sets in the repo before running Stage 1. A dated commit is enough.

### Gaps
- Run counts are illustrative. They depend on the actual sweep levels, and on hours-per-run per room, which I could not measure in this research task. Use Stage-0 timing.
- No found source gives a validated tau threshold for "proxy is good enough". The 0.4 / CI-excludes-0 rule is a heuristic and should be stated as such in the paper.
- How strongly early stopping on success rate biases AUC comparisons depends on the project's AUC implementation. Check that AUC uses the full nominal horizon with last-value carry-forward.
