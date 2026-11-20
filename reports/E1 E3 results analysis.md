# E1 and E3 results analysis

Nov 11, 2026. Both sweeps are complete. Every number below comes from these files on Euler, under `/cluster/scratch/eazevedo/semproj/`:

| Source | Path |
|---|---|
| E1 per-seed metrics | `fork_e1_seeds.tsv`, built from `e1_place_code_audit/eval_metrics/20261108114624/*/*/seed_*/{auc,success_curve,cr_values,spl_values,ndtw_values,npl_values}.npy` |
| E1 seed statistics | `e1_place_code_audit/stats/20261108114624.json` (`seed_stats.py`: Welch, Holm at α = 0.01, probability of improvement) |
| E1 figure data and aliasing | `e1_place_code_audit/figures/20261108114624/fig2_left.csv`, `e1_place_code_audit/aliasing/scores.jsonl` |
| E3 per-seed metrics | `fork_e3_seeds.tsv`, built from `e3_information/eval_metrics/20261110205630/...` |
| E3 statistics and figure data | `e3_information/stats/20261110205630.json`, `e3_information/figures/20261110205630/fig4.csv` |

The helper scripts are `fork_seed_table.py` and `fork_stats.py` in the same folder.

Abbreviations used below:
- **AUC**: success AUC padded to the full budget.
- **IQM**: interquartile mean with a stratified-bootstrap 95% CI.
- **Final**: the last eval success.
- **n_eval**: the number of evals; fewer than 200 means early stopping at the success threshold.
- **CR**: completion rate over 10 × 10 eval episodes.

## Completeness and hidden failures

- **E1:** 120 of 120 runs have an `auc.npy`. `check_results.py` found every result pickle is a success, so no Hydra exception is hiding behind a COMPLETED status. That check covers arrays 20208530–32, 20208549–51, 20347035 and 20347856. The launcher (20092384) ran its stats and figure steps.
- **E3:** 180 of 180 runs. All 36 arrays of 5 completed, and `check_results.py` found no exceptions. The launcher 20092387 ended COMPLETED after 11 h 16 min and saved `stats/20261110205630.json` and `figures/20261110205630/fig4.*`. **Nothing remains to run for E3.**

## E1: colour place code audit

### Per seed (AUC, final success in brackets)

Seeds 100–104, in order. two_room and four_room are left out: every seed of both backbones has AUC 0.96–1.00 and final success 1.00 in every condition.

| Room | Condition | MLP | LSTM |
|---|---|---|---|
| C_maze | global, fixed | 0.84, 0.86, 0.86, 0.60, 0.66 (all 1.00) | 0.17 (0.30), 0.73, 0.81, 0.88, 0.05 (0.18) |
| C_maze | off, fixed | 0.66, 0.64, 0.86, 0.84, 0.97 (all ≥ 0.98) | 0.46, 0.36 (0.92), 0.91, 0.68, 0.96 (rest 1.00) |
| C_maze | off, random | 0.95, 0.96, 0.95, 0.97, 0.98 (all 1.00) | 0.96, 0.75, 0.92, 0.71, 0.96 (all 1.00) |
| medium_maze | global, fixed | 0.42 (0.54), 0.72 (1.00), 0.44 (0.52), 0.86 (0.94), 0.87 (1.00) | 0.16 (0.24), 0.04 (0.04), 0.14 (0.10), 0.21 (0.18), 0.07 (0.02) |
| medium_maze | off, fixed | 0.93, 0.89, 0.82, 0.44 (0.34), 0.95 (rest ≥ 0.98) | 0.08 (0.12), 0.07 (0.06), 0.14 (0.26), 0.24 (0.24), 0.81 (0.98) |
| medium_maze | off, random | 0.59 (0.68), 0.45 (0.44), 0.63 (0.80), 0.58 (1.00), 0.51 (0.60) | 0.04 (0.02), 0.65 (1.00), 0.12 (0.06), 0.22 (0.28), 0.35 (0.62) |

**What the seeds show:**
- **C_maze is a learning-speed gap, apart from colour on with a fixed goal.** In the off/fixed and off/random conditions, every LSTM seed ends at 0.92–1.00, the same as the MLP. Its lower AUC comes from needing more evals before early stopping (LSTM n_eval 33–200, MLP 33–149).
- **C_maze with colour on and a fixed goal has two failed LSTM seeds:** 100 reaches at most 0.62 and 104 at most 0.30, with CR 0.
- **medium_maze is a gap in final success too, not just speed.** Within the 1000-iteration budget, 13 of 15 LSTM seeds end at or below 0.62. Only off/fixed seed 104 (0.98) and off/random seed 101 (1.00) converge. In comparison, 8 of 15 MLP seeds end at or above 0.94: 3 of 5 with colour on, 4 of 5 with colour off and a fixed goal, and 1 of 5 with random goals. The other 7 end at 0.34–0.80.
- **The LSTM failures are graded, unlike E2's all-or-nothing seeds.** In medium_maze the failed LSTM seeds peak at 0.22–0.94 during training (`success_curve.npy` max), so they learn partly and plateau.

### Seed-level tests (`stats/20261108114624.json`, LSTM − MLP)

| Room | Condition | AUC diff | Welch p | Holm p | PoI(LSTM > MLP) [CI] |
|---|---|---|---|---|---|
| C_maze | global, fixed | −0.237 | 0.25 | 1 | 0.36 [0.00, 0.76] |
| C_maze | off, fixed | −0.120 | 0.40 | 1 | 0.40 [0.04, 0.80] |
| C_maze | off, random | −0.101 | 0.13 | 1 | 0.24 [0.00, 0.64] |
| medium_maze | global, fixed | −0.537 | 0.0039 | 0.046 | 0.00 [0.00, 0.00] |
| medium_maze | off, fixed | −0.535 | 0.015 | 0.16 | 0.04 [0.00, 0.16] |
| medium_maze | off, random | −0.274 | 0.060 | 0.60 | 0.20 [0.00, 0.60] |
| two_room, four_room | all | −0.001 to +0.006 | ≥ 0.18 | 1 | 0.44–0.68 |

- **No comparison is significant after Holm at α = 0.01.** The closest is medium_maze global/fixed: Holm p 0.046, and no LSTM seed beats any MLP seed (PoI 0.00).
- **Secondary metrics point the same way.** In medium_maze the LSTM's SPL is 0.10–0.32 lower. The LSTM is ahead in two places only: SPL in two_room (+0.05 to +0.13, PoI 0.76–1.00, Holm p ≥ 0.16) and SPL in C_maze off/random (+0.04, PoI 0.92). In those, both backbones are at ceiling on success, and the LSTM takes slightly shorter paths.

### Colour on vs off (fixed goal, mean AUC, my Welch test on the per-seed values above)

| Backbone | Room | Global | Off | Off − global | Welch p | PoI(off > global) |
|---|---|---|---|---|---|---|
| MLP | C_maze | 0.764 | 0.794 | +0.030 | 0.73 | 0.64 |
| MLP | medium_maze | 0.661 | 0.804 | +0.143 | 0.32 | 0.80 |
| LSTM | C_maze | 0.527 | 0.673 | +0.146 | 0.51 | 0.64 |
| LSTM | medium_maze | 0.124 | 0.268 | +0.145 | 0.36 | 0.68 |

**Aliasing vs memory gain** (`aliasing/scores.jsonl`: eps_frac 0.1, min_pos_dist 100, 2000 samples; memory gain from `fig2_left.csv`):
- **With global colour** the score is exactly 0 in every room.
- **With colour off** it's 0.0003–0.0054, highest in C_maze.
- **Memory gain (LSTM − MLP IQM)** is +0.001 to +0.007 in the small rooms, −0.10 to −0.22 in C_maze and −0.55 to −0.72 in medium_maze.
- **So no relationship can be seen.** The score spans less than 0.006 and the gain is never clearly positive.

### Do the plan's E1 conclusions hold?

- **"Colour isn't the MLP's place code": holds.** Switching colour off never hurts the MLP: +0.03 in C_maze and +0.14 in medium_maze (PoI 0.64 and 0.80; not significant). It doesn't hurt the LSTM either (+0.15 in both rooms), and with colour on, C_maze has the only two LSTM seeds that never solved it.
- **H1 is partly supported in the gap, but not in the mechanism.** The LSTM − MLP gap does shrink from global/fixed to off/random: C_maze −0.24 → −0.10, medium_maze −0.54 → −0.27. But removing colour alone changes it little in medium_maze (−0.537 → −0.535). The shrink there comes from random goals, mostly because the MLP drops (0.80 → 0.55 mean AUC). The gap never flips.
- **"The MLP leads because it learns these reactive mazes faster": holds for C_maze, too weak for medium_maze.**
  - In C_maze the final success is equal, so the gap is learning speed.
  - In medium_maze, 13 of 15 LSTM seeds end at or below 0.62 within 1000 iterations. Whether they would get there with more iterations is untested.
  - Suggested wording: "within the 1000-iteration budget the memoryless agent solves these mazes faster and more often".
- **"The geometry alone already tells the agent where it is": overstated.** The score only says that near-identical views from far-apart places are rare under one threshold (eps_frac 0.1, `scripts/aliasing_score.py`). The values are too small and too narrow to test H7. **H7 is not supported**, and E1b at 90° points the same way.

## E3: information sources

### AUC IQM [95% CI] (`fig4.csv`)

four_room saturates: every condition and backbone is at 0.978–0.996, and LSTM − MLP is −0.010 to +0.008 with Holm p = 1 throughout. The table therefore shows C_maze only.

| Condition | MLP | LSTM | LSTM − MLP [CI] | Holm p |
|---|---|---|---|---|
| I0 nothing extra | 0.927 [0.910, 0.956] | 0.939 [0.868, 0.983] | +0.012 [−0.061, +0.060] | 1 |
| G1 oracle goal | 0.936 [0.918, 0.955] | **0.517 [0.125, 0.822]** | **−0.419 [−0.786, −0.109]** | 0.78 |
| G2a last seen, updated | 0.943 [0.898, 0.963] | 0.969 [0.951, 0.977] | +0.025 [−0.001, +0.069] | 1 |
| G2b last seen, frozen | 0.948 [0.923, 0.970] | 0.825 [0.650, 0.951] | −0.123 [−0.286, +0.013] | 1 |
| P1 xy | 0.921 [0.892, 0.945] | 0.908 [0.608, 0.923] | −0.013 [−0.315, +0.018] | 1 |
| P2 Fourier features | 0.877 [0.758, 0.925] | 0.800 [0.719, 0.905] | −0.076 [−0.179, +0.071] | 1 |
| P3 grid code | 0.879 [0.834, 0.942] | 0.860 [0.719, 0.926] | −0.020 [−0.178, +0.066] | 1 |
| P4 noisy grid code | 0.892 [0.818, 0.953] | 0.731 [0.609, 0.901] | −0.161 [−0.301, +0.008] | 1 |
| P5 agent + goal grid code | 0.907 [0.868, 0.933] | 0.792 [0.704, 0.878] | −0.115 [−0.201, −0.019] | 0.78 |

**Per-seed points (from `fork_e3_seeds.tsv`):**
- **The LSTM with the oracle goal (G1)** has two seeds that fail: 102 (AUC 0.08, final 0.06, CR 56) and 103 (0.21, final 0.18, CR 51). The other three reach 0.52–0.82 AUC.
- **The MLP with the oracle goal** is the best-pathing agent in E3: SPL 0.61–0.76, against 0.45–0.55 at I0.
- **I don't know why the oracle hurts the LSTM.** The goal slot is normalised (distance / room diagonal, `envs/navigation_env.py:431`), so input scale isn't the cause. A plausible explanation, untested: an LSTM that follows the through-wall bearing gets trapped in the C's concave dead end.
- **Position information** (P1–P5) lowers the AUC of both backbones a little compared with I0, the LSTM more. Only LSTM P1 seed 100 (final 0.28) really fails.

### Against the pre-registered hypotheses (plan §3)

- **H3 (being told the goal or the position closes the MLP–recurrent gap in open layouts but not in dead-end ones): untestable as run.** There is no gap at I0 in either layout (four_room +0.008, C_maze +0.012), so there's nothing to close. In C_maze the oracle goal opens a gap in the MLP's favour instead.
- **H4 (frozen goal and noisy position bring back the recurrent advantage; updated goal doesn't): not supported.** G2b and P4 are the conditions where the LSTM is furthest behind (−0.12, −0.16), and G2a is the only one with a slightly positive sign (+0.03, CI touches 0). None is significant.
- **H5 (grid code no better than Fourier features): supported, as a null result.** MLP P3 0.879 vs P2 0.877, LSTM 0.860 vs 0.800, with heavily overlapping CIs.
- **The headline "information substitutes for memory" can't come from E3 as run.** Neither layout needs memory: I0 is at 0.93–0.99 for both backbones.

## Bugs and metric issues noticed (not fixed: this task was read-only)

- **nDTW is degenerate with fixed goals** (`utils/metrics.py:286`, `exp(-DTW / (len(test) * success_radius))` with success_radius = goal radius 16).
  - With fixed goals in C_maze and four_room it's 1e-5 to 0.06 even for seeds with NPL 1.3–1.6.
  - In four_room/fixed it's bimodal per seed (e.g. 0.0005 vs 0.27), which looks like the agent taking the other of two equally short routes.
  - With random goals (shorter paths) it's 0.4–0.7.
  - So nDTW isn't comparable across goal modes or rooms, and a 16-unit threshold is very strict for long paths. My suggestion: report it only within one goal mode, or use a larger threshold such as a corridor width. This is a metric-design issue, not a crash.
- **The `success` metric in `seed_stats.py` output is the completion rate / 100, not the last eval success.** C_maze off/fixed and off/random give identical statistics (diff −0.004, Welch p 0.178), but I checked that this is a coincidence of two 99% seeds in each. No bug there.

## Summary

- **E1 (complete, 120/120 runs, no hidden failures):** colour is not the MLP's place code, since switching it off never hurts. The MLP leads in C_maze and medium_maze, but nothing is significant after Holm at α = 0.01. In C_maze the lead is learning speed (equal final success). In medium_maze it's also final success within the budget: 13 of 15 LSTM seeds end at or below 0.62. LSTM failures there are graded, not all-or-nothing.
- **H1:** the gap shrinks mostly through random goals, not colour, and never flips. **H7:** not supported (aliasing scores ≈ 0, no positive memory gain).
- **E3 (complete, 180/180 runs, no hidden failures):** four_room saturates. In C_maze, I0 is already at 0.93–0.94 with no memory gain, so H3 can't be tested. H4 isn't supported (the frozen-goal and noisy-grid LSTMs are the worst). H5 holds as a null result. The oracle goal hurts the LSTM: 2 of 5 seeds fail, IQM 0.52 vs 0.94.
- **Bugs and issues:** no code bugs found. nDTW at `utils/metrics.py:286` saturates near 0 for long fixed-goal paths, so it isn't comparable across goal modes. The plan's wording "the geometry alone already tells the agent where it is" and "learns faster" (medium_maze) needs softening, as described above.
- **Remaining for E3:** nothing to run. For the paper, E3 as designed can't show substitution. A layout where the I0 recurrent agent beats the MLP would be needed, for example the twin rooms or a narrower FOV.
