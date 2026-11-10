# Venues and Additional Experiments for a Small Empirical RL Indoor-Navigation Workshop Paper

Context: ETH Zurich / IVIA semester project, PPO + egocentric 2D ray casts, one-room to hard maze, comparing encoders, MLP/LSTM/frame stacking, TBPTT windows, aux losses (depth, loop closure), grid cells, curriculum. Today: 2026-10-29. About 1–2 months of project time left.

Labeling convention: **CONFIRMED** = read on an official or organizer page in this session. **ESTIMATE** = based on an earlier edition or an aggregator. **UNVERIFIED** = from prior knowledge and not fetched in this session.

## Q1. Which workshop and conference venues fit, and what are their deadlines?

### Takeaway
The only ML workshop deadlines that are both confirmed and still open are at **AAAI-27 (due Nov 20, 2026; Montréal, Feb 22–23, 2027)**, and the topical fit there is moderate at best. The best-fitting realistic targets are the **ICLR 2027 workshops** (suggested paper deadline Feb 1, 2027; the list of accepted workshops is announced Nov 29, 2026) and **CVPR 2027 embodied-AI workshops** (paper deadlines probably Mar–Apr 2027). On the architecture side, **eCAADe 2027** is the best fit for the timeline: recent editions asked for an abstract around mid-January and the full paper around April (ESTIMATE). NeurIPS 2026, CoRL 2026 and CAADRIA 2027 have all closed.

### Cited Findings

**AAAI-27 workshops (CONFIRMED general dates)**
- Workshops run Feb 22–23, 2027 in Montréal. Papers are due to each workshop's organizers by **Nov 20, 2026**, notification is Dec 2, 2026, and individual workshops may set other deadlines. — [AAAI-27 workshop program](https://aaai.org/conference/aaai/aaai-27/aaai-27-workshop-program/); [AAAI-27 workshops call](https://aaai.org/conference/aaai/aaai-27/workshops-call/)
- The full list has about 45 workshops. None is about the built environment or architecture. The closest topical fits are below. — [AAAI-27 workshop list](https://aaai.org/conference/aaai/aaai-27/aaai-27-workshop-program-list/)
  - **NeuroAI 2.0**: continual learning and adaptive systems. Full papers up to 8 pages are **archival**; short papers up to 4 pages are **non-archival**. The deadline was not stated on the AAAI page. — [site](https://neuroai-multimodal-workshop.github.io). Fit: moderate, for a grid-cell or cognitive-map framing.
  - **Human-Centric Agentic Mobility Services**: up to 4 pages excluding references, non-archival, double-blind, due Nov 20, 2026 AoE. — [site](https://agentic-mobility-workshop.github.io). Fit: weak to moderate (human mobility simulation).
  - **WoMAPF 2027** (Multi-Agent Path Finding): no page limit (AAAI format), non-archival, due Nov 20, CMT submission. Its scope includes learning-based and MARL navigation. — [site](https://wo-mapf.github.io/aaai27/). Fit: weak, since the project is single-agent.
  - **Embodied AI Meta-alignment**: regular papers 6–8 pages, position papers 2–4, non-archival, double-blind. Its scope is RL/IRL for value alignment. — [site](https://embodied-meta-alignment.pages.dev/). Fit: weak.
  - **W3PHIAI-27 Health Intelligence**: full papers up to 8 pages, **archival** (Springer), due Nov 20. Its theme is agentic clinical workflows. — [site](http://w3phiai2027.w3phi.com). Fit: weak (hospital angle only).
  - **CALE-AI (Evaluating AI Interventions)**: full papers up to 8 pages, non-archival, due Nov 20. — [site](https://caleai.github.io/). Fit: weak.
  - **Social Simulation with LLMs**: regular papers up to 7 pages; replication papers up to 4 pages plus artifacts; non-archival; due **Nov 6, 2026**. — [site](https://sites.google.com/view/social-sims-with-llms/social-sim27). Fit: only if the occupant simulation includes an LLM agent.
  - **GSR-VLM (Geometric & Spatial Reasoning in VLMs)**: full papers up to 7 pages, short papers up to 4, deadline TBA. — [site](https://gsr-vlm.org). Fit: only if a VLM/VLA baseline is added.

**ICLR 2027 workshops (CONFIRMED timeline; individual workshops not yet known)**
- Workshop proposals were due Oct 9, 2026. **Acceptance notifications to organizers come Nov 29, 2026.** The suggested paper deadline is **Feb 1, 2027**, and authors must be notified by Feb 26, 2027. — [ICLR 2027 Call for Workshops](https://iclr.cc/Conferences/2027/CallForWorkshops)
- The dates page lists the main conference for Apr 26–28 and workshops for Apr 28–30. The call page body says "April 29 and 30" with a stale year, so the exact days should be treated as uncertain. — [ICLR 2027 Dates](https://iclr.cc/Conferences/2027/Dates)
- ICLR requires every workshop to accept short submissions. — [ICBINB @ ICLR 2026 submit page](https://sites.google.com/view/icbinb-2026/submit)
- ICLR 2026 (Rio, Apr 26–27, 2026) ran about 40 workshops. Relevant ones that may recur in 2027 (ESTIMATE): **World Models (2nd edition)**, **Lifelong Agents**, **MemAgents (memory for LLM agents)**, **New Frontiers in Associative Memories**, and **ICBINB (negative results, LLM-focused in 2026)**. — [ICLR 2026 downloads](https://iclr.cc/Downloads/2026); [MemAgents](https://iclr.cc/virtual/2026/workshop/10000792); [ICBINB 2026](https://sites.google.com/view/icbinb-2026/submit)
- An ICLR 2026 workshop paper on LLM maze navigation was accepted, which shows that small grid and maze navigation studies do land at ICLR workshops. — [Do LLMs Build Spatial World Models?](https://iclr.cc/virtual/2026/10018071)

**CVPR 2027 (ESTIMATE for paper deadlines)**
- Workshop proposals are due Nov 4, 2026 and decisions come Dec 22, 2026, so individual calls will appear in January 2027. — [CVPR 2027 Call for Workshops](https://cvpr.thecvf.com/Conferences/2027/CallForWorkshops)
- CVPR 2026 embodied-AI workshop deadlines, as a proxy:
  - "Deployment of Foundation Models for Embodied AI": due Mar 8, 2026, up to 8 pages, **archival**. — [WDFM-EAI CFP](https://wdfm-eai.github.io/CVPR26/call_for_papers.html)
  - "Embodied Intelligence in the Wild": submissions open Feb 10 – Mar 10, 2026. — [Google Groups CFP](https://groups.google.com/g/ai-robotics/c/2-Fit2Ey4Dc)
  - "Bridging Language, Vision and Action in 3D": 2–8 pages, non-archival, due Apr 26, 2026. — [3D-LLM-VLA](https://3d-llm-vla.github.io)
  - "Foundation Models Meet Embodied Agents": non-archival, so dual submission is allowed. — [FMEA CVPR 2026](https://foundation-models-meet-embodied-agents.github.io/cvpr2026/)
- The flagship "Embodied AI Workshop" (embodied-ai.org) site was not reached in this session. Its deadline and archival status are a gap.

**NeurIPS 2026 (CONFIRMED closed)**
- The suggested workshop paper date was Aug 29, 2026, and the author notification deadline was Sep 29, 2026. Workshops run Dec 11–13, 2026 in Sydney, Paris and Atlanta. — [NeurIPS 2026 Call for Workshops](https://neurips.cc/Conferences/2026/CallForWorkshops)
- The most relevant workshop, NEmo 2026 (long-lived embodied agents), had its deadline on Sep 10, 2026. — [NEmo CFP](https://groups.google.com/g/rl-list/c/QqPMrCP4kcc). No NeurIPS 2026 call with a future deadline was found.

**CoRL 2026 (closed)**
- CoRL 2026 is in Austin. Workshop deadlines were around late September; for example, Continually Self-Improving Robots closed Sep 28 (aggregator source). — [digg aggregator](https://digg.com/ai/1qfgdz89); [CoRL 2026 contributions](https://2026.corl.org/contributions)

**RLC 2027 (not announced; ESTIMATE)**
- RLC 2026 (Montréal, Aug 2026) had its abstract due Mar 1 and full paper due Mar 5, 2026, and the call came out in January 2026. RLC 2027 is therefore likely to have a call in Jan 2027 and a deadline in early March 2027. — [Amii RLC page](https://www.amii.ca/events/reinforcement-learning-conference); [rl-list](https://groups.google.com/g/rl-list/c/I73cS5OOY8I)
- RLC 2026 had workshops such as RLVG (RL and Video Games), which extended its deadline, and "Finding the Frame". — [RLVG extension](https://groups.google.com/g/ML-news/c/JMSJhFeA1AE); [RLVG submission guide](https://sites.google.com/view/rlvg-2026/submission-guide); [Finding the Frame](https://groups.google.com/g/rl-list/c/rt15TqdBgEw)
- The RLC main track is a full-paper venue. It explicitly values careful empirical work, which makes it a plausible follow-up target if the project is extended.

**ICML 2027, ICRA 2027, IROS 2027 (UNVERIFIED, not searched successfully)**
- No 2027 calls were checked. From typical past patterns (prior knowledge, not verified this session):
  - ICML workshop papers are usually due late May to early June.
  - ICRA workshop papers are usually due around March–April.
  - IROS workshop papers are usually due around August–September.
- All of these fall after the project ends. They only matter if the user keeps working on the paper.

**Architecture and simulation venues**
- **CAADRIA 2027** (Suzhou, Mar 31, 2027): papers were due Jul 27, 2026, so it is **closed**. This comes from an aggregator. — [myhuiban CAADRIA](https://www.myhuiban.com/conference/5719?lang=en-us)
- **eCAADe**: the 2025 abstract (300 words) was due **Jan 15, 2025** and the full paper Apr 1, 2025. The 2024 abstract was due Jan 15, 2024. eCAADe 2027 is therefore ESTIMATED at an abstract around mid-January 2027 and a full paper around April 2027. — [eCAADe 2025 submission](https://ecaade2025.metu.edu.tr/submission); [eCAADe 2024 abstracts](https://cyprusconferences.org/ecaade2024/?p=20)
- **SimAUD**: the 2025 edition (inside ANNSIM 2025) had papers due Jan 19, 2025, with full and short paper tracks. Earlier editions had November deadlines that were often extended (e.g., Nov 10, 2017 for 2018; extended to Nov 26, 2018 for 2019). No 2026 or 2027 dates were found. — [onebuilding SimAUD 2025](https://onebuilding.org/archive-part2/bldg-sim-onebuilding.org/2024-October/005136.html); [SimAUD 2019 extension](https://onebuilding.org/archive-part2/bldg-sim-onebuilding.org/2018-November/000548.html); [SimAUD 2018](https://onebuilding.org/archive-part2/bldg-sim-onebuilding.org/2017-November/000084.html)
- **ACADIA**: not searched (gap).

### Inferences
- **Recommended plan**, given about 1–2 months left:
  1. **Primary target: an ICLR 2027 workshop** (paper around Feb 1, 2027, 4-page short-paper track). Check the accepted-workshop list after **Nov 29, 2026** for memory, world-model, agent, NeuroAI/cognitive-map, embodied or negative-results workshops. This gives time to finish experiments in Nov–Dec and write in January.
  2. **Architecture-side twin paper: eCAADe 2027**, which would need a short abstract around mid-January (ESTIMATE). Frame it as "RL occupant agents for wayfinding assessment of hospital plans." Most CAAD venues are archival with ISBN proceedings, so check whether the ICLR workshop allows it. Most ML workshops are non-archival and permit dual submission.
  3. **Optional stretch: AAAI-27 NeuroAI 2.0** (short non-archival track) or **Human-Centric Agentic Mobility**, both due Nov 20, 2026. That is only three weeks away and would mean submitting current results with little extension. It is feasible only if a clean 4-page story already exists.
  4. **Backup: CVPR 2027 embodied workshops** (Mar–Apr 2027), and RLC 2027 workshops in summer 2027 (ESTIMATE).
- Non-archival ML workshops let the work later go to RLC 2027 or a CAAD venue without conflict. Archival workshops (NeuroAI full papers, W3PHIAI, WDFM-EAI) do not.
- RLDM 2027 and CogSci 2027 are plausible extra venues for a grid-cell or cognitive-map angle. Neither was verified in this session (see Gaps).

### Gaps
- No ICLR 2027 workshop titles yet (announced Nov 29, 2026).
- No ICML 2027, RLC 2027, ICRA 2027 or IROS 2027 workshop calls were found; all are presumably not yet released.
- The embodied-ai.org CVPR flagship workshop details were not retrieved.
- SimAUD 2026/2027 dates were not found; simaud.org was not reached.
- ACADIA 2027 dates were not searched.
- The deadline for AAAI NeuroAI 2.0 was not stated on the AAAI page. Check its site.
- RLDM 2027 and CogSci 2027 deadlines are unverified.

## Q2. What makes a small empirical RL workshop paper compelling?

### Takeaway
Reviewers reward three things: a single sharp question (for example, "when does memory or an aux loss actually help in indoor navigation?"), sound statistics (many seeds, interval estimates, aggregate metrics), and a released, simple environment. Negative or surprising results are explicitly welcome at ICBINB-type workshops, and a well-explained negative result is even rewarded there. The "memoryless MLP beats LSTM" finding fits this template well, as long as it comes with a mechanistic explanation of why the rooms do not need memory.

### Cited Findings
- ICBINB asks for papers on "failures and unexpected difficulties". It gives an **Entropic Award** for the most surprising negative result and a **Didactic Award** for the best-explained paper, and it offers a non-archival option. — [ICBINB 2026](https://sites.google.com/view/icbinb-2026/submit)
- The ICLR 2026 workshop paper "Do LLMs Build Spatial World Models? Evidence from Grid-World Maze Tasks" is an example of an accepted small, controlled maze study whose main result is that performance depends on representation and prompt format. — [ICLR 2026 virtual page](https://iclr.cc/virtual/2026/10018071)
- Templates for large-scale "what matters" empirical RL studies (UNVERIFIED this session, well-known papers):
  - Andrychowicz et al., *What Matters in On-Policy RL? A Large-Scale Empirical Study* (PPO design choices). — [arXiv 2006.05990](https://arxiv.org/abs/2006.05990)
  - Henderson et al., *Deep RL that Matters* (seeds, variance). — [arXiv 1709.06560](https://arxiv.org/abs/1709.06560)
  - Agarwal et al., *Deep RL at the Edge of the Statistical Precipice* (rliable: IQM, stratified bootstrap CIs, performance profiles). — [arXiv 2108.13264](https://arxiv.org/abs/2108.13264)
  - Huang et al., *The 37 Implementation Details of PPO* (ICLR blog track). — [blog](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- Memory-benchmark templates (UNVERIFIED this session):
  - POPGym (Morad et al., ICLR 2023) compares many memory models across POMDPs. — [arXiv 2303.01859](https://arxiv.org/abs/2303.01859)
  - Ni et al., *Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs*. — [arXiv 2110.05038](https://arxiv.org/abs/2110.05038)
  - Ni et al., *When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment* (NeurIPS 2023). It separates memory length from credit-assignment length, which is directly useful for explaining when the MLP suffices. — [arXiv 2307.03864](https://arxiv.org/abs/2307.03864)
- Prior work the project directly replicates or extends (UNVERIFIED this session):
  - Mirowski et al., *Learning to Navigate in Complex Environments* (depth-prediction and loop-closure aux losses with LSTM, the exact losses in this project). — [arXiv 1611.03673](https://arxiv.org/abs/1611.03673)
  - Jaderberg et al., UNREAL aux tasks. — [arXiv 1611.05397](https://arxiv.org/abs/1611.05397)
  - Banino et al., grid-like representations in agents (Nature 2018). — [Nature](https://www.nature.com/articles/s41586-018-0102-6)
  - Wijmans et al., *Emergence of Maps in the Memories of Blind Navigation Agents* (ICLR 2023): LSTM agents with only egomotion sensing form map-like memory. This is the closest analogue to ray-cast plus LSTM. — [arXiv 2301.13261](https://arxiv.org/abs/2301.13261)

### Inferences
Checklist for this paper:
1. **One question in the title**, for example "When does memory help? Design choices for RL indoor navigation from one room to mazes."
2. **At least 5, ideally 10, seeds per configuration**, reported with IQM and 95% bootstrap CIs or performance profiles (rliable). Run the same budget across arms.
3. **A complexity axis** (room to hard maze) as the x-axis of the main figure, showing where each design choice stops or starts helping.
4. **A mechanistic explanation** of the negative result. For example, show that the rays already make the state nearly Markov in the current rooms (an "aliasing" metric: the fraction of positions with near-identical observations). Then show that memory starts to matter once aliasing is introduced (similar rooms, colour cues).
5. **Release** a pip-installable Gymnasium env plus configs. This is cheap and reviewers credit it.
6. **The Mirowski-style replication framing**, "aux losses from 2016 do / don't transfer to ray-cast agents", turns the LSTM plus loop-closure failure into a citable finding rather than a bug.

### Gaps
- No curated list of accepted RL-navigation workshop papers from 2025–2026 was retrieved (OpenReview workshop pages were not searched). The ICLR 2026 maze paper is the only directly verified example.

## Q3. Which additional experiments rank highest by (insight × publishability) / cost?

### Takeaway
Ranked shortlist:
1. **Held-out layout generalization** (train on N layouts, test on unseen).
2. **Parameter-scaling curve plus aux-loss-versus-matched-parameters comparison**, run together as one experiment.
3. **Observation ablations** (ray count, FOV).
4. **Aliasing / colour-cue rooms** to show where memory becomes necessary.
5. **GRU** as a cheap extra backbone.
6. **Human-likeness / shortest-path-ratio metrics.**
7. **Sequential multi-goal routes.**
8. **Transformer or SSM backbones.**
9. **Real hospital plans.**
10. **A small VLA or LLM baseline**, low priority.

### Cited Findings and per-experiment assessment

**(c) Generalization to held-out / procedurally generated layouts. Rank 1.**
- Prior evidence: Cobbe et al. showed that RL agents overfit even large training sets of levels and that generalization improves with the number of training levels. These are the "train on N, test on unseen" curves (CoinRun, then Procgen). — [Quantifying Generalization in RL, arXiv 1812.02341](https://arxiv.org/abs/1812.02341); [Procgen, arXiv 1912.01588](https://arxiv.org/abs/1912.01588) (UNVERIFIED this session)
- Pérez-D'Arpino et al. trained RL navigation on simple layouts and tested on unseen reconstructions of real environments. — [Robot Navigation in Constrained Pedestrian Environments](https://ai.stanford.edu/~cdarpino/socialnavconstrained/)
- Cost: low to medium. Generate K random mazes or floorplans and train on N ∈ {1, 4, 16, 64}. Each run costs the same as now.
- Figure: test success versus number of training layouts, one line per architecture (MLP / LSTM / frame stacking), with the train–test gap shaded.
- Why it ranks first: it is directly relevant to Arxitect, where agents must work on new plans the architect draws. It may also overturn the "MLP is best" result, because memorization of a single layout favours an MLP.

**(a) + (b) Parameter scaling and aux loss versus a matched parameter budget. Rank 2, run together.**
- Prior evidence: Hilton et al. fit scaling laws for single-agent RL, with compute-optimal model size growing with compute. — [arXiv 2301.13442](https://arxiv.org/abs/2301.13442)
- In value-based deep RL, heavy pruning keeps or improves performance, which suggests many RL networks are over-parameterized. — [Obando-Ceron et al., arXiv 2402.12479](https://arxiv.org/abs/2402.12479)
- Larger networks often fail to help in deep RL unless the architecture changes (for example, MoE). — [arXiv 2402.08609](https://arxiv.org/abs/2402.08609)
- All three sources are UNVERIFIED this session.
- Cost: low. Sweep hidden width over roughly {8, 16, 32, 64, 128, 256}, giving about 6 sizes × 3 seeds × 2–3 rooms. A small MLP trains in minutes.
- Figure: final success (IQM) versus parameter count on a log axis, one curve per room difficulty. Overlay "base + aux head" points at equal total parameters to find the crossover where an aux loss beats spending the same parameters on width. This is exactly the user's "trade-off point" idea.
- Note: aux heads are discarded at test time, so report both training-time and inference-time parameter counts.

**(h) Observation ablations (ray count, FOV, max ray range, noise). Rank 3.**
- Cost: very low. These are environment parameters, so it is a pure sweep.
- Figure: a heat map of success over ray count × FOV for MLP versus LSTM. The hypothesis is that memory helps more as FOV shrinks, because partial observability increases. This gives an interpretable, controllable partial-observability axis, which reviewers like.
- Prior evidence: Wijmans et al. show that agents with minimal sensing (egomotion only) still navigate by building memory maps, which motivates the low-FOV end. — [arXiv 2301.13261](https://arxiv.org/abs/2301.13261) (UNVERIFIED this session)

**Aliasing / colour cues in similar rooms (user's own idea). Rank 4.**
- This is the needed counterpart to the negative result. Build layouts with perceptually identical rooms where only a remembered cue (colour seen earlier) disambiguates the goal. It is a T-maze / "memory length" task in the style of Ni et al. — [arXiv 2307.03864](https://arxiv.org/abs/2307.03864) (UNVERIFIED this session)
- Cost: low to medium (environment change).
- Figure: success versus cue–decision delay for MLP, frame stacking with k frames, LSTM and GRU. The MLP should fall to chance, and frame stacking should fail once the delay exceeds k.

**(g) Newer memory backbones. GRU is rank 5; transformer/SSM is rank 8.**
- Prior evidence: in POPGym, simple recurrent models such as GRU were among the strongest across POMDP tasks. — [arXiv 2303.01859](https://arxiv.org/abs/2303.01859)
- Transformers in RL need stabilization (GTrXL). — [arXiv 1910.06764](https://arxiv.org/abs/1910.06764)
- SSMs (S4/S5) have been applied to in-context RL. — [arXiv 2303.03982](https://arxiv.org/abs/2303.03982)
- All UNVERIFIED this session.
- Cost: GRU is nearly free as a drop-in for the LSTM. Transformer or SSM means medium to high engineering cost with PPO plus TBPTT, plus tuning risk.
- Recommendation: add GRU only. Cite POPGym and Ni et al. to justify not going further.

**(e) Human-likeness of trajectories. Rank 6; high value for the architecture venue.**
- Prior evidence:
  - The "Navigation Turing Test" (Devlin et al., ICML 2021) learned to evaluate human-likeness of agent navigation and found that standard RL agents are distinguishable from humans. — [arXiv 2105.09637](https://arxiv.org/abs/2105.09637) (UNVERIFIED this session)
  - The cogARCH framework argues that wayfinding assessment needs heuristic, non-shortest-path occupant behaviour. — [cogARCH, Northumbria](https://researchportal.northumbria.ac.uk/en/publications/cogarch-a-cognitive-agent-framework-to-simulate-wayfinding-by-arc/)
  - A deep-RL path simulation for construction workers was reported to match real trajectories 17.8% better than A*. — [AI-HSE publication record](https://ai-hse.inu.ac.kr/publications/cmshnssk6001rqny9ygah0v2l). This comes from the search summary; the exact metric was not verified.
- Cost: low without human data. Compute path-length ratio versus A*, wall-distance histograms (wall hugging), turning-angle statistics, revisit rate and DTW to the shortest path. Real human data would be expensive; a small VR or desktop study is out of scope for 1–2 months.
- Figure: overlaid trajectory heat maps (agent versus A*) on the floorplan, plus a table of path-efficiency and wall-distance stats per architecture.

**(f) Sequential multi-goal tasks (doctor routes). Rank 7.**
- Cost: medium. It needs goal conditioning first, which is already on the user's list.
- Figure: success versus number of sequential goals.
- Value: strong for the hospital story. It is also a natural memory test, because the agent must remember which goals are done. Without goal conditioning it is weak.

**(i) Real hospital floorplans. Rank 9.**
- Prior evidence:
  - Deep Q-learning on a grid of a real hospital floor (Al-Istishari Hospital, Amman) exists as a precedent. — [record](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c272t-9aaa7953b9548176a07fdc064e0be84caec6625f6261eea106095dab34dc7e5a3?lng=en)
  - An agent-based (non-RL) wayfinding study of a multistory hospital (HERD 2024) found that optimized plans reduced distance and time for all user groups. — [record](https://www.bohrium.com/paper-details/utilizing-agent-based-modeling-for-optimization-of-wayfinding-in-hospital-a-case-study/994781438375100418-97279)
- Cost: medium. Vectorizing 2–3 plans is manual unless IVIA/Arxitect already has them. Zero-shot evaluation is cheap once (c) exists.
- Figure: success and path ratio on real plans for policies trained on procedurally generated plans. This works well as a capstone "transfer" figure if (c) is done.

**(d) Small VLA or LLM baseline. Rank 10, not worth it for this paper.**
- Prior evidence:
  - MazeEval: with only coordinate and wall-distance feedback, most LLMs fail beyond 9×9 mazes because of looping; O3 handles up to 30×30. This is a single-author preprint. — [arXiv 2507.20395](https://arxiv.org/abs/2507.20395)
  - LLM maze performance swings widely with representation (Gemini 80–86% with adjacency lists versus 16–34% with visual grids). — [ICLR 2026 workshop paper](https://iclr.cc/virtual/2026/10018071)
  - BALROG benchmarks LLM/VLM agents on games, including navigation-heavy ones. — [arXiv 2411.13543](https://arxiv.org/abs/2411.13543) (UNVERIFIED this session)
- Cost: high. API cost per step times thousands of steps, or a GPU for a small VLA. Prompt-design choices also confound the result, and a ray-cast vector is not a natural VLA input. VLAs such as OpenVLA (7B, [arXiv 2406.09246](https://arxiv.org/abs/2406.09246), UNVERIFIED) are built for robot manipulation with images and would need rendering plus fine-tuning.
- Possible cheap version: a zero-shot LLM given the ray readings as text on 2–3 rooms, about 50 episodes each, reported as one baseline row. Expect it to be poor and slow. It would mainly support the claim that "a 10k-parameter PPO policy beats a billion-parameter generalist at a fraction of the cost". That claim pairs nicely with the parameter-scaling figure, but it is optional.

**Grid cells.** Prior work also argues that grid-cell emergence in trained networks depends strongly on hand-picked supervision and design choices (Schaeffer, Khona & Fiete, NeurIPS 2022, "No Free Lunch from Deep Learning in Neuroscience"; UNVERIFIED, URL not retrieved). This supports reporting a grid-cell-input null result honestly.

### Inferences
- **Minimum viable workshop paper** (fits in 4 weeks of compute plus 2 weeks of writing): the current comparison with ≥5 seeds and rliable stats, plus (c) generalization, plus (a/b) the scaling and aux trade-off curve, plus (h) the FOV/ray sweep, plus the aliasing/colour-cue task with GRU added.
  - This gives 4 main figures: complexity-axis bars, generalization curve, parameter curve with aux overlay, and FOV × memory heat map.
  - The combined story would be "memory and aux losses only pay off when partial observability or layout diversity demands it." That turns the current negative result into a positive, conditional claim.
- **For the eCAADe/CAAD version**, add (e) trajectory metrics and (i) one or two real plans. Lighter statistics are acceptable there, but the occupant-simulation framing is needed.
- **Defer** transformer/SSM, multi-goal and VLA work to future work, unless goal conditioning is already done, in which case multi-goal becomes cheap.

### Gaps
- No per-run compute numbers for the user's setup were available. Cost estimates are relative.
- The 17.8% figure for the construction-worker deep-RL study came from a search summary, not the paper itself.
- No human-trajectory dataset for indoor or hospital wayfinding was identified that could be used off the shelf.

## Q4. What recent (2023–2026) work on RL for building, floorplan occupancy or wayfinding simulation should the paper cite or position against?

### Takeaway
The niche is thin. Prior occupant-simulation work in architecture is mostly **non-RL**: rule-based or A*-based agents (SimArch, cogARCH, hospital ABM) or LLM "generative agents" (TravelAgent). RL in architecture is mostly used to **design layouts** or **evacuation exits**, not to simulate occupants' wayfinding. An empirical study of RL design choices for occupant-like wayfinding agents therefore fills a real gap. Robotics navigation work such as EDEN covers the embodied-ML side.

### Cited Findings
- **SimArch** (2018): a multi-agent system that takes floorplans and produces occupant density maps. It uses an MDP and weighted A* rather than learned policies. Baseline to position against. — [arXiv 1807.03760](https://arxiv.org/abs/1807.03760)
- **cogARCH**: a cognitive-agent framework for pre-occupancy wayfinding evaluation in multilevel buildings, using heuristic rather than optimal shortest-path occupants. It was applied to three design variants of an education building. — [Northumbria portal](https://researchportal.northumbria.ac.uk/en/publications/cogarch-a-cognitive-agent-framework-to-simulate-wayfinding-by-arc/)
- **Zhang, Schwartz, Usman, Faloutsos, Kapadia (2022)**: modify agent policies to produce movement patterns and directional guidance for occupant agents in a 3D engine. Closest prior "learned occupant behaviour for design" work (abstract only seen). — [author page](https://cs.paperswithcode.com/author/petros-faloutsos)
- **TravelAgent (arXiv 2412.18985, Dec 2024)**: LLM generative agents in the built environment. It runs 100 simulations across spatial layouts, reports 76% task completion, and aims at evaluating and refining spatial designs. This is the main LLM-based competitor framing. — [arXiv 2412.18985](https://arxiv.org/pdf/2412.18985)
- **Hospital wayfinding agent-based modelling (HERD, 2024)**: optimized hospital floor plans reduced total distance and time for all user groups versus the existing layout (non-RL). — [record](https://www.bohrium.com/paper-details/utilizing-agent-based-modeling-for-optimization-of-wayfinding-in-hospital-a-case-study/994781438375100418-97279)
- **Deep Q-learning on a real hospital floor** (Al-Istishari Hospital, custom Gym grid, DQN versus Dueling DQN). — [record](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c272t-9aaa7953b9548176a07fdc064e0be84caec6625f6261eea106095dab34dc7e5a3?lng=en)
- **RL for evacuation layout (TUHH, 2024)**: a grid RL environment in which the agent learns escape paths and places exits to optimize fire-safety layout. — [TUHH TORE](https://tore.tuhh.de/entities/publication/c0ca1f2a-fa49-4b06-b9e4-b403b92f6b86)
- **Fitkau & Hartmann (TU Berlin, 2025)**: single-agent PPO learning escape strategies while door placement is modified in a multi-room prototype. — [DepositOnce](https://depositonce.tu-berlin.de/items/1ae547f1-5595-409c-a5aa-fdae010ed3c3/full)
- **SpaceLayoutGym** (Kakooee & Dillenburger, ETH Zurich): an RL environment for architectural space layout design, using imitation-learning pretraining followed by PPO fine-tuning. Same institution, so worth citing and possibly contacting. — [record](https://bohrium.dp.tech/paper/arxiv/25dc413189305063ea5273bb40d601cd719bd46d82cb7e6d80641b00a95b6d99)
- **EDEN (arXiv 2506.03046, 2025)**: entorhinal (grid-cell) driven egocentric navigation trained with PPO, aimed at robotic deployment. A direct comparison point for the grid-cell arm. — [arXiv 2506.03046](https://arxiv.org/pdf/2506.03046)
- **Deep-RL construction-worker path simulation (Automation in Construction, 2023)**: reported trajectories matched real workers 17.8% better than A*. It supports the claim that RL can be more human-like than shortest-path planners. — [AI-HSE record](https://ai-hse.inu.ac.kr/publications/cmshnssk6001rqny9ygah0v2l)
- **Azizi et al. (2020)**: floorplan embeddings as attributed graphs that encode geometry, semantics and occupant behaviour. — [aitopics floor plan tag](https://aitopics.org/tag/floor%20plan)

### Inferences
- Positioning sentence: "Occupant wayfinding simulators in CAAD rely on hand-designed heuristics (cogARCH, SimArch) or costly LLM agents (TravelAgent). Learned RL policies are used mainly for layout or exit design. We provide the first controlled study of which RL design choices (memory, aux losses, encoders) matter for egocentric occupant-like navigation in floorplans."
  - The word "first" should be softened unless a CumInCAD search confirms it.
- The TravelAgent comparison also argues against a costly VLA/LLM baseline. Cite it as the LLM-agent alternative instead of re-running one.

### Gaps
- The CumInCAD database was not searched directly. eCAADe, CAADRIA, ACADIA and SimAUD 2023–2026 proceedings may hold RL-wayfinding papers that were not found here.
- The SpaceLayoutGym year and venue were not confirmed.
- The exact venue for the 2022 Zhang et al. paper was not confirmed.
