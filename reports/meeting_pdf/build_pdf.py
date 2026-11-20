"""Supervisor meeting report: project direction, experiment plan, results so far, problems and fixes."""
import json, os, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image,
                                PageBreak, KeepTogether, CondPageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
FONTS = "/opt/anaconda3/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf"
pdfmetrics.registerFont(TTFont("Sans", f"{FONTS}/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", f"{FONTS}/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Italic", f"{FONTS}/DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("Sans-BoldItalic", f"{FONTS}/DejaVuSans-BoldOblique.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic", boldItalic="Sans-BoldItalic")

INK, INK2, RULE, TINT = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#c3c2b7"), colors.HexColor("#f4f3ef")
ACCENT = colors.HexColor("#2a78d6")
GOOD, PART, BAD, OPEN = colors.HexColor("#1baf7a"), colors.HexColor("#eda100"), colors.HexColor("#d6336c"), colors.HexColor("#8a8984")

S = {
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=19, leading=23, textColor=INK, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=10.5, leading=14, textColor=INK2, spaceAfter=2),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=13.5, leading=17, textColor=INK, spaceBefore=10, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=10.5, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=8.8, leading=12.4, textColor=INK, spaceAfter=4, alignment=TA_LEFT),
    "bullet": ParagraphStyle("bullet", fontName="Sans", fontSize=8.8, leading=12.2, textColor=INK, leftIndent=11,
                             bulletIndent=2, spaceAfter=2.2),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=7.6, leading=9.8, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=7.6, leading=9.8, textColor=INK),
    "caption": ParagraphStyle("caption", fontName="Sans", fontSize=7.6, leading=10, textColor=INK2, spaceBefore=2, spaceAfter=8),
    "box": ParagraphStyle("box", fontName="Sans", fontSize=8.8, leading=12.4, textColor=INK),
    "empty": ParagraphStyle("empty", fontName="Sans-Italic", fontSize=8.4, leading=11.5, textColor=OPEN),
}
W = A4[0] - 36 * mm


def esc(text):
    """W&B is the only bare ampersand in the text; Paragraph reads XML."""
    return text.replace("W&B", "W&amp;B")


def P(text, style="body"):
    return Paragraph(esc(text), S[style])


def bullets(items):
    return [Paragraph(esc(t), S["bullet"], bulletText="•") for t in items]


def table(rows, widths, header=True, zebra=True, status_col=None):
    data = [[c if not isinstance(c, str) else Paragraph(esc(c), S["cellb" if (header and i == 0) else "cell"])
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
             ("RIGHTPADDING", (0, 0), (-1, -1), 3.5), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2),
             ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE)]
    if zebra:
        for i in range(2, len(rows), 2):
            style.append(("BACKGROUND", (0, i), (-1, i), TINT))
    t.setStyle(TableStyle(style))
    return t


def figure(name, width_frac, caption):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return [P(f"[figure {name} missing]", "empty")]
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    w = W * width_frac
    return [KeepTogether([Image(path, width=w, height=w * ih / iw), P(caption, "caption")])]


def callout(text, colour=ACCENT):
    t = Table([[Paragraph(esc(text), S["box"])]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LINEBEFORE", (0, 0), (0, -1), 2.5, colour),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def status(word):
    c = {"Supported": GOOD, "Partly": PART, "Not supported": BAD, "Untestable": OPEN, "Pending": OPEN,
         "Done": GOOD, "Running": PART, "Not started": OPEN, "Fixed": GOOD, "Open": PART, "Explained": GOOD}[word]
    return Paragraph(f'<font color="{c.hexval().replace("0x", "#")}"><b>{word}</b></font>', S["cell"])


def empty(text="Not done yet."):
    return P(text, "empty")


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.2)
    canvas.setFillColor(INK2)
    canvas.drawString(18 * mm, 10 * mm, "Perceive, Remember, or Be Told — direction and results, supervisor meeting")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


c64 = json.load(open(os.path.join(HERE, "c64.json"))) if os.path.exists(os.path.join(HERE, "c64.json")) else None
story = []

# ------------------------------------------------------------------ title + summary
story += [P("Perceive, Remember, or Be Told", "title"),
          P("Project direction, experiment plan and results so far — semester project (ETH IVIA Lab), "
            "supervisor meeting, November 2026", "subtitle"),
          P("Emerson Azevedo Aguiar · state of all runs as of Nov 19, 2026 · "
            "every planned run has finished (E4 Stage 3, E3T at L = 64 and E2 with four rooms final)", "subtitle"), Spacer(1, 6)]

story += [callout(
    "<b>Direction in one sentence.</b> A navigation agent can get the information it needs in three ways: it can "
    "<i>perceive</i> it, <i>remember</i> it, or <i>be told</i> it. The design choices in the project title (observation "
    "space, memory architecture, auxiliary losses, action space, reward) mostly decide which route the agent uses, so "
    "whether a choice helps depends on what the other routes already provide. The project therefore moved from a "
    "leaderboard (\"which configuration wins\") to controlled tests of when memory is needed and what can replace it.")]
story += [Spacer(1, 6), P("Where we stand", "h2")]
story += bullets([
    "<b>Standard floorplans don't test memory.</b> In every room we had (two_room up to hard_maze) the memoryless MLP "
    "matches or beats the LSTM, with or without the colour place code (E1), with a narrower field of view (E1b) and with "
    "goal/position information (E3). No LSTM−MLP difference survives Holm correction.",
    "<b>The twin rooms make memory necessary, and the memory horizon is a clean dial (E2, headline).</b> Memoryless "
    "agents fall to chance (0.5) from cue distance L = 32 on; the LSTM with a 16-step gradient window solves 5/5 seeds "
    "at L = 32, 3/5 at 64 and 0/5 at 128; a 256-step window keeps 4/5 at 128; nothing solves L = 256. Diagnostics "
    "(swap, reset, mirror) show the cue sits in the recurrent state on all 64 solved recurrent seeds. With four rooms "
    "(two bits instead of one) the best backbones solve 7/15 seeds at L = 32 and none at L = 128.",
    "<b>The body is a memory too.</b> Memoryless agents keep the cue in their position in the corridor: FS16 solves "
    "L = 32 on 2/5 seeds this way, and the MLP solves when told the cue only every 4th step by steering through the "
    "blank steps (E3T). A timing check found that E3T's longest interval (64 steps) puts the second telling right at the "
    "decision; rerun with a random telling phase, the LSTM remembers (5/5) and FS16 learns body memory (5/5). "
    "At L = 64 (random phase) the split is clean: FS16 solves only while the last telling fits its 16 frames, the "
    "MLP fails from every-16 on, and LSTM-BPTT256 remembers in every condition (20/20 seeds, swap-verified).",
    "<b>Staged ablation (E4).</b> Stage 1 and 2 are done: CNN encoder, collision penalty and discrete 3-action control "
    "lead their themes; auxiliary losses give no gain against matched references; CNN + LSTM + loop closure collapses "
    "(an interaction with the collision penalty, explained below). Stage 3 (5 configs × 2 rooms × 10 final seeds, 100 runs) is done: best combined beats the baseline in both rooms (+0.36 / +0.54 AUC) and wins 4 of 8 pre-registered tests at α = 0.01.",
])
story += [P("What I'd like to decide in the meeting", "h2")]
story += bullets([
    "Agree on the framing: E2 + E3T (memory horizon and body-as-memory) as the core of the paper, E1/E3 as the "
    "\"standard floorplans don't need memory\" result, E4 as the methodology contribution.",
    "E4 Stage 3 result: report the 4 significant comparisons as the claim and the rest as \"more reliable across "
    "seeds\" (no extra seeds added after seeing the p-values). Agree?",
    "E3T at L = 64 is done and supports H8 (table in the E3T section). E2 with four rooms (K = 4) is done too: two bits "
    "are learned at L = 32 (GRU 4/5, LSTM-BPTT256 3/5 seeds) but at L = 128 by no backbone within the budget. Proposal: "
    "appendix, marked exploratory, no longer-budget rerun. Agree?",
    "Next four weeks: extra seeds only where the pre-registered rules ask for them, then the final rliable statistics and "
    "frozen figures (Dec 10–16), so the writing starts Dec 17. Anything you want run before the figures freeze?",
    "Target venue: ICLR 2027 workshops (list out Nov 29, deadline ~Feb 1) and/or eCAADe 2027 (abstract ~mid-January).",
])
story += [P("All material behind this report (training curves per sweep, room pictures, tables per seed, and rollout videos of whole episodes from the saved checkpoints, best and worst seed per configuration) is in the W&B group <a href='https://wandb.ai/eazevedo-eth-z-rich/semproj/groups/report_nov19' color='#2a78d6'>report_nov19</a>, one run per section; links in Section 6.", "caption")]

story += [PageBreak(), P("Status of all experiments", "h2")]
story += [table([
    ["Phase", "Question it answers", "Runs", "Status"],
    ["E0 Fixes + pilot", "Is the pipeline valid; how long does a run take; which rooms separate agents?", "32", status("Done")],
    ["E1 Place-code audit", "Does the MLP win only because the colour field tells it where it is?", "120", status("Done")],
    ["E1b Field of view", "Does a narrower view create a memory advantage?", "40", status("Done")],
    ["E2 Twin rooms", "At what cue distance L does each memory architecture fail? (+ layers, + four rooms)", "175 + 10 + 30", status("Done")],
    ["E3 Information sources", "Does being told the goal or position replace memory in normal rooms?", "180", status("Done")],
    ["E3T Told vs remembered", "How often must the cue be told so that memory isn't needed?", "105 + 15 + 60", status("Done")],
    ["E4 Stage 1 (screen)", "Which levels of observation/action/reward/capacity/aux survive screening?", "126", status("Done")],
    ["E4 Stage 2 (confirm)", "Do the survivors hold on harder rooms and fresh seeds?", "240 + 26", status("Done")],
    ["E4 Stage 3 (final)", "Best combined vs baseline vs leave-one-theme-out, 10 seeds, α = 0.01", "100", status("Done")],
    ["Stretch / eCAADe", "Held-out layouts; trajectory statistics for architects", "—", status("Not started")],
], [0.2, 0.56, 0.1, 0.14])]
story += [P("Runs are counted per seed; seeds 0–2 are screening only, 100–104 confirmation, 1000–1009 final claims.",
            "caption")]

# ------------------------------------------------------------------ idea, hypotheses
story += [P("1. The idea and the hypotheses", "h1")]
story += [P("The original results showed a memoryless MLP beating recurrent agents. Instead of hiding that, the plan "
            "turns it into a testable, conditional claim: memory, goal information and position codes are substitutes, "
            "each helps only when the others are missing. The gaps this fills: design choices in navigation RL are "
            "studied one at a time in incompatible setups (Mirowski 2017, Banino 2018, Wijmans 2023); \"memoryless wins\" "
            "results are often confounded by information leaks (Ni et al. 2023); and memory demand is rarely a "
            "controllable dial in navigation. The hypotheses were written before the runs (status from the results so far):")]
story += [table([
    ["ID", "Hypothesis", "Exp.", "Status", "Evidence so far"],
    ["H1", "The MLP's lead shrinks or flips once the colour place code is removed and goals are randomised", "E1",
     status("Not supported"), "Colour off never hurts the MLP; the gap shrinks (medium_maze −0.54 → −0.27 AUC) but never flips; nothing significant"],
    ["H2", "In twin rooms each backbone fails once L exceeds its horizon (MLP at chance, FS-k fails for L > k, LSTM past its BPTT window, GRU lasts longest)",
     "E2", status("Partly"), "Horizons hold for MLP and LSTM windows; FS16 bridges L = 32 by body memory; GRU ties LSTM-BPTT256 (4/5 at L = 128); nothing solves L = 256. Two bits (K = 4): 7/15 at L = 32, 0/15 at L = 128"],
    ["H3", "Telling the goal or position closes the MLP–recurrent gap in open layouts but not in dead-end layouts", "E3",
     status("Untestable"), "E3 rooms have no memory gap to close (both backbones 0.93–0.99 AUC); moved to the twin rooms (E3T)"],
    ["H4", "A frozen last-seen goal or noisy path-integrated position brings back the recurrent advantage", "E3",
     status("Not supported"), "LSTM trails most in exactly those conditions (−0.12, −0.16), not significant"],
    ["H5", "A hand-crafted hexagonal grid code is no better than random Fourier features of the same size", "E3",
     status("Supported"), "Null result as predicted"],
    ["H6", "Perception, capacity, action and reward rank consistently from easy to hard rooms; memory and aux don't", "E4",
     status("Pending"), "Stage 1 → 2 τ = 1.0 on medium_maze; Stage 3: best first and baseline last in both rooms, the leave-one-out configs swap order"],
    ["H7", "Memory gain grows with perceptual aliasing and as the field of view narrows", "E1a/b",
     status("Not supported"), "Aliasing score ≈ 0 in every room; at 90° FOV the LSTM drops more than the MLP"],
    ["H8", "Being told the cue replaces memory only as far as the gap between tellings fits the agent's memory", "E3T",
     status("Supported"), "At L = 64: MLP fails from every-16 on, FS16 solves exactly while the telling fits its 16 frames, LSTM remembers throughout. At L = 32 the body bridged gaps (MLP every-4, FS16 every-64 random phase)"],
], [0.05, 0.36, 0.07, 0.12, 0.40])]

# ------------------------------------------------------------------ methods
story += [P("2. Testbed and evaluation", "h1")]
story += [P("<b>Agent.</b> PPO, egocentric 2D ray casts (37 rays × 7 channels: hit flags, distance, colour) plus "
            "proprioception (last speed and turn), continuous actions (turn, speed). Encoders: MLP or CNN over rays. "
            "Backbones: MLP (memoryless), frame stacks of 4 or 16 frames (FS4/FS16), LSTM (2 × 64 units, truncated BPTT "
            "window 16/64/256) and GRU. 256 parallel environments × episode-length steps per iteration, 1000 iterations, "
            "early stopping on eval success. All training runs on ETH Euler (CPU).")]
story += figure("rooms.png", 1.0, "Rooms used in the hard tier and the twin-room memory task (drawn from the room "
                "JSONs). Twin rooms: the cue walls (pink) show one of two colours in the start alcove; a bent shaft hides "
                "them once the agent leaves; a corridor of length L leads to the junction (orange box); the goal is behind "
                "a baffle in one of two identical rooms (green), and entering the wrong room ends the episode.")
story += [P("<b>Metrics.</b> Primary: success AUC over training, padded to the full budget so early-stopped and slow runs "
            "are scored on the same horizon. Secondary: success rate, SPL, nDTW. Evaluation follows your scheme of 10 eval "
            "runs × 10 episodes per seed (mean of means with standard errors); the statistical unit is the training seed. "
            "<b>Statistics:</b> IQM with stratified-bootstrap 95% CIs (rliable), probability of improvement, Welch's t at "
            "α = 0.01 with Holm correction for claims. <b>Seeds:</b> three disjoint, pre-fixed sets — 0–2 for screening only "
            "(never reported), 100–104 for confirmation (E1–E3, E4 Stage 2), 1000–1009 for final claims (Stage 3). "
            "<b>Recurrent rule:</b> LSTM is the default; GRU runs in E2 and elsewhere only if the LSTM falls below the MLP "
            "with non-overlapping CIs (never \"best of LSTM and GRU\").")]

# ------------------------------------------------------------------ E0
story += [P("3. Results by phase", "h1"), P("E0 — Fixes and pilot", "h2")]
story += [P("<b>Tests:</b> whether the pipeline is valid (several bugs had to go first, see Section 5), what a run costs, "
            "and which rooms separate agents. <b>Result:</b> a full 1000-iteration run costs ≈ 2.1 h (MLP) / 2.8 h (LSTM) "
            "in any room; cost is set by iterations, not by the room. Learning rates: LSTM 3e-4, MLP 1e-4 (one seed, "
            "screening only). With colour off and random goals, every room easier than C_maze saturates — the goal is "
            "in view from the start about half the time — so the staged ablation moved to the hard tier.")]
story += [table([
    ["Room", "MLP AUC", "LSTM AUC", "Goal in view at start", "Median path", "Time per run"],
    ["two_room / T_maze / I_maze", "0.96–0.99", "0.99–1.00", "53%", "265", "15–19 min"],
    ["four_room", "0.99", "1.00", "52%", "328", "11–26 min"],
    ["C_maze", "0.93", "0.96", "16%", "721", "26–27 min"],
    ["medium_maze", "0.57", "0.81", "12%", "1124", "1.9–2.6 h"],
    ["hard_maze", "0.18", "0.00", "7%", "685", "2.8–3.2 h"],
], [0.28, 0.12, 0.12, 0.18, 0.13, 0.17])]
story += [P("Room probe, seed 0 (screening). 8192-step episodes were worse than 4096, so the hard tier keeps 4096.", "caption")]

# ------------------------------------------------------------------ E1
story += [CondPageBreak(90 * mm), P("E1 — Information-leak audit (\"perceive\") and E1b — field of view", "h2")]
story += [P("<b>Tests:</b> H1, whether the MLP wins because the global Perlin colour on the walls works as a place code "
            "and the fixed goal makes every room a reactive lookup. 3 conditions (colour on + fixed goal, colour off + fixed, "
            "colour off + random goal) × MLP/LSTM × 4 rooms × seeds 100–104 = 120 runs. E1b: FOV 90/180/270/360° in C_maze "
            "(40 runs). E1a: an aliasing score (share of near-identical observations from far-apart places), no training.")]
story += [table([
    ["Room", "Condition", "MLP AUC (IQM [95% CI])", "LSTM AUC (IQM [95% CI])"],
    ["two_room, four_room", "all three", "0.98–0.99", "0.98–0.99"],
    ["C_maze", "colour on, fixed goal", "0.79 [0.62, 0.86]", "0.57 [0.09, 0.85]"],
    ["C_maze", "colour off, fixed goal", "0.79 [0.64, 0.93]", "0.68 [0.39, 0.92]"],
    ["C_maze", "colour off, random goal", "0.96 [0.95, 0.97]", "0.87 [0.72, 0.96]"],
    ["medium_maze", "colour on, fixed goal", "0.67 [0.43, 0.86]", "0.12 [0.05, 0.19]"],
    ["medium_maze", "colour off, fixed goal", "0.88 [0.57, 0.94]", "0.15 [0.08, 0.62]"],
    ["medium_maze", "colour off, random goal", "0.56 [0.47, 0.62]", "0.23 [0.09, 0.55]"],
], [0.22, 0.26, 0.26, 0.26])]
story += bullets([
    "The place-code explanation doesn't hold: switching colour off never hurts the MLP. The gap to the LSTM shrinks "
    "(C_maze −0.24 → −0.10, medium_maze −0.54 → −0.27) but mostly because random goals make the MLP worse; it never "
    "flips, and no difference survives Holm (closest: medium_maze colour on/fixed, Holm p = 0.046).",
    "In C_maze the gap is learning speed (both end at 0.92–1.00 success); in medium_maze it is also final success "
    "(13 of 15 LSTM seeds end ≤ 0.62 within the budget).",
    "E1a: the aliasing score is ≈ 0 (≤ 0.005) in every room and colour mode, so it can't explain anything. "
    "E1b: narrowing the view doesn't create a memory gain; at 90° the LSTM drops more than the MLP (0.47 vs 0.81).",
    "Consequence: colour stays off for every later experiment, and the memory question needs a task built for it (E2).",
])
story += figure("wb_e1.png", 1.0, "Training curves from W&B (group e1_place_code_audit): eval success per PPO iteration, "
                "mean ± SE over seeds 100–104, smoothed over 5 evals, early-stopped runs carried forward.")
story += figure("fig2_left.png", 1.0, "E1 summary from the evaluation pipeline (paper Fig. 2 left).")
story += figure("fig_e1b_fov.png", 0.55, "E1b: AUC vs field of view in C_maze.")

# ------------------------------------------------------------------ E2
story += [CondPageBreak(120 * mm), P("E2 — Twin rooms with a cue (\"remember\"), headline experiment", "h2")]
story += [P("<b>Tests:</b> H2, whether each architecture fails at a predictable memory horizon. The cue (one of two "
            "colours) is shown only in the start alcove; after a corridor of length L the agent must choose between two "
            "identical rooms. A memoryless agent can only guess (0.5). 5 corridor lengths (L = 8, 32, 64, 128, 256 steps) × "
            "7 backbones (MLP, FS4, FS16, LSTM with BPTT 16/64/256, GRU) × seeds 100–104 = 175 runs.")]
story += [P("<b>Design steps on the way</b> (details in Section 5): the first layout leaked the cue down a straight corridor "
            "(fixed with a bent shaft and a line-of-sight test); the unshaped L = 32 pilot was decided by which seeds "
            "managed to leave the start, not by memory (MLP 0/5, GRU 1/5 seeds even reached the junction), fixed with "
            "potential-based shaping towards the junction, which leaves the optimal policy and the 0.5 ceiling unchanged.")]
story += [table([
    ["Backbone", "L = 8", "L = 32", "L = 64", "L = 128", "L = 256"],
    ["MLP", "3/5 (body)", "0/5", "0/5", "0/5", "0/5"],
    ["FS4", "5/5", "0/5", "0/5", "0/5", "0/5"],
    ["FS16", "5/5", "2/5 (body)", "0/5", "0/5", "0/5"],
    ["LSTM-BPTT16", "5/5", "5/5", "3/5", "0/5", "0/5"],
    ["LSTM-BPTT64", "5/5", "5/5", "3/5", "1/5", "0/5"],
    ["LSTM-BPTT256", "5/5", "5/5", "5/5", "4/5", "0/5"],
    ["GRU (BPTT64)", "5/5", "5/5", "4/5", "4/5", "0/5"],
    ["Junction reached (min over seeds)", "0.97", "0.99", "0.99", "0.97", "0.90"],
], [0.28, 0.12, 0.15, 0.15, 0.15, 0.15])]
story += [P("Seeds solved (success ≥ 0.9) of 5, from 1000 deterministic evaluation episodes per seed.", "caption")]
story += bullets([
    "<b>Every failure is a memory failure.</b> Every seed reaches the junction (0.90–1.00 of episodes); unsolved seeds "
    "pick one room and sit at chance.",
    "<b>The gradient window sets the horizon.</b> LSTM-BPTT16 holds to L = 32 and fades by 64–128; a 256-step window "
    "keeps 4/5 at L = 128; GRU at window 64 matches it. At L = 256 the cue is ~257 steps old at the junction, just past "
    "every window, and nothing solves it.",
    "<b>The cue is in the recurrent state.</b> On all 64 solved recurrent seeds, swapping in the recurrent state of an "
    "episode with the other cue sends the agent to the wrong room (success 0.00–0.07); swapping the pose doesn't "
    "(0.93–1.00). Resetting the state anywhere in the corridor drops success to chance; at the junction much less "
    "(median 0.65–0.93) — the turn is already committed there.",
    "<b>Body memory.</b> At L = 8 the MLP solves 3/5 by steering to one side right after seeing the cue (mirroring its "
    "pose sends seed 100 to 0.04). FS16 at L = 32 keeps the cue in its lateral offset in the corridor (decodable at "
    "0.87–1.00) even though the cue is twice its window old; at L ≥ 64 this no longer works.",
    "<b>More LSTM layers don't help</b> (exploratory check at L = 64, BPTT16): 1 layer 4/5, 2 layers 3/5, 3 layers 2/5 "
    "solved; depth stacks per step and doesn't lengthen the gradient path in time. 2 layers stay.",
])
story += figure("fig3.png", 0.85, "E2 summary from the evaluation pipeline (paper Fig. 3): success vs corridor length per "
                "backbone with the BPTT-window heatmap.")
story += figure("wb_e2.png", 1.0, "Training curves from W&B (group e2_twin_rooms, shaped sweep): eval success per "
                "iteration, mean ± SE over seeds; dashed line = chance.")
story += figure("wb_layers.png", 0.45, "Layer check from W&B: LSTM-BPTT16 at L = 64 with 1, 2 (E2 reference) and 3 layers.")

story += [CondPageBreak(70 * mm), P("E2 with four rooms (K = 4), exploratory", "h2")]
story += [P("<b>Tests:</b> whether memory that holds one bit over L also holds two. Same twin-room task, but each arm ends "
            "in a cross corridor with two identical rooms, and the cue shows one of four colours, so a memoryless policy "
            "sits at 0.25. E2's three best backbones, L = 32 and 128, seeds 100–104, E2's settings; 30 runs, all finished "
            "(Nov 19). Not pre-registered.")]
story += [table([
    ["Backbone", "L = 32, K = 4", "K = 2 (E2)", "L = 128, K = 4", "K = 2 (E2)"],
    ["GRU-BPTT64", "4/5 · 0.57 [0.51–0.99]", "5/5", "0/5 · 0.13 [0.23–0.52]", "4/5"],
    ["LSTM-BPTT256", "3/5 · 0.49 [0.26–1.00]", "5/5", "0/5 · 0.18 [0.24–0.54]", "4/5"],
    ["LSTM-BPTT64", "0/5 · 0.25 [0.25–0.75]", "5/5", "0/5 · 0.19 [0.23–0.26]", "1/5"],
    ["All three", "7/15", "15/15", "0/15", "9/15"],
], [0.2, 0.25, 0.15, 0.25, 0.15])]
story += [P("Seeds solved (success ≥ 0.9) of 5 · AUC IQM [success range over seeds], from 1000 deterministic episodes per "
            "seed on the best checkpoint. Welch + Holm at α = 0.01: no pair differs (closest: GRU vs LSTM-BPTT64 at L = 32, "
            "AUC p = 0.016, p_Holm = 0.09).", "caption")]
story += bullets([
    "<b>Two bits cost a lot at L = 32 and everything at L = 128.</b> 7 of 30 seeds solve, against 24 of 30 with two "
    "rooms. At L = 32 the order is the K = 2 order (GRU ≈ BPTT256 &gt; BPTT64).",
    "<b>Where it is solved, the cue sits in the recurrent state.</b> Every solved seed gets both bits right (0.92–1.00 "
    "each), fails the swap (other cue's recurrent state: 0.00–0.09) and loses 0.42–0.61 to a reset at the junction.",
    "<b>The failures are not \"one bit but not two\".</b> 18 of the 23 unsolved seeds sit at chance and go to one or two "
    "fixed rooms whatever the cue; their swap equals their success, so the state carries no cue. Only five seeds hold "
    "part of it (0.51–0.75). At L = 128 the backbones that hold one bit with two rooms learn none with four: a learning "
    "failure (a 25% reward signal for a 4-way choice made after the corridor), which this design can't separate from a "
    "capacity limit.",
    "<b>The budget binds for the partial seeds:</b> they were still improving when the 1000 iterations ran out (e.g. "
    "LSTM-BPTT256, L = 128, seed 103: 0.29 → 0.34 → 0.54 over the last three quarters); the chance-level seeds were flat "
    "throughout. For the paper: one sentence plus this table in the appendix. A longer budget or a shaped second "
    "junction is the follow-up if a reviewer asks; neither is planned.",
])
story += figure("wb_k4.png", 0.8, "Training curves of the four-room task (W&B group e2_twin_rooms_k4, read from the per-seed "
                "curves on Euler): eval success per iteration, mean ± SE over seeds 100–104, padded to the full budget and "
                "smoothed over 5 evals; dashed line = chance (0.25).")

# ------------------------------------------------------------------ E3
story += [CondPageBreak(70 * mm), P("E3 — Information sources (\"be told\") in normal rooms", "h2")]
story += [P("<b>Tests:</b> H3–H5, whether telling the agent the goal (oracle distance/bearing, last seen frozen or updated "
            "by odometry) or its position (raw xy, Fourier features, hand-made hexagonal grid code, noisy grid code) "
            "replaces memory. 9 conditions × MLP/LSTM × four_room/C_maze × 5 seeds = 180 runs.")]
story += bullets([
    "E3 can't answer its question: with nothing extra, both backbones already reach 0.93–0.99 AUC in four_room and "
    "C_maze, so there is no memory gap to close (H3 untestable). four_room saturates in every condition.",
    "In C_maze the oracle goal hurts the LSTM (IQM 0.52 [0.13, 0.82] vs 0.94 without it; two seeds fail) while the MLP "
    "with it has the best paths (SPL 0.61–0.76). Cause unknown.",
    "H4 not supported (the frozen-goal and noisy-grid LSTMs trail most, not significantly); H5 holds as a null result.",
    "Consequence: the information question moved to the twin rooms, where memory is required and verified (E3T).",
])
story += figure("fig4.png", 0.8, "E3 summary from the evaluation pipeline (paper Fig. 4): conditions × backbones, two layouts.")

# ------------------------------------------------------------------ E3T
story += [PageBreak(), P("E3T — Told or remembered, in the twin rooms", "h2")]
story += [P("<b>Tests:</b> H8, how often the cue must be told for memory to become unnecessary. At L = 32 (pre-registered "
            "rule: L = 64 only if LSTM-BPTT16 solved ≥ 4/5 there in E2; it solved 2–3) the cue is put into the input "
            "every 1, 4, 16 or 64 steps, or never (I0); two controls tell the goal (G1) or the position (P1) instead. "
            "7 conditions × MLP/FS16/LSTM-BPTT16 × 5 seeds = 105 runs; all conditions share one input size.")]
story += [table([
    ["Backbone", "Every step", "Every 4", "Every 16", "Every 64", "Every 64, random phase", "Never", "Told goal", "Told xy"],
    ["MLP", "5/5", "5/5", "0/5", "0/5", "0/5", "0/5", "5/5", "0/5"],
    ["FS16", "5/5", "5/5", "4/5*", "4/5", "5/5", "0/5", "5/5", "2/5"],
    ["LSTM-BPTT16", "5/5", "5/5", "5/5", "5/5", "5/5", "5/5", "5/5", "3/5"],
    ["H8 predicted (M / F / L)", "✓ / ✓ / ✓", "✗ / ✓ / ✓", "✗ / ✓ / ✓", "✗ / ✗ / ✓", "✗ / ✗ / ✓", "✗ / ✗ / ✓", "✓ / ✓ / ✓", "✗ / ? / ✓"],
], [0.17, 0.09, 0.09, 0.09, 0.09, 0.13, 0.09, 0.12, 0.13])]
story += [P("Seeds solved (≥ 0.9) of 5 from 1000 episodes each. *The fifth FS16 seed never reaches the junction "
            "(an exploration failure). LSTM − MLP is significant (Welch + Holm, α = 0.01) when never told (+0.44 AUC) "
            "and told every 16 steps (+0.34). The every-64 column has the telling at the decision; the random-phase rerun is in the timing check.", "caption")]
story += bullets([
    "<b>The LSTM and the told goal behave as predicted.</b> The LSTM solves every interval; told the goal, every backbone "
    "solves and the LSTM stops using its memory (the swap no longer matters).",
    "<b>The MLP beats the prediction with the body.</b> It solves when told every 4th step: between tellings it keeps "
    "steering to the cued side (the cue is decodable from its pose at 0.70–1.00). FS16's 4/5 when told every 64th step "
    "turned out to be a timing artefact (below), not memory.",
    "<b>Told the position</b>, the MLP stays at chance as predicted; FS16 gains 2 seeds (position makes its lateral offset "
    "readable); two LSTM seeds get worse — an extra input that carries nothing destabilises some LSTM seeds, as the "
    "oracle goal did in E3.",
])
story += figure("fig_e3t.png", 0.55, "E3T summary from the evaluation pipeline: success (IQM) vs telling interval per "
                "backbone, plus told-goal and told-xy points.")
story += figure("wb_e3t.png", 1.0, "Training curves from W&B (group e3_twin_rooms): eval success per iteration per "
                "condition, mean ± SE over seeds 100–104; dashed = chance.")
story += [P("Timing check: does \"every 64 steps\" test memory?", "h2")]
story += [P("At \"every 64\", 3 of 5 LSTM seeds ignored the recurrent-state swap halfway down the corridor, which means "
            "they read a telling after that point. The walk from the start to the junction takes about 60–75 steps at "
            "L = 32, so the second telling (step 64) lands right at the decision. To check, each seed ran 1000 episodes "
            "three times: normal, with the cue input zeroed from the bend on, and from halfway down the corridor on.")]
if c64:
    story += [table(c64["rows"], c64["widths"])]
    story += [P(c64["caption"], "caption")] + bullets(c64["bullets"])
else:
    story += [empty("Running on Euler at the time of writing (8 jobs: LSTM, MLP, FS16 at intervals 4–64).")]

story += [P("E3T at L = 64: telling, window and memory separate", "h2")]
story += [P("At L = 32 the body blurred \"told\" vs \"remembered\". At L = 64 E2's FS16 never solves and LSTM-BPTT256 "
            "solves 5/5, so this length separates the backbones. Conditions: never told (I0) and told every 4 / 16 / 64 "
            "steps with a random phase; MLP, FS16, LSTM-BPTT256; seeds 100–104; 60 runs, all finished (Nov 18).")]
story += [table([
    ["Backbone", "Every 4", "Every 16", "Every 64", "Never", "Blinded from the bend"],
    ["MLP", "2/5 · 0.55", "0/5 · 0.33", "0/5 · 0.31", "0/5 · 0.26", "every-4 falls to 0.25–0.50"],
    ["FS16", "5/5 · 0.79", "5/5 · 0.80", "1/5 · 0.44", "0/5 · 0.41", "every-4/16 fall to 0.21–0.49"],
    ["LSTM-BPTT256", "5/5 · 0.89", "5/5 · 0.81", "5/5 · 0.88", "5/5 · 0.89", "no effect (0.98–1.00)"],
    ["H8 predicted (M / F / L)", "? / ✓ / ✓", "✗ / ? / ✓", "✗ / ✗ / ✓", "✗ / ✗ / ✓", ""],
], [0.2, 0.13, 0.13, 0.13, 0.13, 0.28])]
story += [P("Seeds solved (≥ 0.9) of 5 · AUC IQM. \"Blinded from the bend\": the cue input zeroed once the agent enters the "
            "corridor, 1000 episodes per seed. Welch + Holm at α = 0.01: LSTM > MLP every-16, every-64 and never "
            "(+0.44 / +0.54 / +0.61 AUC); FS16 > MLP every-16 (+0.46); LSTM > FS16 every-64 and never (+0.42 / +0.47).",
            "caption")]
story += bullets([
    "<b>FS16 reads tellings inside its window and nothing more.</b> It solves every-4 and every-16 (the open case) but "
    "only with tellings in the corridor, and fails at every-64 (1/5) and never told (0/5).",
    "<b>The L = 32 body memory doesn't come back.</b> No FS16 seed solves with the corridor blinded; one every-64 seed "
    "holds a corridor telling up to 63 steps old through its pose but still needs it.",
    "<b>The LSTM remembers in every condition.</b> Blinding the corridor changes nothing, and swapping its recurrent "
    "state at mid-corridor sends it to the wrong room on all 20 seeds (success 0.00–0.17).",
    "<b>The MLP</b> partly solves every-4 (2/5, 0.66–0.91) by steering between tellings and is at chance from every-16 on.",
])

# ------------------------------------------------------------------ E4
story += [CondPageBreak(110 * mm), P("E4 — Staged ablation of the remaining design choices (\"perceive & act\")", "h2")]
story += [P("<b>Tests:</b> H6, and which observation, action, reward, auxiliary-loss and capacity choices matter, without "
            "a full factorial. Each theme is varied around a base config; a staged protocol promotes survivors: Stage 1 "
            "screens on C_maze + medium_maze with seeds 0–2, Stage 2 confirms on medium_maze + hard_maze with seeds "
            "100–104, Stage 3 compares the best combined config against the baseline and leave-one-theme-out configs "
            "with 10 fresh seeds. Rule: keep the theme leader and every config whose CI overlaps it (at most ⌈n/3⌉), plus "
            "one random wildcard per theme; auxiliary losses are always ranked against the same config without the aux "
            "head (matched reference).")]
story += [P("Stage 1 (108 + 18 runs, screening seeds, AUC IQM [95% CI]; baseline 0.73 [0.65, 0.75])", "h2")]
story += [table([
    ["Theme", "Leader", "Also kept / wildcard", "Dropped"],
    ["Observation", "CNN, FOV 180°: 0.87 [0.80, 0.92]", "CNN FOV 270° (0.85); wildcard MLP FOV 90°", "CNN FOV 90°, MLP FOV 270° (over the cap)"],
    ["Action", "continuous (baseline)", "wildcard discrete 3 (0.56)", "discrete 5 (0.65)"],
    ["Reward", "collision penalty 0.01: 0.86 [0.71, 0.90]", "wildcard γ = 0.99 (0.72)", ""],
    ["Capacity", "width 256: 0.81; 1 hidden layer/module 0.87", "wildcard width 16 (0.47)", "3 layers + junction ReLU (0.68)"],
    ["Aux (masked LSTM 0.12)", "depth + loop closure 0.18", "wildcard loop closure (0.07)", "depth (0.11)"],
], [0.17, 0.3, 0.3, 0.23])]
story += [P("Stage 2 (240 + 26 runs, seeds 100–104, base CNN + MLP 0.84 [0.78, 0.87])", "h2")]
story += [table([
    ["Theme", "Leader", "Also kept / wildcard", "Dropped"],
    ["Observation", "baseline (CNN, FOV 180°)", "CNN FOV 270° (0.80); wildcard MLP w/o penalty (0.43)", "MLP encoder 0.66, MLP FOV 90° 0.14"],
    ["Action", "discrete 3 actions: 0.94 [0.90, 0.96]", "continuous (baseline)", ""],
    ["Reward", "collision penalty 0.01 (baseline)", "wildcard no penalty (0.74)", "γ = 0.99 (0.83, over the cap)"],
    ["Capacity", "width 64 (baseline)", "width 256 (0.81); wildcard 1 layer (0.77)", "width 16 (0.41)"],
    ["Memory", "baseline (CNN + MLP)", "CNN + LSTM 0.62 [0.44, 0.77]", ""],
    ["BPTT window", "16 (0.09)", "wildcard 64 (0.02)", "256 (0.01)"],
    ["Aux", "no gain: every CI overlaps its matched reference", "", ""],
], [0.17, 0.3, 0.3, 0.23])]
story += bullets([
    "Kendall τ between Stage 1 and 2 on medium_maze is 1.0 over the 6 configs that ran unchanged in both: no theme reopens.",
    "<b>BPTT windows:</b> the recurrent PPO did fewer updates at longer windows (128 / 32 / 8 per iteration). At equal "
    "updates no window is clearly better in the mazes; the Stage-2 \"longer is worse\" is an update-count effect.",
    "<b>CNN + LSTM + loop closure collapsed</b> (0.01 vs 0.62 without aux): 9 of 10 runs stopped moving. A 6-run test "
    "shows it needs both the collision penalty and an aux loss that dominates the shared features (aux gradient 7–9× "
    "the task gradient); coefficient 0.1 or no penalty removes it. Reported as that interaction, not as \"loop closure hurts\".",
])
story += figure("wb_e4.png", 1.0, "Training curves of the theme leaders and references on medium_maze and hard_maze, "
                "mean ± SE over seeds 100–104. Same eval success as W&B group e4_stage2, read from the per-seed curves "
                "saved on Euler, because the seeds rerun after the quota loss were only backfilled to W&B as summaries.")
story += [P("Stage 3 (final, seeds 1000–1009)", "h2")]
story += [table([
    ["Config", "Encoder", "Action", "Collision penalty", "Role", "AUC IQM (medium / hard)"],
    ["best", "CNN", "discrete 3", "0.01", "best combined (= Stage 2's discrete 3)", "0.96 / 0.90"],
    ["−action", "CNN", "continuous", "0.01", "leave out action (= Stage 2 base)", "0.88 / 0.71*"],
    ["−encoder", "MLP", "discrete 3", "0.01", "leave out encoder", "0.83* / 0.80"],
    ["−reward", "CNN", "discrete 3", "0", "leave out reward", "0.89 / 0.76"],
    ["baseline", "MLP", "continuous", "0", "Stage 1's base", "0.60* / 0.38*"],
], [0.12, 0.1, 0.13, 0.15, 0.32, 0.18])]
story += [P("100/100 runs, Nov 17. * = best is better at α = 0.01 (Welch's t, Holm over the 8 comparisons). Only the "
            "action leader differs from Stage 2's base, so the FOV, width, memory and aux themes have nothing to leave out. "
            "Best has the highest mean AUC in every comparison (P(best better) 0.78–1.00) and the tightest seeds (0.83–0.98); "
            "the non-significant tests come from failing seeds in the leave-one-out configs (−reward 0.36, 0.12, 0.10; "
            "−encoder 0.09; −action 0.31). Final eval success: best 98 / 96%, baseline 80 / 68%.", "caption")]

# ------------------------------------------------------------------ remaining
story += [P("4. What hasn't been done yet", "h1")]
story += [table([
    ["Item", "Purpose", "Result"],
    ["Fig. 1 testbed diagram", "Agent, rooms, twin task, the three routes", "—"],
    ["Final statistics + figures", "rliable across experiments, figures frozen (planned Dec 10–16)", "—"],
    ["Stretch: held-out layouts", "Train on 4/16 procedural four-room layouts, test on unseen (cleanest test for H5)", "—"],
    ["eCAADe trajectory statistics", "Path ratio, wall distance, revisit rate, occupancy heatmaps per design choice", "—"],
    ["Pre-registration commit", "Hypotheses, metrics, promotion rules and seed sets fixed in git", "—"],
], [0.25, 0.6, 0.15])]
story += [P("Timeline (from the plan; E4 Stage 3, planned for Nov 26 – Dec 2, finished Nov 17): pick the workshop once "
            "the ICLR list is out (Nov 29) · gaps and extra seeds Dec 3–9 · statistics and figures frozen Dec 10–16 · "
            "semester report + 4-page workshop version Dec 17 – Jan 10 · eCAADe abstract ~mid-January, ICLR 2027 workshop "
            "deadline ~Feb 1.", "caption")]

# ------------------------------------------------------------------ problems
story += [PageBreak(), P("5. Problems we ran into and how we fixed them", "h1")]
story += [P("Several results in this project changed because of quiet problems, so every fix is listed. "
            "\"Validity\" problems would have changed a conclusion; \"infrastructure\" problems cost runs or time.")]
story += [P("Validity of the experiments", "h2")]
story += [table([
    ["Problem", "What it would have done", "Fix", "Status"],
    ["Global Perlin wall colour seeded by the RL seed worked as a place code",
     "Confounds memory vs no memory; each seed saw a different colour map",
     "<font face='Sans'>colour_mode</font> (global/off/room-local/per-wall) and a fixed <font face='Sans'>colour_seed</font>; "
     "E1 tested it: not the reason the MLP wins", status("Fixed")],
    ["Fixed goal per room, final eval from one start", "Every room becomes a reactive lookup; metrics from one pair",
     "Per-episode goal sampling; navmesh reference paths for any start/goal pair", status("Fixed")],
    ["AUC computed over each run's own evals", "Early-stopped fast learners and slow learners scored on different horizons",
     "AUC padded to the full budget (last value carried forward)", status("Fixed")],
    ["Depth auxiliary head predicted a channel the policy also sees", "Aux loss reconstructs its input; any aux claim void",
     "Depth masked from the policy when depth aux is on; every aux config ranked against a matched reference", status("Fixed")],
    ["Masked agents with colour off are blind", "Masked LSTM near random; depth aux has nothing to learn from",
     "Masked results not read as evidence about memory; unmasked LSTM ± loop closure added to Stage 2", status("Explained")],
    ["Field of view set the turn rate", "A FOV sweep would also change the action space", "FOV-independent maximum turn", status("Fixed")],
    ["Agents spawned outside rooms (bounding-box spawning)", "Unreachable starts, wasted episodes",
     "Hand-made spawn regions for every maze", status("Fixed")],
    ["Easy rooms saturate (goal in view ~50% of the time)", "Screening would keep everything", "E4 moved to the hard tier (C_maze, medium_maze, hard_maze)", status("Fixed")],
    ["First twin-room layout: straight corridor", "The edge ray of a 180° view saw the cue wall from the junction — memoryless agents could re-read the cue",
     "Bent shaft after the alcove; test with 360° rays from every reachable point", status("Fixed")],
    ["Twin-room exploration bottleneck", "Unshaped pilot measured which seeds got out of the start, not memory",
     "Potential-based shaping to the junction (optimal policy and 0.5 ceiling unchanged)", status("Fixed")],
    ["Recurrent PPO did fewer updates at longer BPTT windows", "BPTT comparison also compared training budgets",
     "Minibatch = 1024 / window, so every window gets 128 updates of 1024 samples", status("Fixed")],
    ["Sweep scripts left the learning rate at the default", "MLP would have trained at the rate the pilot rejected",
     "Scripts use the pilot's rates (MLP 1e-4, LSTM 3e-4)", status("Fixed")],
    ["CNN + LSTM + loop closure stopped moving", "Would read as \"loop closure hurts memory\"",
     "6-run test isolated the interaction (collision penalty × dominant aux gradient)", status("Explained")],
    ["Junction counted by a box only as tall as the corridor", "Corner-cutting agents never \"reached\" the junction; reset-at-junction success inflated",
     "Junction also counts any point past the corridor's end; reset statistics over reset episodes only; all diagnostics rerun", status("Fixed")],
    ["Body memory (FS16, MLP)", "Success above chance without memory could be read as a cue leak",
     "Pose probe, mirror and swap tests in the diagnostics attribute every success", status("Explained")],
    ["\"Every 64 steps\" telling may land at the decision", "E3T's longest interval might not test memory",
     "Blinding/timing check (E3T section)", status("Open") if not c64 else status(c64.get("status", "Explained"))],
    ["Training-time shaping uses the same junction box", "Corner-cutting episodes get a refunded shaping penalty (policy unchanged)",
     "Training latch uses the diagnostics' junction test since Nov 17 (E3T at L = 64, E2 K = 4); earlier twin-room runs kept the box (the shaping telescopes either way)", status("Fixed")],
    ["nDTW degenerate with fixed goals; seed_stats \"success\" is completion rate / 100", "Secondary metrics not comparable across goal modes",
     "Noted; check before the final tables", status("Open")],
], [0.25, 0.27, 0.36, 0.12])]
story += [P("Infrastructure (Euler, Hydra, W&B)", "h2")]
story += [table([
    ["Problem", "Fix"],
    ["Sweep launcher killed by the 1 h default SLURM limit; final evaluation never ran", "Launchers submitted as long sbatch jobs"],
    ["Code passed on the login node but failed in SLURM jobs (circular import via submitit)", "Import fixed; new code tested with a 2-iteration submitit job"],
    ["Runs that hit the time limit or the disk quota showed as COMPLETED in sacct (submitit requeue fails, Hydra stores the exception)",
     "check_results script reads every job's result file; longer timeouts; lost runs rerun"],
    ["Home inode quota exceeded (W&B and Hydra files): 28 Stage-2 runs lost", "W&B and Hydra output moved to scratch in every sweep script"],
    ["Duplicate W&B runs from reruns and backfills", "Results grouped by storage name + seed (as in this report's plots)"],
    ["W&B image keys and storage folder names over the 255-byte limit", "Shortened image keys; run names checked up front"],
    ["Hydra parses timestamps like 20261112_133519 as integers", "Folder names use the integer form; timestamps quoted when reused"],
    ["Diagnostics starved by training jobs under the per-user memory cap", "Lower priority (nice) for pending training arrays"],
    ["<font face='Sans'>conda activate</font> inside <font face='Sans'>sbatch --wrap</font> left jobs on the system Python (no torch)",
     "Jobs call the environment's Python directly"],
], [0.6, 0.4])]


# ------------------------------------------------------------------ material
story += [CondPageBreak(60 * mm), P("6. Where to find the material", "h1")]
story += [P("W&B group <a href='https://wandb.ai/eazevedo-eth-z-rich/semproj/groups/report_nov19' color='#2a78d6'>report_nov19</a> (project semproj), made from the sweeps' "
            "seed folders on Euler by <font face='Sans'>scripts/report_to_wandb.py</font>. Each run has the room pictures, "
            "learning curves (one colour per configuration, thin lines = seeds, thick = mean, dashed = chance), a table per "
            "seed and per configuration, and rollout videos of whole deterministic episodes from the saved checkpoints "
            "(one episode per cue in the twin rooms) with rays, trail and outcome drawn in.")]
story += [table([['Section', 'W&B run', 'Contents'], ['Overview and testbed', '<a href="https://wandb.ai/eazevedo-eth-z-rich/semproj/runs/0lu2gbdd" color="#2a78d6">0lu2gbdd</a>', 'report figures, every room'], ['E1 place-code audit', '<a href="https://wandb.ai/eazevedo-eth-z-rich/semproj/runs/7oq6w6qu" color="#2a78d6">7oq6w6qu</a>', 'curves for 4 rooms, 24 videos (median seed)'], ['E1b field of view', '<a href="https://wandb.ai/eazevedo-eth-z-rich/semproj/runs/cghepytv" color="#2a78d6">cghepytv</a>', 'curves per FOV, 8 videos'], ['E2 twin rooms', '<a href="https://wandb.ai/eazevedo-eth-z-rich/semproj/runs/qqz7tgqy" color="#2a78d6">qqz7tgqy</a>', 'curves per L, layer check, 61 videos (best + worst seed)'], ['E2 four rooms (K = 4)', '<a href="https://wandb.ai/eazevedo-eth-z-rich/semproj/runs/b36i23p5" color="#2a78d6">b36i23p5</a>', 'curves, 11 videos'], ['E3 information sources', '<a href="https://wandb.ai/eazevedo-eth-z-rich/semproj/runs/r4oqzz1j" color="#2a78d6">r4oqzz1j</a>', 'curves per condition, 36 videos'], ['E3T told or remembered', '<a href="https://wandb.ai/eazevedo-eth-z-rich/semproj/runs/k93dcig4" color="#2a78d6">k93dcig4</a>', 'L = 32, random phase and L = 64, 51 videos'], ['E4 staged ablation', '<a href="https://wandb.ai/eazevedo-eth-z-rich/semproj/runs/yna6s47v" color="#2a78d6">yna6s47v</a>', 'Stage 1–3 curves, 10 videos (Stage 3)']], [0.3, 0.16, 0.54])]
story += [P("The training runs themselves stay in their sweep groups (e1_place_code_audit, e2_twin_rooms, …). E0 has no "
            "curves there; E4 Stage 1 has curves for 46 of 126 seeds (the rest were only backfilled as summaries).", "caption")]

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=17 * mm,
                      title="Perceive, Remember, or Be Told — direction and results", author="Emerson Azevedo Aguiar")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=on_page)])
doc.build(story)
print("wrote", OUT)
