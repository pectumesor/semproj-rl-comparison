// Supervisor-meeting deck: framing, ablation plan, key metric per experiment, rollout videos.
//   NODE_PATH=<dir with pptxgenjs, react-icons, react, react-dom, sharp> node build_deck.js <out.pptx> <apply_theme.js>
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const { FaEye, FaBrain, FaBullhorn } = require("react-icons/fa");

const OUT = process.argv[2];
const { applyTheme } = require(process.argv[3]);
const MEDIA = path.join(__dirname, "media");

// Ink slate for title/closing slides, white content slides; the series colours are the report's (MLP blue,
// LSTM orange, GRU green, FS16 pink, LSTM-BPTT256 brown), so the deck reads like the PDF figures
const THEME = {
  name: "Perceive Remember Told",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "16212C", lt1: "FFFFFF", dk2: "1F3B4D", lt2: "EEF2F4",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "E87BA4", accent5: "9E3A12", accent6: "EDA100",
    hlink: "2A78D6", folHlink: "6250D6",
  },
};
const HEX = THEME.colors;
const MUTED = "52514E", GRID = "E6E6E3";
const SERIES = { MLP: HEX.accent1, FS16: HEX.accent4, "LSTM-BPTT16": HEX.accent2, "LSTM-BPTT256": HEX.accent5, GRU: HEX.accent3 };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";       // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Perceive, Remember, or Be Told";
pres.author = "Emerson Azevedo Aguiar";
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "DARK", background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.2, w: 5.4, h: 1.6, fontSize: 38, bold: true,
                                color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.0, w: 5.4, h: 1.6, fontSize: 15,
                                color: "C9D6DF", valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "DARK_TOP", background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 8.8, h: 0.8, fontSize: 30, bold: true,
                                color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.28, w: 8.1, h: 0.75, fontSize: 24, bold: true,
                                color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Perceive, Remember, or Be Told · supervisor meeting, Nov 2026",
              options: { x: 0.5, y: 5.25, w: 6, h: 0.25, fontSize: 9, color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.25, w: 0.5, h: 0.25, fontSize: 9, color: MUTED, align: "right" },
});

// ---------- helpers ----------
async function icon(Component, colour = "FFFFFF") {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Component, { color: "#" + colour, size: 256 }));
  const png = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}
function dataUri(file) {
  return "data:image/png;base64," + fs.readFileSync(path.join(MEDIA, file)).toString("base64");
}
function imageSize(file) {
  const b = fs.readFileSync(path.join(MEDIA, file));
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };          // PNG IHDR
}
// Experiment tag in the top right of every content slide (the deck's motif: rounded pills)
function tag(slide, text) {
  slide.addText(text, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.15, x: 8.75, y: 0.42, w: 0.85, h: 0.42,
                        fill: { color: C.background2 }, color: C.text2, fontSize: 12, bold: true, align: "center",
                        valign: "middle", margin: 0, objectName: "Experiment tag" });
}
// A video with its poster frame, sized to the clip's aspect ratio inside the box, and a caption under it
function video(slide, name, x, y, w, maxH, caption) {
  const poster = `${name}_poster.png`;
  const { w: pw, h: ph } = imageSize(poster);
  let vw = w, vh = w * ph / pw;
  if (vh > maxH) { vh = maxH; vw = maxH * pw / ph; }
  const vx = x + (w - vw) / 2;
  slide.addShape(pres.shapes.RECTANGLE, { x: vx - 0.03, y: y - 0.03, w: vw + 0.06, h: vh + 0.06,
                                          fill: { color: C.background1 }, line: { color: "C3C2B7", width: 0.75 },
                                          objectName: `Frame ${name}` });
  slide.addMedia({ type: "video", path: path.join(MEDIA, `${name}.mp4`), cover: dataUri(poster),
                   x: vx, y, w: vw, h: vh, objectName: `Video ${name}` });
  if (caption) {
    slide.addText(caption, { x, y: y + vh + 0.1, w, h: 0.55, fontSize: 11, color: MUTED, valign: "top", margin: 0,
                             isTextBox: true, objectName: `Caption ${name}` });
  }
  return vh;
}
function bullets(items, opts = {}) {
  return items.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6, ...opts },
  }));
}
function rich(parts) {
  // [[text, bold], ...] -> runs
  return parts.map(([text, bold]) => ({ text, options: { bold: !!bold } }));
}
const CHART_TEXT = { catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontSize: 11,
                     valAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
                     legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: MUTED,
                     valGridLine: { color: GRID, size: 0.75 }, catGridLine: { style: "none" },
                     catAxisLineColor: "C3C2B7", valAxisLineShow: false,
                     titleFontFace: "+mn-lt", titleColor: MUTED, titleFontSize: 12 };
const AXIS_TITLE = { catAxisTitleColor: MUTED, valAxisTitleColor: MUTED, catAxisTitleFontSize: 11,
                     valAxisTitleFontSize: 11, catAxisTitleFontFace: "+mn-lt", valAxisTitleFontFace: "+mn-lt" };

(async () => {
  const eye = await icon(FaEye), brain = await icon(FaBrain), horn = await icon(FaBullhorn);

  // ================= 1. Title =================
  pres.addSection({ title: "Framing" });
  let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Framing" });
  s.addText("Perceive, Remember, or Be Told", { placeholder: "title" });
  s.addText([
    { text: "When does a navigation agent need memory, and what can replace it?", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Semester project, ETH IVIA Lab · Emerson Azevedo Aguiar", options: { breakLine: true, fontSize: 13 } },
    { text: "Supervisor meeting, November 2026", options: { fontSize: 13 } },
  ], { placeholder: "body" });
  {
    const { w: pw, h: ph } = imageSize("room_twin_rooms_L64.png");
    const w = 3.3, h = w * ph / pw;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.25, y: 1.55, w: w + 0.3, h: h + 0.3, rectRadius: 0.12,
                                                fill: { color: C.background1 }, line: { type: "none" }, objectName: "Room card" });
    s.addImage({ path: path.join(MEDIA, "room_twin_rooms_L64.png"), x: 6.4, y: 1.7, w, h,
                 altText: "Twin-room memory task: cue alcove, corridor, two identical rooms" });
    s.addText("The twin-room memory task", { x: 6.25, y: 1.65 + h + 0.35, w: w + 0.3, h: 0.3, fontSize: 11,
                                             color: "C9D6DF", margin: 0, isTextBox: true });
  }
  s.addNotes("The project started as a comparison of design choices for RL navigation. This deck covers how it was " +
             "reframed, the ablation plan that followed, and the results, with the one key metric per experiment and " +
             "rollout videos from the saved checkpoints.");

  // ================= 2. Three routes =================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Framing" });
  s.addText("Three routes to what the agent needs", { placeholder: "title" });
  const routes = [
    [eye, HEX.accent1, "Perceive", "Ray casts: walls, distance, colour. A leak like a colour place code is a route too."],
    [brain, HEX.accent2, "Remember", "Recurrent state (LSTM, GRU) or a frame stack. The horizon is set by the training window."],
    [horn, HEX.accent3, "Be told", "Goal, position or cue put straight into the input, every step or every N steps."],
  ];
  routes.forEach(([img, col, head, body], i) => {
    const x = 0.5 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.3, w: 2.8, h: 2.35, rectRadius: 0.12,
                                                fill: { color: C.background2 }, line: { type: "none" }, objectName: `${head} card` });
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.55, w: 0.7, h: 0.7, fill: { color: col }, line: { type: "none" },
                                   objectName: `${head} icon circle` });
    s.addImage({ data: img, x: x + 0.47, y: 1.72, w: 0.36, h: 0.36, altText: `${head} icon` });
    s.addText(head, { x: x + 0.3, y: 2.38, w: 2.3, h: 0.4, fontSize: 18, bold: true, color: C.text2, margin: 0,
                      fontFace: "Cambria", isTextBox: true });
    s.addText(body, { x: x + 0.3, y: 2.8, w: 2.3, h: 0.8, fontSize: 13, color: C.text1, margin: 0, valign: "top",
                      isTextBox: true });
  });
  s.addText(rich([["The design choices in the title mostly decide which route the agent uses. ", true],
                  ["So whether a choice helps depends on what the other routes already give: not a leaderboard, but " +
                   "controlled tests of when memory is needed and what can replace it.", false]]),
            { x: 0.5, y: 3.95, w: 8.9, h: 0.95, fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("The original results had a memoryless MLP beating the LSTM. Instead of hiding that, the plan turns it " +
             "into a conditional claim: memory, goal information and position codes are substitutes, each helps only " +
             "when the others are missing.");

  // ================= 3. Standard floorplans =================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Framing" });
  s.addText("Standard floorplans don't test memory", { placeholder: "title" });
  tag(s, "E1");
  s.addChart(pres.charts.BAR, [
    { name: "MLP (memoryless)", labels: ["C_maze, colour on", "C_maze, colour off", "medium_maze, colour on", "medium_maze, colour off"],
      values: [0.79, 0.96, 0.67, 0.56] },
    { name: "LSTM", labels: ["C_maze, colour on", "C_maze, colour off", "medium_maze, colour on", "medium_maze, colour off"],
      values: [0.57, 0.87, 0.12, 0.23] },
  ], { x: 0.4, y: 1.15, w: 5.4, h: 3.9, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent1, HEX.accent2],
       showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: MUTED, dataLabelFormatCode: "0.00",
       dataLabelFontFace: "+mn-lt", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00",
       showLegend: true, legendPos: "t", showTitle: true, title: "Success AUC (IQM, 5 seeds); colour off = also random goals", ...CHART_TEXT,
       objectName: "E1 AUC chart" });
  s.addText(bullets([
    "The memoryless MLP matches or beats the LSTM in every room we had, with or without the colour place code.",
    "Same with a narrower view (E1b) and with goal or position information (E3).",
    "No LSTM − MLP difference survives Holm correction.",
  ]), { x: 6.1, y: 1.3, w: 3.5, h: 2.6, fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true });
  s.addText("Memory needs a task built for it → the twin rooms", {
    shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x: 6.1, y: 4.05, w: 3.5, h: 0.8, fill: { color: C.background2 },
    fontSize: 14, bold: true, color: C.text2, margin: 8, valign: "middle", objectName: "E1 takeaway" });
  s.addNotes("E1 tested whether the MLP wins because the global Perlin wall colour works as a place code. Colour off " +
             "never hurts the MLP; the gap shrinks (medium_maze -0.54 to -0.27 AUC) mostly because random goals make the " +
             "MLP worse, but it never flips. Colour off and random goal shown here; colour stays off from here on.");

  // ================= 4. Ablation plan =================
  pres.addSection({ title: "Ablation plan" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Ablation plan" });
  s.addText("Controlled tests, then a staged ablation", { placeholder: "title" });
  const steps = [
    ["E0", "Pilot + fixes", "Valid pipeline, run cost, which rooms separate agents", "32"],
    ["E1", "Leak audit", "Is the colour a place code? FOV (E1b)", "160"],
    ["E2", "Twin rooms", "At what cue distance does each memory fail?", "215"],
    ["E3", "Be told", "Does goal, position or cue input replace memory?", "360"],
    ["E4", "Staged ablation", "Observation, action, reward, capacity, aux", "492"],
  ];
  steps.forEach(([id, head, q, runs], i) => {
    const x = 0.5 + i * 1.82;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.25, w: 1.65, h: 2.3, rectRadius: 0.1,
                                                fill: { color: i === 2 || i === 3 ? "DCE8F7" : C.background2 },
                                                line: { type: "none" }, objectName: `${id} step card` });
    s.addText(id, { x: x + 0.15, y: 1.38, w: 1.35, h: 0.45, fontSize: 22, bold: true, color: C.text2, margin: 0,
                    fontFace: "Cambria", isTextBox: true });
    s.addText(head, { x: x + 0.15, y: 1.83, w: 1.4, h: 0.32, fontSize: 13, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(q, { x: x + 0.15, y: 2.17, w: 1.4, h: 0.85, fontSize: 11, color: C.text1, margin: 0, valign: "top", isTextBox: true });
    s.addText(`${runs} runs`, { x: x + 0.15, y: 3.12, w: 1.4, h: 0.3, fontSize: 11, bold: true, color: MUTED, margin: 0, isTextBox: true });
    if (i < steps.length - 1) {
      s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 1.665, y: 2.3, w: 0.14, h: 0.2,
                                            fill: { color: "C3C2B7" }, line: { type: "none" }, objectName: "Arrow" });
    }
  });
  const chips = [
    ["Metric", "Success AUC over training, padded to the full budget; plus success, SPL, nDTW"],
    ["Seeds", "0–2 screening only · 100–104 confirmation · 1000–1009 final claims"],
    ["Statistics", "IQM + bootstrap CIs; Welch + Holm at α = 0.01; H1–H8 written before the runs"],
  ];
  chips.forEach(([k, v], i) => {
    const x = 0.5 + i * 3.05;
    s.addText([{ text: k + "  ", options: { bold: true, color: C.text2 } }, { text: v, options: { color: C.text1 } }],
              { x, y: 3.85, w: 2.85, h: 1.05, fontSize: 12, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes("About 1260 training runs in all, on ETH Euler. Shaded: the memory core of the paper (E2 and E3T, the told-" +
             "versus-remembered test in the twin rooms). E4 varies one theme at a time around a base config and promotes " +
             "survivors through three stages with fresh seeds.");

  // ================= 5. Twin rooms =================
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Twin rooms turn memory into a dial", { placeholder: "title" });
  tag(s, "E2");
  {
    const { w: pw, h: ph } = imageSize("room_twin_rooms_L64.png");
    const w = 5.0, h = w * ph / pw;
    s.addImage({ path: path.join(MEDIA, "room_twin_rooms_L64.png"), x: 0.5, y: 1.35, w, h,
                 altText: "Twin rooms at L = 64: cue alcove top left, corridor, two identical goal rooms right" });
  }
  const twin = [
    ["1", "Cue", "One of two colours on the alcove walls (magenta), hidden by the bent shaft once the agent leaves."],
    ["2", "Corridor of length L", "8, 32, 64, 128 or 256 steps between seeing the cue and choosing."],
    ["3", "Two identical rooms", "The cued one is success, the other ends the episode. Memoryless = 0.5."],
  ];
  twin.forEach(([n, head, body], i) => {
    const y = 1.3 + i * 1.12;
    s.addShape(pres.shapes.OVAL, { x: 5.9, y, w: 0.42, h: 0.42, fill: { color: C.text2 }, line: { type: "none" },
                                   objectName: `Point ${n}` });
    s.addText(n, { x: 5.9, y, w: 0.42, h: 0.42, fontSize: 14, bold: true, color: C.background1, align: "center",
                   valign: "middle", margin: 0, isTextBox: true });
    s.addText([{ text: head, options: { bold: true, breakLine: true } }, { text: body }],
              { x: 6.45, y: y - 0.05, w: 3.15, h: 1.0, fontSize: 13, color: C.text1, valign: "top", margin: 0, isTextBox: true });
  });
  s.addText("7 backbones × 5 lengths × seeds 100–104 = 175 runs", { x: 0.5, y: 4.75, w: 5, h: 0.3, fontSize: 12,
                                                                     color: MUTED, margin: 0, isTextBox: true });
  s.addNotes("Two design fixes before the sweep: the first layout leaked the cue down a straight corridor (fixed with " +
             "the bent shaft and a line-of-sight test), and potential-based shaping to the junction so the result is " +
             "about memory, not about which seeds manage to leave the start. Shaping leaves the optimal policy and the " +
             "0.5 ceiling unchanged.");

  // ================= 6. E2 key metric =================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("The training window sets the memory horizon", { placeholder: "title" });
  tag(s, "E2");
  const Ls = ["8", "32", "64", "128", "256"];
  const solved = { MLP: [3, 0, 0, 0, 0], FS16: [5, 2, 0, 0, 0], "LSTM-BPTT16": [5, 5, 3, 0, 0],
                   "LSTM-BPTT256": [5, 5, 5, 4, 0], GRU: [5, 5, 4, 4, 0] };
  s.addChart(pres.charts.LINE, Object.entries(solved).map(([name, values]) => ({ name: name === "GRU" ? "GRU (BPTT64)" : name, labels: Ls, values })),
             { x: 0.4, y: 1.15, w: 5.7, h: 3.95, chartColors: Object.keys(solved).map((k) => SERIES[k]),
               lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7, valAxisMinVal: 0, valAxisMaxVal: 5,
               valAxisMajorUnit: 1, showLegend: true, legendPos: "b", showValAxisTitle: true, valAxisTitle: "Seeds solved (of 5)",
               showCatAxisTitle: true, catAxisTitle: "Corridor length L (steps)", ...CHART_TEXT, ...AXIS_TITLE,
               objectName: "E2 seeds solved vs L" });
  s.addText(bullets([
    "Memoryless agents fall to chance from L = 32; frame stacks hold only while the cue fits their window.",
    "A 16-step window holds to L = 32, 256 steps to L = 128; GRU matches it. Nothing solves L = 256.",
    "On all 64 solved recurrent seeds, swapping in the other cue's recurrent state sends the agent to the wrong room.",
  ]), { x: 6.4, y: 1.3, w: 3.2, h: 3.6, fontSize: 13, color: C.text1, valign: "top", margin: 0, isTextBox: true });
  s.addNotes("Solved = success at least 0.9 over 1000 deterministic episodes. Every seed reaches the junction (0.90 to " +
             "1.00), so every failure is a memory failure. FS16 at L = 32 solves 2/5 by body memory: it keeps the cue in " +
             "its lateral offset in the corridor. More LSTM layers don't help (1 layer 4/5, 3 layers 2/5 at L = 64).");

  // ================= 7. E2 videos =================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("At L = 64 the LSTM remembers, the MLP guesses", { placeholder: "title" });
  tag(s, "E2");
  video(s, "E2_twin_rooms_L64_130_104", 0.5, 1.25, 4.35, 3.1, "LSTM-BPTT256, seed 104: both cues lead to the right room (success 1.00)");
  video(s, "E2_twin_rooms_L64_140_103", 5.15, 1.25, 4.35, 3.1, "MLP, seed 103: the same room whatever the cue; right for blue, wrong for red (0.50)");
  s.addNotes("One episode per cue; the goal room is shaded green and the cue walls show the episode's colour. Rays are " +
             "coloured where they hit a coloured wall. Click to play.");

  // ================= 8. E3T key metric =================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Being told replaces memory only inside the window", { placeholder: "title" });
  tag(s, "E3T");
  const intervals = ["Every 4", "Every 16", "Every 64", "Never"];
  s.addChart(pres.charts.BAR, [
    { name: "MLP", labels: intervals, values: [2, 0, 0, 0] },
    { name: "FS16", labels: intervals, values: [5, 5, 1, 0] },
    { name: "LSTM-BPTT256", labels: intervals, values: [5, 5, 5, 5] },
  ], { x: 0.4, y: 1.15, w: 5.7, h: 3.95, barDir: "col", barGapWidthPct: 55, chartColors: [SERIES.MLP, SERIES.FS16, SERIES["LSTM-BPTT256"]],
       showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: MUTED, dataLabelFontFace: "+mn-lt",
       valAxisMinVal: 0, valAxisMaxVal: 5, valAxisMajorUnit: 1, showLegend: true, legendPos: "b",
       showValAxisTitle: true, valAxisTitle: "Seeds solved (of 5)", showCatAxisTitle: true,
       catAxisTitle: "Cue told every N steps (L = 64, random phase)", ...CHART_TEXT, ...AXIS_TITLE, objectName: "E3T seeds solved" });
  s.addText(bullets([
    "FS16 solves while the last telling fits its 16 frames, and only with tellings inside the corridor.",
    "The MLP needs a telling every few steps and bridges them by steering.",
    "The LSTM solves all 20 seeds; zeroing the cue in the corridor changes nothing. H8 supported.",
  ]), { x: 6.4, y: 1.3, w: 3.2, h: 3.6, fontSize: 13, color: C.text1, valign: "top", margin: 0, isTextBox: true });
  s.addNotes("Welch + Holm at alpha 0.01: LSTM beats MLP at every-16, every-64 and never; FS16 beats MLP at every-16; " +
             "LSTM beats FS16 at every-64 and never. At L = 32 the body blurred told versus remembered (the MLP solved " +
             "every-4 by steering, FS16 every-64 by body memory), which is why the clean test is at L = 64.");

  // ================= 9. E3T videos =================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Told or remembered at L = 64", { placeholder: "title" });
  tag(s, "E3T");
  video(s, "E3T_L64_twin_rooms_L64_45_103", 0.5, 1.4, 2.9, 2.6, "MLP, never told: guesses (one cue right, one wrong)");
  video(s, "E3T_L64_twin_rooms_L64_10_101", 3.55, 1.4, 2.9, 2.6, "FS16, told every 16: reads the tellings in the corridor (\"TOLD\" in the banner)");
  video(s, "E3T_L64_twin_rooms_L64_25_104", 6.6, 1.4, 2.9, 2.6, "LSTM-BPTT256, never told: remembers the cue from the alcove");
  s.addText(rich([["Zeroing the cue input once the agent enters the corridor: ", true],
                  ["FS16 falls to chance (0.21–0.49), the LSTM doesn't change (0.98–1.00). Swapping the LSTM's recurrent state " +
                   "mid-corridor sends it to the wrong room on all 20 seeds.", false]]),
            { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x: 0.5, y: 4.15, w: 9.0, h: 0.85,
              fill: { color: C.background2 }, fontSize: 13, color: C.text1, margin: 10, valign: "middle",
              objectName: "E3T blinding takeaway" });
  s.addNotes("The banner shows TOLD on the steps the cue is in the input.");

  // ================= 10. K = 4 =================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Two bits are much harder to learn than one", { placeholder: "title" });
  tag(s, "E2 K4");
  [["7/15", "seeds solved at L = 32", "15/15 with two rooms"], ["0/15", "seeds solved at L = 128", "9/15 with two rooms"]]
    .forEach(([big, label, ref], i) => {
      const y = 1.3 + i * 1.65;
      s.addText(big, { x: 0.5, y, w: 2.6, h: 0.85, fontSize: 54, bold: true, color: C.text2, fontFace: "Cambria",
                       margin: 0, valign: "bottom", isTextBox: true });
      s.addText([{ text: label, options: { breakLine: true } }, { text: ref, options: { color: MUTED } }],
                { x: 0.5, y: y + 0.88, w: 2.6, h: 0.6, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true });
    });
  s.addText("Four identical rooms, cue in one of four colours (chance 0.25); GRU-BPTT64, LSTM-BPTT256, LSTM-BPTT64. Exploratory.",
            { x: 0.5, y: 4.6, w: 2.75, h: 0.6, fontSize: 10, color: MUTED, margin: 0, valign: "top", isTextBox: true });
  video(s, "K4_twin_rooms_K4_L32_20_102", 3.5, 1.25, 3.0, 2.6, "GRU, L = 32, seed 102: all four cues right");
  video(s, "K4_twin_rooms_K4_L128_10_102", 6.65, 1.25, 2.95, 2.6, "LSTM-BPTT256, L = 128, seed 102: one of four right, the state carries no cue");
  s.addNotes("Where it is solved the cue sits in the recurrent state (both bits right, swap fails). 18 of 23 unsolved " +
             "seeds sit at chance: a learning failure with a 25% reward signal, which this design can't separate from a " +
             "capacity limit. Partial seeds were still improving at 1000 iterations. Proposal: appendix, exploratory.");

  // ================= 11. E4 =================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Staged ablation: three choices survive and combine", { placeholder: "title" });
  tag(s, "E4");
  const funnel = [["Stage 1 · screen", "126 runs · seeds 0–2", 2.9], ["Stage 2 · confirm", "266 runs · seeds 100–104", 2.45],
                  ["Stage 3 · final", "100 runs · seeds 1000–1009", 2.0]];
  funnel.forEach(([head, sub, w], i) => {
    const y = 1.3 + i * 0.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5 + (2.9 - w) / 2, y, w, h: 0.8, rectRadius: 0.1,
                                                fill: { color: i === 2 ? C.text2 : C.background2 }, line: { type: "none" },
                                                objectName: `${head} bar` });
    s.addText([{ text: head, options: { bold: true, breakLine: true } }, { text: sub, options: { fontSize: 11 } }],
              { x: 0.5 + (2.9 - w) / 2, y, w, h: 0.8, fontSize: 13, align: "center", valign: "middle", margin: 0,
                color: i === 2 ? C.background1 : C.text1, isTextBox: true });
  });
  s.addText("Best = CNN encoder + 3 discrete actions + collision penalty",
            { x: 0.5, y: 4.2, w: 2.9, h: 0.75, fontSize: 12, bold: true, color: C.text2, margin: 0, valign: "top", isTextBox: true });
  const cfgs = ["best", "−action", "−encoder", "−reward", "baseline"];
  s.addChart(pres.charts.BAR, [
    { name: "medium_maze", labels: cfgs, values: [0.96, 0.88, 0.83, 0.89, 0.60] },
    { name: "hard_maze", labels: cfgs, values: [0.90, 0.71, 0.80, 0.76, 0.38] },
  ], { x: 3.65, y: 1.15, w: 5.95, h: 3.95, barDir: "col", barGapWidthPct: 55, chartColors: [HEX.accent1, HEX.dk2],
       showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: MUTED, dataLabelFormatCode: "0.00",
       dataLabelFontFace: "+mn-lt", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00",
       showLegend: true, legendPos: "t", showTitle: true, title: "Stage 3: success AUC (IQM, 10 seeds), leave one theme out",
       ...CHART_TEXT, objectName: "E4 Stage 3 AUC" });
  s.addNotes("Best has the highest mean AUC in every comparison and the tightest seeds; it beats the baseline in both " +
             "rooms (+0.36 and +0.54 AUC) and wins 4 of 8 pre-registered tests at alpha 0.01 with Holm (baseline in both " +
             "rooms, -action in hard_maze, -encoder in medium_maze). The non-significant ones come from single failing " +
             "seeds in the leave-one-out configs. Auxiliary losses gave no gain against matched references.");

  // ================= 12. E4 videos =================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("hard_maze, median seeds: the best of each family", { placeholder: "title" });
  tag(s, "E4");
  // 3 x 2 grid: MLP backbones (Stage 3, seeds 1000-1009) on top, LSTM backbones (Stage 2, seeds 100-104) below
  const cols = [0.5, 3.55, 6.6], rows = [1.12, 3.2], cw = 2.9, vh = 1.5;
  const cell = (r, c, name, head, rest) => {
    video(s, name, cols[c], rows[r], cw, vh);
    s.addText([{ text: head, options: { bold: true, color: C.text2, breakLine: true } }, { text: rest }],
              { x: cols[c], y: rows[r] + vh + 0.07, w: cw, h: 0.45, fontSize: 10, color: MUTED, valign: "top",
                margin: 0, isTextBox: true, objectName: `Caption ${name}` });
  };
  cell(0, 0, "Stage_3_hard_maze_30_1002", "CNN + MLP · best overall", "discrete 3, collision penalty · AUC 0.90 · 3 of 3");
  cell(0, 1, "Stage_3_hard_maze_50_1009", "MLP + MLP · best with MLP encoder", "discrete 3, collision penalty · AUC 0.79 · 3 of 3");
  cell(0, 2, "Stage_3_hard_maze_40_1008", "MLP + MLP · baseline", "continuous, no penalty · AUC 0.30 · 2 of 3");
  cell(1, 0, "Stage_2_hard_maze_CNN_LSTM_104", "CNN + LSTM · best LSTM", "BPTT 16, continuous, penalty · AUC 0.62 · 3 of 3");
  cell(1, 1, "Stage_2_hard_maze_MLP_LSTM_101", "MLP + LSTM · best with MLP encoder", "BPTT 64, mb 16, continuous · AUC 0.11 · 1 of 3");
  s.addText([
    { text: "Memory never made Stage 3", options: { bold: true, color: C.text2, breakLine: true, fontSize: 12 } },
    { text: "CNN + LSTM lost the Stage 2 memory comparison to CNN + MLP (IQM 0.62 vs 0.84). The bottom row is " +
            "Stage 2: continuous actions, seeds 100–104, so compare within a row." },
  ], { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x: cols[2], y: rows[1], w: cw, h: vh + 0.5,
       fill: { color: C.background2 }, line: { type: "none" }, fontSize: 10.5, color: C.text1, valign: "middle",
       margin: 0.12, objectName: "LSTM takeaway" });
  s.addNotes("Median seed by AUC of each config, three random start/goal episodes each, the same three for every " +
             "config (fixed seeds). Top row: Stage 3 configs, 10 seeds (1000-1009). Bottom row: the best LSTM configs, " +
             "which stopped at Stage 2 (5 seeds, 100-104, lr 3e-4, continuous actions; the action and reward winners " +
             "were never combined with an LSTM). The MLP-encoder LSTM is the best of the BPTT sweep (window 64, " +
             "16-env minibatches) and still near zero in hard_maze. Long episodes are subsampled to at most 600 frames.");

  // ================= 13. Takeaways + decisions =================
  pres.addSection({ title: "Next" });
  s = pres.addSlide({ masterName: "DARK_TOP", sectionTitle: "Next" });
  s.addText("Where this leaves the paper", { placeholder: "title" });
  const col = (x, head, items) => {
    s.addText(head, { x, y: 1.45, w: 4.2, h: 0.4, fontSize: 18, bold: true, color: C.background1, fontFace: "Cambria",
                      margin: 0, isTextBox: true });
    s.addText(bullets(items), { x, y: 1.95, w: 4.2, h: 3.0, fontSize: 14, color: "DCE5EB", valign: "top", margin: 0, isTextBox: true });
  };
  col(0.6, "What we found", [
    "Standard floorplans don't need memory; the twin rooms do, and the training window sets the horizon.",
    "Being told replaces memory only as far as the gap between tellings fits the agent's memory.",
    "The body is a memory too: memoryless agents carry the cue in their position.",
    "E4: CNN + discrete actions + collision penalty wins, 4 of 8 tests significant.",
  ]);
  col(5.2, "To decide today", [
    "Framing: E2 + E3T as the core, E1/E3 as the negative result, E4 as the method.",
    "E4 claim: the 4 significant tests, the rest as \"more reliable across seeds\".",
    "K = 4 in the appendix as exploratory?",
    "Venue: ICLR 2027 workshops (list Nov 29) and/or eCAADe (abstract mid-January).",
  ]);
  s.addNotes("Next: extra seeds only where the pre-registered rules ask for them, final rliable statistics and frozen " +
             "figures Dec 10-16, writing from Dec 17. All curves and videos: W&B group report_nov19.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
