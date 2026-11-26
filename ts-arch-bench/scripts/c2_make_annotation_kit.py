"""Build the cycle-2 human-grading kit (PREREGISTRATION_C2.md, amendment A17).

    python scripts/c2_make_annotation_kit.py \
        --pool ../results-archive/20261113-cycle2-run/data/c2_n100

Writes to `annotation_c2/`:

    annotator_A.html, annotator_B.html, annotator_C.html
        One self-contained page per person. Opens in any browser, works
        offline, saves progress in the browser, and exports a JSON file of
        answers. Each person sees the items in their OWN shuffled order.
    INSTRUCTIONS.md
        What to do, and the rules that keep the grading blind.
    _private/key.json
        The answer key and item metadata. NOT for annotators. The analysis
        script (scripts/c2_analyse_annotation.py) reads it.

WHAT IS GRADED. Every `trend_description` and `event_explanation` item of
the H2 / H3 run at L=128 - all 200 of them - so human answers line up item
by item with every model's. Each person sees what a model saw, minus the
numbers-as-text: a plot of the same 128 values and the SAME question text,
word for word (the question carries the definition of every option and the
answer format), and picks one option per asked slot. No key, no model answer,
no generator setting is in the page.

The page carries a hash of its items. The analysis refuses answer files whose
hash does not match the key, so answers from an edited or different kit
cannot be mixed in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ground_truth as gt                       # noqa: E402
from src.config import load_config                       # noqa: E402
from src.ground_truth import QUALITATIVE_SLOTS, SLOTS_FOR_TYPE_C2  # noqa: E402
from src.pipeline import make_samples                    # noqa: E402

KIT_VERSION = 1
L = 128
ANNOTATORS = ("A", "B", "C")
#: per-annotator order seeds - fixed, so the kit is reproducible
ORDER_SEEDS = {"A": 101, "B": 202, "C": 303}

#: options shown per slot. `position` adds "none", which the question itself
#: tells the reader to write when there is no event (models had it too).
OPTIONS = {s: list(c) for s, c in QUALITATIVE_SLOTS.items()}
OPTIONS["position"] = OPTIONS["position"] + ["none"]


def build_items(pool_dir: str | None):
    cfg = load_config(ROOT / "configs" / "c2_h2.yaml")
    if pool_dir:
        cfg.data["pool_dir"] = str(Path(pool_dir).resolve())
    samples = make_samples(cfg, ROOT)
    items, key = [], {}
    for s in samples:
        if s.query_type not in SLOTS_FOR_TYPE_C2:
            continue
        v = s.view(L)
        truth = gt.qualitative_truth_c2(s, L)
        items.append(dict(
            id=s.sample_id, type=s.query_type, question=v.query,
            slots=list(SLOTS_FOR_TYPE_C2[s.query_type]),
            values=[round(float(x), 4) for x in v.window]))
        key[s.sample_id] = dict(
            type=s.query_type,
            expected=gt.c2_scored_slots(truth, s.query_type),
            pattern=s.pattern,
            truth=s.metadata["truth"])
    items.sort(key=lambda it: it["id"])
    blob = json.dumps(items, sort_keys=True).encode()
    return items, key, hashlib.sha256(blob).hexdigest()[:16]


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Series grading — annotator __ANNOTATOR__</title>
<style>
:root { --bg:#fbfbfa; --fg:#1d1d1b; --mut:#6b6b66; --line:#2b6cb0; --grid:#e3e2dd;
        --card:#ffffff; --acc:#2b6cb0; --ok:#2f855a; --warn:#b7791f; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#191918; --fg:#ecebe6; --mut:#a3a29c; --line:#7fb3ec; --grid:#33332f;
          --card:#22221f; --acc:#7fb3ec; --ok:#68d391; --warn:#f6c26b; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:860px; margin:0 auto; padding:16px; }
header { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center;
         justify-content:space-between; margin-bottom:12px; }
h1 { font-size:18px; margin:0; }
.bar { height:6px; background:var(--grid); border-radius:3px; overflow:hidden; flex:1 1 200px; }
.bar > div { height:100%; background:var(--acc); width:0; }
.card { background:var(--card); border:1px solid var(--grid); border-radius:10px; padding:16px; }
.q { margin:12px 0; }
svg { width:100%; height:auto; display:block; }
fieldset { border:0; padding:0; margin:12px 0; }
legend { font-weight:600; margin-bottom:4px; }
label.opt { display:inline-flex; align-items:center; gap:6px; margin:4px 12px 4px 0;
            padding:6px 10px; border:1px solid var(--grid); border-radius:8px; cursor:pointer; }
label.opt:has(input:checked) { border-color:var(--acc); outline:2px solid var(--acc); }
nav { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; align-items:center; }
button { font:inherit; padding:8px 14px; border-radius:8px; border:1px solid var(--grid);
         background:var(--card); color:var(--fg); cursor:pointer; }
button.primary { background:var(--acc); color:#fff; border-color:var(--acc); }
button:disabled { opacity:.45; cursor:default; }
.mut { color:var(--mut); font-size:14px; }
.done { color:var(--ok); } .warn { color:var(--warn); }
input[type=file] { font-size:14px; }
</style></head>
<body><main>
<header>
  <h1>Series grading — annotator <b>__ANNOTATOR__</b></h1>
  <span class="mut" id="count"></span>
  <div class="bar"><div id="prog"></div></div>
</header>
<p class="mut">Read <code>INSTRUCTIONS.md</code> first. Answer from the plot only.
The question is the exact text the AI models received; its last sentence
(<i>"Reply with exactly two lines…"</i>) told the models how to format a reply —
you just click your choices.</p>
<p class="mut" id="storage">Progress saves automatically in this browser; use <b>Download answers</b> to save a file.</p>
<div class="card">
  <div class="mut" id="pos"></div>
  <div id="plot"></div>
  <p class="q" id="question"></p>
  <form id="form"></form>
  <nav>
    <button id="prev" type="button">← Previous</button>
    <button id="next" type="button" class="primary">Next →</button>
    <button id="jump" type="button">First unanswered</button>
    <span class="mut" id="state"></span>
  </nav>
</div>
<nav>
  <button id="dl" type="button">Download answers</button>
  <label class="mut">Resume from a file: <input id="up" type="file" accept=".json"></label>
</nav>
<p class="mut">Kit __KITHASH__ · version __KITVERSION__</p>
<script>
const ANNOTATOR = "__ANNOTATOR__", KIT = "__KITHASH__", VERSION = __KITVERSION__;
const ITEMS = __ITEMS__;
const ORDER = __ORDER__;
const OPTIONS = __OPTIONS__;
const LABELS = {trend:"Trend", seasonal:"Seasonal", event:"Event", position:"Position"};
const STORE = "tsarch-c2-grading-" + ANNOTATOR + "-" + KIT;
let answers = {}, times = {}, i = 0, shownAt = Date.now();
let canStore = false;
try { localStorage.setItem("__t", "1"); localStorage.removeItem("__t"); canStore = true; } catch (e) {}
if (!canStore) { const el = document.getElementById("storage");
  el.className = "warn"; el.innerHTML = "<b>This browser is not saving your progress.</b> Click <b>Download answers</b> " +
  "every 20 items or so, and use <i>Resume from a file</i> if you reload. (Opening the file in Chrome, Edge or Firefox normally fixes this.)"; }
try { const s = JSON.parse(localStorage.getItem(STORE) || "null");
      if (s && s.kit === KIT) { answers = s.answers || {}; times = s.times || {}; i = s.i || 0; } } catch (e) {}
function save() { try { localStorage.setItem(STORE, JSON.stringify({kit:KIT, answers, times, i})); } catch (e) {} }
const byId = Object.fromEntries(ITEMS.map(it => [it.id, it]));

function plot(vals) {
  const W = 760, H = 280, l = 52, r = 12, t = 26, b = 30;
  const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.06 || 1;
  const y0 = lo - pad, y1 = hi + pad, n = vals.length;
  const X = k => l + (W - l - r) * k / (n - 1), Y = v => t + (H - t - b) * (1 - (v - y0) / (y1 - y0));
  const css = getComputedStyle(document.documentElement);
  const grid = css.getPropertyValue("--grid"), mut = css.getPropertyValue("--mut"), line = css.getPropertyValue("--line");
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="time series plot">`;
  for (let q = 0; q <= 4; q++) { const v = y0 + (y1 - y0) * q / 4, yy = Y(v);
    s += `<line x1="${l}" x2="${W - r}" y1="${yy}" y2="${yy}" stroke="${grid}"/>`;
    s += `<text x="${l - 6}" y="${yy + 4}" font-size="11" text-anchor="end" fill="${mut}">${v.toFixed(1)}</text>`; }
  // thirds of the window, which the event question's "position" refers to
  [n / 3, 2 * n / 3].forEach(k => { const xx = X(k - 0.5);
    s += `<line x1="${xx}" x2="${xx}" y1="${t - 4}" y2="${H - b}" stroke="${mut}" stroke-dasharray="4 4" opacity=".6"/>`; });
  ["early", "middle", "late"].forEach((w, q) => s += `<text x="${X((q + 0.5) * (n - 1) / 3)}" y="${t - 10}" font-size="12" text-anchor="middle" fill="${mut}">${w}</text>`);
  [0, 32, 64, 96, n - 1].forEach(k => s += `<text x="${X(k)}" y="${H - b + 18}" font-size="11" text-anchor="middle" fill="${mut}">${k}</text>`);
  const pts = vals.map((v, k) => `${X(k).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
  s += `<polyline points="${pts}" fill="none" stroke="${line}" stroke-width="1.6"/>`;
  vals.forEach((v, k) => s += `<circle cx="${X(k).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="1.8" fill="${line}"/>`);
  return s + "</svg>";
}
function complete(id) { const a = answers[id], it = byId[id]; return a && it.slots.every(s => a[s]); }
function render() {
  const it = byId[ORDER[i]];
  document.getElementById("pos").textContent = `Item ${i + 1} of ${ORDER.length}`;
  document.getElementById("plot").innerHTML = plot(it.values);
  document.getElementById("question").textContent = it.question;
  const a = answers[it.id] || {};
  document.getElementById("form").innerHTML = it.slots.map(s =>
    `<fieldset><legend>${LABELS[s]}</legend>` + OPTIONS[s].map(o =>
      `<label class="opt"><input type="radio" name="${s}" value="${o}" ${a[s] === o ? "checked" : ""}> ${o}</label>`).join("") +
    `</fieldset>`).join("");
  document.querySelectorAll("#form input").forEach(el => el.addEventListener("change", () => {
    answers[it.id] = Object.assign(answers[it.id] || {}, {[el.name]: el.value});
    times[it.id] = (times[it.id] || 0) + (Date.now() - shownAt) / 1000; shownAt = Date.now();
    save(); status(); }));
  document.getElementById("prev").disabled = i === 0;
  document.getElementById("next").textContent = i === ORDER.length - 1 ? "Finish" : "Next →";
  shownAt = Date.now(); status();
}
function status() {
  const n = ORDER.filter(complete).length;
  document.getElementById("count").textContent = `${n} / ${ORDER.length} answered`;
  document.getElementById("prog").style.width = (100 * n / ORDER.length) + "%";
  const st = document.getElementById("state"), it = byId[ORDER[i]];
  st.textContent = complete(it.id) ? "answered" : "answer every row to continue";
  st.className = complete(it.id) ? "mut done" : "mut warn";
  document.getElementById("next").disabled = !complete(it.id);
}
document.getElementById("prev").onclick = () => { if (i > 0) { i--; save(); render(); } };
document.getElementById("next").onclick = () => {
  if (i < ORDER.length - 1) { i++; save(); render(); }
  else { const n = ORDER.filter(complete).length;
         alert(n === ORDER.length ? "All items answered. Click 'Download answers' and send the file."
                                  : `${ORDER.length - n} items are still unanswered.`); } };
document.getElementById("jump").onclick = () => { const k = ORDER.findIndex(id => !complete(id));
  if (k >= 0) { i = k; save(); render(); } };
document.getElementById("dl").onclick = () => {
  const out = {annotator: ANNOTATOR, kit: KIT, version: VERSION, saved_at: new Date().toISOString(),
               n_answered: ORDER.filter(complete).length, n_items: ORDER.length, answers, seconds: times};
  const url = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  const a = document.createElement("a"); a.href = url; a.download = `answers_${ANNOTATOR}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000); };
document.getElementById("up").onchange = ev => { const f = ev.target.files[0]; if (!f) return;
  f.text().then(txt => { const d = JSON.parse(txt);
    if (d.kit !== KIT || d.annotator !== ANNOTATOR) { alert("That file is for a different kit or annotator."); return; }
    answers = d.answers || {}; times = d.seconds || {}; const k = ORDER.findIndex(id => !complete(id));
    i = k >= 0 ? k : 0; save(); render(); }); };
render();
</script>
</main></body></html>
"""

INSTRUCTIONS = """# Grading instructions — cycle 2 human baseline

You are one of three graders (A, B, C). Open **only your own file**:
`annotator_A.html`, `annotator_B.html` or `annotator_C.html`. It opens in any
browser, works offline, and needs nothing installed.

## What you do

For each of the __N__ items you see a plot of 128 values of one time series
and a question. Pick one option for every row under the question:

- **Trend** questions ask whether the series drifts up (`increasing`), down
  (`decreasing`) or neither (`stable`), and whether it repeats a cycle at a
  fixed interval (`seasonal`: yes / no).
- **Event** questions ask for the single most important event — a `spike`
  (one isolated point far above its neighbours), an `abrupt change` (a
  lasting shift in level) or `no major event` — and which third of the
  series it is in (`early`, `middle`, `late`; the dashed lines mark the
  thirds). If there is no major event, pick position `none`.

The question text on each item states these definitions too; it is exactly
the text the AI models were given. Its last sentence ("Reply with exactly two
lines ...") told the models how to format their reply - ignore it and just
click your choices.

There is no time limit, but go with your judgement: aim for about 20–30
seconds per item. Expect roughly 1.5 hours in total; the page saves your
progress, so you can stop and come back in the same browser. If the page
shows a warning that progress is not being saved, click **Download answers**
every 20 items or so.

## Rules that keep the grading valid

1. **Answer from the plot only.** Do not look up the series, the data files,
   the code, or anything under `_private/` or `results-archive/`.
2. **Do not discuss items with the other graders** until all three of you
   have sent your answers. Your answers are compared with each other's.
3. **Do not change answers after seeing anyone else's**, or any result.
4. If an item genuinely looks ambiguous, still pick the option you think
   fits best — the AI models had to choose too.

## When you finish

Click **Download answers**. It saves `answers_A.json` (or B/C). Send that
file, or put it in `annotation_c2/answers/`. If you switch computers, use
"Resume from a file" with your latest download.

Kit __KIT__ · version __VERSION__
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=None,
                    help="the H2/H3 pool (data/c2_n100); default: config's")
    ap.add_argument("--out", default=str(ROOT / "annotation_c2"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "_private").mkdir(parents=True, exist_ok=True)
    (out / "answers").mkdir(parents=True, exist_ok=True)

    items, key, kit = build_items(args.pool)
    ids = [it["id"] for it in items]
    for ann in ANNOTATORS:
        order = list(np.random.default_rng(ORDER_SEEDS[ann]).permutation(ids))
        page = (PAGE.replace("__ANNOTATOR__", ann)
                .replace("__KITHASH__", kit)
                .replace("__KITVERSION__", str(KIT_VERSION))
                .replace("__ITEMS__", json.dumps(items, separators=(",", ":")))
                .replace("__ORDER__", json.dumps(order))
                .replace("__OPTIONS__", json.dumps(OPTIONS)))
        (out / f"annotator_{ann}.html").write_text(page, encoding="utf-8",
                                                  newline="\n")
    (out / "INSTRUCTIONS.md").write_text(
        INSTRUCTIONS.replace("__N__", str(len(items)))
        .replace("__KIT__", kit).replace("__VERSION__", str(KIT_VERSION)),
        encoding="utf-8", newline="\n")
    (out / "_private" / "key.json").write_text(json.dumps(dict(
        kit=kit, version=KIT_VERSION, L=L, n_items=len(items),
        orders={a: list(np.random.default_rng(ORDER_SEEDS[a]).permutation(ids))
                for a in ANNOTATORS},
        key=key), indent=1, default=str), encoding="utf-8", newline="\n")
    n_td = sum(it["type"] == "trend_description" for it in items)
    print(f"kit {kit}: {len(items)} items ({n_td} trend, "
          f"{len(items) - n_td} event) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
