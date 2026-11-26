"""Build the annotation packet for PREREGISTRATION.md §4.1.

    python scripts/export_annotation.py --results results/h2_query_type \
        --config configs/h2_query_type.yaml --n 50

Writes `<results>/annotation/`:

    instructions.md               what the annotators are told
    taskA_extraction_c{1,2,3}.csv  answer text -> which class does it assert?
    taskB_truth_c{1,2,3}.csv       series -> which class is actually true?
    series/<item_uid>.csv          one series per Task B item, to plot
    _key.json                      MACHINE labels. NOT FOR ANNOTATORS.

WHAT THE ANNOTATORS MUST NOT SEE, AND WHY
-----------------------------------------
`_key.json` holds the machine's labels and stays with you. If an annotator
sees the extractor's answer or the answer key, the agreement this measures is
anchoring, not validity - and a validation that can only confirm is not a
validation. For the same reason each coder's sheet is shuffled with its own
seed, so the three cannot align their work row by row.

Task A sheets carry the ANSWER TEXT and no series: judging what a sentence
asserts must not be contaminated by knowing whether it is true. Task B sheets
carry the SERIES and no answer text, for the mirror reason.

Task B is the one that can embarrass the answer key. `is_seasonal` uses the
generator's TRUE period, which the annotator cannot see and a deployed system
could not either. If humans systematically disagree with the key on
`seasonal`, that is evidence the key encodes information absent from the
data - a finding about the benchmark, not about the annotators.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                      # noqa: E402
import pandas as pd                                     # noqa: E402

from evaluation.human_validation import stratified_sample  # noqa: E402
from evaluation.qualitative import extract_slot          # noqa: E402
from src.config import load_config                       # noqa: E402
from src.ground_truth import (QUALITATIVE_SLOTS, SLOTS_FOR_TYPE,  # noqa: E402
                              qualitative_truth)
from src.pipeline import make_samples                    # noqa: E402

QUAL_TYPES = ("trend_description", "event_explanation")
_BLOCKS = "_.:-=+*#%@"   # ASCII only


def sparkline(x: np.ndarray, width: int = 60) -> str:
    """A crude shape preview, so a sheet is skimmable without opening a plot.

    Deliberately crude: it is an index, not the evidence. The per-item CSV is
    the evidence, and the instructions say to open it.

    ASCII rather than Unicode block characters: the sheets are opened in
    Excel on Windows, where a cp1252 locale turns U+2581..U+2588 into mojibake
    and makes the column useless exactly where it is most likely to be read.
    """
    x = np.asarray(x, float)
    if len(x) > width:
        step = len(x) / width
        x = np.array([x[int(i * step):int((i + 1) * step)].mean()
                      for i in range(width)])
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return _BLOCKS[0] * len(x)
    idx = ((x - lo) / (hi - lo) * (len(_BLOCKS) - 1)).round().astype(int)
    return "".join(_BLOCKS[i] for i in idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True,
                    help="a results dir containing raw_results.csv")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=50,
                    help="items per task (§4.1 asks for 50-100)")
    ap.add_argument("--coders", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()

    res = Path(args.results)
    raw = res / "raw_results.csv"
    if not raw.exists():
        sys.exit(f"no raw_results.csv under {res}")
    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute()
                      else args.config)
    df = pd.read_csv(raw)
    df = df[df.query_type.isin(QUAL_TYPES)].copy()
    if df.empty:
        sys.exit("no qualitative rows in these results")

    # One L only: the answer key depends on the window, so mixing history
    # lengths would make two items with the same sample_id different
    # questions with different truths.
    L = int(sorted(df.L.unique())[len(sorted(df.L.unique())) // 2])
    df = df[df.L == L]
    print(f"qualitative rows at L={L}: {len(df)}")

    out = res / "annotation"
    (out / "series").mkdir(parents=True, exist_ok=True)

    # ---- Task A: answer text -> asserted class -------------------------
    # Sampled across ARMS on purpose: the extractor must cope with the full
    # range of phrasing the three systems produce, not just one arm's style.
    df["item_uid"] = (df.sample_id.astype(str) + "__" + df.system.astype(str))
    a_rows = stratified_sample(
        df.to_dict("records"), n=args.n,
        strata=("query_type", "system"), seed=args.seed)
    print(f"Task A: {len(a_rows)} items over "
          f"{len({r['_stratum'] for r in a_rows})} strata")

    # ---- Task B: series -> true class ----------------------------------
    # Keyed on the ITEM, not the arm: the series and its truth do not depend
    # on which system answered.
    samples = {s.sample_id: s for s in make_samples(cfg, ROOT)}

    uniq = df.drop_duplicates("sample_id").copy()
    uniq["item_uid"] = uniq.sample_id.astype(str)
    b_rows = stratified_sample(
        uniq.to_dict("records"), n=args.n,
        strata=("query_type", "pattern"), seed=args.seed + 1)
    print(f"Task B: {len(b_rows)} items over "
          f"{len({r['_stratum'] for r in b_rows})} strata")

    # ---- machine labels: the key, withheld from annotators -------------
    key = dict(L=L, n_coders=args.coders, seed=args.seed,
               task_a={}, task_b={}, strata={})
    for r in a_rows:
        slots = SLOTS_FOR_TYPE[r["query_type"]]
        pred = {}
        for slot in slots:
            p = extract_slot(str(r.get("answer_text") or ""), slot)
            pred[slot] = (p.predicted if p.status == "one" else p.status)
        key["task_a"][r["item_uid"]] = dict(
            query_type=r["query_type"], system=r["system"], machine=pred,
            stratum=r["_stratum"], weight=r["_sampling_weight"])
    for r in b_rows:
        s = samples[r["sample_id"]]
        truth = qualitative_truth(s, L)
        expected = truth.slots(r["query_type"])
        key["task_b"][r["item_uid"]] = dict(
            query_type=r["query_type"], pattern=r["pattern"],
            machine=dict(expected), stratum=r["_stratum"],
            weight=r["_sampling_weight"])

    # ---- sheets ---------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    for c in range(1, args.coders + 1):
        # Task A
        sheet = []
        for r in a_rows:
            # the question comes from the Sample: raw_results.csv does not
            # carry it, and an annotator judging an answer with no question
            # in front of them is being asked the wrong thing
            row = dict(item_uid=r["item_uid"],
                       question=samples[r["sample_id"]].query,
                       answer_text=str(r.get("answer_text") or ""))
            for slot in SLOTS_FOR_TYPE[r["query_type"]]:
                row[f"label_{slot}"] = ""
            sheet.append(row)
        sa = pd.DataFrame(sheet).sample(frac=1.0,
                                        random_state=args.seed * 100 + c)
        sa.to_csv(out / f"taskA_extraction_c{c}.csv", index=False,
                 encoding="utf-8-sig")

        # Task B
        sheet = []
        for r in b_rows:
            s = samples[r["sample_id"]]
            w = s.window(L)
            row = dict(item_uid=r["item_uid"],
                       question=samples[r["sample_id"]].query,
                       shape=sparkline(w),
                       series_file=f"series/{r['item_uid']}.csv")
            for slot in SLOTS_FOR_TYPE[r["query_type"]]:
                row[f"label_{slot}"] = ""
            sheet.append(row)
            p = out / "series" / f"{r['item_uid']}.csv"
            if not p.exists():
                pd.DataFrame({"t": np.arange(len(w)), "value": w}).to_csv(
                    p, index=False)
        sb = pd.DataFrame(sheet).sample(frac=1.0,
                                        random_state=args.seed * 200 + c)
        sb.to_csv(out / f"taskB_truth_c{c}.csv", index=False,
                 encoding="utf-8-sig")

    (out / "_key.json").write_text(json.dumps(key, indent=2, default=str),
                                   encoding="utf-8")
    (out / "instructions.md").write_text(_instructions(), encoding="utf-8")

    print(f"\nwrote {out}/")
    print(f"  taskA_extraction_c1..c{args.coders}.csv   "
          f"({len(a_rows)} items each)")
    print(f"  taskB_truth_c1..c{args.coders}.csv        "
          f"({len(b_rows)} items each)")
    print(f"  series/  ({len(list((out / 'series').glob('*.csv')))} files)")
    print(f"  instructions.md")
    print(f"  _key.json   <- MACHINE LABELS. Do not send this to annotators.")
    print("\nWhen the sheets come back, put them in the same directory and run:")
    print(f"  python scripts/analyse_annotation.py --annotation {out}")


def _instructions() -> str:
    opts = {k: list(v) for k, v in QUALITATIVE_SLOTS.items()}
    return f"""# Annotation instructions

Thank you. This measures whether an automatic scorer agrees with human
readers. There are **two separate tasks** and they are deliberately
different. Please do them in order and do not go back and revise Task A
after seeing Task B.

You will not be shown the scorer's answers. Please do not discuss items with
the other annotators until both tasks are returned — the whole point is three
independent readings.

Fill in the `label_*` columns. Leave nothing blank; use `unsure` if you
genuinely cannot tell.

---

## Task A — what does this sentence assert?

File: `taskA_extraction_c<N>.csv`

You get a **question** and a model's **answer**. You do **not** get the
series, and you do not need it. Do not judge whether the answer is *true*.
Judge only **what it claims**.

For each `label_<slot>` column, enter one of:

| slot | allowed labels |
|---|---|
| `trend` | {" / ".join(opts["trend"])} / `absent` / `contradictory` |
| `seasonal` | {" / ".join(opts["seasonal"])} / `absent` / `contradictory` |
| `event` | {" / ".join(opts["event"])} / `absent` / `contradictory` |
| `position` | {" / ".join(opts["position"])} / `absent` / `contradictory` |

- use the class the sentence asserts;
- `absent` — the sentence says nothing about this slot;
- `contradictory` — it asserts **two or more** classes of the same slot
  (e.g. "rose then fell" for `trend`, or naming both a spike and an abrupt
  change as *the* main event).

`contradictory` is a real and expected answer. Please do not avoid it.

---

## Task B — what is actually true of this series?

File: `taskB_truth_c<N>.csv`

Now you get the **series** and no model answer. `shape` is a rough preview;
the real data is in `series_file` (`series/<item_uid>.csv`, two columns —
open it in a spreadsheet and plot `value` against `t`). **Please plot it.**
The preview compresses and can mislead.

For each `label_<slot>` column, enter one of:

| slot | allowed labels |
|---|---|
| `trend` | {" / ".join(opts["trend"])} / `unsure` |
| `seasonal` | {" / ".join(opts["seasonal"])} / `unsure` |
| `event` | {" / ".join(opts["event"])} / `unsure` |
| `position` | {" / ".join(opts["position"])} / `unsure` |

Definitions, so three readers apply the same ones:

- **trend** — the overall direction across the whole window. `stable` if
  there is no clear overall direction.
- **seasonal** — is there a repeating cycle of roughly fixed length? Judge
  only from the data in front of you. If you cannot see a repeating period,
  answer `no`, even if the series looks complicated.
- **event** — the single most important thing that happens:
  `spike` (one or a few isolated extreme points, returning to the previous
  level), `abrupt change` (the level shifts and *stays* shifted), or
  `no major event`.
- **position** — where that event occurs: `early` / `middle` / `late` third
  of the window. If you answered `no major event`, answer `unsure` here.

Use `unsure` freely. An item three readers cannot agree on tells us something
important about the benchmark, and guessing would hide it.
"""


if __name__ == "__main__":
    main()
