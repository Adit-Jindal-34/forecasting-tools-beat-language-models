"""Analyse the cycle-2 human grading (PREREGISTRATION_C2.md, amendment A17).

    python scripts/c2_analyse_annotation.py \
        --answers annotation_c2/answers \
        --models ../results-archive/20261113-cycle2-run/results

Reads `annotation_c2/_private/key.json` and every `answers_*.json`, and
reports, in this order (the order A17 declares):

  1. completeness and kit check - an answer file from another kit is refused
  2. agreement BETWEEN the graders, per slot: Krippendorff's alpha (nominal)
     and raw pairwise agreement. Alpha is the primary figure; raw agreement
     is never read alone (see evaluation/human_validation.py)
  3. each grader's accuracy against the generator key, scored with the SAME
     scorer as the models (evaluation/qualitative.py::score_qualitative_mc),
     plus the majority vote of the three
  4. humans beside the models on the identical items: every arm of every
     model found under --models, the constant answer, and the rule applied
     to the numeric evidence block
  5. answerability: items the majority of graders got wrong, by key class
     and noise type, and whether models fail on the same items

Descriptive throughout. Paired comparisons (humans vs a model arm) report a
bootstrap CI and a raw Wilcoxon p, uncorrected.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.human_validation import krippendorff_alpha   # noqa: E402
from evaluation.qualitative import score_qualitative_mc      # noqa: E402
from src import ground_truth as gt                           # noqa: E402
from src.ground_truth import QualitativeTruth                # noqa: E402

SLOTS = ("trend", "seasonal", "event", "position")
CELLS = {"trend_description": ("trend", "seasonal"),
         "event_explanation": ("event", "position")}


def load_answers(folder: Path, kit: str) -> dict[str, dict]:
    out = {}
    for f in sorted(folder.glob("answers_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("kit") != kit:
            raise SystemExit(
                f"{f.name} is for kit {d.get('kit')}, but the key is kit "
                f"{kit}. Answers from a different kit cannot be mixed in.")
        out[d["annotator"]] = d
    return out


def as_text(ans: dict) -> str:
    """A grader's choices in the exact format a model's reply is scored in."""
    return "\n".join(f"{k}: {v}" for k, v in ans.items())


def truth_of(entry: dict, qt: str) -> QualitativeTruth:
    e = entry["expected"]
    return QualitativeTruth(trend=e.get("trend", "stable"),
                            seasonal=e.get("seasonal", "no"),
                            event=e.get("event"), position=e.get("position"))


def score(ans: dict | None, entry: dict) -> dict:
    """{'score':..., slot: 0/1 or None} for one item, as the models are scored."""
    qt = entry["type"]
    t = truth_of(entry, qt)
    if qt == "event_explanation" and "event" not in entry["expected"]:
        raise ValueError("event key missing")
    r = score_qualitative_mc(as_text(ans or {}), t, qt)
    row = dict(score=r["score"])
    for s in SLOTS:
        row[s] = r["per_slot"][s]["correct"] if s in r["per_slot"] else None
    return row


def boot(d, n=5000, seed=20261114):
    d = np.asarray(d, float)
    m = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=str(ROOT / "annotation_c2"))
    ap.add_argument("--answers", default=None)
    ap.add_argument("--models", default=None,
                    help="results dir holding c2_h3/ and c2_h2x_*/ (optional)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    kit_dir = Path(args.kit)
    K = json.loads((kit_dir / "_private" / "key.json").read_text(
        encoding="utf-8"))
    key = K["key"]
    ans = load_answers(Path(args.answers or kit_dir / "answers"), K["kit"])
    out = Path(args.out or kit_dir / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    report: dict = dict(kit=K["kit"], graders=sorted(ans))

    # ---------------------------------------------------------------- 1
    print(f"=== 1. kit {K['kit']}: {len(key)} items, graders {sorted(ans)}")
    comp = {}
    for g, d in sorted(ans.items()):
        n_full = sum(1 for i, e in key.items()
                     if all((d["answers"].get(i) or {}).get(s)
                            for s in CELLS[e["type"]]))
        secs = list((d.get("seconds") or {}).values())
        comp[g] = dict(complete=n_full, of=len(key),
                       median_seconds=float(np.median(secs)) if secs else None)
        print(f"  {g}: {n_full}/{len(key)} complete, median "
              f"{comp[g]['median_seconds'] or float('nan'):.0f} s per item")
    report["completeness"] = comp
    if len(ans) < 2:
        print("\nfewer than two graders: agreement cannot be computed yet")

    # per-item scored table
    rows = []
    for i, e in key.items():
        for g, d in ans.items():
            a = (d["answers"] or {}).get(i)
            r = score(a, e)
            rows.append(dict(item=i, type=e["type"], grader=g, **r,
                             **{f"ans_{s}": (a or {}).get(s) for s in SLOTS}))
    T = pd.DataFrame(rows)

    # ---------------------------------------------------------------- 2
    print("\n=== 2. agreement between graders (alpha is the primary figure)")
    agree = {}
    for qt, slots in CELLS.items():
        for s in slots:
            units = {}
            for i, e in key.items():
                if e["type"] != qt:
                    continue
                units[i] = [((ans[g]["answers"] or {}).get(i) or {}).get(s)
                            for g in sorted(ans)]
            a = krippendorff_alpha(units)
            pairs = []
            for x, y in itertools.combinations(sorted(ans), 2):
                both = [(u[sorted(ans).index(x)], u[sorted(ans).index(y)])
                        for u in units.values()]
                both = [(p, q) for p, q in both if p and q]
                if both:
                    pairs.append(np.mean([p == q for p, q in both]))
            agree[s] = dict(alpha=a.get("alpha"),
                            raw_pairwise=float(np.mean(pairs)) if pairs
                            else float("nan"), n_units=a.get("n_units"))
            print(f"  {s:<9} alpha={agree[s]['alpha']:+.3f}  raw pairwise "
                  f"agreement={agree[s]['raw_pairwise']:.3f}  "
                  f"(n={agree[s]['n_units']})")
    report["agreement"] = agree

    # majority vote
    maj_rows = []
    for i, e in key.items():
        vote = {}
        for s in CELLS[e["type"]]:
            c = Counter(((ans[g]["answers"] or {}).get(i) or {}).get(s)
                        for g in ans)
            c.pop(None, None)
            top = c.most_common()
            # a strict majority only; a three-way split is no answer
            if top and top[0][1] > len(ans) / 2:
                vote[s] = top[0][0]
        maj_rows.append(dict(item=i, type=e["type"], grader="majority",
                             **score(vote, e)))
    M = pd.DataFrame(maj_rows)
    A = pd.concat([T, M], ignore_index=True)

    # ---------------------------------------------------------------- 3
    print("\n=== 3. accuracy against the key (same scorer as the models)")
    acc = (A.groupby(["grader", "type"]).score.mean().unstack().round(3))
    slot_acc = A.groupby("grader")[list(SLOTS)].mean().round(3)
    print(acc.to_string())
    print("\n  per slot (position only on items that have an event):")
    print(slot_acc.to_string())
    report["accuracy_by_cell"] = acc.to_dict()
    report["accuracy_by_slot"] = slot_acc.to_dict()

    # ---------------------------------------------------------------- 4
    human = T.groupby(["item", "type"]).score.mean().rename("humans")
    comp_rows = []
    if args.models:
        res = Path(args.models)
        runs = {}
        h3 = res / "c2_h3" / "raw_results.csv"
        if h3.exists():
            df = pd.read_csv(h3, low_memory=False)
            for bb, sub in df.groupby("backbone"):
                runs[bb] = sub
        for d in sorted(res.glob("c2_h2x_*")):
            if (d / "raw_results.csv").exists():
                runs[d.name[len("c2_h2x_"):]] = pd.read_csv(
                    d / "raw_results.csv", low_memory=False)
        hm = human.reset_index().set_index("item")
        for model, df in runs.items():
            q = df[df.query_type.isin(list(CELLS))]
            for arm in ("bare_llm", "bare_features", "harness"):
                p = q[q.system == arm].set_index("sample_id").score
                for qt in CELLS:
                    j = hm[hm.type == qt].join(p.rename("m"), how="inner")
                    if j.empty:
                        continue
                    d = (j.humans - j.m).to_numpy()
                    lo, hi = boot(d)
                    comp_rows.append(dict(
                        model=model, arm=arm, cell=qt,
                        model_score=float(j.m.mean()),
                        humans=float(j.humans.mean()),
                        humans_minus_model=float(d.mean()),
                        ci=f"[{lo:+.3f},{hi:+.3f}]",
                        p=float(wilcoxon(d).pvalue) if np.any(d != 0)
                        else float("nan")))
    # references that need no model run
    ref = []
    for qt, slots in CELLS.items():
        items = {i: e for i, e in key.items() if e["type"] == qt}
        exp = [e["expected"] for e in items.values()]
        best = 0.0
        for combo in itertools.product(
                *(gt.QUALITATIVE_SLOTS[s] for s in slots)):
            c = dict(zip(slots, combo))
            sc = np.mean([np.mean([c[k] == v for k, v in x.items()])
                          for x in exp])
            best = max(best, sc)
        ref.append(dict(cell=qt, reference="best constant answer",
                        score=float(best)))
    print("\n=== 4. humans beside the models on the same items")
    print("  references:", {f"{r['cell']} constant": round(r["score"], 3)
                            for r in ref})
    if comp_rows:
        C = pd.DataFrame(comp_rows)
        C.to_csv(out / "humans_vs_models.csv", index=False)
        show = C[C.arm != "bare_features"].copy()
        print(show.round(3).to_string(index=False))
    else:
        print("  (no model results given; pass --models)")
    report["references"] = ref

    # ---------------------------------------------------------------- 5
    print("\n=== 5. answerability: items the majority of graders got wrong")
    Mi = M.set_index("item")
    hard = []
    for i, e in key.items():
        tr = e["truth"]
        hard.append(dict(item=i, type=e["type"],
                         majority_score=float(Mi.loc[i, "score"]),
                         mean_score=float(human.loc[(i, e["type"])]),
                         trend=tr["trend"], seasonal=tr["seasonal"],
                         event=tr["event"], noise=tr["noise"]))
    H = pd.DataFrame(hard)
    H.to_csv(out / "items.csv", index=False)
    for qt, col in (("trend_description", "trend"),
                    ("event_explanation", "event")):
        s = H[H.type == qt]
        print(f"  {qt}: mean human score by true {col} / noise")
        print("   ", s.groupby(col).mean_score.mean().round(3).to_dict(),
              s.groupby("noise").mean_score.mean().round(3).to_dict())
    report["n_items_majority_wrong"] = int((H.majority_score < 0.5).sum())
    print(f"  items the majority got mostly wrong (score < 0.5): "
          f"{report['n_items_majority_wrong']} of {len(H)}")
    if comp_rows:
        # do models fail on the items humans find hard, or on easy ones?
        easy = set(H[H.majority_score >= 0.5].item)
        split = []
        for model, df in runs.items():
            for arm in ("bare_llm", "harness"):
                q = df[(df.system == arm) & df.query_type.isin(list(CELLS))]
                q = q.set_index("sample_id").score
                e = q[q.index.isin(easy)]
                h = q[~q.index.isin(easy)]
                split.append(dict(model=model, arm=arm,
                                  on_human_easy=float(e.mean()),
                                  on_human_hard=float(h.mean())
                                  if len(h) else float("nan"),
                                  n_hard=int(len(h))))
        S = pd.DataFrame(split)
        S.to_csv(out / "models_on_easy_vs_hard.csv", index=False)
        print("\n  model score on items humans found easy vs hard:")
        print(S.round(3).to_string(index=False))
    T.to_csv(out / "per_grader_items.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2,
                                                default=str),
                                     encoding="utf-8")
    print(f"\nwrote {out}/report.json, items.csv, per_grader_items.csv"
          + (", humans_vs_models.csv" if comp_rows else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
