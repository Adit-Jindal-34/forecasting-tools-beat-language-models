"""Read completed annotation sheets and report §4.1, flattering or not.

    python scripts/analyse_annotation.py --annotation results/h2_query_type/annotation

Reports, per task and per slot:

  * Krippendorff's alpha across the annotators - the CEILING. Machine
    agreement cannot be trusted beyond the humans' agreement with each
    other: if three readers cannot agree what a series shows, "the scorer
    matches the humans" is not a meaningful claim.
  * machine vs human consensus - raw agreement AND alpha, because on a
    skewed slot (`event` is often "no major event") the raw rate flatters.
  * the confusion matrix, so a systematic direction is visible rather than
    averaged away.
  * items with no human consensus, counted not dropped.

NOTHING HERE THRESHOLDS A RESULT. §4.1 requires the correlation be reported
whether or not it is flattering, so the script prints Krippendorff's own
reading of each alpha and stops. Deciding what a low alpha means for the
confirmatory family is a protocol question, not an arithmetic one, and it
belongs in PREREGISTRATION.md §5 as a dated entry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd                                     # noqa: E402

from evaluation.human_validation import (krippendorff_alpha,  # noqa: E402
                                         consensus, machine_vs_human)

TASKS = {"task_a": ("taskA_extraction_c*.csv", "extraction: text -> class"),
         "task_b": ("taskB_truth_c*.csv", "answer key: series -> class")}


def _load(adir: Path, pattern: str) -> dict:
    """coder id -> DataFrame, by filename glob."""
    out = {}
    for p in sorted(adir.glob(pattern)):
        cid = p.stem.rsplit("_c", 1)[-1]
        out[cid] = pd.read_csv(p).set_index("item_uid")
    return out


def analyse_task(sheets: dict, key: dict, task: str) -> dict:
    if not sheets:
        return dict(error="no completed sheets found")
    slots = sorted({c[len("label_"):] for df in sheets.values()
                    for c in df.columns if c.startswith("label_")})
    uids = sorted(key.keys())
    result = dict(n_coders=len(sheets), coders=sorted(sheets),
                  n_items=len(uids), slots={})

    for slot in slots:
        col = f"label_{slot}"
        units, machine, human = {}, {}, {}
        for u in uids:
            applicable = slot in (key[u]["machine"] or {})
            if not applicable:
                continue                       # slot not used by this type
            codings = []
            for cid, df in sheets.items():
                v = (df[col].get(u) if col in df.columns else None)
                v = None if (v is None or pd.isna(v) or str(v).strip() == ""
                             ) else str(v).strip().lower()
                codings.append(v)
            if not any(c is not None for c in codings):
                continue
            units[u] = codings
            lab, _ = consensus(codings)
            human[u] = lab
            machine[u] = str(key[u]["machine"][slot]).strip().lower()

        if not units:
            result["slots"][slot] = dict(note="no codings for this slot")
            continue

        inter = krippendorff_alpha(units, level="nominal")
        vs = machine_vs_human(machine, human)
        n_unsure = sum(1 for cs in units.values()
                       for c in cs if c in ("unsure", "absent"))
        result["slots"][slot] = dict(
            inter_annotator=inter, machine_vs_human=vs,
            n_units=len(units), n_unsure_or_absent_codings=n_unsure)
    return result


def _print(task: str, label: str, r: dict) -> None:
    print(f"\n{'=' * 74}")
    print(f"{task.upper()}  -  {label}")
    print("=" * 74)
    if "error" in r:
        print(f"  {r['error']}")
        return
    print(f"coders: {r['n_coders']} {r['coders']}   items: {r['n_items']}")
    hdr = (f"{'slot':<12}{'n':>5}{'alpha_human':>13}{'machine_agr':>13}"
           f"{'alpha_m_vs_h':>14}{'no_cons':>9}")
    print(hdr)
    print("-" * len(hdr))
    for slot, s in r["slots"].items():
        if "note" in s:
            print(f"{slot:<12}  {s['note']}")
            continue
        ia, mv = s["inter_annotator"], s["machine_vs_human"]
        f = lambda v: "nan" if v != v else f"{v:.3f}"       # noqa: E731
        print(f"{slot:<12}{s['n_units']:>5}{f(ia['alpha']):>13}"
              f"{f(mv['agreement']):>13}"
              f"{f(mv['alpha_machine_vs_consensus']):>14}"
              f"{mv['n_no_consensus']:>9}")
    for slot, s in r["slots"].items():
        if "note" in s:
            continue
        print(f"\n  {slot}: {s['inter_annotator']['note']}")
        conf = s["machine_vs_human"].get("confusion") or {}
        dis = {k: v for k, v in conf.items()
               if k.split("|")[0].split("=")[1] != k.split("|")[1].split("=")[1]}
        if dis:
            print(f"    disagreements: "
                  + ", ".join(f"{k} x{v}" for k, v in
                              sorted(dis.items(), key=lambda kv: -kv[1])[:6]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation", required=True)
    args = ap.parse_args()
    adir = Path(args.annotation)
    keyf = adir / "_key.json"
    if not keyf.exists():
        sys.exit(f"no _key.json under {adir}; run export_annotation.py first")
    key = json.loads(keyf.read_text(encoding="utf-8"))

    report = dict(L=key.get("L"), source=str(adir))
    for task, (pattern, label) in TASKS.items():
        sheets = _load(adir, pattern)
        # drop sheets that came back entirely blank: an unreturned sheet is
        # not a coder who agreed with everything
        sheets = {c: df for c, df in sheets.items()
                  if df.filter(like="label_").notna().any().any()}
        r = analyse_task(sheets, key.get(task) or {}, task)
        report[task] = r
        _print(task, label, r)

    outf = adir / "validation_report.json"
    outf.write_text(json.dumps(report, indent=2, default=str),
                    encoding="utf-8")

    print(f"\n{'=' * 74}")
    print("HOW TO READ THIS")
    print("=" * 74)
    print("""
alpha_human is the ceiling. The scorer cannot be more valid than the humans
are consistent, so read it first; a low value there is a finding about the
TASK (or the instructions), not about the scorer.

Then alpha_m_vs_h, not machine_agr. Raw agreement on a slot whose classes are
unbalanced can be high while carrying no information.

The two tasks fail differently and have different remedies:
  task_a low  -> the regexes in evaluation/qualitative.py misread text.
                 Fixable in the extractor, and PARSER_VERSION covers it, so
                 a re-score is free - no model calls.
  task_b low  -> the answer key disagrees with human readers of the same
                 data. Not fixable in the extractor. If `seasonal` is the
                 slot, the likely cause is that `is_seasonal` uses the
                 generator's true period, which neither a human nor a
                 deployed system can see.

Whatever comes out, it is a §5 entry before any qualitative number is
reported. Two of the three confirmatory cells are scored by this metric.""")
    print(f"\nwrote {outf}")


if __name__ == "__main__":
    main()
