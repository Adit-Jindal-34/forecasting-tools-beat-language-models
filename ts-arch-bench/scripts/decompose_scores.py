"""Report why items scored zero, and how much of a contrast that explains.

    python scripts/decompose_scores.py --results results/h2_query_type
    python scripts/decompose_scores.py --results results/h3_backbone_strength \
        --arms bare_llm harness

Reads `raw_results.csv` if present, otherwise concatenates `shards/*.csv`, so
it works on a partial grid while rungs are still landing.

This prints a DESCRIPTIVE decomposition: no CI, no p-value, no verdict. The
confirmatory numbers come from `--stage analyse` and are not recomputed
here. What this answers is a different question - whether a contrast is
carrying instrument behaviour - and it answers it by reading the slot
columns (`n_absent`, `n_ambiguous`, `n_slots`) rather than inferring cause
from the shape of the score distribution, which is easy to get backwards.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd                                      # noqa: E402

from evaluation.decomposition import (                    # noqa: E402
    contrast_table, decompose, non_slot_readability, zero_taxonomy)


def _load(results: Path) -> pd.DataFrame:
    raw = results / "raw_results.csv"
    if raw.exists():
        return pd.read_csv(raw)
    shards = sorted((results / "shards").glob("*.csv"))
    if not shards:
        sys.exit(f"no raw_results.csv and no shards under {results}")
    print(f"(no raw_results.csv; concatenating {len(shards)} shards)")
    return pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--arms", nargs=2, default=["bare_llm", "harness"],
                    metavar=("ARM_A", "ARM_B"),
                    help="contrast is ARM_B - ARM_A (default: harness - bare_llm)")
    args = ap.parse_args()

    df = _load(Path(args.results))
    print(f"loaded {len(df)} rows")
    if "backbone" in df.columns and df.backbone.nunique() > 1:
        print(f"NOTE: {df.backbone.nunique()} backbones pooled "
              f"({sorted(df.backbone.unique())}). Pass a single-backbone "
              f"results dir if you want a per-rung view.")

    tax = zero_taxonomy(df)
    print("\n=== why slot-scored items scored zero (counts) ===")
    if tax.empty:
        print("  (no slot-scored rows)")
    else:
        print(tax.to_string())
        print("\n  nothing_extracted       -> unscoreable; a measurement gap")
        print("  ambiguous               -> read, but named two classes of "
              "one exclusive slot")
        print("  extracted_but_all_wrong -> read cleanly and wrong; genuine "
              "capability")

    dec = decompose(df)
    if not dec.empty:
        print("\n=== mean score under three readings ===")
        cols = ["query_type", "system", "n_items", "mean_score",
                "n_unscoreable", "n_ambiguous_zero",
                "mean_excl_unscoreable", "mean_excl_instrument"]
        print(dec[cols].round(4).to_string(index=False))

    a, b = args.arms
    ct = contrast_table(df, a, b)
    if not ct.empty:
        print(f"\n=== contrast {b} - {a} under each reading ===")
        print(ct.round(4).to_string(index=False))
        print("\n  If delta_score and delta_excl_instrument agree, the "
              "contrast is not an")
        print("  artifact of how zeroes are attributed. If they diverge, the "
              "headline number")
        print("  is carrying instrument behaviour and the paper must say so.")
        print("  delta_excl_instrument is an UPPER BOUND on the instrument's "
              "possible role:")
        print("  it drops items because they scored badly, which is "
              "selection on the outcome.")

    ns = non_slot_readability(df)
    if not ns.empty:
        print("\n=== non-slot cells (no slots; answered-rate only) ===")
        print(ns.round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("DESCRIPTIVE ONLY. No CI, no p-value, no verdict - by design.")
    print("The confirmatory result comes from `--stage analyse`.")
    print("=" * 70)


if __name__ == "__main__":
    main()
