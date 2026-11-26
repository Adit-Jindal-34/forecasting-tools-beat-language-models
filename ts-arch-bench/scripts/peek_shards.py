"""Descriptive health check on whatever shards exist. NOT a hypothesis test.

    python scripts/peek_shards.py --results results/h3_backbone_strength

WHY THIS IS SEPARATE FROM `--stage analyse`
-------------------------------------------
`--stage analyse` refuses a partial grid on purpose, so a family is never
computed over whatever happens to be on disk. But while rungs are still
landing you do need to know whether the data is sane - and that is a
different question from whether the hypothesis holds.

So this prints DIAGNOSTICS ONLY: per-arm cell means, coverage, the failure
taxonomy, routing accuracy and call counts. It deliberately computes

  * no paired contrast,
  * no p-value,
  * no Holm correction,
  * no verdict,

because H3's confirmatory difference-in-differences needs only the two ENDS
of the ladder, which may well already be present. Being able to compute it
is not a reason to look at it: seeing the confirmatory result before the
ladder is complete would mean any later decision to re-run a rung was taken
with knowledge of the outcome. That is the researcher degree of freedom the
pre-registration exists to remove, and it costs nothing to avoid.

Read this to answer "is the pipeline producing believable data?" - then run
`--stage analyse` once every rung is in, and read the verdict exactly once.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd                                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    sd = Path(args.results) / "shards"
    shards = sorted(sd.glob("bb=*.csv"))
    if not shards:
        sys.exit(f"no shards under {sd}")

    frames = []
    print("=== shards present ===")
    for p in shards:
        d = pd.read_csv(p)
        frames.append(d)
        print(f"  {p.name:42} {len(d):5} rows  "
              f"{d.system.nunique()} arms  {d.query_type.nunique()} types")
    df = pd.concat(frames, ignore_index=True)
    print(f"\ntotal {len(df)} rows | backbones: {sorted(df.backbone.unique())}")

    print("\n=== mean score by cell and arm (DESCRIPTIVE) ===")
    piv = df.pivot_table(index="query_type", columns=["backbone", "system"],
                         values="score", aggfunc="mean")
    print(piv.round(3).to_string())

    print("\n=== coverage: fraction answered ===")
    cov = df.pivot_table(index="system", columns="backbone",
                         values="answered", aggfunc="mean")
    print(cov.round(3).to_string())

    print("\n=== failure taxonomy ===")
    f = df[df.failure.notna() & (df.failure != "")]
    if len(f):
        print(f.groupby(["backbone", "system", "failure"]).size()
              .unstack(fill_value=0).to_string())
    else:
        print("  (no failures)")

    print("\n=== routing accuracy (abstain counts as wrong) ===")
    rt = df[df.route_source == "llm"]
    if len(rt):
        r = rt.pivot_table(index="query_type", columns="backbone",
                           values="route_correct", aggfunc="mean")
        print(r.round(3).to_string())
        print("\n  overall by backbone:")
        print(rt.groupby("backbone")["route_correct"].mean().round(3)
              .to_string())
        print("\n  route_status counts:")
        print(rt.groupby(["backbone", "route_status"]).size()
              .unstack(fill_value=0).to_string())
    else:
        print("  (no self-routed rows)")

    print("\n=== cost ===")
    print(df.groupby(["backbone", "system"])
          .agg(model_calls=("n_model_calls", "sum"),
               billed=("n_billed_calls", "sum"),
               latency_s=("latency_s", "sum")).round(1).to_string())

    print("\n" + "=" * 70)
    print("DIAGNOSTICS ONLY. No contrast, no p-value, no verdict - by design.")
    print("Run `--stage analyse` once every rung has landed, and read the")
    print("confirmatory result exactly once.")
    print("=" * 70)


if __name__ == "__main__":
    main()
