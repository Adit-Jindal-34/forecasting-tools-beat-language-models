"""H3 sensitivity analysis, using the project's OWN confirmatory_did.

    python scripts/h3_sensitivity.py

Writes results-archive/20261110-h2-seed2-replication/h3_sensitivity.json
and prints the four readings side by side. Needs no GPU: it re-analyses the
archived H3 shards.


Two variants, both POST-HOC and both labelled as such:

  (a) FLOOR AT 3B. The declared weak endpoint, 0.5B, routes correctly on
      7.2% of `trend_description` items and 9.2% of `event_explanation`,
      and the self-routed harness there is WORSE than the bare arm. A
      contrast anchored on a model that cannot operate the architecture
      measures routing competence. Re-anchoring on 3B keeps the same
      question with a weak endpoint that can route.

  (b) ORACLE ARM. Replaces (bare_llm, harness) with (bare_llm,
      harness_oracle), which holds tool selection correct by construction.
      `harness_oracle` is a DECLARED secondary ablation (D2), so this is a
      legitimate secondary contrast rather than a new arm invented now.

Neither is confirmatory. Both are reported whichever way they fall.
"""
import copy
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path("D:/TS-ARCH-BENCH-main/ts-arch-bench")
ARCH = pathlib.Path("D:/TS-ARCH-BENCH-main/results-archive/"
                    "20261107-all-three-complete/results/h3_backbone_strength")
sys.path.insert(0, str(ROOT))

from runners import _common as C                                 # noqa: E402
from src.config import load_config                               # noqa: E402

cfg = load_config(str(ROOT / "configs/h3_backbone_strength.yaml"))
df = pd.concat([pd.read_csv(f) for f in sorted((ARCH / "shards").glob("*.csv"))],
               ignore_index=True)
print(f"loaded {len(df)} rows, backbones={sorted(df.backbone.unique())}")
print(f"declared contrast : {tuple(cfg.analysis['confirmatory_contrast'])}")
print(f"declared family   : {list(cfg.analysis['confirmatory_family'])}")

LADDER = list(cfg.analysis["ladder"]) if "ladder" in cfg.analysis else None
print(f"declared ladder   : {LADDER}")

CELLS = list(cfg.analysis["confirmatory_family"])


def run(label, first, second, contrast=None):
    c = copy.deepcopy(cfg)
    if contrast is not None:
        c.analysis["confirmatory_contrast"] = list(contrast)
    res = C.confirmatory_did(df, c, axis="backbone", first=first,
                             second=second)
    pair = tuple(c.analysis["confirmatory_contrast"])
    print(f"\n{'=' * 74}")
    print(f"{label}")
    print(f"  arms={pair[0]} vs {pair[1]}   Delta = gap({first}) - gap({second})")
    print(f"  POSITIVE = architectural gap SHRINKS as the backbone gets stronger")
    print(f"{'=' * 74}")
    print(f"  {'cell':<20}{'Delta':>9}{'CI':>22}{'p_raw':>11}{'p_adj':>11}  win")
    out = {}
    for cell in CELLS:
        v = res["cells"][cell]
        out[cell] = v
        print(f"  {cell:<20}{v['mean_diff']:>+9.4f}"
              f"   [{v['ci_low']:+.4f},{v['ci_high']:+.4f}]"
              f"{v['p_raw']:>11.4g}{v['p_adj']:>11.4g}  {v['win']}")
    return out


declared = run("(0) DECLARED, as pre-registered", "qwen2_5_0_5b",
               "qwen2_5_72b")
floor3b = run("(a) POST-HOC: floor at 3B, self-routed", "qwen2_5_3b",
              "qwen2_5_72b")
oracle = run("(b) POST-HOC: oracle-routed, declared endpoints",
             "qwen2_5_0_5b", "qwen2_5_72b",
             contrast=("bare_llm", "harness_oracle"))
both = run("(c) POST-HOC: oracle-routed AND floor at 3B", "qwen2_5_3b",
           "qwen2_5_72b", contrast=("bare_llm", "harness_oracle"))

print(f"\n{'=' * 74}")
print("SUMMARY: sign of Delta under each reading "
      "(+ = gap shrinks with strength)")
print(f"{'=' * 74}")
print(f"  {'cell':<20}{'declared':>11}{'3B floor':>11}{'oracle':>11}"
      f"{'both':>11}   stable sign?")
for cell in CELLS:
    vals = [declared[cell]["mean_diff"], floor3b[cell]["mean_diff"],
            oracle[cell]["mean_diff"], both[cell]["mean_diff"]]
    sigs = [declared[cell]["win"], floor3b[cell]["win"], oracle[cell]["win"],
            both[cell]["win"]]
    signs = {v > 0 for v in vals}
    print(f"  {cell:<20}" + "".join(
        f"{v:>+10.4f}{'*' if s else ' '}" for v, s in zip(vals, sigs))
        + f"   {'YES' if len(signs) == 1 else 'NO - REVERSES'}")
print("  (* = detected after Holm within that reading)")

out = {k: {c: {kk: (float(vv) if isinstance(vv, (int, float)) else vv)
               for kk, vv in v.items()}
           for c, v in d.items()}
       for k, d in (("declared_0_5b_vs_72b", declared),
                    ("posthoc_3b_vs_72b", floor3b),
                    ("posthoc_oracle_0_5b_vs_72b", oracle),
                    ("posthoc_oracle_3b_vs_72b", both))}
dest = pathlib.Path("D:/TS-ARCH-BENCH-main/results-archive/"
                    "20261110-h2-seed2-replication/h3_sensitivity.json")
dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print(f"\nwrote {dest}")
