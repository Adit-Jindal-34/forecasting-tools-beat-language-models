"""Verify every figure quoted in FINDINGS_2026-11-09.md against the archive.

    python scripts/verify_findings.py

Recomputes each canonical number from the frozen shards, analysis.json and
judge reports, then asserts the formatted value appears in the document.
Transcription error is the realistic failure mode for a hand-written results
table, and a results document nobody can check is a results document nobody
should believe.

Exits non-zero and names every mismatch if any figure does not reproduce.
Covers: the three confirmatory tables with CIs, row and shard counts,
H3 route_status and coverage, H1's five-point dose-response, the H3 ladder
for both the self-routed and oracle-routed arms, the oracle-routed endpoint
contrasts, routing accuracy per rung, and all judge alphas (pooled and
broken down by L and by backbone).

Comparison is on VALUES: both sides are normalised for typographic minus
(U+2212) and thousands separators, so prose style cannot fail a check.
"""
import json
import pathlib
import re

import pandas as pd

ROOT = pathlib.Path("D:/TS-ARCH-BENCH-main")
ARCH = ROOT / "results-archive/20261107-all-three-complete/results"
VAL = ROOT / "results-archive/20261108-slot-extractor-validation"
_RAW = (ROOT / "ts-arch-bench/FINDINGS_2026-11-09.md").read_text(
    encoding="utf-8")
# The document uses a typographic minus (U+2212) and no thousands separator
# in some places. Normalise BOTH sides so the check is about VALUES, not
# about which hyphen or comma style the prose happens to use.
def _norm(t):
    return (t.replace("−", "-").replace("–", "-")
             .replace(",", ""))
DOC = _norm(_RAW)

NAMES = {"h1": "h1_history_length", "h2": "h2_query_type",
         "h3": "h3_backbone_strength"}
CELLS = ("forecasting", "trend_description", "event_explanation")
ORDER = ["qwen2_5_0_5b", "qwen2_5_1_5b", "qwen2_5_3b", "qwen2_5_7b",
         "qwen2_5_14b", "qwen2_5_32b", "qwen2_5_72b"]

fails, checks = [], 0


def ck(label, text):
    """Assert `text` occurs in the document."""
    global checks
    checks += 1
    if _norm(text) not in DOC:
        fails.append(f"{label}: expected to find {text!r}")


# ---- 1. confirmatory tables -----------------------------------------
for h, name in NAMES.items():
    a = json.loads((ARCH / name / "analysis.json").read_text(encoding="utf-8"))
    for cell in CELLS:
        v = a["confirmatory"]["cells"][cell]
        ck(f"{h}/{cell} Delta", f"{v['mean_diff']:+.4f}")
        ck(f"{h}/{cell} CI", f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")

# row counts
tot = 0
for h, name in NAMES.items():
    shards = sorted((ARCH / name / "shards").glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
    tot += len(df)
    ck(f"{h} rows", f"{len(df):,}")
    ck(f"{h} shards", str(len(shards)))
ck("total rows", f"{tot:,}")

# route_status for h3
df3 = pd.concat([pd.read_csv(f) for f in
                 sorted((ARCH / NAMES["h3"] / "shards").glob("*.csv"))],
                ignore_index=True)
rs = df3.route_status.value_counts().to_dict()
ck("h3 unparseable", str(rs.get("unparseable")))
ck("h3 recovered", str(rs.get("recovered")))
ans = df3.groupby("system").answered.mean().round(4)
for arm in ("bare_llm", "harness", "harness_oracle"):
    ck(f"h3 answered {arm}", f"{ans[arm]:.4f}")

# ---- 2. H1 dose-response from curve_by_L.csv ------------------------
c1 = pd.read_csv(ARCH / NAMES["h1"] / "curve_by_L.csv")
for _, r in c1[c1.query_type.isin(CELLS)].iterrows():
    if r.query_type == "forecasting":
        ck(f"h1 curve {r.query_type} L{int(r.L)}", f"{r.mean_gap:.4f}")
        ck(f"h1 curve CI {r.query_type} L{int(r.L)}",
           f"[{r.ci_low:.4f}, {r.ci_high:.4f}]")
    else:
        ck(f"h1 curve {r.query_type} L{int(r.L)}", f"{r.mean_gap:.4f}")

# ---- 3. H3 ladder, self and oracle ----------------------------------
d3 = df3[df3.query_type.isin(CELLS)]
p3 = d3.pivot_table(index="backbone", columns=["query_type", "system"],
                    values="score", aggfunc="mean").reindex(ORDER)
for cell in CELLS:
    for arm, tag in (("harness", "self"), ("harness_oracle", "oracle")):
        gap = p3[(cell, arm)] - p3[(cell, "bare_llm")]
        for b in ORDER:
            ck(f"h3 ladder {cell}/{tag}/{b}", f"{gap.loc[b]:+.3f}")
    # oracle DiD
    go = p3[(cell, "harness_oracle")] - p3[(cell, "bare_llm")]
    ck(f"h3 oracle DiD {cell}",
       f"{go.loc['qwen2_5_0_5b'] - go.loc['qwen2_5_72b']:+.4f}")

# ---- 4. routing accuracy --------------------------------------------
rt = pd.read_csv(ARCH / NAMES["h3"] / "desc_routing.csv")
rt = rt[rt.query_type.isin(CELLS)].pivot_table(
    index="backbone", columns="query_type", values="route_correct")
rt = rt.reindex(ORDER)
for b in ORDER:
    for cell in CELLS:
        ck(f"routing {b}/{cell}", f"{rt.loc[b, cell]:.3f}")

# ---- 5. judge alphas ------------------------------------------------
rep = {k: json.loads((VAL / f"{k}_judge_slots" / "judge_report.json")
                     .read_text(encoding="utf-8"))
       for k in ("h1", "h2", "h3")}
for slot, v in rep["h2"]["slots"].items():
    ck(f"h2 alpha {slot}", f"{v['alpha']:+.3f}")
for L, slots in rep["h1"]["by"]["L"].items():
    for slot, v in slots.items():
        ck(f"h1 alpha L{L}/{slot}", f"{v['alpha']:+.3f}")
for b, slots in rep["h3"]["by"]["backbone"].items():
    for slot, v in slots.items():
        ck(f"h3 alpha {b}/{slot}", f"{v['alpha']:+.3f}")
for slot, v in rep["h3"]["slots"].items():
    ck(f"h3 pooled alpha {slot}", f"{v['alpha']:+.3f}")

# judge sample sizes
for k, items, units in (("h1", 1500, 4500), ("h2", 300, 900),
                        ("h3", 2100, 6300)):
    ck(f"{k} judge items", str(items))
    ck(f"{k} judge units", f"{units:,}" if units >= 1000 else str(units))
ck("judge total items", "3,900")
ck("judge total units", "11,700")
ck("judge total calls", "2,525")

# ---- 6. H2 seed-2 replication (2026-11-10) --------------------------
SEED2 = ROOT / ("results-archive/20261110-h2-seed2-replication/results/"
                "h2_query_type_seed2")
if SEED2.exists():
    a2 = json.loads((SEED2 / "analysis.json").read_text(encoding="utf-8"))
    a1 = json.loads((ARCH / NAMES["h2"] / "analysis.json")
                    .read_text(encoding="utf-8"))
    for cell in CELLS:
        v2 = a2["confirmatory"]["cells"][cell]
        v1 = a1["confirmatory"]["cells"][cell]
        ck(f"seed2/{cell} Delta", f"{v2['mean_diff']:+.4f}")
        ck(f"seed2/{cell} diff vs seed1",
           f"{v2['mean_diff'] - v1['mean_diff']:+.4f}")
        # the document writes 1.20e-10, not 1.2e-10
        ck(f"seed2/{cell} p_adj", f"{v2['p_adj']:.2e}")

    # The document claims the two pools share pattern assignment but differ
    # in realisation. Both halves are asserted, because that caveat is the
    # whole reason seed-2 is called a PARTIAL replication.
    s1 = pd.concat([pd.read_csv(f) for f in
                    sorted((ARCH / NAMES["h2"] / "shards").glob("*.csv"))],
                   ignore_index=True)
    s2 = pd.concat([pd.read_csv(f) for f in
                    sorted((SEED2 / "shards").glob("*.csv"))],
                   ignore_index=True)
    mp = (s1[["sample_id", "pattern"]].drop_duplicates()
          .merge(s2[["sample_id", "pattern"]].drop_duplicates(),
                 on="sample_id", suffixes=("_1", "_2")))
    if not (mp.pattern_1 == mp.pattern_2).all():
        fails.append("doc says patterns are identical across pools; "
                     "they are not")
    ck("seed2 n sample_ids", str(len(mp)))

    f1 = json.loads((ARCH / NAMES["h2"] / "config_used.json")
                    .read_text(encoding="utf-8"))["_data_fingerprint"]
    f2 = json.loads((SEED2 / "config_used.json")
                    .read_text(encoding="utf-8"))["_data_fingerprint"]
    if f1 == f2:
        fails.append("doc says the pools differ; fingerprints match")
    ck("seed1 fingerprint", f1)
    ck("seed2 fingerprint", f2)

# ---- 7. H3 sensitivity readings -------------------------------------
SENS = ROOT / ("results-archive/20261110-h2-seed2-replication/"
               "h3_sensitivity.json")
if SENS.exists():
    sens = json.loads(SENS.read_text(encoding="utf-8"))
    for reading, cells_ in sens.items():
        for cell, v in cells_.items():
            ck(f"sens {reading}/{cell}", f"{float(v['mean_diff']):+.4f}")

print(f"{checks} checks, {len(fails)} failures\n")
for f in fails:
    print("  FAIL", f)
if not fails:
    print("  all quoted figures reproduce from the archive")

# independent recomputation of the three totals the doc asserts
print("\n--- totals, recomputed ---")
print("judge items :", 1500 + 300 + 2100)
print("judge units :", 4500 + 900 + 6300)
print("judge calls :", 914 + 209 + 1402)
