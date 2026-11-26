"""H2 REFRAMED - series character. Does the gap depend on the SERIES?

PREREGISTRATION_C3.md C3-6. One backbone, one history length, one cell, two
STRATA of series.

WHY THIS RUNNER EXISTS INSTEAD OF REUSING h2_query_type.py
----------------------------------------------------------
Cycle 2's H2 crossed nine query types and asked whether the architectural
gap depends on what is asked. On real data there is exactly one legitimate
label - the held-out future - so eight of those nine arms do not exist and
the question has no contrast left.

The question that replaces it is the one cycle 2 could only answer post
hoc: at 32B the harness gained +0.170 on seasonal series against +0.014 on
non-seasonal ones. That was the largest conditional effect in the cycle and
it was never a declared test. Here it is.

    within stratum s:   d_i = score(harness)_i - score(bare_llm)_i
    contrast:           mean d over WEAK  -  mean d over STRONG

⚠ THE CONTRAST IS TWO-SAMPLE, NOT PAIRED, AND THAT IS NOT A DETAIL.
H1 and H3 sweep axes along which the same items appear at both ends
(invariants I1/I2), so their Delta_i exists and `confirmatory_did` is
correct. A weak-seasonality series and a strong-seasonality one are
different series: there is no item to difference. `confirmatory_did`
refuses `axis="stratum"` for exactly this reason, and this runner uses
`src.stats.two_sample_compare`. The WITHIN-stratum gaps are still paired -
each is a harness-versus-bare contrast on the same item - so those use the
ordinary paired test.

⚠ STRATUM IS PARTIALLY CONFOUNDED WITH ARCHIVE, BY NECESSITY.
Weak-seasonality series long enough for the L=512 geometry come
overwhelmingly from `weather`. So the pooled contrast could be reporting a
difference between archives rather than between series characters. This
runner therefore computes the WITHIN-DATASET contrast for every dataset
that supplies at least `min_items_per_stratum_for_within_dataset` items in
both strata, and prints it beside the pooled number. Where the two
disagree, the within-dataset version is the one that carries the claim -
that rule is fixed here, before the run, not chosen once the numbers are
in.

    python -m runners.h2_series_character --config configs/c3_h2_strata.yaml \
        --stage inference --backbone qwen2_5_32b

    python -m runners.h2_series_character --config configs/c3_h2_strata.yaml \
        --stage analyse
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import _common as C                        # noqa: E402
from runners import _c2 as X                            # noqa: E402
from src.stats import two_sample_compare                # noqa: E402

CELL = "forecasting"


def _gaps_by_stratum(df: pd.DataFrame, pair, strata) -> dict:
    """stratum -> (sample_ids, per-item gap vector). Strict: a missing arm
    is an error, never a dropped row (the `.dropna()` rule, §1.2)."""
    out = {}
    for s in strata:
        sub = df[df.stratum == s]
        if sub.empty:
            out[s] = (np.array([]), np.array([]))
            continue
        ids, _, d = C.item_gap(sub, pair, CELL)
        out[s] = (ids, d)
    return out


def stratum_report(df: pd.DataFrame, cfg, strata, label: str) -> dict:
    """Within-stratum gaps (paired) plus the between-stratum contrast
    (two-sample)."""
    pair = tuple(cfg.analysis["confirmatory_contrast"])
    n_boot = int(cfg.analysis.get("n_boot", 5000))
    gaps = _gaps_by_stratum(df, pair, strata)
    within = {}
    for s, (_, d) in gaps.items():
        if not len(d):
            continue
        res = C.paired_compare(d, np.zeros_like(d), n_boot=n_boot,
                               seed=int(cfg.seed))
        sub = df[(df.stratum == s) & (df.query_type == CELL)]
        within[s] = dict(
            asdict(res),
            arm_means={a: float(sub[sub.system == a].score.mean())
                       for a in sorted(sub.system.unique())})
    first, second = strata
    between = two_sample_compare(gaps[first][1], gaps[second][1],
                                 n_boot=n_boot, seed=int(cfg.seed))
    return dict(
        scope=label, strata=list(strata), within_stratum=within,
        between=dict(asdict(between)),
        interpretation=(
            f"between.mean_diff = mean gap on {first} - mean gap on "
            f"{second}; positive means the harness's advantage is LARGER on "
            f"{first}-seasonality series. Two-sample (different series at "
            "the two ends), so Cliff's delta is the effect size - it is NOT "
            "comparable with the paired_sign_effect reported inside each "
            "stratum."))


def analyse(cfg, out: Path, L: int) -> dict:
    df = C.load_shards(out, expect_backbones=sorted(cfg.backbones),
                       expect_lengths=[L])
    backbones = sorted(df.backbone.dropna().unique())
    if len(backbones) != 1:
        raise RuntimeError(
            f"this is a single-backbone design but the shards carry "
            f"{backbones}.")
    bb = backbones[0]
    df = df[df.L == int(L)]
    if "stratum" not in df.columns or df.stratum.isna().all():
        raise RuntimeError(
            "the shards carry no `stratum` column. This runner needs a REAL "
            "pool (data.name: real); a synthetic pool has no series "
            "character to stratify on.")

    strata = tuple(cfg.analysis.get("stratum_contrast") or ("weak", "strong"))
    have = set(df.stratum.dropna().unique())
    missing = [s for s in strata if s not in have]
    if missing:
        raise RuntimeError(
            f"analysis.stratum_contrast names strata absent from the pool: "
            f"{missing}; the pool holds {sorted(have)}. Rebuild with "
            "data-real/build_pool_c3.py, which balances them by "
            "construction.")

    samples = C.make_samples(C.narrow_config(cfg, bb), ROOT)
    screen = C.screen_for(cfg, df, samples, L)
    pool = C.pool_note(cfg, samples)

    pooled = stratum_report(df, cfg, strata, "pooled over datasets")

    # The confound control. A dataset qualifies by a rule fixed in the
    # config, not by how its numbers come out.
    floor = int(cfg.analysis.get(
        "min_items_per_stratum_for_within_dataset", 20))
    counts = (df.drop_duplicates("sample_id")
              .groupby(["dataset_id", "stratum"]).size()
              .unstack(fill_value=0))
    qualifying = [ds for ds in counts.index
                  if all(counts.loc[ds].get(s, 0) >= floor for s in strata)]
    within_dataset = {
        ds: stratum_report(df[df.dataset_id == ds], cfg, strata,
                           f"within {ds}")
        for ds in qualifying}

    # The headline gap and the baselines, reported per stratum so a reader
    # can see the floor each arm is being judged against. The declared-period
    # seasonal-naive baseline is the bar that matters on real data.
    arms = sorted(df.system.unique())
    baselines = {
        s: X.forecast_baseline_report(
            cfg, df[df.stratum == s],
            [x for x in samples if x.metadata.get("stratum") == s], L, arms)
        for s in strata}

    overall = C.confirmatory_single(df, cfg, backbone=bb, L=L, cells=[CELL])
    secondary = C.secondary_ablation(df, cfg, "sample_id", backbone=bb, L=L)
    clust = C.clustering_sensitivity(df, cfg, C.confirmatory_single, [CELL],
                                     backbone=bb, L=L)

    # ---------------------------------------------------------------- print
    C.print_header(cfg, f"H2 REFRAMED - series character   backbone={bb}  "
                        f"L={L}  strata={list(strata)}")
    C.print_pool_note(pool)
    print("\n=== items per (dataset, stratum) - the confound, measured ===")
    print(counts.to_string())
    print(f"\nqualifying for a within-dataset contrast (>= {floor} items in "
          f"BOTH strata): {qualifying or 'NONE'}")
    if not qualifying:
        print("  ⚠ no dataset supplies both strata at this floor, so the "
              "pooled contrast CANNOT be separated from an archive effect. "
              "Report it as confounded; do not call it a series-character "
              "finding.")

    C.print_confirmatory(overall, "overall gap at this L (not the stratum "
                                  "test; the floor it sits on)")
    _print_stratum(pooled)
    for ds, rep in within_dataset.items():
        _print_stratum(rep)
    print("\n=== declared-period seasonal naive, per stratum (the bar) ===")
    for s in strata:
        rep = baselines.get(s) or {}
        if not rep:
            continue
        print(f"  --- {s} ---")
        X.print_forecast_baselines(rep)
    C.print_secondary(secondary)
    C.print_clustering_sensitivity(clust)

    verdict = _verdict(pooled, within_dataset, qualifying,
                       float(cfg.analysis.get("min_effect", 0.02)))
    print("\n=== verdict ===")
    for line in verdict["reading"]:
        print("  " + line)

    C.emit(cfg, out, df, screen, overall, secondary, [],
           extra=dict(hypothesis="H2-reframed (series character)",
                      backbone=bb, L=L, strata=list(strata),
                      stratum_contrast=pooled,
                      within_dataset=within_dataset,
                      qualifying_datasets=qualifying,
                      items_per_dataset_stratum=counts.to_dict(),
                      per_stratum_baselines=baselines,
                      clustering_sensitivity=clust, real_pool=pool,
                      verdicts=verdict))
    print(f"\nwrote {out}/analysis.json, screening.json, desc_*.csv")
    return verdict


def _print_stratum(rep: dict) -> None:
    print(f"\n=== gap by stratum: {rep['scope']} ===")
    hdr = (f"{'stratum':<10}{'n':>5}{'bare':>8}{'harness':>9}"
           f"{'mean_gap':>10}{'95% CI':>19}{'p_raw':>10}")
    print(hdr)
    print("-" * len(hdr))
    for s, v in rep["within_stratum"].items():
        am = v["arm_means"]
        ci = f"[{v['ci_low']:+.3f},{v['ci_high']:+.3f}]"
        p = "nan" if v["p_raw"] != v["p_raw"] else f"{v['p_raw']:.4g}"
        print(f"{s:<10}{v['n']:>5}{am.get('bare_llm', float('nan')):>8.3f}"
              f"{am.get('harness', float('nan')):>9.3f}"
              f"{v['mean_diff']:>+10.4f}{ci:>19}{p:>10}")
    b = rep["between"]
    ci = f"[{b['ci_low']:+.3f},{b['ci_high']:+.3f}]"
    p = "nan" if b["p_raw"] != b["p_raw"] else f"{b['p_raw']:.4g}"
    print(f"\n  between strata (TWO-SAMPLE): {b['mean_diff']:+.4f} {ci}  "
          f"p={p}  Cliff's delta={b['cliffs_delta']:+.3f}  "
          f"Hedges g={b['hedges_g']:+.3f}   n={b['n_a']} vs {b['n_b']}")


def _verdict(pooled: dict, within_dataset: dict, qualifying: list,
             min_effect: float) -> dict:
    """Both readings, side by side, never substituted (the §2 rule)."""
    b = pooled["between"]
    moved = bool(np.isfinite(b["ci_low"]) and np.isfinite(b["ci_high"])
                 and (b["ci_low"] > 0 or b["ci_high"] < 0)
                 and abs(b["mean_diff"]) >= min_effect)
    per_ds = {ds: bool(
        np.isfinite(r["between"]["ci_low"])
        and np.isfinite(r["between"]["ci_high"])
        and (r["between"]["ci_low"] > 0 or r["between"]["ci_high"] < 0)
        and abs(r["between"]["mean_diff"]) >= min_effect)
        for ds, r in within_dataset.items()}
    reading = [
        f"pooled contrast        : {'MOVED' if moved else 'did not move'} "
        f"({b['mean_diff']:+.4f}, CI [{b['ci_low']:+.3f},"
        f"{b['ci_high']:+.3f}])",
    ]
    if not qualifying:
        reading.append(
            "within-dataset control : UNAVAILABLE - no dataset supplies both "
            "strata, so the pooled contrast cannot be separated from an "
            "archive effect. This is a confounded result and must be "
            "reported as one.")
    else:
        for ds, ok in per_ds.items():
            r = within_dataset[ds]["between"]
            reading.append(
                f"within {ds:<16}: {'MOVED' if ok else 'did not move'} "
                f"({r['mean_diff']:+.4f}, CI [{r['ci_low']:+.3f},"
                f"{r['ci_high']:+.3f}])")
        if moved and not any(per_ds.values()):
            reading.append(
                "⚠ the pooled contrast moved and NO within-dataset contrast "
                "did. The pre-registered rule says the within-dataset "
                "version carries the claim: report this as an ARCHIVE "
                "effect, not a series-character effect.")
    return dict(pooled_moved=moved, within_dataset_moved=per_ds,
                qualifying_datasets=qualifying, min_effect=min_effect,
                reading=reading)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(ap)
    args = ap.parse_args()
    cfg, out = C.resolve(args)

    L = int(args.only_L if args.only_L else cfg.history_lengths[0])
    if args.stage in ("inference", "both"):
        banner = C.mock_banner(cfg)
        if banner:
            print(banner)
        targets = [args.backbone] if args.backbone else sorted(cfg.backbones)
        for bb in targets:
            C.run_cell(cfg, bb, L, out, resume=not args.no_resume,
                       no_cache=args.no_cache, cache_dir=args.cache_dir)
    if args.stage in ("analyse", "both"):
        analyse(cfg, out, L)


if __name__ == "__main__":
    main()
