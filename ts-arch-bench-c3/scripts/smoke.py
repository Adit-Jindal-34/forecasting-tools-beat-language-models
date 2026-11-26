"""Pipeline smoke run: three arms, one history length, mock or real backbone.

    python scripts/smoke.py --config configs/smoke_mock.yaml

This is NOT a hypothesis runner and produces no verdict. It exists to prove
the pipeline executes, to print the pre-test screening report, and to show
the failure taxonomy and trivial baselines beside every cell mean - the
three things the old build never surfaced.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from evaluation.metrics import selective_report  # noqa: E402
from src.config import load_config, save_config_snapshot  # noqa: E402
from src.pipeline import (Registry, make_samples, run_inference,  # noqa: E402
                          evaluate_run, fixed_denominators, save_csv,
                          save_json)
from src.screening import (screen_report, routing_variance_within_framing,
                           clustering_decision)  # noqa: E402

_MOCK_BANNER = """
================================================================================
!!  MOCK BACKBONE - THESE NUMBERS VALIDATE PLUMBING ONLY  !!

    Backbone(s) using provider="mock": {names}

    MockLLM imports src.ground_truth and synthesises answers at a target
    accuracy. It does not read the series the way a model does; it is a
    behavioural placeholder with the answer key in hand.

    NO TABLE FROM THIS RUN MAY BE QUOTED, in a paper, a slide or a message.
    Every number below is a pipeline check. Use a real backbone for results.
================================================================================
"""


def _mock_banner(cfg) -> None:
    names = [n for n, b in cfg.backbones.items() if b.provider == "mock"]
    if names:
        print(_MOCK_BANNER.format(names=", ".join(sorted(names))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/smoke_mock.yaml")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute()
                      else args.config)
    out = (Path(cfg.out_dir) if Path(cfg.out_dir).is_absolute()
           else ROOT / cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, out / "config_used.json")

    _mock_banner(cfg)

    L = cfg.history_lengths[0]
    samples = make_samples(cfg, ROOT)
    reg = Registry(cfg, cache_dir=out / "cache", enabled=not args.no_cache)
    print(f"pool: {len(samples)} samples | systems: {list(reg.systems)} | L={L}")
    fam = cfg.family_report()
    print(f"declared Holm family: {fam['n_tests']} tests, "
          f"{fam['confirmatory_contrast']} over {fam['confirmatory_family']}")
    if fam["saturated_cells_kept_in_family"]:
        print("  (!) tool-saturated cells kept in family by override: "
              f"{fam['saturated_cells_kept_in_family']}")

    results = run_inference(cfg, samples, L, reg.systems, reg)
    rows = evaluate_run(cfg, samples, L, results, judge=reg.judge,
                        denominators=fixed_denominators(cfg, samples))
    save_csv(rows, out / "raw_results.csv")
    df = pd.DataFrame(rows)

    print("\n=== mean score by cell (scored_mean: failures count as 0) ===")
    piv = df.pivot_table(index="query_type", columns="system", values="score",
                         aggfunc="mean")
    print(piv.round(3).to_string())

    print("\n=== coverage (fraction answered) ===")
    cov = df.pivot_table(index="query_type", columns="system",
                         values="answered", aggfunc="mean")
    print(cov.round(3).to_string())

    print("\n=== failure taxonomy ===")
    tax = (df[df.failure != ""].groupby(["system", "failure"]).size()
           .unstack(fill_value=0))
    print(tax.to_string() if len(tax) else "  (no failures)")

    print("\n=== routing (self-routing arm) ===")
    h = df[df.route_source == "llm"]
    if len(h):
        print(h.groupby(["system", "route_status"]).size()
              .unstack(fill_value=0).to_string())
        # BOTH columns. `route_correct` counts an abstention as a routing
        # failure (primary); `route_correct_selective` is NaN on abstention
        # and so is conditional on having routed at all. Reporting only the
        # second is the same bug class as dropping unanswered items.
        acc = h.groupby("system").agg(
            routing_accuracy=("route_correct", "mean"),
            routing_accuracy_selective=("route_correct_selective", "mean"),
            abstain_rate=("abstained", "mean")).round(3)
        print("\n" + acc.to_string())

        print("\n=== within-framing routing variance (clustering rule) ===")
        var = routing_variance_within_framing(rows)
        for key in sorted(var):
            v = var[key]
            print(f"  {key:36} {v['n_framings_with_variation']:>3}/"
                  f"{v['n_framings']:<3} framings vary  "
                  f"(distinct ops {v['distinct_route_ops']})")
        dec = clustering_decision(
            var, float(cfg.analysis["framing_variance_threshold"]))
        print(f"  -> cluster on {dec['cluster_unit']}; effective routing "
              f"n/type = {dec['effective_routing_n_per_type']} "
              f"(varying fraction "
              f"{dec['mean_fraction_of_framings_with_routing_variation']:.3f}"
              f" vs threshold {dec['threshold']})")

    print("\n=== call accounting ===")
    print(df.groupby("system").agg(
        architectural=("n_model_calls", "sum"),
        billed=("n_billed_calls", "sum")).astype(int).to_string())
    print("  architectural calls are NOT comparable across arms: the oracle "
          "arm issues 0 classify calls by design.")

    screen = screen_report(
        rows, samples, L,
        contrast=tuple(cfg.analysis["confirmatory_contrast"]),
        framing_threshold=float(cfg.analysis["framing_variance_threshold"]))
    save_json(screen, out / "screening.json")
    print("\n=== pre-test screen ===")
    print(f"decidable cells: {screen['n_decidable']}/{screen['n_cells']}")
    for qt, why in screen["undecidable_cells"].items():
        print(f"  UNDECIDABLE {qt}: {why}")
    print(f"saturated cells (one arm pinned, separate axis): "
          f"{screen['saturated_cells']}")
    cv = screen.get("confirmatory_contrast_screen") or {}
    if cv:
        print(f"\nconfirmatory contrast {screen['confirmatory_contrast']}:")
        for qt in sorted(cv):
            d = cv[qt]
            gap = d["gap"]
            print(f"  {qt:20} gap={gap if gap is None else round(gap, 3):>7}"
                  f"  decidable={str(d['decidable']):>5}"
                  f"  saturated={str(d['saturated']):>5}")

    print("\n=== trivial baselines (what a constant answer scores) ===")
    for qt, base in sorted(screen["trivial_baselines"].items()):
        b = base.get("best_constant_score")
        print(f"  {qt:20} {('%.3f' % b) if b is not None else 'n/a':>6}"
              f"   ({base['kind']})")

    sel = []
    for (sysname, qt), sub in df.groupby(["system", "query_type"]):
        rep = selective_report(sub["score"].tolist(), sub["answered"].tolist())
        sel.append(dict(system=sysname, query_type=qt, **rep))
    save_csv(sel, out / "selective_summary.csv")

    print(f"\nwrote {out}/raw_results.csv, selective_summary.csv, "
          f"screening.json, config_used.json")
    print("cache:", reg.cache_stats())


if __name__ == "__main__":
    main()
