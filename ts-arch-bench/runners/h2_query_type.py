"""H2 - query type. Does the architectural gap depend on what is asked?

PREREGISTRATION.md §6, H2. One backbone, one history length, nine query
types, three arms. The confirmatory test is §3 applied directly: Wilcoxon
signed-rank on the per-item gap within each of the three declared cells,
m = 3 passed explicitly to Holm (the non-saturated set; PREREG D6).

    # inference (one GPU job)
    python -m runners.h2_query_type --config configs/h2_query_type.yaml \
        --stage inference --backbone qwen2_5_7b

    # analysis (no GPU, re-run as often as you like)
    python -m runners.h2_query_type --config configs/h2_query_type.yaml \
        --stage analyse

The four non-family cells are still run and still reported - with their
`saturated` flag and trivial baseline beside them - because a reader needs to
see why a large gap on `statistics` is not evidence about architecture. They
are simply not in the family.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import _common as C                        # noqa: E402
from runners import _c2 as X                            # noqa: E402


def analyse(cfg, out: Path, L: int) -> dict:
    df = C.load_shards(out, expect_backbones=sorted(cfg.backbones),
                       expect_lengths=[L])
    backbones = sorted(df.backbone.dropna().unique())
    if len(backbones) != 1:
        raise RuntimeError(
            f"H2 is a single-backbone design but the shards carry "
            f"{backbones}. Use H3 for a ladder, or point --out at a "
            "directory holding one backbone's shards.")
    bb = backbones[0]
    df = df[df.L == int(L)]

    samples = C.make_samples(C.narrow_config(cfg, bb), ROOT)
    # the screen is written BEFORE any test is read (§2)
    screen = C.screen_for(cfg, df, samples, L)
    declared = list(cfg.analysis["confirmatory_family"])
    screened = C.decidable_cells(screen, declared)
    clustering = screen.get("clustering_decision") or {}
    cluster_unit = clustering.get("cluster_unit", "sample_id")

    # Cycle 2 (PREREGISTRATION_C2.md): the family is the declared cells that
    # SURVIVE the constant-answer rule (C2-7), and m is its size. In cycle 1
    # `family == declared` and nothing below changes.
    c2 = {}
    family = declared
    pair = tuple(cfg.analysis["confirmatory_contrast"])
    if X.is_c2(cfg):
        rule = X.constant_rule(df, screen, declared, pair)
        family = X.surviving_family(declared, rule)
        screened = [q for q in screened if q in family]
        arms = sorted(df.system.unique())
        c2 = dict(
            constant_rule=rule, declared_family=declared, family=family,
            absolute_means=X.absolute_means(df, declared),
            forecast_baselines=X.forecast_baseline_report(
                cfg, df, samples, L, arms),
            extra_contrasts=X.extra_contrasts(df, cfg, declared),
            routing=X.routing_table(df).to_dict("records"))

    confirm = C.confirmatory_single(df, cfg, backbone=bb, L=L, cells=family)
    secondary = C.secondary_ablation(df, cfg, cluster_unit,
                                     backbone=bb, L=L)
    grid = C.sensitivity_grid(df, cfg, screened, C.confirmatory_single,
                              declared=family, backbone=bb, L=L)

    C.print_header(cfg, f"H2 - query type   backbone={bb}  L={L}")
    print(f"\nclustering rule: f={clustering.get('mean_fraction_of_framings_with_routing_variation')}"
          f" vs threshold {clustering.get('threshold')}"
          f" -> unit={cluster_unit},"
          f" effective routing n/type={clustering.get('effective_routing_n_per_type')}")
    print(f"saturated cells : {screen.get('saturated_cells')}")
    print(f"decidable cells : {screened}")
    for qt, why in (screen.get("undecidable_cells") or {}).items():
        print(f"  UNDECIDABLE {qt}: {why}")

    if c2:
        X.print_constant_rule(c2["constant_rule"])
        print(f"family after C2-7 : {family}  (m={len(family)}; declared "
              f"{declared})")
    wins = C.print_confirmatory(confirm, "confirmatory family (H2)")
    verdicts = C.print_verdicts(cfg, wins, family, screened)
    C.print_secondary(secondary)
    if c2:
        X.print_extra_contrasts(c2["extra_contrasts"])
        X.print_forecast_baselines(c2["forecast_baselines"])
        X.print_routing(X.routing_table(df))

    print("\n=== trivial baselines (what a constant answer scores) ===")
    for qt, base in sorted((screen.get("trivial_baselines") or {}).items()):
        b = base.get("best_constant_score")
        print(f"  {qt:<20}{('%.3f' % b) if b is not None else 'n/a':>7}"
              f"   ({base.get('kind')})")

    print("\n" + C.cost_note())

    C.emit(cfg, out, df, screen, confirm, secondary, grid,
           extra=dict(hypothesis="H2", backbone=bb, L=L,
                      screened_cells=screened, clustering=clustering,
                      verdicts=verdicts, **({"c2": c2} if c2 else {})))
    print(f"\nwrote {out}/screening.json, analysis.json, "
          "sensitivity_grid.csv, desc_*.csv, raw_results.csv")
    return verdicts


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
