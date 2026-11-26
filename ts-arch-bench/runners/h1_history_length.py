"""H1 - history length. Does the gap depend on how much history is visible?

PREREGISTRATION.md §6, H1. One backbone, five history lengths, nine query
types, three arms. The pool is FIXED across L (invariants I1/I2), so the same
item is tracked along the sweep rather than resampled.

CONFIRMATORY TEST: item-level difference-in-differences

    Delta_i = d_i(L=max) - d_i(L=min)        d = harness - bare_llm

Wilcoxon signed-rank on Delta within each of the three declared cells,
m = 3, Holm (the non-saturated set; see PREREG deviation D6). This is
n = 100 per cell. It is deliberately NOT a rank test
over the five L values: `min_detectable_rho(5)` is ~1, so a 5-point Spearman
could only ever return "not detected". The five-point curve is still
reported - it is where a crossover would be visible - but as shape, not as
the test.

    # inference: one GPU job does every L for one backbone (model loads once)
    python -m runners.h1_history_length --config configs/h1_history_length.yaml \
        --stage inference --backbone qwen2_5_7b

    # or one array task per L, if the queue prefers short jobs
    python -m runners.h1_history_length --config ... --stage inference \
        --backbone qwen2_5_7b --only-L 512

    python -m runners.h1_history_length --config ... --stage analyse
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


def analyse(cfg, out: Path) -> dict:
    lengths = sorted(int(x) for x in cfg.history_lengths)
    df = C.load_shards(out, expect_backbones=sorted(cfg.backbones),
                       expect_lengths=lengths)
    backbones = sorted(df.backbone.dropna().unique())
    if len(backbones) != 1:
        raise RuntimeError(
            f"H1 is a single-backbone design but the shards carry "
            f"{backbones}. Separate the output directories per backbone.")
    bb = backbones[0]
    lo, hi = lengths[0], lengths[-1]
    samples = C.make_samples(C.narrow_config(cfg, bb), ROOT)

    # Screen at BOTH ends: the confirmatory test is a DiD between them, so a
    # cell with no headroom at either end cannot show the gap moving.
    screens = {L: C.screen_for(cfg, df[df.L == L], samples, L)
               for L in (lo, hi)}
    declared = list(cfg.analysis["confirmatory_family"])
    screened = C.decidable_at_both_ends(screens, declared)

    ref_L = int(cfg.analysis.get("reference_L", lengths[len(lengths) // 2]))
    ref_screen = screens.get(ref_L) or C.screen_for(
        cfg, df[df.L == ref_L], samples, ref_L)
    clustering = ref_screen.get("clustering_decision") or {}
    cluster_unit = clustering.get("cluster_unit", "sample_id")

    # SIGN CONVENTION - read this before interpreting `mean_d`.
    #
    # `confirmatory_did` computes Delta_i = d_i(first) - d_i(second), and here
    # first=hi, second=lo, so
    #
    #     Delta = gap(L_max) - gap(L_min)
    #     POSITIVE => the harness advantage GROWS as history lengthens
    #     NEGATIVE => the harness advantage SHRINKS as history lengthens
    #
    # ⚠ H3 PASSES THESE THE OTHER WAY ROUND (`first=weak, second=strong`), so
    # a positive number there means the advantage SHRINKS with strength. The
    # same function, the same column name, opposite meanings. Do not read an
    # H1 sign by analogy with H3. The orders are left as they are rather than
    # unified because changing either would flip the sign of results already
    # recorded in results/*/analysis.json and quoted in STATUS_2026-11-07.md.
    # Cycle 2: see runners/h2_query_type.py. For H1 the C2 configs declare
    # `forecasting` alone (the qualitative key depends on whether window(L)
    # contains the injected event, so its DiD compares different questions),
    # and each arm's own change is reported so the mechanism is explicit.
    c2 = {}
    family = declared
    pair = tuple(cfg.analysis["confirmatory_contrast"])
    if X.is_c2(cfg):
        rules = {L: X.constant_rule(df[df.L == L], screens[L], declared, pair)
                 for L in (lo, hi)}
        family = X.surviving_family(declared, *rules.values())
        screened = [q for q in screened if q in family]
        arms = sorted(df.system.unique())
        all_cells = sorted(df.query_type.unique())
        c2 = dict(
            constant_rule={str(k): v for k, v in rules.items()},
            declared_family=declared, family=family,
            arm_changes=X.arm_changes(
                df, cfg, "L", hi, lo,
                [q for q in all_cells
                 if q in ("forecasting",) + X.QUALITATIVE_CELLS], arms),
            absolute_means={str(L): X.absolute_means(
                df[df.L == L], ("forecasting",) + X.QUALITATIVE_CELLS)
                for L in lengths},
            forecast_baselines={str(L): X.forecast_baseline_report(
                cfg, df[df.L == L], samples, L, arms) for L in lengths},
            extra_contrasts=X.extra_contrasts(
                df[df.L == ref_L], cfg,
                ("forecasting",) + X.QUALITATIVE_CELLS),
            routing=X.routing_table(df).to_dict("records"))

    confirm = C.confirmatory_did(df, cfg, axis="L", first=hi, second=lo,
                                 cells=family)
    secondary = C.secondary_ablation(df, cfg, cluster_unit,
                                     backbone=bb, L=ref_L)
    grid = C.sensitivity_grid(df, cfg, screened, C.confirmatory_did,
                              declared=family, axis="L", first=hi, second=lo)

    curve_rows = C.curve(df, cfg, "L", lengths)
    shape = C.rank_shape(curve_rows, {L: float(L) for L in lengths}, "L")

    C.print_header(cfg, f"H1 - history length   backbone={bb}  "
                        f"L in {lengths}")
    print(f"\nDiD ends        : first=L{hi}, second=L{lo} "
          f"(positive Delta = gap wider at the longer history)")
    print(f"screened at both: {screened}")
    for L in (lo, hi):
        und = screens[L].get("undecidable_cells") or {}
        if und:
            print(f"  L={L} undecidable: {sorted(und)}")
    print(f"clustering (L={ref_L}): unit={cluster_unit}, "
          f"f={clustering.get('mean_fraction_of_framings_with_routing_variation')}")

    C.print_curve(curve_rows, "L")
    if c2:
        for L in (lo, hi):
            X.print_constant_rule(c2["constant_rule"][str(L)], f"L={L}")
        print(f"family after C2-7 : {family}  (m={len(family)}; declared "
              f"{declared})")
    wins = C.print_confirmatory(confirm, "confirmatory family (H1, DiD)")
    verdicts = C.print_verdicts(cfg, wins, family, screened)
    if c2:
        X.print_arm_changes(c2["arm_changes"], f"L={hi}", f"L={lo}")
        for L in (lo, hi):
            print(f"\n--- L={L} ---")
            X.print_forecast_baselines(c2["forecast_baselines"][str(L)])
        X.print_extra_contrasts(c2["extra_contrasts"])
    C.print_rank_shape(shape, "L")
    C.print_secondary(secondary)

    print("\n=== coverage and failures by L (the bare arm fails "
          "DIFFERENTLY as prompts grow) ===")
    cov = df.pivot_table(index="L", columns="system", values="answered",
                         aggfunc="mean")
    print(cov.round(3).to_string())
    fails = df[df.failure != ""].groupby(["L", "system", "failure"]).size()
    print("\n" + (fails.to_string() if len(fails) else "  (no failures)"))

    print("\n" + C.cost_note())

    C.emit(cfg, out, df, screens[hi], confirm, secondary, grid,
           extra=dict(hypothesis="H1", backbone=bb, lengths=lengths,
                      did_first=hi, did_second=lo, reference_L=ref_L,
                      screened_cells=screened, clustering=clustering,
                      screens_by_L={str(k): v for k, v in screens.items()},
                      curve=curve_rows, rank_shape=shape,
                      verdicts=verdicts, **({"c2": c2} if c2 else {})))
    C.write_csv(curve_rows, out / "curve_by_L.csv")
    print(f"\nwrote {out}/screening.json (L={hi}), analysis.json "
          "(both ends), curve_by_L.csv, sensitivity_grid.csv, desc_*.csv")
    return verdicts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(ap)
    args = ap.parse_args()
    cfg, out = C.resolve(args)

    if args.stage in ("inference", "both"):
        banner = C.mock_banner(cfg)
        if banner:
            print(banner)
        targets = [args.backbone] if args.backbone else sorted(cfg.backbones)
        lengths = ([int(args.only_L)] if args.only_L
                   else sorted(int(x) for x in cfg.history_lengths))
        for bb in targets:
            for L in lengths:
                C.run_cell(cfg, bb, L, out, resume=not args.no_resume,
                           no_cache=args.no_cache,
                           cache_dir=args.cache_dir)
    if args.stage in ("analyse", "both"):
        analyse(cfg, out)


if __name__ == "__main__":
    main()
