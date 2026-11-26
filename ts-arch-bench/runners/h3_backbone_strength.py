"""H3 - backbone strength. Does the gap shrink as the backbone gets stronger?

PREREGISTRATION.md §6, H3. One history length, nine query types, three arms,
repeated across a single-family ladder. One family (Qwen2.5 0.5B -> 72B) is
deliberate: mixing families would confound capability with tokenizer and
training recipe, and the x-axis is meant to be capability alone.

CONFIRMATORY TEST: item-level difference-in-differences across the ladder ends

    Delta_i = d_i(weakest) - d_i(strongest)      d = harness - bare_llm

Wilcoxon signed-rank on Delta within each of the three declared cells, m = 3,
Holm (the non-saturated set; PREREG D6). Positive Delta = the harness helps the weak backbone more, i.e. the
shrinking-gap claim. n = 100 per cell.

The seven-rung Spearman is reported as SHAPE ONLY, with
`min_detectable_rho(7)` printed beside it, because a 7-point rank test
cannot support an inferential claim at this n.

THE X-AXIS IS EXTERNAL AND MUST BE FIXED BEFORE THE RUN
-------------------------------------------------------
`BackboneSpec.strength` is a published capability score, never anything
measured by this benchmark - regressing in-benchmark accuracy on itself
would be circular. The metric, source and retrieval date live in the config
and are echoed into every snapshot. Changing them after seeing results is a
§5 protocol deviation.

    # one array task per rung; each loads its model once
    python -m runners.h3_backbone_strength --config configs/h3_backbone_strength.yaml \
        --stage inference --backbone qwen2_5_7b

    python -m runners.h3_backbone_strength --config ... --stage analyse
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


def ladder(cfg) -> list:
    """The confirmatory ladder, weakest first, excluding robustness arms.

    THE DECLARED ORDER IS AUTHORITATIVE, and deliberately so.

    `analysis.ladder` is written weakest-to-strongest in the config, and that
    written order IS the pre-registration: it is what fixes which two rungs
    the confirmatory difference-in-differences contrasts. Re-sorting by
    `strength` would let the capability SCORE silently choose the DiD ends -
    so while those scores are still placeholders (see the config's warning),
    a placeholder could decide the headline test. It must not.

    `strength` therefore enters exactly one place: the descriptive Spearman
    in `rank_shape`. A wrong score can invalidate that curve; it cannot move
    the confirmatory test.

    If the two disagree, that is worth knowing, so it is reported rather than
    resolved: a mismatch means either the declared order or the scores are
    wrong, and only the author can say which.
    """
    declared = cfg.analysis.get("ladder")
    robust = set(cfg.analysis.get("robustness_backbones") or [])
    names = [n for n in (declared or sorted(cfg.backbones))
             if n not in robust]
    unknown = [n for n in names if n not in cfg.backbones]
    if unknown:
        raise ValueError(
            f"analysis.ladder names backbones absent from the config: "
            f"{unknown}")
    if not declared:
        # nothing was declared, so strength is all we have to order by
        return sorted(names, key=lambda n: float(cfg.backbones[n].strength))

    by_strength = sorted(names, key=lambda n: float(cfg.backbones[n].strength))
    if by_strength != names:
        print("*** WARNING: analysis.ladder order disagrees with the "
              "`strength` values.\n"
              f"    declared    : {names}\n"
              f"    by strength : {by_strength}\n"
              "    The DECLARED order is used for the confirmatory DiD. "
              "Reconcile them before reporting: one of the two is wrong.")
    return names


def analyse(cfg, out: Path, L: int) -> dict:
    rungs = ladder(cfg)
    robust = [n for n in (cfg.analysis.get("robustness_backbones") or [])
              if n in cfg.backbones]
    df = C.load_shards(out, expect_backbones=rungs + robust,
                       expect_lengths=[L])
    df = df[df.L == int(L)]
    if len(rungs) < 2:
        raise RuntimeError(
            f"H3 needs at least two rungs to contrast; have {rungs}")
    weak, strong = rungs[0], rungs[-1]
    # PREREGISTRATION_C2.md, H3-C2: the endpoints of the confirmatory
    # contrast may be DECLARED in the config (`analysis.did_endpoints`)
    # instead of defaulting to the ladder's ends. C2 declares 3B and 72B,
    # because cycle 1 showed the 0.5B rung cannot operate the harness at all
    # (it routed `trend_description` correctly on 7% of items), so a contrast
    # anchored there measures routing collapse, not architecture. The full
    # ladder is still run and plotted. Absent the key, behaviour is cycle 1's.
    ends = cfg.analysis.get("did_endpoints")
    if ends:
        if len(ends) != 2 or any(e not in rungs for e in ends):
            raise ValueError(
                f"analysis.did_endpoints must name two ladder rungs; got "
                f"{ends}, ladder is {rungs}")
        weak, strong = ends
    samples = C.make_samples(C.narrow_config(cfg, weak), ROOT)

    # screen at both ends of the ladder, for the same reason as H1
    screens = {bb: C.screen_for(cfg, df[df.backbone == bb], samples, L)
               for bb in (weak, strong)}
    declared = list(cfg.analysis["confirmatory_family"])
    screened = C.decidable_at_both_ends(screens, declared)
    clustering = screens[strong].get("clustering_decision") or {}
    cluster_unit = clustering.get("cluster_unit", "sample_id")

    # SIGN CONVENTION - read this before interpreting `mean_d`.
    #
    # `confirmatory_did` computes Delta_i = d_i(first) - d_i(second), and here
    # first=weak, second=strong, so
    #
    #     Delta = gap(weakest rung) - gap(strongest rung)
    #     NEGATIVE => the harness advantage GROWS with backbone strength
    #     POSITIVE => the harness advantage SHRINKS with backbone strength
    #
    # ⚠ H1 PASSES THESE THE OTHER WAY ROUND (`first=hi, second=lo`), so a
    # positive number there means the advantage GROWS with history length.
    # The same function, the same column name, opposite meanings. Do not read
    # an H3 sign by analogy with H1. The orders are left as they are rather
    # than unified because changing either would flip the sign of results
    # already recorded in results/*/analysis.json and quoted in
    # STATUS_2026-11-07.md.
    #
    # Measured 2026-11-07: trend_description -0.3650 (advantage grows with
    # strength) and event_explanation +0.1225 (advantage shrinks) are both
    # significant, i.e. the two cells disagree in direction. "win" in the
    # output means "the gap moved detectably", NOT "the harness won" - the
    # test is two-sided on the DiD.
    # Cycle 2: see runners/h2_query_type.py. A cell must survive the
    # constant-answer rule at BOTH ends to stay in the family.
    c2 = {}
    family = declared
    pair = tuple(cfg.analysis["confirmatory_contrast"])
    if X.is_c2(cfg):
        rules = {bb: X.constant_rule(df[df.backbone == bb], screens[bb],
                                     declared, pair)
                 for bb in (weak, strong)}
        family = X.surviving_family(declared, *rules.values())
        screened = [q for q in screened if q in family]
        arms = sorted(df.system.unique())
        c2 = dict(
            constant_rule=rules, declared_family=declared, family=family,
            arm_changes=X.arm_changes(df, cfg, "backbone", weak, strong,
                                      declared, arms),
            absolute_means={bb: X.absolute_means(df[df.backbone == bb],
                                                 declared) for bb in rungs},
            forecast_baselines={bb: X.forecast_baseline_report(
                cfg, df[df.backbone == bb], samples, L, arms)
                for bb in (weak, strong)},
            extra_contrasts={bb: X.extra_contrasts(
                df[df.backbone == bb], cfg, declared)
                for bb in (weak, strong)},
            routing=X.routing_table(df).to_dict("records"))

    confirm = C.confirmatory_did(df, cfg, axis="backbone",
                                 first=weak, second=strong, cells=family)
    secondary = C.secondary_ablation(df, cfg, cluster_unit,
                                     backbone=strong, L=L)
    grid = C.sensitivity_grid(df, cfg, screened, C.confirmatory_did,
                              declared=family, axis="backbone", first=weak,
                              second=strong)

    conf_df = df[df.backbone.isin(rungs)]
    curve_rows = C.curve(conf_df, cfg, "backbone", rungs)
    xs = {n: float(cfg.backbones[n].strength) for n in rungs}
    shape = C.rank_shape(curve_rows, xs, "backbone")

    # Pre-registered SECONDARY axis (§6 H3). Declared in the config before
    # the run, and actually computed here - a secondary axis no code reads
    # would be a declaration in name only.
    alt_map = cfg.analysis.get("strength_alt") or {}
    shape_alt, xs_alt = None, None
    if all(n in alt_map for n in rungs):
        xs_alt = {n: float(alt_map[n]) for n in rungs}
        shape_alt = C.rank_shape(curve_rows, xs_alt, "backbone")

    C.print_header(cfg, f"H3 - backbone strength   L={L}  "
                        f"{len(rungs)} rungs")
    print("\n=== declared strength axis (EXTERNAL; fixed before the run) ===")
    print(f"metric: {cfg.analysis.get('strength_metric', 'UNDECLARED')}")
    print(f"source: {cfg.analysis.get('strength_source', 'UNDECLARED')}")
    print(f"retrieved: {cfg.analysis.get('strength_retrieved', 'UNDECLARED')}")
    for n in rungs:
        print(f"  {n:<22}{xs[n]:>6.3f}   {cfg.backbones[n].model}")
    if robust:
        print(f"cross-family robustness (NOT in the family): {robust}")
    if any(cfg.analysis.get(k) in (None, "", "UNDECLARED")
           for k in ("strength_metric", "strength_source",
                     "strength_retrieved")):
        print("\n*** WARNING: the strength axis is not fully declared. "
              "Fill strength_metric / strength_source / strength_retrieved "
              "in the config before reporting anything from this run. ***")

    print(f"\nDiD ends        : first={weak}, second={strong} "
          "(positive Delta = harness helps the weaker backbone more)")
    print(f"screened at both: {screened}")
    for bb in (weak, strong):
        und = screens[bb].get("undecidable_cells") or {}
        if und:
            print(f"  {bb} undecidable: {sorted(und)}")

    C.print_curve(curve_rows, "backbone")
    if c2:
        for bb in (weak, strong):
            X.print_constant_rule(c2["constant_rule"][bb], bb)
        print(f"family after C2-7 : {family}  (m={len(family)}; declared "
              f"{declared})")
    wins = C.print_confirmatory(confirm, "confirmatory family (H3, DiD)")
    verdicts = C.print_verdicts(cfg, wins, family, screened)
    if c2:
        X.print_arm_changes(c2["arm_changes"], weak, strong)
        for bb in (weak, strong):
            print(f"\n--- {bb} ---")
            X.print_forecast_baselines(c2["forecast_baselines"][bb])
            X.print_extra_contrasts(c2["extra_contrasts"][bb])
        X.print_routing(X.routing_table(df))
    C.print_rank_shape(shape, "declared strength")
    if shape_alt is not None:
        C.print_rank_shape(
            shape_alt, f"SECONDARY axis: "
                       f"{cfg.analysis.get('strength_alt_metric', 'alt')}")
    C.print_secondary(secondary)

    print("\n=== routing accuracy by rung (abstain=0 primary) ===")
    rt = df[df.route_source == "llm"]
    if len(rt):
        piv = rt.pivot_table(index="backbone", columns="query_type",
                             values="route_correct", aggfunc="mean")
        print(piv.reindex(rungs).round(3).to_string())

    print("\n" + C.cost_note())

    C.emit(cfg, out, df, screens[strong], confirm, secondary, grid,
           extra=dict(hypothesis="H3", L=L, ladder=rungs,
                      robustness_backbones=robust,
                      did_first=weak, did_second=strong,
                      strength_axis=xs,
                      strength_metric=cfg.analysis.get("strength_metric"),
                      strength_source=cfg.analysis.get("strength_source"),
                      strength_retrieved=cfg.analysis.get(
                          "strength_retrieved"),
                      strength_alt_metric=cfg.analysis.get(
                          "strength_alt_metric"),
                      strength_axis_alt=xs_alt,
                      rank_shape_alt=shape_alt,
                      screened_cells=screened, clustering=clustering,
                      screens_by_backbone={k: v for k, v in screens.items()},
                      curve=curve_rows, rank_shape=shape,
                      verdicts=verdicts, **({"c2": c2} if c2 else {})))
    C.write_csv(curve_rows, out / "curve_by_backbone.csv")
    print(f"\nwrote {out}/screening.json ({strong}), analysis.json "
          "(both ends), curve_by_backbone.csv, sensitivity_grid.csv")
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
        targets = ([args.backbone] if args.backbone
                   else ladder(cfg) + list(
                       cfg.analysis.get("robustness_backbones") or []))
        for bb in targets:
            C.run_cell(cfg, bb, L, out, resume=not args.no_resume,
                       no_cache=args.no_cache, cache_dir=args.cache_dir)
    if args.stage in ("analyse", "both"):
        analyse(cfg, out, L)


if __name__ == "__main__":
    main()
