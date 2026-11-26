"""Cycle-2 analysis additions (PREREGISTRATION_C2.md).

Everything here is ANALYSIS-STAGE: it reads shards and the sample pool and
never calls a model, so it can be re-run freely. The cycle-1 statistics in
`runners/_common.py` are reused unchanged; this module adds what C2 declared
on top of them:

  C2-5   `bare_features` and any other declared extra contrast, reported
         with effects and CIs, OUTSIDE the correction
  C2-6   classical forecasting baselines on the same items and the same MASE
         denominator, as first-class rows
  C2-7   the constant-answer rule: a qualitative cell where the best
         constant answer scores at least as high as BOTH arms is
         UNDECIDABLE and leaves the confirmatory family
  C2-8   strict and functional routing side by side
  H1/H3  each arm's own change along the axis, so "the gap moved" is never
         reported without saying which arm moved
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from src import ground_truth as gt
from src.pipeline import fixed_denominators
from src.stats import paired_compare

QUALITATIVE_CELLS = ("trend_description", "event_explanation")


def is_c2(cfg) -> bool:
    return cfg.design == "c2"


# --------------------------------------------------------------------------
# C2-7: the constant-answer rule
# --------------------------------------------------------------------------
def constant_rule(df: pd.DataFrame, screen: dict, cells, pair) -> dict:
    """Per qualitative cell: does a constant answer match or beat both arms?

    `df` must already be restricted to ONE (backbone, L). Forecasting and
    the numeric cells are not subject to the rule (their floors are reported
    by the classical baselines and the trivial-baseline table instead).
    """
    out = {}
    base = screen.get("trivial_baselines") or {}
    for qt in cells:
        if qt not in QUALITATIVE_CELLS:
            continue
        const = (base.get(qt) or {}).get("best_constant_score")
        sub = df[df.query_type == qt]
        means = {arm: float(sub[sub.system == arm].score.mean())
                 for arm in pair}
        dominated = bool(const is not None and
                         all(const >= m for m in means.values()))
        out[qt] = dict(
            best_constant_score=const,
            best_constant_answer=(base.get(qt) or {}).get(
                "best_constant_answer"),
            arm_means=means, undecidable=dominated,
            reading=("UNDECIDABLE: a constant answer scores at least as "
                     "high as both arms, so the cell cannot carry a "
                     "confirmatory claim" if dominated else
                     "decidable: at least one arm beats the best constant "
                     "answer"))
    return out


def surviving_family(declared, *rules: dict) -> list:
    """Declared cells minus any that a constant-rule report marks undecidable.

    With several reports (the two ends of an H1/H3 contrast) a cell must
    survive ALL of them: a difference-in-differences needs a meaningful gap
    at both ends.
    """
    dead = {qt for r in rules for qt, v in r.items() if v["undecidable"]}
    return [q for q in declared if q not in dead]


def print_constant_rule(rule: dict, label: str = "") -> None:
    if not rule:
        return
    print(f"\n=== constant-answer rule (C2-7){' ' + label if label else ''} ===")
    for qt, v in rule.items():
        arms = "  ".join(f"{a}={m:.3f}" for a, m in v["arm_means"].items())
        c = v["best_constant_score"]
        print(f"  {qt:<20} constant={c:.3f}  {arms}  -> "
              f"{'UNDECIDABLE' if v['undecidable'] else 'decidable'}")


# --------------------------------------------------------------------------
# C2-6: classical forecasting baselines
# --------------------------------------------------------------------------
def _estimate_period(x: np.ndarray) -> int:
    """Best detrended-ACF lag if it is convincing, else 1 (no seasonality)."""
    lag, r = gt.best_lag_acf_detrended(x)
    return int(lag) if r > 0.5 else 1


def _snaive(x: np.ndarray, period: int, horizon: int) -> np.ndarray:
    if period < 2 or len(x) < period:
        return np.full(horizon, x[-1])
    return np.array([x[len(x) - period + (h % period)]
                     for h in range(horizon)])


BASELINE_METHODS = ("last_value", "window_mean",
                    "seasonal_naive_estimated_period",
                    "seasonal_naive_true_period")


def forecast_baseline_scores(cfg, samples, L: int) -> pd.DataFrame:
    """One row per (forecasting item, baseline), scored like any arm.

    Same target, same MASE denominator (`fixed_denominators`), same
    `1/(1+MASE)` score. `seasonal_naive_true_period` uses the generator's
    period, which no system can see; it is an upper reference for the
    seasonal-naive family, not a fair competitor.
    """
    dens = fixed_denominators(cfg, samples) or {}
    rows = []
    for s in samples:
        if s.query_type != "forecasting":
            continue
        x = np.asarray(s.window(L), float)
        y = np.asarray(s.target(), float)
        h = len(y)
        den = dens.get(s.sample_id)
        if den is None:
            den = gt.naive_window_error(x, s.seasonal_period)
        preds = dict(
            last_value=np.full(h, x[-1]),
            window_mean=np.full(h, x.mean()),
            seasonal_naive_estimated_period=_snaive(
                x, _estimate_period(x), h),
            seasonal_naive_true_period=_snaive(
                x, int(s.seasonal_period), h))
        for name, p in preds.items():
            mase = float(np.mean(np.abs(p - y)) / max(float(den), 1e-9))
            rows.append(dict(sample_id=s.sample_id, baseline=name,
                             mase=mase, score=1.0 / (1.0 + mase)))
    return pd.DataFrame(rows)


def forecast_baseline_report(cfg, df: pd.DataFrame, samples, L: int,
                             arms) -> dict:
    """Baseline means, and every arm against every baseline (paired).

    DESCRIPTIVE: effects and CIs with a raw p, no correction, no verdict.
    `df` must already be restricted to one (backbone, L).
    """
    base = forecast_baseline_scores(cfg, samples, L)
    if base.empty:
        return {}
    piv = base.pivot(index="sample_id", columns="baseline", values="score")
    sub = df[df.query_type == "forecasting"]
    arm_piv = sub.pivot_table(index="sample_id", columns="system",
                              values="score", aggfunc="first")
    out = dict(
        baseline_mean_score={b: float(piv[b].mean()) for b in piv.columns},
        baseline_median_mase={
            b: float(base[base.baseline == b].mase.median())
            for b in piv.columns},
        arm_mean_score={a: float(arm_piv[a].mean())
                        for a in arms if a in arm_piv.columns},
        versus={})
    for a in arms:
        if a not in arm_piv.columns:
            continue
        for b in piv.columns:
            j = pd.concat([arm_piv[a].rename("a"), piv[b].rename("b")],
                          axis=1).dropna()
            res = paired_compare(j["a"].to_numpy(), j["b"].to_numpy(),
                                 n_boot=int(cfg.analysis.get("n_boot", 5000)),
                                 seed=int(cfg.seed))
            out["versus"][f"{a}|{b}"] = dict(
                n=res.n, mean_diff=res.mean_diff, ci_low=res.ci_low,
                ci_high=res.ci_high, p_raw=res.p_raw)
    return out


def print_forecast_baselines(rep: dict) -> None:
    if not rep:
        return
    print("\n=== forecasting: classical baselines (C2-6; same items, same "
          "MASE denominator) ===")
    for b, m in rep["baseline_mean_score"].items():
        print(f"  {b:<34} score={m:.3f}  median MASE="
              f"{rep['baseline_median_mase'][b]:.3f}")
    for a, m in rep["arm_mean_score"].items():
        print(f"  {a:<34} score={m:.3f}")
    print("  arm - baseline (paired; descriptive, uncorrected):")
    for key, v in rep["versus"].items():
        a, b = key.split("|")
        p = "nan" if v["p_raw"] != v["p_raw"] else f"{v['p_raw']:.3g}"
        print(f"    {a:<16} vs {b:<34} {v['mean_diff']:+.4f} "
              f"[{v['ci_low']:+.3f},{v['ci_high']:+.3f}]  p={p}")


# --------------------------------------------------------------------------
# C2-5: declared extra contrasts (bare_features), outside the correction
# --------------------------------------------------------------------------
def extra_contrasts(df: pd.DataFrame, cfg, cells) -> dict:
    """`analysis.extra_contrasts`: [[lo, hi], ...], reported per cell.

    The two declared in the C2 configs decompose the confirmatory gap:
        bare_llm -> bare_features   does the COMPUTATION help?
        bare_features -> harness    does the SCAFFOLDING (routing, tool
                                    dispatch) add anything on top of it?
    `df` must already be restricted to one (backbone, L).
    """
    out = {}
    have = set(df.system.unique())
    for lo, hi in (cfg.analysis.get("extra_contrasts") or []):
        if lo not in have or hi not in have:
            continue
        per = {}
        for qt in cells:
            piv = df[df.query_type == qt].pivot_table(
                index="sample_id", columns="system", values="score",
                aggfunc="first")
            if lo not in piv.columns or hi not in piv.columns:
                continue
            both = piv[[lo, hi]].dropna()
            res = paired_compare(both[hi].to_numpy(), both[lo].to_numpy(),
                                 n_boot=int(cfg.analysis.get("n_boot", 5000)),
                                 seed=int(cfg.seed))
            per[qt] = dict(asdict(res), mean_lo=float(both[lo].mean()),
                           mean_hi=float(both[hi].mean()))
        out[f"{lo}|{hi}"] = per
    return out


def print_extra_contrasts(rep: dict) -> None:
    if not rep:
        return
    print("\n=== declared secondary contrasts (OUTSIDE the correction) ===")
    for key, per in rep.items():
        lo, hi = key.split("|")
        print(f"  {hi} - {lo}")
        for qt, v in per.items():
            p = "nan" if v["p_raw"] != v["p_raw"] else f"{v['p_raw']:.3g}"
            print(f"    {qt:<20} {v['mean_lo']:.3f} -> {v['mean_hi']:.3f}  "
                  f"d={v['mean_diff']:+.4f} "
                  f"[{v['ci_low']:+.3f},{v['ci_high']:+.3f}]  p={p}")


# --------------------------------------------------------------------------
# H1/H3: which ARM moved?
# --------------------------------------------------------------------------
def arm_changes(df: pd.DataFrame, cfg, axis: str, first, second, cells,
                arms) -> dict:
    """Each arm's own paired change `score(first) - score(second)` per cell.

    Cycle 1 reported H1's forecasting gap widening and attributed it to the
    bare arm degrading; the paired data showed the forecaster improving
    (+0.050, p < 0.001) and the bare arm not moving (-0.023, p = 0.87). A
    difference-in-differences cannot tell those apart, so C2 always prints
    this table beside it.
    """
    key = dict(L="L", backbone="backbone")[axis]
    out = {}
    for qt in cells:
        sub = df[df.query_type == qt]
        per = {}
        for arm in arms:
            a = sub[(sub.system == arm) & (sub[key] == first)]
            b = sub[(sub.system == arm) & (sub[key] == second)]
            j = pd.concat([a.set_index("sample_id").score.rename("a"),
                           b.set_index("sample_id").score.rename("b")],
                          axis=1).dropna()
            if j.empty:
                continue
            res = paired_compare(j["a"].to_numpy(), j["b"].to_numpy(),
                                 n_boot=int(cfg.analysis.get("n_boot", 5000)),
                                 seed=int(cfg.seed))
            per[arm] = dict(n=res.n, mean_first=float(j["a"].mean()),
                            mean_second=float(j["b"].mean()),
                            mean_diff=res.mean_diff, ci_low=res.ci_low,
                            ci_high=res.ci_high, p_raw=res.p_raw)
        out[qt] = per
    return out


def print_arm_changes(rep: dict, first, second) -> None:
    if not rep:
        return
    print(f"\n=== each arm's own change: score({first}) - score({second}) "
          "(descriptive; says WHICH arm moved) ===")
    for qt, per in rep.items():
        for arm, v in per.items():
            p = "nan" if v["p_raw"] != v["p_raw"] else f"{v['p_raw']:.3g}"
            print(f"  {qt:<20}{arm:<16} {v['mean_second']:.3f} -> "
                  f"{v['mean_first']:.3f}  d={v['mean_diff']:+.4f} "
                  f"[{v['ci_low']:+.3f},{v['ci_high']:+.3f}]  p={p}")


# --------------------------------------------------------------------------
# C2-8: routing, strict and functional
# --------------------------------------------------------------------------
def routing_table(df: pd.DataFrame) -> pd.DataFrame:
    rt = df[df.route_source == "llm"]
    if rt.empty or "route_equivalent" not in rt.columns:
        return pd.DataFrame()
    return (rt.groupby(["backbone", "L", "query_type"])
            .agg(route_correct=("route_correct", "mean"),
                 route_equivalent=("route_equivalent", "mean"),
                 n=("route_correct", "size")).reset_index())


def print_routing(tab: pd.DataFrame) -> None:
    if tab.empty:
        return
    print("\n=== routing: strict vs functional (C2-8; describe and explain "
          "are interchangeable) ===")
    print(tab.round(3).to_string(index=False))


def absolute_means(df: pd.DataFrame, cells) -> dict:
    """Every arm's mean score per cell - printed beside every gap."""
    sub = df[df.query_type.isin(list(cells))]
    piv = sub.pivot_table(index="query_type", columns="system",
                          values="score", aggfunc="mean")
    return {qt: {a: float(v) for a, v in row.items() if v == v}
            for qt, row in piv.iterrows()}
