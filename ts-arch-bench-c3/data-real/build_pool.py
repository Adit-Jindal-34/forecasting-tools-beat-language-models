"""Build a stratified real-data pool for the H1/H2/H3 forecasting cell.

Design decisions, all of them consequences of the existing protocol:

* **One window per series.** Two windows cut from one series are not
  independent, and the paired tests assume items are. Taking a single window
  per series keeps `item` a legitimate clustering unit; `dataset` is still the
  *conservative* unit and is recorded for `paired_compare_clustered`.

* **A common cut origin, suffix windows.** Every item stores the last
  `L_MAX + H_MAX` points of its series. The history at L is the final L
  points before the cut, so the windows at L=32..512 are nested suffixes
  ending at the *same* instant and the forecast target never changes. That is
  invariant I1/I2 - "the same item tracked along the sweep" - carried over
  from the synthetic pool.

* **The MASE denominator is fixed and public.** It is the mean absolute
  seasonal difference over the L_MAX history using the period the *archive*
  declares for the frequency (`tsf.FREQ_PERIOD`), computed once per item and
  identical at every L. So it cannot drift with the window an arm is shown,
  which is what `scoring.fixed_mase_denominator` requires. The arms are still
  not told the period - estimating it stays part of the task.

* **Strata are descriptive, never a key.** `seasonal_strength` and `trend_t`
  are recorded so H2 can be reframed onto series character, and so results
  can be read by stratum. They must NOT become an answer key: the evidence
  block computes the same family of statistics, and a key drawn from them
  would recreate exactly the cycle-1 defect where the tool and the key shared
  code. Real data has a label for forecasting and for nothing else.

Run:  python data-real/build_pool.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsf import read_tsf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

L_MAX = 512          # the longest history the sweep uses
H_MAX = 48           # target stored; a config may use any horizon <= H_MAX
SEED = 20261114
N_PER_DATASET = 75   # 4 datasets -> n = 300, no dataset over 25% of the pool

SOURCES = {
    "m4_hourly": "m4_hourly_dataset.tsf",
    "nn5_daily": "nn5_daily_dataset_without_missing_values.tsf",
    "electricity_hourly": "electricity_hourly_dataset.tsf",
    "traffic_hourly": "traffic_hourly_dataset.tsf",
}


def seasonal_naive_mase_denominator(hist: np.ndarray, period: int) -> float:
    """Mean absolute seasonal difference over the in-sample history.

    The standard MASE denominator (Hyndman & Koehler). Bounded away from zero
    by the caller: a series whose seasonal differences all vanish is rejected
    rather than divided by.
    """
    if period < 1 or len(hist) <= period:
        return float("nan")
    return float(np.mean(np.abs(hist[period:] - hist[:-period])))


#: Every history the sweep will ever show an arm. The exclusion rule below
#: must hold at ALL of them, not just at L_MAX: a shorter history is a
#: *suffix* of the long one, so it can be flat while the full window is not.
#: Real sensor feeds do exactly this - a traffic loop reads 0 all night.
SWEEP_LS = (32, 64, 128, 256, 512)

#: `src.ground_truth.REL_EPS` is the floor the feature code divides by. When
#: a window's MAD or local noise scale collapses to it, `series_evidence`
#: does not fail - it emits `largest_local_peak_robust_z=35800000.00` into the
#: prompt. Garbage evidence is worse than a rejected item, so such windows are
#: excluded here, by a rule fixed before any real-data result is seen.
MIN_SCALE = 1e-6


def refuse_overwrite(out_dir: str) -> None:
    """Stop before replacing a pool that already exists.

    The configs and PREREGISTRATION_C3.md declare each pool by the sha256
    of its `windows.npz`. A rebuild on another machine gives the SAME
    windows and denominators, but `seasonal_strength` and `trend_t` come
    out different in the last decimal places (measured 2026-11-14: up to
    1.7e-13, from the BLAS behind `np.dot` / `np.polyfit`). That changes
    the file hash, so every config would then refuse to load the pool.
    The pools shipped with the code are the declared ones; a run needs
    nothing rebuilt. `--force` is for deliberately making a NEW pool,
    which also means new hashes in the configs and a dated amendment.
    """
    target = os.path.join(out_dir, "windows.npz")
    if os.path.exists(target) and "--force" not in sys.argv[1:]:
        raise SystemExit(
            f"{target} already exists - not overwritten.\n"
            "The shipped pool is the one the configs declare by hash, and a "
            "rebuild on a different machine changes that hash (last-digit "
            "float differences in the strata columns). Nothing needs "
            "rebuilding to run cycle 3.\n"
            "Check the pools instead:  python data-real/check_pools.py\n"
            "To really build a new pool:  add --force (then the configs' "
            "pool_sha256 and the pre-registration must change too).")


def degenerate_at_no_L(window: np.ndarray) -> bool:
    """True if any history in the sweep is unusable.

    Checks, at every L: non-constant values, a usable spread, and a non-
    collapsed robust scale (MAD). The last one is what the synthetic
    generator made impossible and real data makes routine.
    """
    for L in SWEEP_LS:
        hist = window[L_MAX - L:L_MAX]
        if len(hist) < L:
            return True
        if not np.all(np.isfinite(hist)):
            return True
        if float(np.std(hist)) <= MIN_SCALE:
            return True
        mad = float(np.median(np.abs(hist - np.median(hist))))
        if mad <= MIN_SCALE:
            return True
    return False


def detrend(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=float)
    a, b = np.polyfit(t, x, 1)
    return x - (a * t + b)


def seasonal_strength(hist: np.ndarray, period: int) -> float:
    """Autocorrelation of the detrended history at the declared period.

    Descriptive only - a stratification variable for the reframed H2, chosen
    because it is the quantity the archive's own frequency label implies.
    """
    if period < 2 or len(hist) <= 2 * period:
        return float("nan")
    d = detrend(hist)
    d = d - d.mean()
    denom = float(np.dot(d, d))
    if denom <= 0:
        return float("nan")
    return float(np.dot(d[period:], d[:-period]) / denom)


def trend_t(hist: np.ndarray) -> float:
    """t-statistic of the fitted slope over the history. Descriptive only."""
    n = len(hist)
    t = np.arange(n, dtype=float)
    a, b = np.polyfit(t, hist, 1)
    resid = hist - (a * t + b)
    dof = n - 2
    if dof <= 0:
        return float("nan")
    s2 = float(np.dot(resid, resid)) / dof
    sxx = float(np.sum((t - t.mean()) ** 2))
    if s2 <= 0 or sxx <= 0:
        return float("inf") if a > 0 else float("-inf")
    return float(a / np.sqrt(s2 / sxx))


def main() -> None:
    rng = np.random.default_rng(SEED)
    need = L_MAX + H_MAX

    items, report = [], []
    for ds, fn in SOURCES.items():
        path = os.path.join(RAW, fn)
        f = read_tsf(path)
        lengths = f.lengths()
        period = f.period

        eligible = []
        for i, s in enumerate(f.series):
            if len(s) < need or np.isnan(s).any():
                continue
            w = s[-need:]
            hist = w[:L_MAX]
            denom = seasonal_naive_mase_denominator(hist, period)
            if not np.isfinite(denom) or denom <= 0:
                continue
            if not degenerate_at_no_L(w):
                eligible.append((i, w, denom))

        take = min(N_PER_DATASET, len(eligible))
        pick = rng.choice(len(eligible), size=take, replace=False)
        for k in sorted(pick):
            i, w, denom = eligible[k]
            hist = w[:L_MAX]
            items.append(dict(
                item_id=f"{ds}-{f.names[i]}",
                dataset=ds,
                series_name=f.names[i],
                frequency=f.frequency,
                period=period,
                archive_horizon=f.horizon,
                window=w,
                mase_denominator=denom,
                seasonal_strength=seasonal_strength(hist, period),
                trend_t=trend_t(hist),
            ))

        report.append(dict(
            dataset=ds, file=fn, n_series=int(len(lengths)),
            frequency=f.frequency, declared_period=period,
            archive_horizon=f.horizon,
            len_min=int(lengths.min()), len_median=int(np.median(lengths)),
            len_max=int(lengths.max()),
            n_eligible=len(eligible), n_taken=take,
            sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
        ))

    windows = np.stack([it["window"] for it in items]).astype(np.float64)
    out = os.path.join(HERE, "pool_real_n%d" % len(items))
    refuse_overwrite(out)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(
        os.path.join(out, "windows.npz"),
        windows=windows,
        mase_denominator=np.array([it["mase_denominator"] for it in items]),
        period=np.array([it["period"] for it in items], dtype=int),
        seasonal_strength=np.array([it["seasonal_strength"] for it in items]),
        trend_t=np.array([it["trend_t"] for it in items]),
    )

    meta = dict(
        built="2026-11-14",
        seed=SEED,
        n_items=len(items),
        L_MAX=L_MAX,
        H_MAX=H_MAX,
        layout=("windows[i] has L_MAX + H_MAX points. History at L is "
                "windows[i][L_MAX-L : L_MAX]; the target for horizon h is "
                "windows[i][L_MAX : L_MAX+h]. All L are nested suffixes "
                "ending at one cut, so the target never changes with L."),
        mase_denominator=("mean |x[t] - x[t-period]| over windows[i][:L_MAX] "
                          "with the archive-declared period; identical at "
                          "every L by construction"),
        strata_warning=("seasonal_strength and trend_t are DESCRIPTIVE "
                        "stratification variables. They must never be used "
                        "as an answer key: the harness's evidence block "
                        "computes the same family of statistics, and a key "
                        "drawn from them reproduces the cycle-1 defect."),
        items=[{k: v for k, v in it.items() if k != "window"} for it in items],
        datasets=report,
    )
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    print(f"pool: {out}  n={len(items)}  windows={windows.shape}")
    for r in report:
        print(f"  {r['dataset']:20} eligible {r['n_eligible']:4}/"
              f"{r['n_series']:<4} taken {r['n_taken']:3}  "
              f"period={r['declared_period']}")


if __name__ == "__main__":
    main()
