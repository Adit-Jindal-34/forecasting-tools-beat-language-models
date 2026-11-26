"""Diagnostics on the real pool, before a single GPU hour is spent.

Answers three questions the synthetic pool answered badly:

1. **Is the cell non-degenerate?** What do the classical baselines score, and
   is there headroom between the best trivial answer and a perfect one?
2. **Does history length buy anything at all?** A seasonal-naive baseline
   that must *estimate* its period cannot work at L=32 when the period is
   24, and can at L=512. If that baseline does not improve with L, neither
   a forecaster nor an LLM is likely to, and H1 has nothing to measure -
   which is exactly what amendment A15 diagnosed in the synthetic pool.
3. **Are the H2 strata populated?** The reframed H2 needs both strongly and
   weakly seasonal items.

Scores use the project's convention: `1/(1+MASE)`, with the pool's fixed
denominator, so these numbers are directly comparable to the cycle-2 tables
(where 0.5 is the seasonal-naive forecast the denominator is built from).

Run:  python data-real/verify_pool.py                       # every pool
      python data-real/verify_pool.py --pool pool_clean_n300 --horizon 24
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pool_c3 import STRATUM_CUTS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LS = (32, 64, 128, 256, 512)

#: Default horizon. PREREGISTRATION_C3.md C3-2 declares 24 as PRIMARY - one
#: full declared cycle for the hourly majority - with 8 as the sensitivity
#: arm. 8 is cycle 2's value and is a third of a cycle on period-24 data,
#: which is seasonal naive's easiest regime; that is why its bar is so high.
DEFAULT_HORIZON = 24


def estimate_period(hist: np.ndarray, lo: int = 2) -> int:
    """Best detrended-autocorrelation lag, searched over the window only.

    Deliberately the *estimating* variant: it is what an arm can do without
    being told the frequency, so its behaviour along L is the headroom test.
    """
    n = len(hist)
    hi = max(lo + 1, n // 3)
    t = np.arange(n, dtype=float)
    a, b = np.polyfit(t, hist, 1)
    d = hist - (a * t + b)
    d = d - d.mean()
    denom = float(np.dot(d, d))
    if denom <= 0:
        return 1
    best, best_r = 1, -np.inf
    for lag in range(lo, min(hi, n - 1)):
        r = float(np.dot(d[lag:], d[:-lag]) / denom)
        if r > best_r:
            best, best_r = lag, r
    return best


def forecasts(hist: np.ndarray, h: int, declared: int) -> dict:
    last = np.full(h, hist[-1])
    wmean = np.full(h, hist.mean())

    def snaive(p: int) -> np.ndarray:
        p = max(1, min(p, len(hist)))
        reps = int(np.ceil(h / p))
        return np.tile(hist[-p:], reps)[:h]

    return {
        "last_value": last,
        "window_mean": wmean,
        "seasonal_naive_declared": snaive(declared),
        "seasonal_naive_estimated": snaive(estimate_period(hist)),
    }


def report(pool: str, h: int) -> None:
    z = np.load(os.path.join(pool, "windows.npz"))
    meta = json.load(open(os.path.join(pool, "meta.json"), encoding="utf-8"))
    W = z["windows"]
    denom = z["mase_denominator"]
    period = z["period"]
    lmax = meta["L_MAX"]
    if h > meta["H_MAX"]:
        raise SystemExit(f"horizon {h} exceeds the pool's H_MAX "
                         f"{meta['H_MAX']}")

    target = W[:, lmax:lmax + h]
    print("=" * 78)
    print(f"{os.path.basename(pool)}   n={len(W)}  horizon={h}  "
          f"(pool stores up to {meta['H_MAX']})")
    print("=" * 78)

    print("mean score 1/(1+MASE) by history length "
          "- 0.5 would be an exactly-naive forecast")
    names = ["last_value", "window_mean",
             "seasonal_naive_declared", "seasonal_naive_estimated"]
    print(f"{'L':>5} " + "".join(f"{n[:24]:>26}" for n in names))
    first = {}
    for L in LS:
        row = {n: [] for n in names}
        for i in range(len(W)):
            hist = W[i, lmax - L:lmax]
            for n, f in forecasts(hist, h, int(period[i])).items():
                mase = float(np.mean(np.abs(f - target[i]))) / denom[i]
                row[n].append(1.0 / (1.0 + mase))
        means = {n: float(np.mean(v)) for n, v in row.items()}
        if L == LS[0]:
            first = means
        print(f"{L:>5} " + "".join(f"{means[n]:>26.4f}" for n in names))
    print(f"{'chg':>5} " + "".join(
        f"{means[n] - first[n]:>+26.4f}" for n in names))

    print("\nheadroom: best trivial answer vs a perfect forecast (score 1.0)")
    best = max(means[n] for n in names)
    print(f"  best classical baseline at L=512: {best:.4f} "
          f"-> {1.0 - best:.4f} of range left to win")

    ss = z["seasonal_strength"]
    # The cuts are IMPORTED, never restated: a third copy of a
    # pre-registered threshold is a third chance to drift from
    # `src.real_data.STRATUM_CUTS`, which is what
    # tests/test_real_data.py::test_stratum_cuts_agree_* exists to stop.
    wlo, shi = STRATUM_CUTS
    print(f"\nH2 strata (seasonal_strength = detrended ACF at the declared "
          f"period; cuts: weak < {wlo}, strong >= {shi})")
    for lo, hi, label in [(-np.inf, wlo, "weak   "), (wlo, shi, "medium "),
                          (shi, np.inf, "strong ")]:
        m = (ss >= lo) & (ss < hi)
        print(f"  {label}: {int(m.sum()):4} items")

    print("\nby dataset")
    ds = [it["dataset"] for it in meta["items"]]
    for d in sorted(set(ds)):
        m = np.array([x == d for x in ds])
        print(f"  {d:20} n={int(m.sum()):3}  "
              f"seasonal_strength median {float(np.median(ss[m])):+.3f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=None,
                    help="pool directory name under data-real/; "
                         "default: every pool present")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                    help=f"forecast horizon (default {DEFAULT_HORIZON}, the "
                         "C3-2 primary; 8 is the declared sensitivity arm "
                         "and cycle 2's value)")
    args = ap.parse_args()

    pools = ([os.path.join(HERE, args.pool)] if args.pool else
             sorted(os.path.dirname(p) for p in
                    glob.glob(os.path.join(HERE, "pool_*", "windows.npz"))))
    if not pools:
        raise SystemExit("no pool found; run build_pool.py / build_pool_c3.py")
    for pool in pools:
        report(pool, args.horizon)


if __name__ == "__main__":
    main()
