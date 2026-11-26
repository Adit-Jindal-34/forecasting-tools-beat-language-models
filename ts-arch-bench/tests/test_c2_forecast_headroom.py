"""Does the FORECASTING cell have room to show anything? (amendment A15)

A9 keyed the qualitative cells on what the generator injected, and A13/A14
then had to measure, per slot and per L, whether the window actually shows
it. Nobody had asked the same question of the one cell that carries a
confirmatory claim in all three hypotheses.

The answer on the first C2 generator was "barely". The score is
`1/(1+MASE)` against an in-sample seasonal-naive denominator, so seasonal
naive sits at about 0.50 by construction and a forecaster that knew the
generating process exactly scores only about 0.56: the whole usable range is
narrow. And with seasonal periods of 6, 8 and 12, a 32-point window already
showed the full pattern, so that ceiling rose by just +0.033 from L=32 to
L=512 - which bounds H1's only confirmatory contrast.

A15 lengthened the periods to 12/24/48 and strengthened the cycle. These
tests pin what that bought, using an INFORMED forecaster: a least-squares
fit that is told the true period and the true event location. It is not a
competitor - no arm has that knowledge - it is the ceiling, and the
question is only whether the ceiling leaves room.

If a later generator change shrinks that room again, this file fails and
says so before any GPU time is spent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import ground_truth as gt, synthetic             # noqa: E402
from src.stats import paired_mde                          # noqa: E402

N, H = 512, 8
LENGTHS = (32, 64, 128, 256, 512)


def informed_forecast(x: np.ndarray, truth: dict, L: int) -> np.ndarray:
    """Least squares on window(L): level + slope + a harmonic at the TRUE
    period (+ a step at the TRUE event index; a known spike is ignored)."""
    w = x[N - L:N]
    t = np.arange(L, dtype=float)
    tf = np.arange(L, L + H, dtype=float)
    cols, cols_f = [np.ones(L), t], [np.ones(H), tf]
    if truth["seasonal"] and L >= truth["period"]:
        for f in (np.sin, np.cos):
            cols.append(f(2 * np.pi * t / truth["period"]))
            cols_f.append(f(2 * np.pi * tf / truth["period"]))
    weight = np.ones(L)
    if truth["event_index"] is not None:
        loc = truth["event_index"] - (N - L)
        if truth["event"] == "abrupt change" and 2 <= loc < L - 2:
            cols.append((t > loc).astype(float))
            cols_f.append(np.ones(H))
        if truth["event"] == "spike" and 0 <= loc < L:
            weight[loc] = 0.0
    A = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(A * weight[:, None], w * weight, rcond=None)
    return np.column_stack(cols_f) @ beta


@pytest.fixture(scope="module")
def scores():
    """{name: {L: per-item score array}} on 360 generated series."""
    rng = np.random.default_rng(20261111)
    pool = [synthetic.generate_base_series_c2(N, c, rng, horizon=H)
            for c in synthetic.c2_design_cells(360, rng)]
    out = {k: {L: [] for L in LENGTHS}
           for k in ("informed", "last_value", "seasonal_naive")}
    for x, truth in pool:
        y = x[N:]
        # the pipeline's own denominator: seasonal naive on the longest
        # window, with the true period (evaluation.forecast_denominators)
        den = gt.naive_window_error(x[:N], truth["period"])

        def score(pred):
            return 1.0 / (1.0 + np.mean(np.abs(pred - y)) / den)

        for L in LENGTHS:
            w = x[N - L:N]
            p = truth["period"]
            sn = (np.array([w[L - p + (h % p)] for h in range(H)])
                  if 2 <= p <= L else np.full(H, w[-1]))
            out["informed"][L].append(score(informed_forecast(x, truth, L)))
            out["last_value"][L].append(score(np.full(H, w[-1])))
            out["seasonal_naive"][L].append(score(sn))
    return {k: {L: np.asarray(v) for L, v in d.items()}
            for k, d in out.items()}


def test_seasonal_naive_sits_at_the_score_midpoint(scores):
    """Sanity on the metric: MASE = 1 for the forecast its denominator is
    built from, so the true-period seasonal naive scores about 0.5."""
    assert scores["seasonal_naive"][512].mean() == pytest.approx(0.5,
                                                                 abs=0.02)


def test_h2_h3_have_room_at_128(scores):
    """At L=128 the ceiling must clear the naive floors by a margin a paired
    test on 100 items can see (Holm, m=3)."""
    ceiling = scores["informed"][128]
    for floor_name, min_gap in (("last_value", 0.09),
                                ("seasonal_naive", 0.05)):
        gap = ceiling - scores[floor_name][128]
        dz = gap.mean() / gap.std(ddof=1)
        assert gap.mean() > min_gap, (
            f"ceiling - {floor_name} at L=128 is only {gap.mean():+.3f}")
        assert dz > 1.5 * paired_mde(100, n_tests=3), (
            f"ceiling - {floor_name}: dz {dz:.2f} is too close to what "
            "n=100 can detect")


def test_history_length_matters_for_forecasting(scores):
    """H1's premise. The ceiling must rise from L=32 to L=512 by clearly
    more than the 0.02 effect floor, and by enough for 300 items to see.

    On the first C2 generator this was +0.033 (dz 0.41): the contrast H1
    tests could not have moved by more than that.
    """
    change = scores["informed"][512] - scores["informed"][32]
    dz = change.mean() / change.std(ddof=1)
    assert change.mean() > 0.06, (
        f"the forecasting ceiling rises by only {change.mean():+.3f} from "
        "L=32 to L=512; H1 has nothing to measure")
    assert dz > 2.5 * paired_mde(300, n_tests=1), (
        f"dz {dz:.2f} against an MDE of {paired_mde(300, 1):.2f} at n=300")


def test_the_gain_is_front_loaded_and_documented(scores):
    """Recorded, not hidden: almost all of the ceiling's gain is between
    L=32 and L=64. The H1 config header says to read the whole curve; this
    pins the fact that makes that instruction necessary."""
    c = {L: scores["informed"][L].mean() for L in LENGTHS}
    total = c[512] - c[32]
    early = c[64] - c[32]
    assert early / total > 0.6
    assert c[512] >= c[128] - 0.005          # and it never gets worse


def test_naive_floors_do_not_improve_with_history(scores):
    """The floors must be flat in L, or a widening gap could come from the
    yardstick. Last-value is identical at every L by construction."""
    lv = [scores["last_value"][L].mean() for L in LENGTHS]
    assert max(lv) - min(lv) < 1e-12
