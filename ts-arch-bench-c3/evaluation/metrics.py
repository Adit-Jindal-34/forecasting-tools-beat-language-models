"""Metric primitives. Pure, testable, no I/O.

THE PROBLEM THIS MODULE FIXES
-----------------------------
The previous build scored every numeric cell with

    score = 1 / (1 + |v - gt| / max(|gt|, 1e-9))

which divides by the TRUTH. On the n100 pool that made the identical
physical error cost ~0.0005 in the `change` cell and up to 0.95 in
`statistics`, because the series are near zero-mean and |gt| collapses.
46/100 `statistics` items and 30/100 `correlation` items had
|gt| < 0.1 * series_sd. A single 0.02 effect floor was then applied to all
nine cells, and H3 averaged across them.

This is the failure Hyndman & Koehler (2006) identified when they introduced
MASE: percentage errors are undefined or explosive as actuals approach zero,
and the remedy is a denominator BOUNDED AWAY FROM ZERO.

THE FIX
-------
Error is expressed as a fraction of the quantity's own NATURAL SCALE, chosen
per operation and never derived from the target value:

    mean/median/std/max/min   series units   -> std(window)
    pct_change                percent units  -> 100*std(window)/|x[0]|
    lag1_autocorr             bounded [-1,1] -> 1.0 (i.e. absolute error)
    count_spikes              integer count  -> exact match, no scale
    change_point              index          -> localization tolerance

Because every scaled error is then "fraction of natural scale", ONE
tolerance (default 0.05) is meaningful in every cell, and the binary
`within_tol` column is genuinely comparable across cells.

STILL FORBIDDEN: averaging scores across KINDS. A forecast score, a numeric
score, a localization hit and a qualitative slot accuracy are different
measurements. `normalized_score` returns a per-kind number for plotting; it
is not a currency.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9

#: default tolerance, in units of the operation's natural scale
DEFAULT_TOL = 0.05

#: default localization tolerance, in index steps
DEFAULT_LOC_TOL = 2


# --------------------------------------------------------------------------
# natural scale per operation
# --------------------------------------------------------------------------
def natural_scale(op: str, window: np.ndarray) -> float:
    """The denominator for `op`, bounded away from zero, never using the GT.

    Raises for ops that are not scored by scaled error (`count_spikes`,
    `change_point`) so a caller cannot silently get a wrong denominator.
    """
    x = np.asarray(window, float)
    if op in ("mean", "median", "std", "max", "min"):
        return max(float(np.std(x)), EPS)
    if op == "pct_change":
        # percent units: one standard deviation expressed as a percentage of
        # the anchor point the operation divides by
        return max(100.0 * float(np.std(x)) / max(abs(float(x[0])), EPS), EPS)
    if op == "lag1_autocorr":
        return 1.0          # bounded quantity -> plain absolute error
    raise ValueError(
        f"op {op!r} is not scored by scaled error; "
        "count_spikes uses exact match and change_point uses loc_hit")


SCALED_ERROR_OPS = frozenset(
    {"mean", "median", "std", "max", "min", "pct_change", "lag1_autocorr"})
EXACT_MATCH_OPS = frozenset({"count_spikes"})


def scaled_error(value: float, truth: float, op: str,
                 window: np.ndarray) -> float:
    """|value - truth| in units of the operation's natural scale."""
    return abs(float(value) - float(truth)) / natural_scale(op, window)


# --------------------------------------------------------------------------
# forecast
# --------------------------------------------------------------------------
def mae(pred, true) -> float:
    p, t = np.asarray(pred, float), np.asarray(true, float)
    return float(np.mean(np.abs(p - t)))


def rmse(pred, true) -> float:
    p, t = np.asarray(pred, float), np.asarray(true, float)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mase(pred, true, denominator: float) -> float:
    """MASE against a PRE-COMPUTED denominator.

    The caller supplies the denominator so that a history-length sweep can
    hold it FIXED across L (computed once from each series' longest window).
    A per-L denominator makes scores at different L incomparable, because the
    yardstick moves with the condition under test.
    """
    return mae(pred, true) / max(float(denominator), EPS)


# --------------------------------------------------------------------------
# localization
# --------------------------------------------------------------------------
def loc_hit(pred_idx: int, gt_idx: int, tol: int = DEFAULT_LOC_TOL) -> bool:
    return abs(int(pred_idx) - int(gt_idx)) <= int(tol)


def loc_hit_curve(pred_idx: int, gt_idx: int,
                  tols=(0, 1, 2, 5)) -> dict[str, int]:
    """hit@k for several k. A curve is far more informative than one k, and
    costs nothing to compute."""
    return {f"hit@{k}": int(loc_hit(pred_idx, gt_idx, k)) for k in tols}


def index_in_range(pred_idx, L: int) -> bool:
    """A change-point index must address a step i -> i+1 inside the window."""
    try:
        i = int(pred_idx)
    except (TypeError, ValueError, OverflowError):
        return False
    return 0 <= i <= L - 2


# --------------------------------------------------------------------------
# per-kind normalised score (plotting / within-kind comparison only)
# --------------------------------------------------------------------------
def score_numeric(value: float, truth, op: str, window: np.ndarray,
                  tol: float = DEFAULT_TOL) -> dict:
    """Score one numeric answer. Returns the full record, not just a number."""
    if op in EXACT_MATCH_OPS:
        ok = int(round(float(value))) == int(truth)
        return dict(op=op, value=float(value), gt=float(truth),
                    abs_err=abs(float(value) - float(truth)),
                    scaled_err=float("nan"), within_tol=int(ok),
                    score=float(ok), scoring="exact_match")
    se = scaled_error(value, truth, op, window)
    # tolerance compared with a float guard: an error of exactly `tol` must
    # count as within tolerance, and 0.55 - 0.50 is 0.050000000000000044
    return dict(op=op, value=float(value), gt=float(truth),
                abs_err=abs(float(value) - float(truth)),
                scaled_err=se, within_tol=int(se <= tol * (1 + 1e-9) + 1e-12),
                score=1.0 / (1.0 + se), scoring="scaled_error")


def score_forecast(pred, truth, denominator: float) -> dict:
    m = mase(pred, truth, denominator)
    return dict(mae=mae(pred, truth), rmse=rmse(pred, truth), mase=m,
                beats_naive=int(m < 1.0), score=1.0 / (1.0 + m),
                scoring="mase_fixed_denominator")


def score_localization(pred_idx: int, gt_idx: int, L: int,
                       tol: int = DEFAULT_LOC_TOL) -> dict:
    rec = dict(pred_idx=int(pred_idx), gt_idx=int(gt_idx),
               exact=int(int(pred_idx) == int(gt_idx)),
               score=float(loc_hit(pred_idx, gt_idx, tol)),
               abs_idx_err=abs(int(pred_idx) - int(gt_idx)),
               scoring=f"hit_at_{tol}")
    rec.update(loc_hit_curve(pred_idx, gt_idx))
    return rec


# --------------------------------------------------------------------------
# selective prediction: coverage and risk
# --------------------------------------------------------------------------
def selective_report(scores, answered) -> dict:
    """Coverage and selective risk for one system x cell.

    The previous build returned NaN for a parse failure and then dropped the
    row, so a system was never penalised for failing to answer and every
    reported mean was silently conditional on parsing. Report instead:

      coverage        fraction of items the system actually answered
      selective_mean  mean score among ANSWERED items (the old number)
      scored_mean     mean score with unanswered items scored 0 (honest)
      selective_risk  1 - selective_mean

    Pre-register `scored_mean` as primary and show the other two beside it.
    """
    s = np.asarray(scores, float)
    a = np.asarray(answered, bool)
    n = len(s)
    if n == 0:
        return dict(n=0, n_answered=0, coverage=float("nan"),
                    selective_mean=float("nan"), scored_mean=float("nan"),
                    selective_risk=float("nan"))
    answered_scores = s[a & ~np.isnan(s)]
    filled = np.where(a & ~np.isnan(s), np.nan_to_num(s, nan=0.0), 0.0)
    sel = float(answered_scores.mean()) if len(answered_scores) else float("nan")
    return dict(n=int(n), n_answered=int(a.sum()),
                coverage=float(a.sum()) / n,
                selective_mean=sel, scored_mean=float(filled.mean()),
                selective_risk=(1.0 - sel) if sel == sel else float("nan"))
