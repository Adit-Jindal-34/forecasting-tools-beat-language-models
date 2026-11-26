"""Ground truth — the answer key. Nothing here ever sees a model output.

Every ground truth is a pure function of (sample, L):
  * forecast     -> the fixed future beyond the window end (invariant in L)
  * observation  -> computed from window(L) only (invariant I4)
  * qualitative  -> a STRUCTURED slot record, not a bag of keyphrases

WHAT CHANGED FROM THE PREVIOUS BUILD
------------------------------------
1. Qualitative truth is now `QualitativeTruth`, a record of closed-set slots,
   instead of `list[str]`. The old flat list was scored by substring recall
   with no precision penalty, so a single constant string naming every
   possible keyphrase scored 1.000 on every item. Slots make the task a set
   of mutually exclusive classification decisions, which cannot be gamed by
   saying everything. See evaluation/qualitative.py.

2. Ground truth may use privileged generator parameters (the true seasonal
   period). Systems may not. That asymmetry is correct and deliberate: the
   answer key is allowed to know how the data was made.

ON "MEASUREMENT CIRCULARITY"
----------------------------
`systems/tools.py` imports from this module. For `mean`, `max` and
`pct_change` that is not circularity: the question asks for the arithmetic
mean of the visible series, and `np.mean(window)` is simply the correct
answer. Re-deriving it from the generating process would make the key WRONG.

The real issue is narrower and is recorded in OPERATION_NOTES: for
`count_spikes` and `change_point` the key is not a canonical quantity but an
arbitrary operationalisation (a 3-MAD threshold; argmax of |diff|). A system
can answer the question sensibly and still miss the key because it guessed a
different threshold. That is a property to REPORT, not to patch away.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from src.samples import Sample

REL_EPS = 1e-9

#: Precise prose for the methods section. The type LABEL is broader than the
#: OPERATION in the three cases flagged below; say so in the paper rather
#: than renaming and breaking continuity with earlier runs.
OPERATION_NOTES: dict[str, str] = {
    "forecasting": "Point forecast of the next `horizon` values.",
    "statistics": "Arithmetic mean of the visible window. LABEL IS BROADER "
                  "than the operation: 'statistics' means the mean only.",
    "counting": "Number of local maxima exceeding median + 3*MAD. "
                "OPERATIONALISED, not canonical: a different spike "
                "definition yields a different key.",
    "extremum": "Maximum of the visible window.",
    "change": "Percent change between the FIRST and LAST points only. "
              "LABEL IS BROADER: the interior of the window is ignored.",
    "correlation": "Lag-1 autocorrelation of the series with itself. "
                   "LABEL IS BROADER: no second series is involved.",
    "localization": "Index i maximising |x[i+1] - x[i]|. OPERATIONALISED: "
                    "ties and near-ties are resolved by argmax.",
    "trend_description": "Trend direction and seasonality, as slots.",
    "event_explanation": "Trend, seasonality, event kind and event position, "
                         "as slots.",
}


# --------------------------------------------------------------------------
# primitive observations
# --------------------------------------------------------------------------
def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (1.4826 * mad + REL_EPS)


def count_spikes(x: np.ndarray) -> int:
    """Local maxima above median + 3*MAD."""
    if len(x) < 3:
        return 0
    z = _robust_z(x)
    n = 0
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1] and z[i] > 3.0:
            n += 1
    return n


def change_point_index(x: np.ndarray) -> int:
    """Index i of the largest |x[i+1]-x[i]|; the step from i to i+1.

    Valid range is [0, len(x)-2]. Predictions outside it are type errors,
    not merely wrong answers — see evaluation/evaluate.py.
    """
    if len(x) < 2:
        return 0
    return int(np.argmax(np.abs(np.diff(x))))


def lag1_autocorr(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    a, b = x[:-1], x[1:]
    if a.std() < REL_EPS or b.std() < REL_EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def pct_change(x: np.ndarray) -> float:
    """Net percent change between the endpoints of the window."""
    if len(x) < 2:
        return 0.0
    return float((x[-1] - x[0]) / max(abs(x[0]), REL_EPS) * 100.0)


def trend_direction(x: np.ndarray) -> str:
    """'increasing' / 'decreasing' / 'stable' from the slope of the last third."""
    seg = x[-max(3, len(x) // 3):]
    if len(seg) < 3:
        return "stable"
    scale = max(float(np.std(x)), REL_EPS)
    slope = (seg[-1] - seg[0]) / (len(seg) - 1)
    if slope > 0.02 * scale:
        return "increasing"
    if slope < -0.02 * scale:
        return "decreasing"
    return "stable"


def is_seasonal(x: np.ndarray, period: int) -> bool:
    """Strong ACF at the TRUE period. Privileged: key-side only."""
    if period < 2 or len(x) < 2 * period:
        return False
    a, b = x[:-period], x[period:]
    if a.std() < REL_EPS or b.std() < REL_EPS:
        return False
    return float(np.corrcoef(a, b)[0, 1]) > 0.5


def level_shift(x: np.ndarray, idx: int, width: int = 4) -> float:
    """|mean(after) - mean(before)| around idx; 0.0 when either side is empty."""
    n = len(x)
    lo, hi = max(0, idx - width), min(n, idx + 1 + width)
    if idx <= lo or hi <= idx + 1:
        return 0.0
    return float(abs(np.mean(x[idx + 1:hi]) - np.mean(x[lo:idx])))


def local_noise_scale(x: np.ndarray) -> float:
    """Robust point-to-point noise scale: 1.4826 * MAD(diff) / sqrt(2).

    WHY NOT std(x). The old build tested for a level shift with

        level_shift(x, idx) > 3.0 * np.std(x)

    which is self-defeating: a two-level series has std ~= shift/2, so the
    threshold 3*std ~= 1.5*shift can never be cleared by the shift that
    created it. On the real pool the "abrupt change" label therefore fired
    ZERO times - one of the three event classes was unreachable, exactly the
    same failure mode as the unreachable `no_split` verdict branch.

    A median over consecutive differences ignores the single large jump, so
    it measures the noise floor rather than the event, and the test can
    actually fire.
    """
    if len(x) < 3:
        return REL_EPS
    d = np.abs(np.diff(np.asarray(x, float)))
    sigma = 1.4826 * float(np.median(d)) / np.sqrt(2.0)
    return max(sigma, REL_EPS)


def has_level_shift(x: np.ndarray, idx: int) -> bool:
    """A step at `idx` large against the LOCAL noise floor."""
    return len(x) >= 8 and level_shift(x, idx) > 4.0 * local_noise_scale(x)


def event_position(index: int, n: int) -> str:
    if n <= 0:
        return "middle"
    frac = index / max(n - 1, 1)
    if frac < 1 / 3:
        return "early"
    if frac > 2 / 3:
        return "late"
    return "middle"


def naive_window_error(x: np.ndarray, period: int) -> float:
    """In-sample naive error, the MASE denominator.

    Seasonal naive when the window holds >= 2 full periods, else drift naive.
    Hyndman & Koehler's point: this denominator is bounded away from zero as
    long as the series moves at all, which is why MASE survives near-zero
    actuals where percentage errors do not.
    """
    if period >= 2 and len(x) >= 2 * period:
        return float(np.mean(np.abs(x[period:] - x[:-period])))
    if len(x) >= 2:
        return float(np.mean(np.abs(np.diff(x))))
    return float(np.std(x))


# --------------------------------------------------------------------------
# qualitative truth: closed-set slots
# --------------------------------------------------------------------------
#: slot name -> the mutually exclusive classes it admits. A prediction must
#: select exactly one class per applicable slot; naming several is ambiguity,
#: not partial credit. This is what makes the metric ungameable.
QUALITATIVE_SLOTS: dict[str, tuple[str, ...]] = {
    "trend": ("increasing", "decreasing", "stable"),
    "seasonal": ("yes", "no"),
    "event": ("spike", "abrupt change", "no major event"),
    "position": ("early", "middle", "late"),
}

#: which slots each query type is scored on
SLOTS_FOR_TYPE: dict[str, tuple[str, ...]] = {
    "trend_description": ("trend", "seasonal"),
    "event_explanation": ("trend", "seasonal", "event", "position"),
}

#: C2 (PREREGISTRATION_C2.md §C2-4): score ONLY what the question asks.
#:
#: Cycle 1 scored `event_explanation` on four slots while the question asked
#: for two (the event and whether it was early/middle/late), and an
#: unmentioned slot counted as wrong. The harness mentioned trend and
#: seasonality only because its feature block listed them, so two of the four
#: slots rewarded reciting the tool output rather than answering. Measured on
#: cycle 1's 32B archive, asked-slots-only vs the full score:
#:     bare_llm 0.480 / 0.275   harness 0.760 / 0.520   oracle 0.845 / 0.568
SLOTS_FOR_TYPE_C2: dict[str, tuple[str, ...]] = {
    "trend_description": ("trend", "seasonal"),
    "event_explanation": ("event", "position"),
}


def fitted_slope(x: np.ndarray) -> tuple[float, float, float]:
    """OLS slope of x on time, with its standard error and t statistic.

    Returns (slope_per_step, standard_error, t). The whole window is used,
    not a segment.

    WHY THIS REPLACES THE CYCLE-1 RULE. `trend_direction` below reads two
    points - the first and last of the final third:

        slope = (seg[-1] - seg[0]) / (len(seg) - 1)

    Two points carry no averaging, so a single spike at either end sets the
    label. On cycle 1's own pool the `trend` generator pattern - which is
    always built with an upward slope - was labelled `stable` 8 times and
    `decreasing` once in 15 items. A model reading that series correctly was
    marked wrong in 9 of 15 cases. The key was mostly noise, which is the
    D7 Task B failure cycle 1 anticipated on `seasonal` and got on `trend`.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return 0.0, float("inf"), 0.0
    t = np.arange(n, dtype=float)
    tc = t - t.mean()
    denom = float(np.dot(tc, tc))
    if denom < REL_EPS:
        return 0.0, float("inf"), 0.0
    slope = float(np.dot(tc, x - x.mean()) / denom)
    resid = x - (x.mean() + slope * tc)
    dof = n - 2
    if dof <= 0:
        return slope, float("inf"), 0.0
    sigma2 = float(np.dot(resid, resid)) / dof
    se = float(np.sqrt(max(sigma2, 0.0) / denom))
    if se < REL_EPS:
        # Zero residual. t is infinite, and its SIGN matters: a
        # noiseless rising line is the most significant increase
        # there is, not an absence of one.
        if abs(slope) < REL_EPS:
            return slope, se, 0.0
        return slope, se, float("inf") * (1.0 if slope > 0 else -1.0)
    return slope, se, slope / se


#: Two-sided t critical value at alpha=0.05. Fixed rather than looked up:
#: every C2 window has n >= 32, so dof >= 30 and the 0.05 two-sided critical
#: value is within 0.05 of 1.96 for every length used. Declared in advance so
#: the threshold cannot be tuned after seeing labels.
TREND_T_CRIT = 1.96


def trend_direction_fitted(x: np.ndarray) -> str:
    """Computed trend label: sign of a SIGNIFICANT fitted slope, else 'stable'.

    Since amendment A9 this is NOT the C2 key (the generator's record is);
    it is the reference rule a rule-applier would use on the evidence.

    The decision rule - comparing t against TREND_T_CRIT - lives here and
    NOWHERE under systems/. The C2 tool reports `slope_per_step` and
    `slope_se` as numbers and must apply this threshold itself, which is the
    inference step the benchmark is supposed to measure. Cycle 1's tool
    called `trend_direction` directly and handed the model the label.
    """
    _, _, t = fitted_slope(x)
    # NaN is the only "no information" case. +/-inf is a zero-residual
    # fit - maximal significance - and compares correctly against the
    # critical value, so it must NOT be short-circuited to 'stable'.
    if np.isnan(t):
        return "stable"
    if t > TREND_T_CRIT:
        return "increasing"
    if t < -TREND_T_CRIT:
        return "decreasing"
    return "stable"


def detrend(x: np.ndarray) -> np.ndarray:
    """Remove the OLS linear fit. Required before any seasonality test.

    Autocorrelation of a trending series is high at EVERY lag - the trend
    dominates the covariance - so an ACF test on raw values reports trend as
    seasonality. On the archived pool the un-detrended C2 key called 28/30
    `trend` series and 28/28 `abrupt_change` series seasonal at L=512, and
    the effect grew with window length, so the slot's meaning drifted along
    H1's sweep axis.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return x - (x.mean() if n else 0.0)
    slope, _, _ = fitted_slope(x)
    t = np.arange(n, dtype=float)
    return x - (x.mean() + slope * (t - t.mean()))


def best_lag_acf(x: np.ndarray, max_lag: int | None = None) -> tuple[int, float]:
    """Lag in [2, max_lag] with the highest autocorrelation, and that value.

    Observable: uses only the window. This is the evidence the C2 tool
    reports, and the basis of the C2 `seasonal` key below.
    """
    x = np.asarray(x, float)
    n = len(x)
    cap = int(min(max_lag if max_lag is not None else n // 2, n // 2))
    if cap < 2:
        return 1, 0.0
    rs: dict[int, float] = {}
    for lag in range(2, cap + 1):
        a, b = x[:-lag], x[lag:]
        if a.std() < REL_EPS or b.std() < REL_EPS:
            continue
        rs[lag] = float(np.corrcoef(a, b)[0, 1])
    if not rs:
        return 1, 0.0
    best_r = max(rs.values())
    if best_r <= 0.0:
        return 1, best_r
    # Prefer the FUNDAMENTAL period, not a harmonic. A period-12 sine has
    # near-identical ACF at 12, 24 and 36, and argmax picks whichever wins by
    # floating-point noise - reporting `best_lag=36` for a 12-point cycle is
    # misleading evidence for the model to reason from. The smallest lag
    # within a small margin of the maximum is the fundamental.
    fundamental = min(l for l, r in rs.items() if r >= best_r - 0.02)
    return fundamental, rs[fundamental]


def best_lag_acf_detrended(x: np.ndarray,
                           max_lag: int | None = None) -> tuple[int, float]:
    """`best_lag_acf` on the detrended window. The C2 seasonality evidence."""
    return best_lag_acf(detrend(x), max_lag)


def acf_profile(x: np.ndarray, max_lag: int | None = None) -> dict:
    """Autocorrelation at every lag from 1 to max_lag, on the given series."""
    x = np.asarray(x, float)
    n = len(x)
    cap = int(min(max_lag if max_lag is not None else n // 2, n // 2))
    out: dict[int, float] = {}
    for lag in range(1, max(cap, 1) + 1):
        a, b = x[:-lag], x[lag:]
        if len(a) < 2 or a.std() < REL_EPS or b.std() < REL_EPS:
            continue
        out[lag] = float(np.corrcoef(a, b)[0, 1])
    return out


def seasonal_peak(x: np.ndarray) -> tuple[int, float]:
    """Smallest lag >= 2 that is a LOCAL MAXIMUM of the detrended ACF.

    Returns (lag, r), or (1, r_at_lag_1) when no interior peak exists.

    WHY A PEAK AND NOT A THRESHOLD. A persistent-but-aperiodic series has a
    smoothly decaying ACF, so its maximum over lags >= 2 is always lag 2 and
    "r > 0.5" fires on smoothness. Measured on the archived pool, 18 of 82
    `seasonal=yes` labels at L=32 and 22 of 98 at L=512 sat at lag 2. A
    genuine cycle makes the ACF rise into its period and fall after, which
    monotone decay cannot imitate.
    """
    prof = acf_profile(detrend(x))
    if not prof:
        return 1, 0.0
    lags = sorted(prof)
    best = (1, prof.get(1, 0.0))
    for k in lags:
        if k < 2:
            continue
        prev, nxt = prof.get(k - 1), prof.get(k + 1)
        if prev is None:
            continue
        rises = prof[k] > prev
        falls = nxt is None or prof[k] >= nxt
        if rises and falls:
            return k, prof[k]
    return best


def is_seasonal_observable(x: np.ndarray) -> bool:
    """Computed seasonal label: strong ACF at the best OBSERVABLE lag.

    Since amendment A9 this is NOT the C2 key (the generator's record is);
    it is the reference rule a rule-applier would use on the evidence.
    The history below is why it stopped being the key.

    Cycle 1's key was `is_seasonal(x, sample.seasonal_period)` - the
    generator's TRUE period, which cycle 1's own §4.1 warned "encodes
    information absent from the data". A key no reader of the series can
    reach does not make the slot hard, it makes it unanswerable, and an
    unanswerable slot cannot carry a confirmatory claim. The true-period
    version is kept as a declared SECONDARY (`seasonal_true` below).
    """
    # DETRENDED (amendment A2). A PEAK requirement was implemented and
    # MEASURED (A6) after review #3 observed that a lag-2 maximum is
    # persistence rather than a cycle. The observation is right, but
    # the remedy is worse: against the generator's own
    # `seasonal_period > 1`, requiring a local maximum takes precision
    # to 1.00 at L<=128 and drops recall to 0.59-0.69, so F1 falls at
    # every length (0.88->0.78 at L=32, 0.84->0.57 at L=512).
    # `seasonal_peak` is kept as evidence and as the record of that
    # measurement; the key stays on the detrended maximum.
    return best_lag_acf_detrended(x)[1] > 0.5


def qualitative_computed_c2(x: np.ndarray) -> dict:
    """Labels COMPUTED from a window by the rules above. NOT the C2 key.

    This was the C2 key in the first draft. Reviews #2-#4 showed a
    rule-applier reading the tool's evidence reproduced it on 100% of items
    for every slot, i.e. the cells were saturated, so amendment A9 replaced
    it with the generator's record (`qualitative_truth_c2`). It is kept as
    the REFERENCE RULE-APPLIER: `tests/test_c2_saturation_rate.py` scores it
    against the real key to show the evidence no longer determines the
    answer, and `MockLLM` uses it as its simulated "reading" of a series.
    """
    x = np.asarray(x, float)
    idx = change_point_index(x)
    if count_spikes(x) >= 1:
        event = "spike"
    elif has_level_shift(x, idx):
        event = "abrupt change"
    else:
        event = "no major event"
    return dict(trend=trend_direction_fitted(x),
                seasonal="yes" if is_seasonal_observable(x) else "no",
                event=event, position=event_position(idx, len(x)))


#: an injected event counts as visible only with this many points before it
#: inside the window; a shift at the very first point is not observable
C2_MIN_POINTS_BEFORE_EVENT = 2


def c2_event_in_window(truth: dict, max_history: int,
                       L: int) -> tuple[str, int | None]:
    """(event class, index local to window(L)) for a generator record.

    An event that lies before the window start is, for that window, not an
    event: the correct answer is `no major event`. This containment rule is
    why H1's qualitative cells are descriptive in C2 - the same item can
    hold its event at one L and not at another.
    """
    event, idx = truth["event"], truth.get("event_index")
    if event == "no major event" or idx is None:
        return "no major event", None
    local = int(idx) - (int(max_history) - int(L))
    if local < C2_MIN_POINTS_BEFORE_EVENT or local > L - 2:
        return "no major event", None
    return event, local


def qualitative_truth_c2(sample: Sample, L: int) -> QualitativeTruth:
    """C2 answer key: WHAT THE GENERATOR INJECTED (amendment A9).

      trend     the injected drift direction, or `stable` if none
      seasonal  whether a seasonal cycle was injected
      event     the injected event if window(L) contains it, else
                `no major event`
      position  the third of window(L) holding that event; None (and not
                scored) when there is no event

    Nothing here is computed from the window, so no tool reading the window
    can reproduce it by construction - which is the property cycle 1 and the
    first C2 draft lacked. Requires a pool generated with `design: c2`.
    """
    truth = (sample.metadata or {}).get("truth")
    if truth is None:
        raise ValueError(
            f"sample {sample.sample_id} carries no generator record. The C2 "
            "key needs a pool built with data.design: c2; a cycle-1 pool "
            "cannot be scored with it.")
    trend = truth["trend"]
    seasonal = "yes" if truth["seasonal"] else "no"
    if sample.query_type == "trend_description":
        return QualitativeTruth(trend=trend, seasonal=seasonal)
    event, local = c2_event_in_window(truth, sample.max_history, L)
    position = event_position(local, L) if local is not None else None
    return QualitativeTruth(trend=trend, seasonal=seasonal, event=event,
                            position=position)


def c2_scored_slots(truth: "QualitativeTruth", query_type: str) -> dict:
    """slot -> expected class, for the slots actually scored on this item.

    `position` is scored only when there is an event to locate; asking where
    nothing happened has no right answer, so it is not counted either way.
    """
    out = {}
    for name in SLOTS_FOR_TYPE_C2[query_type]:
        value = getattr(truth, name)
        if value is None:
            if name == "position":
                continue
            raise ValueError(f"slot {name!r} missing for {query_type!r}")
        out[name] = value
    return out


def seasonal_true(sample: Sample, L: int) -> str:
    """The cycle-1 `seasonal` key, retained as a declared C2 SECONDARY."""
    return "yes" if is_seasonal(sample.window(L), sample.seasonal_period) \
        else "no"


@dataclass(frozen=True)
class QualitativeTruth:
    trend: str
    seasonal: str                 # "yes" | "no"
    event: str | None = None
    position: str | None = None

    def slots(self, query_type: str) -> dict[str, str]:
        """The slots actually scored for this query type."""
        out: dict[str, str] = {}
        for name in SLOTS_FOR_TYPE[query_type]:
            value = getattr(self, name)
            if value is None:
                raise ValueError(f"slot {name!r} missing for {query_type!r}")
            out[name] = value
        return out

    def as_dict(self) -> dict:
        return asdict(self)


def qualitative_truth(sample: Sample, L: int) -> QualitativeTruth:
    x = sample.window(L)
    trend = trend_direction(x)
    seasonal = "yes" if is_seasonal(x, sample.seasonal_period) else "no"
    if sample.query_type == "trend_description":
        return QualitativeTruth(trend=trend, seasonal=seasonal)

    idx = change_point_index(x)
    if count_spikes(x) >= 1:
        event = "spike"
    elif has_level_shift(x, idx):
        event = "abrupt change"
    else:
        event = "no major event"
    return QualitativeTruth(trend=trend, seasonal=seasonal, event=event,
                            position=event_position(idx, len(x)))


# --------------------------------------------------------------------------
# the answer key
# --------------------------------------------------------------------------
def ground_truth(sample: Sample, L: int) -> Any:
    """The answer key for (sample, L)."""
    x = sample.window(L)
    qt = sample.query_type
    if qt == "forecasting":
        return list(sample.target())
    if qt == "statistics":
        return float(np.mean(x))
    if qt == "counting":
        return int(count_spikes(x))
    if qt == "extremum":
        return float(np.max(x))
    if qt == "change":
        return pct_change(x)
    if qt == "correlation":
        return lag1_autocorr(x)
    if qt == "localization":
        return int(change_point_index(x))
    if qt in ("trend_description", "event_explanation"):
        return qualitative_truth(sample, L)
    raise ValueError(f"no ground truth defined for query type {qt!r}")
