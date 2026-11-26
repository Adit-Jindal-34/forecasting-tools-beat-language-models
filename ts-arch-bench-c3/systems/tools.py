"""Deterministic tools available to the harness, plus period estimation.

Each tool takes the visible window and returns an exact numeric result.
Exactness here is the point: it is what lets a tool-augmented harness win
exact numeric queries by construction, and therefore also what makes those
cells uninformative (see src/screening.py).

PERIOD ESTIMATION IS NOT FREE
-----------------------------
The old build passed `sample.seasonal_period` - a GENERATOR PARAMETER - into
the harness's feature extractor, so the harness was handed the true period
while the bare LLM had to infer it. That is the same class of asymmetry as
the `op=` prompt hint. Here the harness ESTIMATES the period from the window
like any deployed system would; only the answer key may use the true one.
"""
from __future__ import annotations

import numpy as np

from src import ground_truth as gt

# op name -> (function, returns_integer)
TOOLS: dict = {
    "mean": (lambda x: float(np.mean(x)), False),
    "median": (lambda x: float(np.median(x)), False),
    "std": (lambda x: float(np.std(x)), False),
    "max": (lambda x: float(np.max(x)), False),
    "min": (lambda x: float(np.min(x)), False),
    "count_spikes": (lambda x: int(gt.count_spikes(x)), True),
    "pct_change": (lambda x: gt.pct_change(x), False),
    "lag1_autocorr": (lambda x: gt.lag1_autocorr(x), False),
    "change_point": (lambda x: int(gt.change_point_index(x)), True),
}

TOOL_OPS = frozenset(TOOLS)


def run_tool(op: str, window: np.ndarray) -> tuple:
    """Run a deterministic tool. Returns (value, returns_integer)."""
    if op not in TOOLS:
        raise KeyError(f"unknown tool {op!r}; available: {sorted(TOOL_OPS)}")
    fn, is_int = TOOLS[op]
    return fn(np.asarray(window, float)), is_int


def estimate_period(x: np.ndarray, max_period: int | None = None) -> int:
    """Estimate the dominant seasonal period from the window alone.

    Returns the lag in [2, max_period] with the highest autocorrelation, or 1
    when nothing clears the threshold. Uses only observable data.
    """
    x = np.asarray(x, float)
    n = len(x)
    cap = max_period or max(2, n // 3)
    cap = min(cap, n // 2)
    if n < 8 or cap < 2:
        return 1
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom < gt.REL_EPS:
        return 1
    best_lag, best_r = 1, 0.0
    for lag in range(2, cap + 1):
        r = float(np.dot(xc[:-lag], xc[lag:])) / denom
        if r > best_r:
            best_lag, best_r = lag, r
    return best_lag if best_r > 0.3 else 1


def series_features(window: np.ndarray) -> dict:
    """Feature bundle the harness hands to the LLM for qualitative queries.

    Everything here is computed from the window. No generator parameters.
    """
    x = np.asarray(window, float)
    period = estimate_period(x)
    idx = gt.change_point_index(x)
    has_spike = gt.count_spikes(x) >= 1
    has_shift = gt.has_level_shift(x, idx)
    return dict(
        trend=gt.trend_direction(x),
        seasonal=gt.is_seasonal(x, period),
        estimated_period=period,
        spike=has_spike,
        abrupt_change=(has_shift and not has_spike),
        position=gt.event_position(idx, len(x)),
    )


def features_to_block(f: dict) -> str:
    return ("computed features: "
            f"trend={f['trend']} "
            f"seasonal={'yes' if f['seasonal'] else 'no'} "
            f"estimated_period={f['estimated_period']} "
            f"spike={'yes' if f['spike'] else 'no'} "
            f"abrupt_change={'yes' if f['abrupt_change'] else 'no'} "
            f"position={f['position']}")

# --------------------------------------------------------------------------
# C2: evidence, not verdicts  (PREREGISTRATION_C2.md §C2-1)
# --------------------------------------------------------------------------
# WHY THIS EXISTS. `series_features` above returns `trend=increasing`,
# `seasonal=yes`, `spike=yes`, `position=early` - the exact class values the
# answer key uses, computed by calling the key's own functions on the key's
# own input. With `assist_qualitative` the model was shown that block INSTEAD
# of the series, so it only had to copy. Measured on cycle 1's archive: the
# harness and the oracle arm each scored 100/100 on the `trend` slot at 32B
# against 23/100 for the bare arm, and the oracle held 93-100% at every rung
# of the ladder. That is not an architecture beating a baseline, it is a
# metric reading itself.
#
# C2 reports NUMBERS ONLY, and (amendment A9) the answer key is no longer a
# function of these numbers at all: it is the generator's record of what was
# injected. So the block is EVIDENCE in the ordinary sense - usually pointing
# the right way, sometimes not (autocorrelated noise inflates the slope t; a
# level shift produces a slope and long-lag autocorrelation; a spike on a
# seasonal trough has a small robust z) - and reaching a label takes a
# judgement the model has to make with the series in front of it.
# `tests/test_c2_saturation_rate.py` measures how often a mechanical
# rule-applier reading only this block matches the key, and fails the build
# if that reaches 0.95 on any confirmatory cell.
#
# Nothing here calls a key DECISION function.
# `tests/test_saturation_guard.py::test_c2_evidence_does_not_call_key_decision_functions` enforces that.
# (An earlier comment credited `tests/test_leakage.py`, which checks
# oracle-label imports and never had this check - review #3, claim 5.)

#: Fields a C2 evidence block may contain. Asserted by the leakage test, so
#: adding a categorical field cannot pass review by accident.
C2_EVIDENCE_FIELDS: tuple[str, ...] = (
    "n_points", "mean", "std", "noise_scale",
    "slope_per_step", "slope_se", "slope_t",
    "best_lag", "acf_at_best_lag", "acf_at_lag_1",
    "max_abs_step", "max_abs_step_index", "max_abs_step_frac",
    "step_over_noise", "level_shift_over_noise",
    "max_local_peak_robust_z",
    # cycle 3, 1 or 0. See SCALE_FLOOR below. Numeric, like every other
    # field, so the "evidence is numbers, never the key's vocabulary"
    # guard in tests/test_saturation_guard.py still holds.
    "robust_scale_usable",
)

#: Below this the robust scale has collapsed and every quantity DIVIDED by it
#: is meaningless.
#:
#: WHY THIS IS NEEDED AT ALL. `gt._robust_z` divides by `1.4826*MAD +
#: gt.REL_EPS` and `gt.local_noise_scale` bottoms out at `gt.REL_EPS = 1e-9`.
#: On a flat stretch - a traffic loop reading 0 all night, which real sensor
#: feeds do routinely - the division explodes. Measured on real Monash data,
#: `traffic_hourly-T12` at L=32 rendered
#:
#:     noise_scale=1e-09  largest_local_peak_robust_z=35800000.00
#:
#: into the prompt. It raised nothing, failed no test and showed up in no
#: coverage column: the item simply scored, silently, on garbage evidence.
#: The synthetic generator always injects noise, so this could not fire in
#: cycle 1 or cycle 2.
#:
#: The PRIMARY fix is in the pool, not here: `data-real/build_pool.py`
#: rejects a series if ANY L in the sweep is degenerate (4.8% of candidates).
#: This is the second line - the next pool builder may not apply that rule,
#: and the next archive will do this again.
#:
#: WHY THE VALUE IS SAFE FOR THE CACHE. The flag only changes what
#: `evidence_to_block` RENDERS, and only when it is 0. Cycle 2's noise sigma
#: is drawn from [0.5, 2.0] and the real pool's smallest surviving noise
#: scale is 1.7e-3, so both sit six orders of magnitude above this floor and
#: every existing prompt renders byte-identically. The cycle-2 call cache
#: therefore still replays exactly - which matters, because
#: `evidence_to_block` output is inside the cache key.
SCALE_FLOOR = 1e-6


def _max_local_peak_z(x: np.ndarray, z: np.ndarray) -> float:
    """Largest robust z among interior local maxima; 0.0 if none."""
    best = 0.0
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1]:
            best = max(best, float(z[i]))
    return best


def series_evidence(window: np.ndarray) -> dict:
    """Numeric evidence for the qualitative queries. No labels, no verdicts.

    Everything is computed from the window. No generator parameters.
    """
    x = np.asarray(window, float)
    n = len(x)
    slope, se, t = gt.fitted_slope(x)
    # Detrended (A2): the raw ACF of a trending series is high at
    # every lag, so un-detrended evidence would point the model at
    # the trend when asked about seasonality. This must report the
    # SAME quantity the key thresholds, or the slot is unanswerable -
    # which is why it reverted with the key when the peak variant
    # (A6) was measured and rejected.
    best_lag, acf_best = gt.best_lag_acf_detrended(x)
    idx = gt.change_point_index(x)
    noise = gt.local_noise_scale(x)
    step = float(abs(np.diff(x)[idx])) if n >= 2 else 0.0
    z = gt._robust_z(x)
    return dict(
        n_points=n,
        mean=float(np.mean(x)),
        std=float(np.std(x)),
        noise_scale=float(noise),
        slope_per_step=float(slope),
        slope_se=float(se),
        slope_t=float(t),
        best_lag=int(best_lag),
        acf_at_best_lag=float(acf_best),
        # Detrended too (review #3, claim 5): a raw lag-1 value is
        # dominated by the trend and would contradict the detrended
        # peak reported beside it.
        acf_at_lag_1=float(gt.lag1_autocorr(gt.detrend(x))),
        max_abs_step=step,
        max_abs_step_index=int(idx),
        max_abs_step_frac=float(idx / max(n - 1, 1)),
        step_over_noise=float(step / noise) if noise > gt.REL_EPS else 0.0,
        # The robust z of the largest INTERIOR LOCAL MAXIMUM, which
        # is exactly the set `gt.count_spikes` scans. Reporting the
        # max over ALL points would be unusable evidence: the key's
        # rule is "a local maximum above median + 3*MAD", so a model
        # told that rule but handed the global max could apply it
        # correctly and still be wrong.
        # The key's ACTUAL abrupt-change quantity: `has_level_shift`
        # compares the mean of the 4 points after the step with the 4
        # before, NOT the one-step change. The prompt previously stated
        # the one-step rule, which the key never used - of the items the
        # key calls `abrupt change`, that rule agreed on 0 of 44 at
        # L=32 (amendment A1). Reporting the real quantity is what lets
        # the stated rule and the key be the same function.
        level_shift_over_noise=(
            float(gt.level_shift(x, idx) / noise)
            if noise > gt.REL_EPS else 0.0),
        max_local_peak_robust_z=_max_local_peak_z(x, z),
        # 1 normally. 0 when the window is flat enough that `1.4826*MAD` or
        # the point-to-point noise scale has underflowed to `gt.REL_EPS`,
        # which makes every ratio below a division by epsilon. The raw
        # values stay in the dict - they are diagnostics, and the test
        # harness reads them - but `evidence_to_block` refuses to put them
        # in a prompt. See SCALE_FLOOR.
        robust_scale_usable=int(
            float(np.median(np.abs(x - np.median(x)))) > SCALE_FLOOR
            and float(noise) > SCALE_FLOOR),
    )


#: What replaces a scale-divided number when the scale underflowed.
#: Deliberately a WORD, not a sentinel number: a model handed `-1` or `0`
#: would reason with it, and a model handed `35800000.00` already did.
UNUSABLE = "unavailable_flat_window"


def evidence_to_block(e: dict) -> str:
    """Render evidence for the prompt. Fixed field order, fixed precision.

    Deterministic text matters: the call cache is keyed on the prompt, so a
    dict-ordering change would silently invalidate every cached reply. For
    the same reason the degenerate branch below is ADDITIVE: when
    `robust_scale_usable` is 1 - every synthetic window ever generated, and
    every item in the real pool after its exclusion rule - the returned
    string is byte-for-byte what cycle 2 produced.
    """
    if not int(e.get("robust_scale_usable", 1)):
        # Flat window: report the scale-free facts, say plainly that the
        # rest could not be computed, and emit nothing a model could
        # mistake for a measurement.
        return ("computed evidence (from the visible window only):\n"
                f"  points={e['n_points']} mean={e['mean']:.4g} "
                f"sd={e['std']:.4g} noise_scale={UNUSABLE}\n"
                f"  fitted_slope_per_step={e['slope_per_step']:.4g} "
                f"standard_error={e['slope_se']:.4g} "
                f"t={e['slope_t']:.3f}\n"
                f"  best_lag={e['best_lag']} "
                f"autocorr_at_best_lag={e['acf_at_best_lag']:.3f} "
                f"autocorr_at_lag_1={e['acf_at_lag_1']:.3f}\n"
                f"  largest_one_step_change={e['max_abs_step']:.4g} "
                f"at_index={e['max_abs_step_index']} "
                f"({e['max_abs_step_frac']:.2f} of the way through)\n"
                f"  that_change_over_noise_scale={UNUSABLE}"
                f" level_shift_over_noise={UNUSABLE}\n"
                f"  largest_local_peak_robust_z={UNUSABLE}\n"
                "  note: this window is almost constant, so the robust "
                "scale these ratios divide by underflowed. Read the series "
                "itself; the missing ratios are not small, they are "
                "undefined.")
    return ("computed evidence (from the visible window only):\n"
            f"  points={e['n_points']} mean={e['mean']:.4g} "
            f"sd={e['std']:.4g} noise_scale={e['noise_scale']:.4g}\n"
            f"  fitted_slope_per_step={e['slope_per_step']:.4g} "
            f"standard_error={e['slope_se']:.4g} "
            f"t={e['slope_t']:.3f}\n"
            f"  best_lag={e['best_lag']} "
            f"autocorr_at_best_lag={e['acf_at_best_lag']:.3f} "
            f"autocorr_at_lag_1={e['acf_at_lag_1']:.3f}\n"
            f"  largest_one_step_change={e['max_abs_step']:.4g} "
            f"at_index={e['max_abs_step_index']} "
            f"({e['max_abs_step_frac']:.2f} of the way through)\n"
            f"  that_change_over_noise_scale={e['step_over_noise']:.2f}"
            f" level_shift_over_noise="
            f"{e['level_shift_over_noise']:.2f}\n"
            f"  "
            f"largest_local_peak_robust_z="
            f"{e['max_local_peak_robust_z']:.2f}")
