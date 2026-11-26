"""Saturation guard — PREREGISTRATION_C2.md §C2-10.

THE DEFECT THIS EXISTS TO CATCH
-------------------------------
Cycle 1 declared, in `src/config.py`:

    TOOL_SATURATED_OPS = frozenset({"mean", ..., "change_point"})
    #: Operations whose tool IS the answer-key function ... it follows from
    #: `systems/tools.py` importing `src.ground_truth`

`describe` and `explain` were absent from that set, yet
`systems/tools.py::series_features` called `gt.trend_direction`,
`gt.count_spikes`, `gt.has_level_shift` and `gt.event_position` - the same
functions, on the same window, that `qualitative_truth` uses to build the
key - and the harness was shown that block INSTEAD of the series. Measured on
cycle 1's archive: harness and oracle both scored 100/100 on the `trend` slot
at 32B against 23/100 for the bare arm.

Nothing failed. The inconsistency between a declared criterion and its
implementation was invisible to the test suite, and ~12,400 GPU completions
were spent before an external reviewer found it.

These tests make that failure mode loud.
"""
from __future__ import annotations

import ast
import inspect
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.qualitative import score_qualitative_mc          # noqa: E402
from models import prompts as LM                                 # noqa: E402
from src import ground_truth as gt                               # noqa: E402
from src.ground_truth import (QUALITATIVE_SLOTS, SLOTS_FOR_TYPE_C2,  # noqa: E402
                              QualitativeTruth)
from systems import tools                                        # noqa: E402

#: Functions that turn evidence into a CLASS. A system that calls one of
#: these is not inferring, it is reading the key.
KEY_DECISION_FUNCTIONS = (
    "trend_direction",
    "trend_direction_fitted",
    "is_seasonal",
    "is_seasonal_observable",
    "event_position",
    "has_level_shift",
    "qualitative_truth",
    "qualitative_truth_c2",
    "qualitative_computed_c2",
    "c2_event_in_window",
    "seasonal_true",
)

ALL_CLASSES = {c.lower() for cs in QUALITATIVE_SLOTS.values() for c in cs}


def _pool(n: int = 60, L: int = 128, seed: int = 7):
    """A small set of windows spanning the generator's shapes."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t = np.arange(L, dtype=float)
        kind = i % 6
        if kind == 0:
            x = 0.4 * t
        elif kind == 1:
            x = -0.3 * t
        elif kind == 2:
            x = np.zeros(L)
        elif kind == 3:
            x = 8 * np.sin(2 * np.pi * t / 12)
        elif kind == 4:
            x = np.zeros(L)
            x[L // 2:] += 20
        else:
            x = np.zeros(L)
            x[int(0.7 * L)] += 40
        out.append(x + 50 + rng.normal(0, 1.5, L))
    return out


# ---------------------------------------------------------------------------
# 1. the C2 evidence must not BE a verdict
# ---------------------------------------------------------------------------
def test_c2_evidence_is_numeric_only():
    """No evidence field may hold a slot class, or be non-numeric."""
    for x in _pool():
        e = tools.series_evidence(x)
        assert set(e) == set(tools.C2_EVIDENCE_FIELDS)
        for k, v in e.items():
            assert isinstance(v, (int, float)) and not isinstance(v, bool), (
                f"evidence field {k!r} is {type(v).__name__}; C2 evidence "
                "must be numeric, because any categorical field risks being "
                "the key's own vocabulary")
            assert str(v).lower() not in ALL_CLASSES


def _called_names(fn) -> set[str]:
    """Function names CALLED inside `fn`, from its AST.

    Deliberately not a substring scan over `inspect.getsource`: that text
    includes comments and docstrings, so explaining why a key function
    matters tripped the guard meant to catch calling it. A guard that fires
    on its own documentation gets deleted by the next person to hit it.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute):
            out.add(f.attr)
        elif isinstance(f, ast.Name):
            out.add(f.id)
    return out


def test_c2_evidence_does_not_call_key_decision_functions():
    """Static check on the C2 feature path.

    Scoped to `series_evidence` on purpose: cycle 1's `series_features` does
    call these, and it is kept verbatim so cycle 1 stays reproducible. The
    guard is that the C2 path does not.
    """
    called = _called_names(tools.series_evidence) | _called_names(
        tools.evidence_to_block)
    leaked = sorted(set(KEY_DECISION_FUNCTIONS) & called)
    assert not leaked, (
        f"systems/tools.py::series_evidence calls {leaked}, which are key "
        "DECISION functions. That is the cycle-1 defect: the tool would hand "
        "the model the answer instead of the evidence for it.")


def test_cycle1_features_would_fail_this_guard():
    """The guard must actually be able to go red.

    A test that cannot fail is worthless - the same argument the cycle-1 null
    fixture made about its positive control. Cycle 1's `series_features` is
    the known-bad case, so it must trip the same CALL check.
    """
    hits = sorted(set(KEY_DECISION_FUNCTIONS)
                  & _called_names(tools.series_features))
    assert hits, ("cycle 1's series_features no longer CALLS any key "
                  "decision function - if it was deliberately changed, this "
                  "guard has lost its known-bad case and needs a new one")


# ---------------------------------------------------------------------------
# 2. the evidence must not be COPYABLE into a correct answer
# ---------------------------------------------------------------------------
def _class_tokens_in(text: str) -> set[str]:
    """Slot class names appearing as whole words in `text`."""
    low = text.lower()
    return {c for c in ALL_CLASSES
            if re.search(rf"\b{re.escape(c)}\b", low)}


def test_c2_evidence_block_contains_no_scoring_vocabulary():
    """The block must not name a single option the model is asked to choose.

    This is the operational form of "evidence, not a verdict". It is a
    CONTENT check, not a format check: an earlier version of this test
    asserted that copying the block scores 0 under the MC scorer, which both
    cycle 1's block and C2's satisfy - the MC scorer wants `slot: value` and
    cycle 1 wrote `trend=increasing`. That version passed for the known-bad
    case, so it proved nothing.
    """
    for x in _pool(n=24):
        block = tools.evidence_to_block(tools.series_evidence(x))
        leaked = _class_tokens_in(block)
        assert not leaked, (
            f"C2 evidence block names scoring option(s) {sorted(leaked)}:\n"
            f"{block}")


def test_cycle1_feature_block_leaks_scoring_vocabulary():
    """The known-bad case must trip the check above.

    Cycle 1's block reads `trend=increasing seasonal=yes ... position=early`
    - every one of those is a slot class the model was then scored on for
    repeating, while `assist_qualitative` hid the series from it.
    """
    leaked_any = set()
    for x in _pool(n=24):
        block = tools.features_to_block(tools.series_features(x))
        leaked_any |= _class_tokens_in(block)
    assert leaked_any, (
        "cycle 1's feature block no longer names any slot class; this guard "
        "has lost the known-bad case that justifies it")


def test_copying_the_evidence_block_scores_zero():
    """Weaker companion to the content check, kept for the end-to-end path."""
    for x in _pool(n=12):
        truth = QualitativeTruth(
            trend=gt.trend_direction_fitted(x),
            seasonal="yes" if gt.is_seasonal_observable(x) else "no",
            event="spike", position="late")
        block = tools.evidence_to_block(tools.series_evidence(x))
        for qt in SLOTS_FOR_TYPE_C2:
            assert score_qualitative_mc(block, truth, qt)["score"] == 0.0


# ---------------------------------------------------------------------------
# 3. (removed) stated rules
# ---------------------------------------------------------------------------
# Section 3 used to pin the thresholds quoted in `QUAL_RULES` to the key's
# code. Amendment A9 replaced the computed key with the generator's record,
# so there is no threshold to quote: the question now defines each option in
# words (src/samples.py, C2_CLAUSE; pinned in tests/test_prompts.py) and the
# options-match-the-scorer check lives in tests/test_c2_scoring.py.


# ---------------------------------------------------------------------------
# 4. the fitted-slope EVIDENCE must not be the two-point rule
# ---------------------------------------------------------------------------
def test_c2_trend_key_is_robust_where_cycle1_was_not():
    """§1.5: cycle 1 labelled flat noise by reading two points.

    On a pure-noise series the two-point rule returns a direction far more
    often than a fitted slope does. The C2 key must call it `stable`
    essentially always; cycle 1's must not.
    """
    rng = np.random.default_rng(11)
    flat = [rng.normal(50, 3, 128) for _ in range(120)]
    c2_stable = sum(gt.trend_direction_fitted(x) == "stable" for x in flat)
    c1_stable = sum(gt.trend_direction(x) == "stable" for x in flat)
    assert c2_stable >= 114, (
        f"C2 trend key called pure noise 'stable' only {c2_stable}/120 times")
    assert c1_stable < c2_stable, (
        "cycle 1's two-point rule is no longer the weaker one; this guard "
        "has lost its known-bad case")


def test_c2_trend_key_still_detects_a_real_slope():
    """Robustness must not become insensitivity."""
    rng = np.random.default_rng(12)
    for slope, want in ((0.4, "increasing"), (-0.4, "decreasing")):
        hits = sum(
            gt.trend_direction_fitted(slope * np.arange(128)
                                      + rng.normal(0, 3, 128)) == want
            for _ in range(40))
        assert hits == 40, f"{want}: only {hits}/40 detected"


def test_c2_trend_key_handles_a_noiseless_slope():
    """A zero-residual fit gives t = +/-inf; that is certainty, not absence.

    REGRESSION. The first version of `trend_direction_fitted` read

        if not np.isfinite(t): return "stable"

    so a perfectly straight rising line - the most significant slope
    possible - was labelled `stable`. The generator can emit a noiseless
    `trend` pattern, and every test pool here adds noise, so nothing caught
    it until the symbols were exercised by hand.
    """
    assert gt.trend_direction_fitted(np.arange(50.0)) == "increasing"
    assert gt.trend_direction_fitted(-np.arange(50.0)) == "decreasing"
    assert gt.trend_direction_fitted(np.full(50, 7.0)) == "stable"

    # sign must survive into fitted_slope itself
    assert gt.fitted_slope(np.arange(50.0))[2] == float("inf")
    assert gt.fitted_slope(-np.arange(50.0))[2] == float("-inf")

    # a slope below REL_EPS is numerically flat, not a direction
    assert gt.trend_direction_fitted(np.arange(50.0) * 1e-12) == "stable"
