"""Metric validity: the gaming attack, scale comparability, failure accounting."""
from __future__ import annotations

import numpy as np
import pytest

from evaluation import metrics as M
from evaluation.qualitative import (score_qualitative, extract_slot,
                                    GAMING_PROBE)
from src.ground_truth import QualitativeTruth
from conftest import toy_window


# --------------------------------------------------------------------------
# the defect that scored 1.000
# --------------------------------------------------------------------------
ALL_TRUTHS = [
    ("trend_description", QualitativeTruth(t, s))
    for t in ("increasing", "decreasing", "stable") for s in ("yes", "no")
] + [
    ("event_explanation", QualitativeTruth(t, s, e, p))
    for t in ("increasing", "decreasing", "stable") for s in ("yes", "no")
    for e in ("spike", "abrupt change", "no major event")
    for p in ("early", "middle", "late")
]


@pytest.mark.parametrize("query_type", ["trend_description",
                                        "event_explanation"])
def test_constant_contradictory_answer_cannot_score_well(query_type):
    """Under the old coverage metric this exact string scored mean 1.0000
    (min 1.0000) on every item of both qualitative cells. Averaged over all
    possible truths it must now be near the floor.

    It cannot reach exactly zero: `seasonal` is a presence/absence slot, so
    a contradictory answer still reads as "yes" there and collects chance
    accuracy. That residual is documented in evaluation/qualitative.py and
    is why human validation is still required.
    """
    truths = [t for qt, t in ALL_TRUTHS if qt == query_type]
    scores = [score_qualitative(GAMING_PROBE, t, query_type)["score"]
              for t in truths]
    mean = sum(scores) / len(scores)
    assert mean <= 0.25, (
        f"the gaming probe averages {mean:.3f} on {query_type}; the metric "
        "still rewards an answer that asserts everything")


@pytest.mark.parametrize("query_type", ["trend_description",
                                        "event_explanation"])
def test_contradictory_answer_is_flagged_incoherent(query_type):
    truth = [t for qt, t in ALL_TRUTHS if qt == query_type][0]
    rec = score_qualitative(GAMING_PROBE, truth, query_type)
    assert rec["incoherent"] == 1
    assert rec["n_ambiguous"] >= 1
    assert rec["per_slot"]["trend"]["status"] == "ambiguous"


def test_exclusive_slots_detect_contradiction():
    p = extract_slot("the series is both increasing and decreasing", "trend")
    assert p.status == "ambiguous" and p.predicted is None


def test_correct_answer_scores_one():
    truth = QualitativeTruth("increasing", "no", "spike", "early")
    answer = ("The series rises steadily and is not seasonal; a sudden spike "
              "occurs early in the window.")
    assert score_qualitative(answer, truth, "event_explanation")["score"] == 1.0


def test_negation_beats_presence_on_the_seasonal_slot():
    assert extract_slot("clearly not seasonal", "seasonal").predicted == "no"
    assert extract_slot("a strong seasonal cycle", "seasonal").predicted == "yes"


def test_empty_and_irrelevant_answers_score_zero():
    truth = QualitativeTruth("increasing", "no")
    for bad in ("", "I don't know.", "42"):
        assert score_qualitative(bad, truth, "trend_description")["score"] == 0.0


# --------------------------------------------------------------------------
# numeric scale comparability
# --------------------------------------------------------------------------
def test_natural_scale_never_divides_by_the_target():
    """The old metric divided by |gt|, so a near-zero truth exploded the
    error. Scale must depend only on the window."""
    x = toy_window(128)
    for op in ("mean", "max", "min", "median", "std"):
        assert M.natural_scale(op, x) == pytest.approx(max(np.std(x), 1e-9))
    assert M.natural_scale("lag1_autocorr", x) == 1.0
    with pytest.raises(ValueError):
        M.natural_scale("count_spikes", x)


def test_same_physical_error_costs_the_same_across_numeric_cells():
    """The core comparability property. In the old build an error equal to 2%
    of the series scale cost 0.0005 in `change` and up to 0.95 in
    `statistics`; here the two must agree."""
    x = toy_window(128)
    sd = float(np.std(x))
    truth_mean = float(np.mean(x))
    s_mean = M.score_numeric(truth_mean + 0.02 * sd, truth_mean, "mean", x)

    anchor = float(x[0])
    pct_scale = 100.0 * sd / abs(anchor)
    truth_pct = 2.5
    s_pct = M.score_numeric(truth_pct + 0.02 * pct_scale, truth_pct,
                            "pct_change", x)

    assert s_mean["scaled_err"] == pytest.approx(0.02, abs=1e-9)
    assert s_pct["scaled_err"] == pytest.approx(0.02, abs=1e-9)
    assert s_mean["score"] == pytest.approx(s_pct["score"], abs=1e-9)


def test_near_zero_truth_does_not_explode_the_score():
    """A mean of ~0 used to make any error look catastrophic."""
    x = toy_window(128) - np.mean(toy_window(128))     # centred: mean ~ 0
    truth = float(np.mean(x))
    rec = M.score_numeric(truth + 0.01 * float(np.std(x)), truth, "mean", x)
    assert rec["scaled_err"] < 0.02 and rec["score"] > 0.95


def test_counting_uses_exact_match_not_relative_error():
    x = toy_window(128)
    assert M.score_numeric(3, 3, "count_spikes", x)["score"] == 1.0
    assert M.score_numeric(4, 3, "count_spikes", x)["score"] == 0.0
    # the old pathology: gt == 0 made every nonzero answer score exactly 0
    # while 0 scored exactly 1, silently turning the cell into exact match
    assert M.score_numeric(0, 0, "count_spikes", x)["score"] == 1.0


def test_bounded_quantity_uses_absolute_error():
    x = toy_window(128)
    rec = M.score_numeric(0.55, 0.50, "lag1_autocorr", x)
    assert rec["scaled_err"] == pytest.approx(0.05)
    assert rec["within_tol"] == 1


# --------------------------------------------------------------------------
# localization
# --------------------------------------------------------------------------
def test_index_range_validation_catches_the_d3_case():
    """A mis-routed percentage was scored as pred_idx = -125 or 321 on a
    128-point series, with nothing checking the range."""
    assert M.index_in_range(0, 128)
    assert M.index_in_range(126, 128)
    assert not M.index_in_range(127, 128)      # no step from 127 to 128
    assert not M.index_in_range(-125, 128)
    assert not M.index_in_range(321, 128)


def test_localization_reports_a_hit_curve():
    rec = M.score_localization(10, 12, 128, tol=2)
    assert rec["hit@0"] == 0 and rec["hit@2"] == 1 and rec["hit@5"] == 1
    assert rec["score"] == 1.0


# --------------------------------------------------------------------------
# forecast
# --------------------------------------------------------------------------
def test_mase_denominator_is_supplied_not_recomputed():
    truth = np.arange(8.0)
    pred = truth + 1.0
    assert M.score_forecast(pred, truth, 2.0)["mase"] == pytest.approx(0.5)
    assert M.score_forecast(pred, truth, 0.5)["mase"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# selective prediction
# --------------------------------------------------------------------------
def test_failures_are_scored_not_dropped():
    """The `.dropna()` bug: a system was never penalised for not answering."""
    scores = [1.0, 1.0, 0.0, float("nan")]
    answered = [True, True, True, False]
    rep = M.selective_report(scores, answered)
    assert rep["coverage"] == pytest.approx(0.75)
    assert rep["selective_mean"] == pytest.approx(2 / 3)   # the old number
    assert rep["scored_mean"] == pytest.approx(0.5)        # the honest one
    assert rep["scored_mean"] < rep["selective_mean"]
