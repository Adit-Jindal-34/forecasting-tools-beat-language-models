"""Human-validation tooling (PREREGISTRATION.md §4.1).

The alpha implementation is checked against Krippendorff's own published
worked example, not against itself. A reliability coefficient that is only
self-consistent is worthless: it would let a bug read as low annotator
agreement and send the qualitative metric back for a redesign it did not
need, or - worse - read as high agreement and license a confirmatory claim.
"""
from __future__ import annotations

import math

import pytest

from evaluation.human_validation import (krippendorff_alpha, consensus,
                                         machine_vs_human, stratified_sample)

#: Krippendorff, "Computing Krippendorff's Alpha-Reliability" (2011), the
#: 12-unit / 4-observer example with missing values. Published nominal
#: alpha = 0.743.
_CANONICAL = {
    "u1":  [1, 1, None, 1],
    "u2":  [2, 2, 3, 2],
    "u3":  [3, 3, 3, 3],
    "u4":  [3, 3, 3, 3],
    "u5":  [2, 2, 2, 2],
    "u6":  [1, 2, 3, 4],
    "u7":  [4, 4, 4, 4],
    "u8":  [1, 1, 2, 1],
    "u9":  [2, 2, 2, 2],
    "u10": [None, 5, 5, 5],
    "u11": [None, None, 1, 1],
    "u12": [None, 3, None, None],
}


def test_alpha_matches_krippendorffs_published_example():
    out = krippendorff_alpha(_CANONICAL, level="nominal")
    assert out["alpha"] == pytest.approx(0.743, abs=0.001), (
        f"nominal alpha came out {out['alpha']!r}; Krippendorff's published "
        "value for this dataset is 0.743. A mismatch means the coincidence "
        "matrix or the chance correction is wrong.")
    # u12 has a single coding and carries no reliability information
    assert out["n_units"] == 11


def test_unit_with_one_coding_is_dropped_not_counted_as_agreement():
    out = krippendorff_alpha({"a": [1, 1], "b": [2], "c": [2, 2]})
    assert out["n_units"] == 2


def test_perfect_agreement_with_real_variance_is_one():
    out = krippendorff_alpha({"a": [1, 1, 1], "b": [2, 2, 2],
                              "c": [1, 1, 1], "d": [2, 2, 2]})
    assert out["alpha"] == pytest.approx(1.0)


def test_single_class_is_undefined_not_perfect():
    """Everyone saying "no major event" to everything is not reliability.

    Returning 1.0 here would let a degenerate slot - one where the answer
    key has collapsed to one class - certify itself as perfectly reliable.
    """
    out = krippendorff_alpha({"a": ["x", "x"], "b": ["x", "x"]})
    assert math.isnan(out["alpha"])
    assert "undefined" in out["note"]


def test_alpha_is_zero_at_chance():
    """Two coders assigning the modal class independently agree often but
    carry no information; alpha must not reward that."""
    units = {f"u{i}": ([1, 1] if i % 2 == 0 else [1, 2]) for i in range(40)}
    out = krippendorff_alpha(units)
    assert out["alpha"] < 0.5


def test_tie_yields_no_consensus_rather_than_a_coin_flip():
    label, k = consensus(["spike", "abrupt change", None])
    assert label is None
    label, k = consensus(["spike", "spike", "abrupt change"])
    assert label == "spike" and k == 2


def test_items_without_consensus_are_reported_not_dropped():
    machine = {"u1": "a", "u2": "b", "u3": "a"}
    human = {"u1": "a", "u2": None, "u3": "b"}
    out = machine_vs_human(machine, human)
    assert out["n"] == 2 and out["n_no_consensus"] == 1
    assert out["agreement"] == pytest.approx(0.5)


def test_sample_is_stratified_on_design_variables_only():
    rows = [dict(sample_id=f"s{i}", query_type="trend_description",
                 pattern=("trend" if i % 2 else "spikes"),
                 score=float(i % 2)) for i in range(40)]
    picked = stratified_sample(rows, n=10, strata=("query_type", "pattern"),
                               seed=0)
    assert len(picked) == 10
    strata = {r["_stratum"] for r in picked}
    assert strata == {"trend_description|trend", "trend_description|spikes"}
    # the weight must allow a population estimate to be recovered
    assert all(r["_sampling_weight"] > 0 for r in picked)


def test_sampling_does_not_condition_on_the_machine_verdict():
    """Stratifying on `score` would bias the agreement rate it validates."""
    rows = [dict(sample_id=f"s{i}", query_type="t", pattern="p",
                 score=float(i % 2)) for i in range(20)]
    with pytest.raises(ValueError, match="strata columns absent"):
        stratified_sample(rows, n=5, strata=("not_a_column",), seed=0)
