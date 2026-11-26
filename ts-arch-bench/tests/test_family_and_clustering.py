"""The declared Holm family, the saturation axis, and cluster-aware tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import a_priori_saturated, DEFAULT_CONFIRMATORY_FAMILY
from src.screening import (screen_cells, clustering_decision,
                           routing_variance_within_framing)
from src.stats import (holm, benjamini_hochberg, paired_compare,
                       paired_compare_clustered)
from conftest import make_config


# --------------------------------------------------------------------------
# declared family
# --------------------------------------------------------------------------
def test_declared_m_makes_holm_independent_of_how_many_cells_collapse():
    """Degenerate comparisons return NaN. With `m=None` they leave the
    family, shrinking it and making the SURVIVORS easier to call
    significant - the bias runs anti-conservative. A declared `m` pins it."""
    few_survive = [0.02] + [float("nan")] * 8            # 1 testable of 9
    more_survive = [0.02, 0.9, 0.95] + [float("nan")] * 6  # 3 testable of 9
    assert holm(few_survive, m=9)[0] == pytest.approx(
        holm(more_survive, m=9)[0]), (
        "the adjustment moved because other cells collapsed")


def test_auto_m_is_more_permissive_than_the_declared_family():
    collapsed = [0.02] + [float("nan")] * 8
    assert holm(collapsed)[0] == pytest.approx(0.02)       # m inferred as 1
    assert holm(collapsed, m=9)[0] == pytest.approx(0.18)  # declared family
    assert holm(collapsed)[0] < holm(collapsed, m=9)[0]


def test_holm_rejects_an_undersized_declared_family():
    with pytest.raises(ValueError, match="mis-declared"):
        holm([0.01, 0.02, 0.03], m=2)


def test_bh_honours_declared_m():
    ps = [0.01, 0.02, float("nan")]
    assert benjamini_hochberg(ps, m=3)[0] >= benjamini_hochberg(ps)[0]
    with pytest.raises(ValueError, match="mis-declared"):
        benjamini_hochberg(ps, m=1)


def test_saturation_is_an_a_priori_property(cfg):
    """Membership of the family must be derivable before any data exists."""
    sat = a_priori_saturated(cfg.query_types)
    assert set(sat) == {"statistics", "counting", "extremum", "change",
                        "correlation", "localization"}
    assert "trend_description" not in sat and "forecasting" not in sat


def test_declared_family_is_named_and_sized(cfg):
    rep = cfg.family_report()
    assert rep["confirmatory_family"] == list(DEFAULT_CONFIRMATORY_FAMILY)
    assert rep["n_tests"] == len(DEFAULT_CONFIRMATORY_FAMILY) == 3
    assert rep["confirmatory_contrast"] == ["bare_llm", "harness"]


def test_answer_accuracy_family_excludes_every_saturated_cell(cfg):
    """D6: the family is the complement of `a_priori_saturated`.

    This replaces an earlier assertion that two saturated cells were kept
    deliberately. The decomposed-metrics argument (PREREGISTRATION.md §3,
    deviation D6) is that in a saturated cell a correctly-routed harness
    scores exactly 1.000, so the per-item answer score IS the routing
    outcome and the gap reduces to (route_correct - bare_accuracy) - which
    is already reported by the routing columns and the oracle ablation.
    Mixing it into the answer-accuracy family double-counts it.

    Asserting the structural invariant rather than a literal list means a
    newly added query type cannot quietly land in the confirmatory family
    while being pinned at 1.000 by construction.
    """
    rep = cfg.family_report()
    sat = set(a_priori_saturated(cfg.query_types))
    fam = set(rep["confirmatory_family"])
    assert not (fam & sat), (
        f"saturated cells in the answer-accuracy family: {sorted(fam & sat)}. "
        "Their gap is redundant with route_correct; report them there.")
    all_types = {q.name for q in cfg.query_types}
    assert fam == all_types - sat, (
        "the family must be exactly the non-saturated cells; "
        f"missing={sorted((all_types - sat) - fam)} "
        f"extra={sorted(fam - (all_types - sat))}")
    assert rep["saturated_cells_kept_in_family"] == []


def test_family_must_name_known_query_types(tmp_path):
    cfg = make_config(tmp_path)
    cfg.analysis["confirmatory_family"] = ["counting", "not_a_type"]
    with pytest.raises(ValueError, match="unknown query types"):
        cfg.validate()


def test_empty_family_is_refused(tmp_path):
    cfg = make_config(tmp_path)
    cfg.analysis["confirmatory_family"] = []
    with pytest.raises(ValueError, match="confirmatory_family is empty"):
        cfg.validate()


# --------------------------------------------------------------------------
# saturation as a separate axis
# --------------------------------------------------------------------------
def _saturated_cell_rows(n=100, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append(dict(query_type="statistics", system="bare_llm",
                         sample_id=f"s{i}",
                         score=float(np.clip(rng.normal(0.64, 0.18), 0, 1))))
        for name in ("harness", "harness_oracle"):
            rows.append(dict(query_type="statistics", system=name,
                             sample_id=f"s{i}", score=1.0))
    return rows


def test_tool_saturated_cell_is_flagged_but_still_decidable():
    """The C2 contradiction, resolved by splitting the axes.

    A cell with one arm pinned at 1.000 and the other at 0.64 CAN show a
    difference - the gap is real - so calling it undecidable would discard a
    true result. What it cannot do is speak about architecture.
    """
    c = screen_cells(_saturated_cell_rows())[0]
    assert c.saturated is True
    assert set(c.saturated_systems) == {"harness", "harness_oracle"}
    assert c.decidable is True
    assert c.max_abs_gap > 0.3


def test_per_pair_gaps_do_not_let_the_oracle_rescue_a_cell():
    """A cell-level max over ALL pairs lets a third arm make a cell look
    decidable when the confirmatory contrast inside it is not."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(60):
        # real response variance, so the cell is not screened out for being
        # degenerate; only the PAIR gap should decide
        # one shared draw per item: every arm has real response variance,
        # while the BETWEEN-arm gaps stay exactly 0.005 and 0.40
        noise = float(rng.normal(0, 0.05))
        rows.append(dict(query_type="x", system="bare_llm",
                         sample_id=f"s{i}", score=0.50 + noise))
        rows.append(dict(query_type="x", system="harness",
                         sample_id=f"s{i}", score=0.505 + noise))
        rows.append(dict(query_type="x", system="harness_oracle",
                         sample_id=f"s{i}", score=0.90 + noise))
    c = screen_cells(rows)[0]
    assert c.decidable is True                       # cell-level, via oracle
    assert c.decidable_pairs["bare_llm|harness"] is False
    assert c.pair_gaps["bare_llm|harness"] == pytest.approx(0.005, abs=1e-6)


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------
def test_cluster_bootstrap_uses_clusters_as_the_unit():
    """100 items sharing 4 routing decisions is n=4, not n=100."""
    a = np.repeat([1.0, 1.0, 0.0, 0.0], 25)
    b = np.zeros(100)
    clusters = np.repeat([0, 1, 2, 3], 25)
    item = paired_compare(a, b, n_boot=400, min_n=5)
    clust = paired_compare_clustered(a, b, clusters, n_boot=400, min_n=2)
    assert clust.n_clusters == 4 and clust.cluster_unit == "cluster"
    assert clust.mean_diff == pytest.approx(item.mean_diff)
    width = lambda r: r.ci_high - r.ci_low            # noqa: E731
    assert width(clust) > width(item), (
        "the clustered CI must be wider: pretending 100 correlated items are "
        "independent is exactly the pseudo-replication this guards against")


def test_cluster_bootstrap_reduces_to_item_level_when_every_item_differs():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=60), rng.normal(size=60)
    clusters = np.arange(60)
    item = paired_compare(a, b, n_boot=600, seed=1)
    clust = paired_compare_clustered(a, b, clusters, n_boot=600, seed=1)
    assert clust.n_clusters == 60
    assert clust.mean_diff == pytest.approx(item.mean_diff)
    assert abs((clust.ci_high - clust.ci_low)
               - (item.ci_high - item.ci_low)) < 0.1


def test_identical_arms_yield_no_test():
    a = np.array([0.3, 0.7, 1.0] * 10)
    r = paired_compare(a, a.copy(), n_boot=200, min_n=5)
    assert r.mean_diff == 0.0 and np.isnan(r.p_raw)
    assert r.n_zero_diff == len(a)


def test_paired_sign_effect_is_named_correctly():
    """`mean(sign(d))` is P(d>0)-P(d<0) for paired data, not Cliff's delta."""
    r = paired_compare(np.array([1.0] * 20), np.zeros(20), n_boot=100,
                       min_n=5)
    assert r.paired_sign_effect == pytest.approx(1.0)
    assert not hasattr(r, "cliffs_delta")


# --------------------------------------------------------------------------
# the pre-registered clustering rule
# --------------------------------------------------------------------------
def test_clustering_rule_picks_framing_when_leads_are_cosmetic():
    report = {f"h|t{i}": dict(fraction_varying=0.0, n_framings=25,
                              n_framings_with_variation=0,
                              distinct_route_ops=1, system="h",
                              query_type=f"t{i}") for i in range(3)}
    d = clustering_decision(report, threshold=0.20)
    assert d["cluster_unit"] == "framing_id"
    assert d["effective_routing_n_per_type"] == 25


def test_clustering_rule_picks_item_when_leads_matter():
    report = {f"h|t{i}": dict(fraction_varying=0.6, n_framings=25,
                              n_framings_with_variation=15,
                              distinct_route_ops=3, system="h",
                              query_type=f"t{i}") for i in range(3)}
    d = clustering_decision(report, threshold=0.20)
    assert d["cluster_unit"] == "sample_id"
    assert d["effective_routing_n_per_type"] == 100


def test_routing_variance_needs_the_right_columns():
    with pytest.raises(ValueError):
        routing_variance_within_framing([dict(query_type="a", system="b")])
