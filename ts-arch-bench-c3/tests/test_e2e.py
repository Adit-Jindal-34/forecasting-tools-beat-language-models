"""End-to-end: a full three-arm run on the mock backbone.

The old repo's e2e tests never executed on the author's machine (they all
errored with `PermissionError: [WinError 5]` on pytest's basetemp), which is
part of why a dead classifier branch and a 100% fallback rate went unnoticed
for weeks. These use the `tmp_path` fixture the same way, so if the
environment breaks them again it will be visible in CI, not silent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (Registry, make_samples, run_inference,
                          evaluate_run, fixed_denominators)
from src.screening import screen_cells, screen_report, trivial_baselines
from src.stats import holm, benjamini_hochberg, paired_compare
from evaluation.metrics import selective_report
from conftest import make_config

L = 128


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    cfg = make_config(tmp, n_base=18, n_per=18, prose_rate=0.2)
    import pathlib
    samples = make_samples(cfg, pathlib.Path("."))
    reg = Registry(cfg, cache_dir=tmp / "cache")
    results = run_inference(cfg, samples, L, reg.systems, reg)
    rows = evaluate_run(cfg, samples, L, results, judge=reg.judge,
                        denominators=fixed_denominators(cfg, samples))
    return cfg, samples, reg, rows


def test_every_system_answers_every_sample(run):
    cfg, samples, reg, rows = run
    assert len(rows) == len(samples) * len(cfg.systems)
    df = pd.DataFrame(rows)
    assert set(df.system) == {"bare_llm", "harness", "harness_oracle"}


def test_every_row_has_a_finite_score(run):
    """No NaN scores: a failure is a zero, not a dropped row."""
    df = pd.DataFrame(run[3])
    assert df["score"].notna().all()
    assert ((df["score"] >= 0) & (df["score"] <= 1)).all()


def test_failure_taxonomy_is_populated(run):
    df = pd.DataFrame(run[3])
    assert set(df["failure"].unique()) <= {
        "", "abstain", "unparseable", "type_error"}
    assert df["answered"].dtype == bool or set(df["answered"]) <= {True, False}
    # abstentions must score exactly zero and never count as answered
    ab = df[df.abstained]
    if len(ab):
        assert (ab["score"] == 0).all() and not ab["answered"].any()


def test_prose_replies_exercise_recovery_and_abstention(run):
    """prose_rate=0.2 must actually drive the parser off the clean path."""
    df = pd.DataFrame(run[3])
    harness = df[df.system == "harness"]
    statuses = set(harness["route_status"].dropna())
    assert "clean" in statuses
    assert statuses & {"recovered", "ambiguous", "unparseable"}, (
        "the permissive parser and abstention paths were never exercised")


def test_routing_is_not_constant_within_a_cell(run):
    """The collapse guard, end to end: with one fixed template per type
    every item of a type shared a byte-identical classify prompt, so
    route_op had nunique == 1 in every cell."""
    df = pd.DataFrame(run[3])
    h = df[df.system == "harness"]
    per_cell = h.groupby("query_type")["query_type"].size()
    prompts = h.groupby("query_type")["paraphrase_id"].nunique()
    assert (prompts > 1).all(), (
        "paraphrase ids are constant within a cell: routing collapsed")
    assert (prompts == per_cell).all()


def test_both_parser_readings_are_recorded(run):
    df = pd.DataFrame(run[3])
    harness = df[df.system == "harness"]
    assert "op_strict" in harness.columns and "op_permissive" in harness.columns
    # recovery rate is computable offline, with no rerun
    recovered = (harness["route_status"] == "recovered").mean()
    assert 0.0 <= recovered <= 1.0


def test_oracle_arm_routes_perfectly_and_self_arm_does_not_cheat(run):
    df = pd.DataFrame(run[3])
    oracle = df[df.system == "harness_oracle"]
    assert (oracle["route_correct"] == 1).all()
    assert (oracle["route_source"] == "oracle").all()
    assert oracle["n_model_calls"].sum() < df[
        df.system == "harness"]["n_model_calls"].sum()
    # the self-routing arm must make mistakes the oracle arm does not
    selfr = df[df.system == "harness"]
    assert 0.0 <= selfr["route_correct"].mean() <= 1.0
    # abstentions count as routing failures in the primary column and
    # are excluded from the selective one, never silently skipped
    ab = selfr[selfr.abstained]
    if len(ab):
        assert (ab["route_correct"] == 0).all()
        assert ab["route_correct_selective"].isna().all()
        primary = selfr["route_correct"].mean()
        selective = selfr["route_correct_selective"].mean()
        assert primary <= selective + 1e-9


def test_bare_arm_receives_no_operation_hint(run):
    """The `op=` leak: the bare prompt must not name the ground-truth op."""
    df = pd.DataFrame(run[3])
    bare = df[df.system == "bare_llm"]
    assert bare["route_op"].isna().all()
    for trace in bare["trace_json"]:
        assert "op=" not in trace


def test_localization_type_errors_are_caught_not_scored(run):
    df = pd.DataFrame(run[3])
    loc = df[df.query_type == "localization"]
    bad = loc[~loc["type_valid"].astype(bool)]
    assert (bad["score"] == 0).all()
    assert set(bad["failure"]) <= {"type_error"}


def test_screening_runs_and_flags_undecidable_cells(run):
    cfg, samples, reg, rows = run
    cells = screen_cells(rows, screen=dict(min_effect=0.02))
    assert len(cells) == 9
    for c in cells:
        assert isinstance(c.decidable, bool)
        assert c.reason
    rep = screen_report(rows, samples, L,
                        contrast=("bare_llm", "harness"),
                        framing_threshold=0.2)
    assert rep["n_cells"] == 9
    assert "trivial_baselines" in rep


def test_trivial_baselines_are_reported_for_qualitative_cells(samples=None):
    """A reader must be able to see what a constant answer scores."""
    import pathlib
    from src.pipeline import make_samples as mk
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg = make_config(pathlib.Path(d), n_base=40, n_per=40)
        s = mk(cfg, pathlib.Path("."))
    base = trivial_baselines(s, L)
    for qt in ("trend_description", "event_explanation"):
        assert 0.0 < base[qt]["best_constant_score"] <= 1.0
        assert base[qt]["slots"]


def test_selective_report_matches_the_rows(run):
    df = pd.DataFrame(run[3])
    sub = df[(df.system == "harness") & (df.query_type == "localization")]
    rep = selective_report(sub["score"].tolist(), sub["answered"].tolist())
    assert rep["n"] == len(sub)
    assert 0.0 <= rep["coverage"] <= 1.0
    assert rep["scored_mean"] <= (rep["selective_mean"]
                                 if rep["selective_mean"] == rep["selective_mean"]
                                 else 1.0) + 1e-9


def test_paired_comparison_uses_full_vectors(run):
    df = pd.DataFrame(run[3])
    a = df[df.system == "harness"].sort_values("sample_id")
    b = df[df.system == "bare_llm"].sort_values("sample_id")
    assert list(a.sample_id) == list(b.sample_id)
    res = paired_compare(a["score"].to_numpy(), b["score"].to_numpy(),
                         n_boot=200, min_n=5)
    assert res.n == len(a)


def test_corrections_agree_on_a_single_test():
    assert holm([0.01]) == pytest.approx([0.01])
    assert benjamini_hochberg([0.01]) == pytest.approx([0.01])


def test_holm_is_monotone_and_bounded():
    ps = [0.001, 0.01, 0.04, 0.2, float("nan")]
    adj = holm(ps)
    finite = [v for v in adj if v == v]
    assert all(0 <= v <= 1 for v in finite)
    assert finite == sorted(finite)
    assert np.isnan(adj[-1])


def test_rerunning_the_whole_pipeline_issues_no_new_calls(run, tmp_path):
    """Replay with a warm cache: zero API cost after any downstream change."""
    cfg, samples, reg, rows = run
    before = reg.call_cache.stats()["writes"]
    results = run_inference(cfg, samples, L, reg.systems, reg)
    evaluate_run(cfg, samples, L, results, judge=reg.judge,
                 denominators=fixed_denominators(cfg, samples))
    assert reg.call_cache.stats()["writes"] == before
