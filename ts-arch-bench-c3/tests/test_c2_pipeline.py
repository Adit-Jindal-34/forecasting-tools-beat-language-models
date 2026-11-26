"""Cycle 2, wired end to end (PREREGISTRATION_C2.md, amendment A9).

These tests cover what the C2 design claims about itself:

  * the generator records what it injects, balanced and near the end;
  * that record is the qualitative key, and no system can reach it;
  * every arm is shown the same question, definitions and answer format;
  * the harness's model sees the series AND numeric evidence, never a label;
  * `describe` and `explain` are functionally identical;
  * the analysis additions (constant rule, baselines, arm changes) compute;
  * none of this disturbs a cycle-1 pool or a cycle-1 prompt.

The mock backbone is used throughout, so every number here validates
plumbing only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import make_config                          # noqa: E402
from runners import _c2 as X                              # noqa: E402
from src import ground_truth as gt, synthetic             # noqa: E402
from src.config import SystemSpec                         # noqa: E402
from src.pipeline import (Registry, make_samples,         # noqa: E402
                          run_inference, evaluate_run,
                          fixed_denominators, route_equivalent)
from src.screening import screen_report                   # noqa: E402
from systems import tools                                 # noqa: E402

L = 128


def c2_config(tmp_path, n=36, max_history=256):
    cfg = make_config(tmp_path, n_base=n, n_per=n, max_history=max_history)
    cfg.data.update(design="c2", event_zone=128)
    cfg.systems.insert(1, SystemSpec("bare_features", "bare_llm", "mock_mid",
                                     params=dict(features=True)))
    cfg.analysis.update(extra_contrasts=[["bare_llm", "bare_features"],
                                         ["bare_features", "harness"]])
    cfg.validate()
    return cfg


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("c2")
    cfg = c2_config(tmp)
    samples = make_samples(cfg, ROOT)
    reg = Registry(cfg, cache_dir=tmp / "cache")
    results = run_inference(cfg, samples, L, reg.systems, reg)
    rows = evaluate_run(cfg, samples, L, results,
                        denominators=fixed_denominators(cfg, samples))
    return cfg, samples, reg, pd.DataFrame(rows), tmp


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------
def _generated(n=360, N=512, seed=3):
    rng = np.random.default_rng(seed)
    return [synthetic.generate_base_series_c2(N, c, rng, 8)
            for c in synthetic.c2_design_cells(n, rng)], N


def test_design_is_balanced():
    pairs, _ = _generated()
    truths = [t for _, t in pairs]
    for key, n_classes in (("trend", 3), ("event", 3), ("noise", 2)):
        counts = pd.Series([t[key] for t in truths]).value_counts()
        assert len(counts) == n_classes
        assert counts.max() - counts.min() <= 1, (key, dict(counts))
    seas = sum(t["seasonal"] for t in truths)
    assert abs(seas - len(truths) / 2) <= 1


def test_events_sit_inside_the_end_zone_with_margin():
    """Cycle 1 put level shifts where the L<=128 window never saw them."""
    pairs, N = _generated()
    idx = [t["event_index"] for _, t in pairs
           if t["event"] != "no major event"]
    assert idx and min(idx) >= N - 128 + synthetic.C2_EVENT_MARGIN
    assert max(idx) < N - synthetic.C2_EVENT_MARGIN
    for _, t in pairs:
        if t["event"] != "no major event":
            assert gt.c2_event_in_window(t, N, 128)[0] == t["event"]
            assert gt.c2_event_in_window(t, N, 512)[0] == t["event"]


def test_no_spike_is_injected_into_the_forecast_horizon():
    """C2-9. Regenerate each series with and without its event and compare
    the horizon: a spike must leave it untouched; a level shift must move it
    by exactly the shift."""
    N = 256
    for event in ("spike", "abrupt change"):
        cell = ("stable", False, event, "white")
        a, ta = synthetic.generate_base_series_c2(
            N, cell, np.random.default_rng(5), 8)
        b, _ = synthetic.generate_base_series_c2(
            N, ("stable", False, "no major event", "white"),
            np.random.default_rng(5), 8)
        # same rng stream up to the event draw, so the noise is identical
        # and the level offsets differ only by the rng draws the event used
        da = a[N:] - a[N - 200]
        db = b[N:] - b[N - 200]
        if event == "spike":
            assert np.allclose(da, db), "a spike leaked into the horizon"
        else:
            assert np.allclose(da - db, ta["event_size"])


def test_seasonal_cycle_fits_the_confirmatory_window():
    """A15. H2 and H3 run at L=128, which must hold at least two full cycles
    of the longest period. Shorter windows deliberately do NOT (that is what
    makes history matter for forecasting), so this is asserted both ways."""
    assert max(synthetic.C2_PERIODS) * 2 <= 128
    assert max(synthetic.C2_PERIODS) > 32, (
        "every seasonal cycle fits inside L=32 again; that is the design "
        "A15 replaced because it left H1 nothing to measure")


def test_position_classes_are_all_reachable_at_128():
    pairs, N = _generated()
    seen = set()
    for _, t in pairs:
        ev, local = gt.c2_event_in_window(t, N, 128)
        if local is not None:
            seen.add(gt.event_position(local, 128))
    assert seen == {"early", "middle", "late"}


def test_an_event_before_the_window_is_not_an_event_for_that_window():
    truth = dict(event="spike", event_index=400)
    assert gt.c2_event_in_window(truth, 512, 128) == ("spike", 16)
    assert gt.c2_event_in_window(truth, 512, 64) == ("no major event", None)


# ---------------------------------------------------------------------------
# pool: persistence, determinism, isolation from cycle 1
# ---------------------------------------------------------------------------
def test_pool_is_deterministic_and_survives_a_reload(tmp_path):
    cfg = c2_config(tmp_path)
    first = make_samples(cfg, ROOT)            # generates
    second = make_samples(cfg, ROOT)           # rehydrates from disk
    assert [s.sample_id for s in first] == [s.sample_id for s in second]
    for a, b in zip(first, second):
        assert np.array_equal(a.series, b.series)
        assert a.query == b.query
        assert a.metadata["truth"] == b.metadata["truth"]
    manifest = json.loads((Path(cfg.data["pool_dir"]) / "manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["design"] == "c2"


def test_cycle1_pool_and_manifest_are_untouched(tmp_path):
    """A cycle-1 config must produce a manifest with NO design key, so every
    archived cycle-1 manifest still compares equal and is never rebuilt."""
    cfg = make_config(tmp_path)
    assert cfg.design == "c1"
    samples = make_samples(cfg, ROOT)
    manifest = json.loads((Path(cfg.data["pool_dir"]) / "manifest.json")
                          .read_text(encoding="utf-8"))
    assert "design" not in manifest and "event_zone" not in manifest
    assert all("truth" not in s.metadata for s in samples)
    assert "one sentence" in next(
        s.query for s in samples if s.query_type == "trend_description")


def test_c2_and_cycle1_never_share_a_pool(tmp_path):
    a = make_config(tmp_path)
    b = c2_config(tmp_path)
    assert a.data_fingerprint() != b.data_fingerprint()


def test_c2_key_refuses_a_cycle1_pool(tmp_path):
    samples = make_samples(make_config(tmp_path), ROOT)
    s = next(x for x in samples if x.query_type == "trend_description")
    with pytest.raises(ValueError, match="generator record"):
        gt.qualitative_truth_c2(s, L)


# ---------------------------------------------------------------------------
# leakage: the key is the generator record, and no system can see it
# ---------------------------------------------------------------------------
def test_inference_sample_carries_no_generator_record(run):
    _, samples, _, _, _ = run
    view = samples[0].view(L)
    assert not hasattr(view, "metadata")
    assert set(vars(view)) == {"sample_id", "query", "window", "L", "horizon"}
    assert "truth" in samples[0].metadata


def test_key_is_not_a_function_of_the_window(run):
    """Two items with the same window statistics can have different keys -
    and the key equals the record, not the computed label, on at least some
    items. (If they always agreed the cell would be saturated again.)"""
    _, samples, _, _, _ = run
    differ = 0
    for s in samples:
        if s.query_type != "trend_description":
            continue
        key = gt.qualitative_truth_c2(s, L)
        comp = gt.qualitative_computed_c2(s.window(L))
        assert key.trend == s.metadata["truth"]["trend"]
        differ += (key.trend != comp["trend"]) or (
            key.seasonal != comp["seasonal"])
    assert differ > 0


# ---------------------------------------------------------------------------
# prompts: same question for every arm; series + evidence for the harness
# ---------------------------------------------------------------------------
def _prompts(reg):
    return [json.loads(line) for line in
            (Path(reg.call_cache.path)).read_text(encoding="utf-8")
            .splitlines() if line.strip()]


def test_every_arm_is_shown_the_same_question_text(run):
    _, samples, _, _, _ = run
    for s in samples:
        if s.query_type in ("trend_description", "event_explanation"):
            assert "Reply with exactly two lines" in s.query
            assert "\n" not in s.query


def test_harness_model_sees_series_and_numeric_evidence_only(run, tmp_path):
    """Capture the prompts the arms actually build."""
    cfg, samples, _, _, _ = run
    seen = {}

    class Spy:
        name = model = "spy"

        def __init__(self, key):
            self.key = key

        def complete(self, system_prompt, user_prompt):
            seen.setdefault(self.key, []).append(user_prompt)
            if "[task:classify]" in user_prompt:
                return "describe"
            return "trend: stable\nseasonal: no"

    from systems.bare_llm import BareLLMSystem
    from systems.harness_tsfm import HarnessTSFMSystem
    s = next(x for x in samples if x.query_type == "trend_description")
    view = s.view(L)
    BareLLMSystem("bare", Spy("bare"), "spy",
                  dict(design="c2")).answer(view)
    BareLLMSystem("feat", Spy("feat"), "spy",
                  dict(design="c2", features=True)).answer(view)
    HarnessTSFMSystem("h", Spy("harness"), None, "spy",
                      dict(cfg.harness, design="c2")).answer(view)

    bare, feat = seen["bare"][0], seen["feat"][0]
    classify, final = seen["harness"]
    series_line = "Series (oldest to newest, 128 values):"
    assert series_line in bare and "computed evidence" not in bare
    assert series_line in feat and "computed evidence" in feat
    assert series_line in final and "computed evidence" in final, (
        "the harness's model must see the series beside the evidence")
    # the router sees the question and the described menu, never the series
    assert series_line not in classify
    assert "describe - " in classify and "explain - " in classify
    # no label of the scoring vocabulary inside the evidence block
    block = final.split("Series (oldest")[0]
    for word in ("increasing", "decreasing", "stable", "spike",
                 "no major event", "early", "middle", "late"):
        assert word not in block.lower(), word
    # every arm got the identical question
    assert all(s.query in p for p in (bare, feat, final, classify))
    # and the bare arm got no task header naming the operation
    assert "[task:describe]" not in bare and "[task:explain]" not in bare


def test_describe_and_explain_build_the_same_prompt_bar_the_header(run):
    cfg, samples, _, _, _ = run
    from models import llm as LM
    s = next(x for x in samples if x.query_type == "event_explanation")
    e = tools.evidence_to_block(tools.series_evidence(s.window(L))) + "\n"
    kw = dict(L=L, evidence_block=e, series="1, 2", question=s.query)
    a = LM.QUAL_PROMPT_C2.format(task="describe", **kw)
    b = LM.QUAL_PROMPT_C2.format(task="explain", **kw)
    assert a.replace("[task:describe]", "") == b.replace("[task:explain]", "")
    assert route_equivalent("describe", "explain")
    assert route_equivalent("explain", "describe")
    assert not route_equivalent("change_point", "explain")
    assert not route_equivalent("pct_change", "change_point")


# ---------------------------------------------------------------------------
# end to end on the mock
# ---------------------------------------------------------------------------
def test_all_four_arms_answer_every_item(run):
    cfg, samples, _, df, _ = run
    assert set(df.system) == {"bare_llm", "bare_features", "harness",
                              "harness_oracle"}
    assert len(df) == len(samples) * 4
    assert df.score.between(0, 1).all()


def test_qualitative_rows_use_the_mc_scorer_and_the_c2_key(run):
    _, samples, _, df, _ = run
    q = df[df.query_kind == "qualitative"]
    assert set(q.scoring.dropna()) == {"slot_accuracy_mc"}
    by_id = {s.sample_id: s for s in samples}
    td = q[q.query_type == "trend_description"]
    for _, r in td.iterrows():
        truth = by_id[r.sample_id].metadata["truth"]
        assert r.slot_trend_expected == truth["trend"]
        assert r.slot_seasonal_expected == ("yes" if truth["seasonal"]
                                            else "no")
    ee = q[(q.query_type == "event_explanation") & q.answered]
    no_event = ee[ee.slot_event_expected == "no major event"]
    assert len(no_event) and (no_event.n_slots == 1).all()
    assert (ee[ee.slot_event_expected != "no major event"].n_slots == 2).all()
    # the mock follows the format, so the parser must read it
    # (`position: none` is a legitimate absent, so check the slots that
    # every reply must name)
    oracle = q[q.system == "harness_oracle"]
    assert (oracle[oracle.query_type == "trend_description"]
            .slot_trend_status == "one").all()
    assert (oracle[oracle.query_type == "event_explanation"]
            .slot_event_status == "one").all()


def test_routing_columns_are_both_present(run):
    _, _, _, df, _ = run
    rt = df[df.route_source == "llm"]
    assert rt.route_equivalent.notna().all()
    assert (rt.route_equivalent >= rt.route_correct).all()
    assert df[df.route_source == "none"].route_equivalent.isna().all()


def test_constant_baseline_is_scored_like_an_item(run):
    cfg, samples, _, df, _ = run
    screen = screen_report(df.to_dict("records"), samples, L,
                           contrast=("bare_llm", "harness"), design="c2")
    tb = screen["trivial_baselines"]
    for qt in X.QUALITATIVE_CELLS:
        b = tb[qt]
        assert 0 < b["best_constant_score"] < 0.75, (
            f"{qt}: a constant answer scores {b['best_constant_score']:.2f};"
            " the pool is not balanced enough to leave room")
        assert set(b["best_constant_answer"]) == set(
            gt.SLOTS_FOR_TYPE_C2[qt])
    rule = X.constant_rule(df, screen,
                           ["forecasting", "trend_description",
                            "event_explanation"], ("bare_llm", "harness"))
    assert set(rule) == set(X.QUALITATIVE_CELLS)
    fam = X.surviving_family(["forecasting", "trend_description",
                              "event_explanation"], rule)
    assert "forecasting" in fam


def test_constant_rule_removes_a_dominated_cell():
    df = pd.DataFrame(dict(
        query_type=["trend_description"] * 4,
        system=["bare_llm", "bare_llm", "harness", "harness"],
        score=[0.2, 0.4, 0.5, 0.5]))
    screen = dict(trivial_baselines=dict(
        trend_description=dict(best_constant_score=0.55)))
    rule = X.constant_rule(df, screen, ["trend_description"],
                           ("bare_llm", "harness"))
    assert rule["trend_description"]["undecidable"]
    assert X.surviving_family(["forecasting", "trend_description"],
                              rule) == ["forecasting"]
    screen["trivial_baselines"]["trend_description"][
        "best_constant_score"] = 0.45
    rule = X.constant_rule(df, screen, ["trend_description"],
                           ("bare_llm", "harness"))
    assert not rule["trend_description"]["undecidable"]


def test_forecast_baselines_use_the_arms_own_denominator(run):
    cfg, samples, _, df, _ = run
    rep = X.forecast_baseline_report(cfg, df, samples, L,
                                     sorted(df.system.unique()))
    assert set(rep["baseline_mean_score"]) == set(X.BASELINE_METHODS)
    assert all(0 < v <= 1 for v in rep["baseline_mean_score"].values())
    assert "harness|seasonal_naive_estimated_period" in rep["versus"]
    # last-value recomputed by hand against the row's own denominator
    base = X.forecast_baseline_scores(cfg, samples, L)
    row = df[(df.query_type == "forecasting") & (df.system == "harness")
             & df.answered].iloc[0]
    s = next(x for x in samples if x.sample_id == row.sample_id)
    mase = np.mean(np.abs(s.window(L)[-1] - s.target())) / row.mase_denominator
    got = base[(base.sample_id == row.sample_id)
               & (base.baseline == "last_value")].mase.iloc[0]
    assert got == pytest.approx(mase)


def test_extra_contrasts_and_arm_changes_compute(run):
    cfg, _, _, df, _ = run
    rep = X.extra_contrasts(df, cfg, ["forecasting", "trend_description"])
    assert set(rep) == {"bare_llm|bare_features", "bare_features|harness"}
    assert "trend_description" in rep["bare_llm|bare_features"]
    two = pd.concat([df.assign(L=32), df.assign(L=128)])
    ch = X.arm_changes(two, cfg, "L", 128, 32, ["forecasting"],
                       ["bare_llm", "harness"])
    assert ch["forecasting"]["harness"]["mean_diff"] == pytest.approx(0.0)


def test_rerun_is_a_pure_cache_replay(run):
    cfg, samples, reg, _, tmp = run
    before = reg.call_cache.stats()["writes"]
    run_inference(cfg, samples, L, reg.systems, reg)
    assert reg.call_cache.stats()["writes"] == before


# ---------------------------------------------------------------------------
# the configs that will actually be submitted
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["c2_h1", "c2_h2", "c2_h3"])
def test_real_c2_configs_are_not_mock_runs(name):
    """A GPU run must be the real experiment, never the plumbing check.

    The `smoke_c2_*` configs use the mock backbone and the mock forecaster
    on purpose. The three configs below are the ones RUNNING.md submits, and
    they must name real served models, the real forecaster, the full pool,
    the four arms, and the declared analysis - so a mock cannot reach a
    result table by a copy-paste slip.
    """
    from src.config import load_config
    from runners._common import mock_banner
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg.design == "c2"
    assert mock_banner(cfg) is None, "a mock backbone is configured"
    for bb in cfg.backbones.values():
        assert bb.provider == "openai_compatible"
        assert bb.model.startswith("Qwen/Qwen2.5-")
        assert bb.params["temperature"] == 0.0
    assert cfg.tsfm["name"] == "chronos"
    assert cfg.tsfm["model_id"] == "amazon/chronos-t5-small"
    # H1 has its own larger pool (amendment A15); H2 and H3 share one
    n, pool = ((300, "data/c2_h1_n300") if name == "c2_h1"
               else (100, "data/c2_n100"))
    assert int(cfg.data["n_base_series"]) == n
    assert int(cfg.data["n_per_query_type"]) == n
    assert cfg.data["pool_dir"] == pool
    assert int(cfg.analysis["n_boot"]) == 5000
    assert [s.name for s in cfg.systems] == [
        "bare_llm", "bare_features", "harness", "harness_oracle"]
    assert cfg.analysis["confirmatory_contrast"] == ["bare_llm", "harness"]


def test_real_configs_pools_seeds_and_families():
    """H2 and H3 share one pool and one cache; H1 has its own (A15)."""
    from src.config import load_config
    h1, h2, h3 = [load_config(ROOT / "configs" / f"{n}.yaml")
                  for n in ("c2_h1", "c2_h2", "c2_h3")]
    assert h2.data_fingerprint() == h3.data_fingerprint()
    assert h1.data_fingerprint() != h2.data_fingerprint()
    assert h1.seed == h2.seed == h3.seed
    # H1: forecasting is the one confirmatory cell, on 300 items, and only
    # the three family-type cells are run
    assert h1.analysis["confirmatory_family"] == ["forecasting"]
    assert [q.name for q in h1.query_types] == [
        "forecasting", "trend_description", "event_explanation"]
    assert h1.history_lengths == [32, 64, 128, 256, 512]
    # H2/H3: all nine cells, three in the declared family
    for c in (h2, h3):
        assert len(c.query_types) == 9
        assert c.analysis["confirmatory_family"] == [
            "forecasting", "trend_description", "event_explanation"]
    assert h3.analysis["did_endpoints"] == ["qwen2_5_3b", "qwen2_5_72b"]
    # H2's backbone is a rung of H3's ladder, so one cache serves both
    assert set(h2.backbones) <= set(h3.backbones)
    assert (h2.backbone_fingerprint("qwen2_5_32b")
            == h3.backbone_fingerprint("qwen2_5_32b"))
    # every arm, backbone setting and analysis rule H1 shares with H2
    assert [s.name for s in h1.systems] == [s.name for s in h2.systems]
    assert (h1.backbone_fingerprint("qwen2_5_32b")
            == h2.backbone_fingerprint("qwen2_5_32b"))


def test_a_pool_from_an_older_generator_is_rebuilt(tmp_path):
    """The manifest carries the generator version, so changing the generator
    cannot leave a stale pool being served under an unchanged config."""
    cfg = c2_config(tmp_path)
    first = make_samples(cfg, ROOT)
    mpath = Path(cfg.data["pool_dir"]) / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    assert manifest["generator_version"] == synthetic.C2_GENERATOR_VERSION
    # simulate a pool written by the previous generator
    manifest["generator_version"] = synthetic.C2_GENERATOR_VERSION - 1
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    np.savez_compressed(Path(cfg.data["pool_dir"]) / "pool.npz",
                        **{f"s{s.metadata['base_index']}": s.series * 0
                           for s in first})
    again = make_samples(cfg, ROOT)
    assert np.array_equal(again[0].series, first[0].series), (
        "a stale pool was reused instead of being regenerated")


def test_event_class_is_balanced_at_128_and_skewed_below(tmp_path):
    """Amendment A13: containment, and therefore class balance, moves with L.

    H2 and H3 run at L=128 and must be exactly balanced. H1 sweeps below
    that, where the skew is real and is why its qualitative cells are
    descriptive only. This pins both halves so neither is forgotten.
    """
    from collections import Counter

    rng = np.random.default_rng(20261111)
    MH, ZONE = 512, 128
    cells = synthetic.c2_design_cells(360, rng)
    truths = [synthetic.generate_base_series_c2(
        MH, c, rng, horizon=8, event_zone=ZONE)[1] for c in cells]

    def share(L):
        seen = [gt.c2_event_in_window(t, MH, L)[0] for t in truths]
        c = Counter(seen)
        return max(c.values()) / len(seen)

    # balanced where the confirmatory cells live
    for L in (128, 256, 512):
        assert abs(share(L) - 1 / 3) < 0.02, (
            f"L={L} is a confirmatory length for H2/H3 and the event class "
            f"is not balanced: majority share {share(L):.2f}")

    # and genuinely skewed below it - the fact A13 records
    assert share(32) > 0.70, (
        "A13 records an 84% majority at L=32; if the generator changed so "
        "that short windows contain their events, A13 should be withdrawn "
        "rather than left in the document")
    assert share(64) > share(128)


def test_trend_is_answerable_at_128_and_not_below(tmp_path):
    """Amendment A14: the injected drift is not visible in a short window.

    A9 keys `trend` on what was injected, so the opposite of saturation
    becomes the risk - a fact the window does not show. The benchmark here
    is the strongest simple detector, a fitted-slope t test, which is also
    the quantity the evidence block reports: it upper-bounds what any arm
    could score.

    L=128 is a confirmatory length for H2 and H3 and must be well clear of
    the constant line. Below it the slot is at chance, which is why H1's
    qualitative cells are descriptive only.
    """
    rng = np.random.default_rng(20261111)
    MH, ZONE = 512, 128
    cells = synthetic.c2_design_cells(360, rng)
    pool = [synthetic.generate_base_series_c2(
        MH, c, rng, horizon=8, event_zone=ZONE) for c in cells]

    def accuracy(L):
        ok = sum(gt.trend_direction_fitted(
            np.asarray(x[MH - L:MH], float)) == tr["trend"]
            for x, tr in pool)
        return ok / len(pool)

    # balanced by construction, so a constant answer scores 1/3
    assert accuracy(128) > 0.60, (
        f"trend accuracy at L=128 is {accuracy(128):.2f}; L=128 is a "
        "confirmatory length for H2 and H3, so the slot must be clearly "
        "answerable there")
    assert accuracy(512) > accuracy(128)

    # and the fact A14 records: at chance for the two shortest lengths
    assert accuracy(32) < 0.45, (
        f"trend accuracy at L=32 is {accuracy(32):.2f}. If the generator "
        "changed so short windows show their drift, withdraw A14 rather "
        "than leaving it in the document."
    )
