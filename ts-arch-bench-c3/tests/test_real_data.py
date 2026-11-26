"""Cycle 3: the real-data pool loader and everything it is allowed to feed.

The synthetic path builds the pool and its answer keys in one pass, so the
whole of `tests/test_leakage.py` and `tests/test_synthetic.py` reason about
a generator. None of that applies here: a Monash series has no generator and
exactly one legitimate label, the held-out future. These tests pin the
properties that replace those guarantees.

  * the sweep invariants I1/I2 still hold, now by POOL GEOMETRY rather than
    by construction in the generator;
  * the MASE denominator is the pool's, fixed, and identical at every L;
  * the qualitative cells are REFUSED rather than fed empty keys;
  * the pool is content-addressed, so a silently rebuilt pool fails the run
    instead of breaking the paired design quietly;
  * the degenerate-window guard that real data forced into existence fires
    when it should and is INVISIBLE when it should not - the last one
    matters because `evidence_to_block` output is inside the call cache key.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src import real_data as RD
from src.config import Config, QueryTypeSpec, BackboneSpec, SystemSpec
from systems import tools

ROOT = Path(__file__).resolve().parents[1]
POOLS = ("pool_real_n300", "pool_clean_n300", "pool_h2_strata_n240")


def pool_dirs():
    return [ROOT / "data-real" / p for p in POOLS
            if (ROOT / "data-real" / p / "windows.npz").exists()]


requires_pool = pytest.mark.skipif(
    not pool_dirs(),
    reason="no real pool on disk; run data-real/fetch.py && build_pool*.py")


def real_config(pool: Path, horizon=24, max_history=512,
                query_types=(("forecasting", "forecast"),)) -> Config:
    cfg = Config(seed=20261114)
    cfg.data = dict(name="real", design="c2", pool_dir=str(pool),
                    pool_sha256=RD.pool_sha256(pool),
                    max_history=max_history, horizon=horizon)
    cfg.history_lengths = [128]
    cfg.query_types = [QueryTypeSpec(n, k) for n, k in query_types]
    cfg.backbones = {"mock_mid": BackboneSpec("mock_mid", "mock", "mock-llm",
                                              0.6)}
    cfg.systems = [
        SystemSpec("bare_llm", "bare_llm", "mock_mid"),
        SystemSpec("harness", "harness_tsfm", "mock_mid"),
        SystemSpec("harness_oracle", "harness_tsfm", "mock_mid",
                   routing="oracle")]
    # Real data carries ONE cell, so the Holm family is m = 1.
    cfg.analysis.update(min_n_per_cell=5, n_boot=50,
                        confirmatory_family=["forecasting"])
    return cfg


# ===========================================================================
# the loader
# ===========================================================================
@requires_pool
@pytest.mark.parametrize("pool", pool_dirs(), ids=lambda p: p.name)
def test_loader_emits_forecasting_samples(pool):
    cfg = real_config(pool)
    cfg.validate()
    from src.pipeline import make_samples
    samples = make_samples(cfg, ROOT)
    meta = json.loads((pool / "meta.json").read_text(encoding="utf-8"))
    assert len(samples) == meta["n_items"]
    for s in samples:
        assert s.query_type == "forecasting"
        assert s.query_kind == "forecast"
        assert len(s.series) == cfg.data["max_history"] + cfg.data["horizon"]
        assert np.all(np.isfinite(s.series))


@requires_pool
def test_sample_ids_are_opaque_and_unique():
    pool = pool_dirs()[0]
    from src.pipeline import make_samples
    samples = make_samples(real_config(pool), ROOT)
    ids = [s.sample_id for s in samples]
    assert len(set(ids)) == len(ids)
    datasets = {s.metadata["dataset_id"] for s in samples}
    for s in samples:
        # The id must not name its archive: a system can read
        # `InferenceSample.sample_id`, and the dataset name is exactly the
        # contamination-sensitive token a model could recall.
        assert not any(d in s.sample_id for d in datasets)
        assert s.metadata["item_id"].startswith(s.metadata["dataset_id"])
        # `pattern` has no factorial meaning here; it carries provenance so
        # the conservative clustering unit survives into the shard.
        assert s.pattern == s.metadata["dataset_id"]


# ===========================================================================
# invariants I1/I2 - now a property of the POOL GEOMETRY
# ===========================================================================
@requires_pool
def test_I1_windows_are_nested_suffixes_and_I2_target_is_fixed():
    """Every L ends at the SAME cut, so the sweep tracks one item.

    This is what makes H1 a difference-in-differences rather than a
    comparison of five different questions. On the synthetic pool it held
    because the generator was asked for one long series; here it holds
    because `build_pool.py` stores `L_MAX + H_MAX` points with a common cut
    origin and the loader slices rather than re-cuts.
    """
    pool = pool_dirs()[0]
    from src.pipeline import make_samples
    samples = make_samples(real_config(pool), ROOT)
    for s in samples[:25]:
        target = s.target()
        for a, b in zip((32, 64, 128, 256), (64, 128, 256, 512)):
            wa, wb = s.window(a), s.window(b)
            assert np.array_equal(wa, wb[-a:]), "I1 violated"
            assert np.array_equal(s.target(), target), "I2 violated"


@requires_pool
def test_I5_inference_sample_carries_no_provenance():
    """A system must not be able to read the dataset, the declared period or
    the stratum off what it is handed."""
    pool = pool_dirs()[0]
    from src.pipeline import make_samples
    s = make_samples(real_config(pool), ROOT)[0]
    view = s.view(128)
    exposed = json.dumps({k: str(v) for k, v in vars(view).items()
                          if k != "window"}).lower()
    for secret in ("dataset", "period", "stratum", "seasonal_strength",
                   "mase", s.metadata["dataset_id"].lower()):
        assert secret not in exposed, f"{secret!r} leaked into the view"
    assert not hasattr(view, "metadata")


# ===========================================================================
# the MASE denominator
# ===========================================================================
@requires_pool
def test_denominator_comes_from_the_pool_and_is_identical_at_every_L():
    """`scoring.fixed_mase_denominator` requires a yardstick that does not
    move with the condition under test.

    Recomputing it per config would be worse than wrong: `naive_window_error`
    falls back to a DRIFT naive when the window holds fewer than two periods,
    so a short `max_history` would silently switch denominator families for
    part of the pool.
    """
    pool = pool_dirs()[0]
    from src.pipeline import make_samples, fixed_denominators
    z = np.load(pool / "windows.npz")
    want = {}
    for max_history in (128, 512):
        cfg = real_config(pool, max_history=max_history)
        cfg.history_lengths = [128]
        samples = make_samples(cfg, ROOT)
        dens = fixed_denominators(cfg, samples)
        want[max_history] = dens
        for k, s in enumerate(samples):
            assert dens[s.sample_id] == pytest.approx(
                float(z["mase_denominator"][k]), rel=1e-12)
    assert want[128] == want[512]


@requires_pool
def test_every_denominator_is_strictly_positive():
    """A zero denominator would make the score 1/(1+inf) = 0 for every arm
    and silently kill the item. The POOL BUILDER excludes those series; the
    loader refuses a pool that slipped one through."""
    for pool in pool_dirs():
        d = np.load(pool / "windows.npz")["mase_denominator"]
        assert np.all(np.isfinite(d)) and np.all(d > 0)


# ===========================================================================
# what real data may NOT be asked
# ===========================================================================
@requires_pool
@pytest.mark.parametrize("qt", ["trend_description", "event_explanation"])
def test_qualitative_cells_are_refused_with_a_reason(qt):
    """Not fed empty keys, not silently dropped - refused, loudly.

    The cycle-2 key for these is the generator's injected record (A9). Real
    data has no generator, and deriving a replacement from window statistics
    would make the answer key and the harness's evidence block the same
    function, which is cycle 1's retracted defect.
    """
    pool = pool_dirs()[0]
    cfg = real_config(pool, query_types=(("forecasting", "forecast"),
                                         (qt, "qualitative")))
    with pytest.raises(ValueError, match=qt):
        cfg.validate()
    with pytest.raises(ValueError, match="cannot carry"):
        RD.materialize(cfg, pool_dir=pool)


@requires_pool
def test_tool_saturated_cells_are_refused_too():
    """They would *work* - the keys are functions of the visible window -
    but the harness scores 1.000 by construction, so they cost calls and
    measure the bare arm's arithmetic."""
    pool = pool_dirs()[0]
    cfg = real_config(pool, query_types=(("forecasting", "forecast"),
                                         ("statistics", "numeric_exact")))
    with pytest.raises(ValueError, match="statistics"):
        cfg.validate()


def test_config_refuses_generator_keys_on_a_real_pool(tmp_path):
    """`data_fingerprint()` hashes the whole data block, so a leftover
    `event_zone` would define a cache namespace that means nothing."""
    cfg = Config(seed=1)
    cfg.data = dict(name="real", design="c2", pool_dir=str(tmp_path),
                    pool_sha256="x" * 64, max_history=512, horizon=24,
                    event_zone=128)
    cfg.history_lengths = [128]
    cfg.query_types = [QueryTypeSpec("forecasting", "forecast")]
    cfg.backbones = {"m": BackboneSpec("m", "mock", "mock-llm", 0.6)}
    cfg.systems = [SystemSpec("bare_llm", "bare_llm", "m")]
    with pytest.raises(ValueError, match="event_zone"):
        cfg.validate()


def test_config_requires_c2_design_on_a_real_pool(tmp_path):
    cfg = Config(seed=1)
    cfg.data = dict(name="real", design="c1", pool_dir=str(tmp_path),
                    pool_sha256="x" * 64, max_history=512, horizon=24)
    cfg.history_lengths = [128]
    cfg.query_types = [QueryTypeSpec("forecasting", "forecast")]
    cfg.backbones = {"m": BackboneSpec("m", "mock", "mock-llm", 0.6)}
    cfg.systems = [SystemSpec("bare_llm", "bare_llm", "m")]
    with pytest.raises(ValueError, match="c2"):
        cfg.validate()


def test_real_only_cluster_units_are_refused_on_synthetic(tmp_path):
    """A synthetic run writes `dataset_id` empty, so clustering on it would
    silently fall back to the item and report an interval that is too
    narrow."""
    from tests.conftest import make_config
    cfg = make_config(tmp_path)
    cfg.analysis["cluster_unit"] = "dataset_id"
    with pytest.raises(ValueError, match="real pool"):
        cfg.validate()


# ===========================================================================
# the pool is content-addressed
# ===========================================================================
@requires_pool
def test_a_changed_pool_fails_the_run_rather_than_the_design():
    pool = pool_dirs()[0]
    cfg = real_config(pool)
    cfg.data["pool_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        RD.materialize(cfg, pool_dir=pool)


@requires_pool
def test_an_undeclared_pool_hash_is_refused_and_the_message_gives_it():
    pool = pool_dirs()[0]
    cfg = real_config(pool)
    cfg.data.pop("pool_sha256")
    with pytest.raises(ValueError) as exc:
        RD.materialize(cfg, pool_dir=pool)
    assert RD.pool_sha256(pool) in str(exc.value)


@requires_pool
def test_horizon_beyond_the_stored_target_is_refused():
    """Padding a target would invent data."""
    pool = pool_dirs()[0]
    meta = json.loads((pool / "meta.json").read_text(encoding="utf-8"))
    cfg = real_config(pool, horizon=meta["H_MAX"] + 1)
    with pytest.raises(ValueError, match="H_MAX"):
        RD.materialize(cfg, pool_dir=pool)


@requires_pool
def test_max_history_beyond_the_stored_window_is_refused():
    pool = pool_dirs()[0]
    meta = json.loads((pool / "meta.json").read_text(encoding="utf-8"))
    cfg = real_config(pool, max_history=meta["L_MAX"] + 1)
    cfg.history_lengths = [128]
    with pytest.raises(ValueError, match="L_MAX"):
        RD.materialize(cfg, pool_dir=pool)


@requires_pool
def test_pool_fingerprint_changes_with_the_pool_not_with_its_path():
    """Two checkouts may hold the same pool at different paths; a path is a
    location, not content."""
    pool = pool_dirs()[0]
    a = real_config(pool)
    b = real_config(pool)
    b.data["pool_dir"] = "/somewhere/else/" + pool.name
    assert a.data_fingerprint() == b.data_fingerprint()
    c = real_config(pool)
    c.data["pool_sha256"] = "f" * 64
    assert c.data_fingerprint() != a.data_fingerprint()


# ===========================================================================
# the exclusion rule that real data forced into existence
# ===========================================================================
@requires_pool
def test_exclusion_rule_holds_at_EVERY_L_not_just_the_longest():
    """The bug this encodes: a shorter history is a SUFFIX of the long one,
    so it can be flat while the full window is fine. Real sensor feeds do
    this routinely - a traffic loop reads 0 all night - and the first
    version of the rule only checked L=512.
    """
    import sys
    sys.path.insert(0, str(ROOT / "data-real"))
    from build_pool import SWEEP_LS, MIN_SCALE, L_MAX
    for pool in pool_dirs():
        W = np.load(pool / "windows.npz")["windows"]
        for L in SWEEP_LS:
            hist = W[:, L_MAX - L:L_MAX]
            sd = hist.std(axis=1)
            mad = np.median(np.abs(hist - np.median(hist, axis=1,
                                                    keepdims=True)), axis=1)
            assert sd.min() > MIN_SCALE, f"{pool.name}: flat sd at L={L}"
            assert mad.min() > MIN_SCALE, f"{pool.name}: flat MAD at L={L}"


@requires_pool
def test_evidence_block_is_finite_on_every_real_window_at_every_L():
    """The symptom the exclusion rule exists to prevent: a real item at
    L=32 rendering `largest_local_peak_robust_z=35800000.00` into the prompt
    without raising, failing a test, or showing up in coverage."""
    import sys
    sys.path.insert(0, str(ROOT / "data-real"))
    from build_pool import SWEEP_LS, L_MAX
    for pool in pool_dirs():
        W = np.load(pool / "windows.npz")["windows"]
        for L in SWEEP_LS:
            for row in W[::37]:
                e = tools.series_evidence(row[L_MAX - L:L_MAX])
                assert e["robust_scale_usable"] == 1
                assert e["max_local_peak_robust_z"] < 1e4, (
                    f"{pool.name} L={L}: robust z exploded to "
                    f"{e['max_local_peak_robust_z']:.3g}")


def test_stratum_cuts_agree_between_the_builder_and_the_loader():
    """Two copies of a pre-registered threshold are two chances to drift."""
    import sys
    sys.path.insert(0, str(ROOT / "data-real"))
    from build_pool_c3 import STRATUM_CUTS as BUILDER_CUTS
    assert tuple(BUILDER_CUTS) == tuple(RD.STRATUM_CUTS)


def test_stratum_assignment_matches_the_declared_cuts():
    lo, hi = RD.STRATUM_CUTS
    assert RD.seasonality_stratum(lo - 1e-9) == "weak"
    assert RD.seasonality_stratum(lo) == "medium"
    assert RD.seasonality_stratum(hi - 1e-9) == "medium"
    assert RD.seasonality_stratum(hi) == "strong"
    assert RD.seasonality_stratum(float("nan")) == "unknown"


# ===========================================================================
# the degenerate-window guard (handoff §4.3)
# ===========================================================================
def test_floor_guard_fires_on_a_flat_window():
    x = np.zeros(64)
    x[40] = 3.0                      # one spike in an otherwise flat feed
    e = tools.series_evidence(x)
    assert e["robust_scale_usable"] == 0
    block = tools.evidence_to_block(e)
    assert tools.UNUSABLE in block
    assert "largest_local_peak_robust_z=unavailable_flat_window" in block
    # and nothing a model could read as a measurement
    assert "e+0" not in block and "35800000" not in block


def test_floor_guard_is_invisible_on_an_ordinary_window():
    """The guard changes `evidence_to_block` OUTPUT, which is inside the call
    cache key. If it altered a well-scaled block by one byte, every cached
    cycle-2 reply would be invalidated. Synthetic sigma is drawn from
    [0.5, 2.0] and the real pool's smallest surviving noise scale is 1.7e-3,
    so both sit far above the floor and must render unchanged."""
    rng = np.random.default_rng(0)
    for seed_shift in range(20):
        x = (np.cumsum(rng.normal(0, 1.0, 128)) + 50.0 + seed_shift)
        e = tools.series_evidence(x)
        assert e["robust_scale_usable"] == 1
        block = tools.evidence_to_block(e)
        assert tools.UNUSABLE not in block
        # the exact cycle-2 rendering, reconstructed independently
        assert block.startswith("computed evidence (from the visible window "
                                "only):\n  points=128 ")
        assert block.rstrip().endswith(
            f"largest_local_peak_robust_z={e['max_local_peak_robust_z']:.2f}")


def test_floor_guard_field_is_numeric_like_every_other():
    """C2 evidence must stay numbers only: a categorical field risks being
    the key's own vocabulary (tests/test_saturation_guard.py)."""
    e = tools.series_evidence(np.zeros(32))
    assert "robust_scale_usable" in tools.C2_EVIDENCE_FIELDS
    v = e["robust_scale_usable"]
    assert isinstance(v, int) and not isinstance(v, bool)


# ===========================================================================
# the two-sample machinery the reframed H2 needs
# ===========================================================================
def test_two_sample_compare_detects_a_real_shift():
    from src.stats import two_sample_compare
    rng = np.random.default_rng(7)
    a = rng.normal(0.30, 0.1, 150)
    b = rng.normal(0.05, 0.1, 150)
    r = two_sample_compare(a, b, n_boot=500, seed=0)
    assert r.mean_diff == pytest.approx(0.25, abs=0.05)
    assert r.ci_low > 0 and r.p_raw < 1e-6
    assert r.cliffs_delta > 0.7


def test_two_sample_compare_is_null_on_one_distribution():
    from src.stats import two_sample_compare
    rng = np.random.default_rng(11)
    r = two_sample_compare(rng.normal(0, 1, 200), rng.normal(0, 1, 200),
                           n_boot=500, seed=0)
    assert r.ci_low < 0 < r.ci_high
    assert r.p_raw > 0.05
    assert abs(r.cliffs_delta) < 0.2


def test_did_refuses_the_stratum_axis():
    """L and backbone hold the same items at both ends; strata do not, so
    there is no Delta_i and pairing would be a fiction."""
    import pandas as pd
    from runners import _common as C
    from tests.conftest import make_config
    cfg = make_config(Path("."))
    with pytest.raises(ValueError, match="two-sample|different series"):
        C.confirmatory_did(pd.DataFrame(), cfg, axis="stratum",
                           first="weak", second="strong",
                           cells=["forecasting"])
