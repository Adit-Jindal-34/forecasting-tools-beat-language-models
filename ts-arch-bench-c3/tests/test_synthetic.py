"""Item design: the pool must not contain a degenerate cell.

These assertions are the reason the old `counting` cell (83/100 zeros) and
the unreachable `abrupt change` label cannot come back. A regression here
fails the build rather than reaching a paper.
"""
from __future__ import annotations

import collections

import numpy as np
import pytest

from src.ground_truth import (ground_truth, qualitative_truth,
                              SLOTS_FOR_TYPE, QUALITATIVE_SLOTS,
                              has_level_shift, local_noise_scale,
                              change_point_index)
from src.pipeline import make_samples
from conftest import make_config

L = 128


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pool100")
    cfg = make_config(tmp, n_base=100, n_per=100, max_history=256)
    import pathlib
    return cfg, make_samples(cfg, pathlib.Path("."))


def _of_type(samples, qt):
    return [s for s in samples if s.query_type == qt]


def test_counting_is_not_degenerate(pool):
    """Was 0 in 83/100 items, which silently turned the cell into
    'will the model say zero?'."""
    _, samples = pool
    vals = [int(ground_truth(s, L)) for s in _of_type(samples, "counting")]
    zero_rate = vals.count(0) / len(vals)
    assert zero_rate < 0.55, f"counting is {zero_rate:.0%} zeros"
    assert len(set(vals)) >= 4, f"only {len(set(vals))} distinct counts"


def test_statistics_and_change_are_well_conditioned(pool):
    """Near-zero truths made the old relative-error metric explode. The
    metric no longer divides by the truth, but a pool centred on zero is
    still a bad item design."""
    _, samples = pool
    for qt in ("statistics", "extremum"):
        g = np.array([float(ground_truth(s, L)) for s in _of_type(samples, qt)])
        assert np.all(np.abs(g) > 1.0), f"{qt} has truths at zero"
    anchors = np.array([float(s.window(L)[0])
                        for s in _of_type(samples, "change")])
    assert np.all(np.abs(anchors) > 1.0), "pct_change anchor sits near zero"


def test_every_qualitative_class_is_reachable(pool):
    """`abrupt change` never fired in the old pool because the shift test
    compared against std(x), which the shift itself inflates."""
    _, samples = pool
    truths = [qualitative_truth(s, L)
              for s in _of_type(samples, "event_explanation")]
    for slot in SLOTS_FOR_TYPE["event_explanation"]:
        seen = {getattr(t, slot) for t in truths}
        missing = set(QUALITATIVE_SLOTS[slot]) - seen
        assert not missing, f"slot {slot}: class(es) {missing} never occur"


def test_no_qualitative_slot_is_overwhelmingly_one_class(pool):
    """A slot whose majority class exceeds 85% carries almost no signal."""
    _, samples = pool
    truths = [qualitative_truth(s, L)
              for s in _of_type(samples, "event_explanation")]
    for slot in SLOTS_FOR_TYPE["event_explanation"]:
        counts = collections.Counter(getattr(t, slot) for t in truths)
        rate = max(counts.values()) / len(truths)
        assert rate <= 0.85, (
            f"slot {slot} is {rate:.0%} one class: {dict(counts)}")


def test_localization_answers_are_spread(pool):
    _, samples = pool
    idx = [int(ground_truth(s, L)) for s in _of_type(samples, "localization")]
    assert len(set(idx)) >= 20, "change points cluster on a few indices"
    assert all(0 <= i <= L - 2 for i in idx)


def test_level_shift_test_can_actually_fire():
    """Direct regression on the self-defeating threshold."""
    x = np.concatenate([np.zeros(64), np.full(64, 10.0)])
    x += np.random.default_rng(0).normal(0, 0.2, 128)
    idx = change_point_index(x)
    assert has_level_shift(x, idx), (
        "a clean 50-sigma level shift is not detected; the threshold is "
        "scaled by a quantity the shift itself inflates")
    flat = np.random.default_rng(1).normal(0, 1.0, 128)
    assert not has_level_shift(flat, change_point_index(flat))


def test_local_noise_scale_ignores_a_single_jump():
    base = np.random.default_rng(2).normal(0, 1.0, 200)
    jumped = base.copy()
    jumped[100:] += 50.0
    assert local_noise_scale(jumped) == pytest.approx(
        local_noise_scale(base), rel=0.25)


def test_pool_is_deterministic_and_shared_across_types(tmp_path):
    cfg = make_config(tmp_path, n_base=20, n_per=20)
    import pathlib
    a = make_samples(cfg, pathlib.Path("."))
    b = make_samples(cfg, pathlib.Path("."))
    assert [s.sample_id for s in a] == [s.sample_id for s in b]
    assert np.array_equal(a[0].series, b[0].series)
    # every query type draws from the same base pool (so cells are paired)
    by_type = collections.defaultdict(set)
    for s in a:
        by_type[s.query_type].add(s.metadata["base_index"])
    sizes = {len(v) for v in by_type.values()}
    assert sizes == {20}


def test_sample_ids_do_not_collide_on_a_prefix(tmp_path):
    """The old ids used `query_type[:4]`, which collides the moment someone
    adds a type."""
    cfg = make_config(tmp_path, n_base=20, n_per=20)
    import pathlib
    samples = make_samples(cfg, pathlib.Path("."))
    ids = [s.sample_id for s in samples]
    assert len(ids) == len(set(ids))
    for s in samples:
        assert s.query_type in s.sample_id
