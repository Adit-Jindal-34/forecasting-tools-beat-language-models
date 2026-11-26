"""The paraphrase bank: the regression guard on the routing collapse.

Before this bank existed, every item of a query type produced a
byte-identical classify prompt, so the whole 900-item benchmark contained
NINE distinct routing decisions. Routing accuracy per cell could only be 0%
or 100%, and any self-versus-oracle test counted one decision a hundred
times. These tests fail if that comes back.
"""
from __future__ import annotations

import collections

import pytest

from models.llm import MockLLM
from models.prompts import CLASSIFY_PROMPT
from src.config import QUERY_OP
from src.pipeline import make_samples
from src.samples import (build_query, framing_of, lead_of, N_FRAMINGS,
                         N_LEADS, N_VARIANTS, PARAPHRASE_FRAMES,
                         REQUIRED_PHRASES, OPERATIONAL_CLAUSE)
from systems.harness_tsfm import ROUTABLE_OPS
from conftest import make_config

ALL_TYPES = sorted(PARAPHRASE_FRAMES)


def test_bank_shape():
    assert N_VARIANTS == N_FRAMINGS * N_LEADS == 100
    for qt in ALL_TYPES:
        assert len(PARAPHRASE_FRAMES[qt]) == N_FRAMINGS


@pytest.mark.parametrize("qt", ALL_TYPES)
def test_all_variants_render_distinct(qt):
    rendered = {build_query(qt, 8, v) for v in range(N_VARIANTS)}
    assert len(rendered) == N_VARIANTS, (
        f"{qt} collapses to {len(rendered)} distinct queries")


@pytest.mark.parametrize("qt", ALL_TYPES)
def test_operational_clause_is_preserved_verbatim(qt):
    """The mechanical guard on the authoring rule.

    Paraphrase the framing; never the operation. The MAD definition and the
    0-based-index convention ARE the answer key - reword one and the task
    silently changes under a benchmark that still claims to measure the old
    one. Human review does not scale to 900 renderings; this does.
    """
    clause = OPERATIONAL_CLAUSE[qt]
    for v in range(N_VARIANTS):
        text = build_query(qt, 8, v)
        if clause:
            assert clause.strip() in text, f"{qt} v{v} lost its clause"
        for phrase in REQUIRED_PHRASES[qt]:
            assert phrase.format(h=8) in text, (
                f"{qt} v{v} is missing required phrase {phrase!r}:\n  {text}")


@pytest.mark.parametrize("qt", ALL_TYPES)
def test_classify_prompts_vary_per_item(qt):
    """THE regression guard. At least 20 distinct classify prompts per type."""
    prompts = {CLASSIFY_PROMPT.format(ops=", ".join(sorted(ROUTABLE_OPS)),
                                      question=build_query(qt, 8, v))
               for v in range(N_VARIANTS)}
    assert len(prompts) >= 20, (
        f"{qt} yields only {len(prompts)} distinct classify prompts; routing "
        "has collapsed to a handful of decisions replicated across items")


def test_variant_decomposition():
    for v in range(N_VARIANTS):
        assert framing_of(v) == v % N_FRAMINGS
        assert lead_of(v) == v // N_FRAMINGS


@pytest.mark.parametrize("qt", ALL_TYPES)
def test_mock_infers_the_right_op_for_every_paraphrase(qt):
    """A mock that cannot route makes every mock run meaningless.

    The old keyword table matched surface wording ("describe the overall
    trend"), which the bank varies. Inference must key on the operational
    clauses the bank guarantees.
    """
    want = QUERY_OP[qt]
    bad = [v for v in range(N_VARIANTS)
           if MockLLM._infer_op(build_query(qt, 8, v)) != want]
    assert not bad, f"{qt}: mock mis-infers op on variants {bad[:5]}"


def test_pool_spreads_variants_over_both_axes(tmp_path):
    """`k % N_VARIANTS` would emit only variants 0..n-1 below full size, all
    sharing lead_id=0, making the within-framing variance check vacuous."""
    import pathlib
    cfg = make_config(tmp_path, n_base=20, n_per=20)
    samples = make_samples(cfg, pathlib.Path("."))
    loc = [s for s in samples if s.query_type == "localization"]
    assert len({s.paraphrase_id for s in loc}) == 20
    assert len({s.lead_id for s in loc}) > 1, (
        "every item shares one lead; the compositional axis is unexercised")
    assert len({s.query for s in loc}) == 20


def test_full_size_pool_gives_one_variant_per_item(tmp_path):
    import pathlib
    cfg = make_config(tmp_path, n_base=100, n_per=100, max_history=256)
    samples = make_samples(cfg, pathlib.Path("."))
    for qt in ALL_TYPES:
        items = [s for s in samples if s.query_type == qt]
        assert len({s.query for s in items}) == 100, qt
        counts = collections.Counter(s.framing_id for s in items)
        assert set(counts.values()) == {N_LEADS}, (
            f"{qt}: framings are not balanced across leads")


def test_paraphrase_id_survives_a_pool_reload(tmp_path):
    """Rebuilding the query from scratch on reload would drift from the pool
    that was actually run."""
    import pathlib
    cfg = make_config(tmp_path, n_base=20, n_per=20)
    a = make_samples(cfg, pathlib.Path("."))
    b = make_samples(cfg, pathlib.Path("."))
    assert [s.paraphrase_id for s in a] == [s.paraphrase_id for s in b]
    assert [s.query for s in a] == [s.query for s in b]


def test_query_version_is_in_the_data_fingerprint(tmp_path):
    import src
    cfg = make_config(tmp_path)
    before = cfg.data_fingerprint()
    original = src.QUERY_VERSION
    try:
        src.QUERY_VERSION = original + 1
        assert cfg.data_fingerprint() != before, (
            "editing the paraphrase bank would serve a stale pool under "
            "unchanged sample ids")
    finally:
        src.QUERY_VERSION = original
