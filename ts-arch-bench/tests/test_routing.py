"""Routing, parsing and abstention — the defects that invalidated H2 and H3."""
from __future__ import annotations

import pytest

from models import MockLLM, MockTSFM
from models.llm import CLASSIFY_PROMPT, GENERIC_PROMPT, M_CLASSIFY
from src.samples import Sample, build_query
from systems.harness_tsfm import HarnessTSFMSystem, ROUTABLE_OPS
from systems.routing import (parse_route, oracle_route, VALID_OPS,
                             STATUS_CLEAN, STATUS_RECOVERED,
                             STATUS_AMBIGUOUS, STATUS_UNPARSEABLE)
from conftest import toy_window


def make_view(query_type="statistics", kind="numeric_exact", n=128):
    s = Sample("sid", toy_window(n + 8), n, 8, query_type,
               build_query(query_type, 8), kind, "trend", 1)
    return s.view(n)


# --------------------------------------------------------------------------
# the dead-branch defect
# --------------------------------------------------------------------------
def test_classify_prompt_contains_its_dispatch_marker():
    """The old template rendered `[task:classify request=s_0001]` while the
    mock looked for `[task:classify]`, so the branch was unreachable and the
    oracle fallback fired on 100% of mock samples."""
    rendered = CLASSIFY_PROMPT.format(ops="mean, max", question="q?")
    assert M_CLASSIFY in rendered


def test_mock_actually_routes_rather_than_falling_through():
    """A high-strength mock must route via the LLM path with a clean parse."""
    llm = MockLLM("bb", strength=0.95, prose_rate=0.0, seed=1)
    sysobj = HarnessTSFMSystem("h", llm, MockTSFM(), "bb")
    res = sysobj.answer(make_view("statistics"))
    cls = res.trace_dicts()[0]["detail"]
    assert cls["source"] == "llm", "routing fell through to a fallback"
    assert cls["status"] == STATUS_CLEAN
    assert cls["op"] == "mean"
    assert res.n_model_calls == 1


def test_mock_raises_on_an_unrecognised_prompt():
    """Silent catch-alls are how the old dead branch hid. Fail loudly."""
    with pytest.raises(ValueError, match="no recognised task marker"):
        MockLLM("bb").complete("sys", "a prompt with no marker at all")


def test_every_routable_op_is_valid():
    assert set(ROUTABLE_OPS) == set(VALID_OPS)


# --------------------------------------------------------------------------
# parser taxonomy
# --------------------------------------------------------------------------
def test_clean_parse():
    r = parse_route("mean")
    assert (r.op, r.status, r.source) == ("mean", STATUS_CLEAN, "llm")
    assert r.op_strict == "mean"


def test_prose_reply_is_recovered_not_dropped():
    """D2: `raw.strip().lower().split()[0]` took the first token only, so
    every prose reply fell through to the oracle fallback."""
    r = parse_route("To find the index where the change occurs, use "
                    "change_point.")
    assert r.op == "change_point"
    assert r.status == STATUS_RECOVERED
    assert r.op_strict is None and r.op_permissive == "change_point"


def test_two_candidates_abstain_rather_than_guess():
    r = parse_route("It could be pct_change or change_point.")
    assert r.op is None and r.status == STATUS_AMBIGUOUS
    assert r.abstained and set(r.candidates) == {"pct_change", "change_point"}


def test_no_candidate_abstains():
    r = parse_route("I cannot determine the answer.")
    assert r.op is None and r.status == STATUS_UNPARSEABLE and r.abstained


def test_empty_reply_abstains():
    for bad in ("", "   ", None):
        assert parse_route(bad).abstained


def test_op_names_match_on_word_boundaries():
    """`max` must not be found inside `maximum`, and `change` must not be
    matched inside `pct_change`."""
    assert parse_route("the maximum of the series").status == STATUS_UNPARSEABLE
    r = parse_route("pct_change")
    assert r.op == "pct_change" and r.candidates == ("pct_change",)


def test_both_readings_are_always_logged():
    """F2: log op_strict and op_permissive on every call so parser recovery
    is measurable offline, with no rerun."""
    for reply in ("mean", "use the mean please", "mean or max", "nothing"):
        d = parse_route(reply).as_detail()
        assert "op_strict" in d and "op_permissive" in d and "status" in d


# --------------------------------------------------------------------------
# abstention end to end
# --------------------------------------------------------------------------
def test_unroutable_reply_produces_an_abstention_not_an_oracle_answer():
    """The D1 regression test. A classifier that says nothing useful must
    yield an abstention, NOT the correct operation."""
    class Mute(MockLLM):
        def complete(self, sp, up):
            if M_CLASSIFY in up:
                return "I cannot determine the answer."
            return super().complete(sp, up)

    sysobj = HarnessTSFMSystem("h", Mute("bb", strength=0.9), MockTSFM(), "bb")
    res = sysobj.answer(make_view("statistics"))
    assert res.abstained and not res.answered
    assert res.parsed is None
    assert res.failure == "route_unparseable"
    assert res.extra["route_op"] is None


def test_oracle_arm_routes_correctly_and_is_labelled():
    sysobj = HarnessTSFMSystem("h", MockLLM("bb"), MockTSFM(), "bb",
                               routing="oracle")
    res = sysobj.answer(make_view("statistics"), oracle_op="mean")
    assert res.extra["route_source"] == "oracle"
    assert res.extra["route_op"] == "mean"
    assert res.n_model_calls == 0        # the ablation skips the classify call


def test_oracle_route_rejects_an_invalid_op():
    with pytest.raises(ValueError):
        oracle_route("not_an_operation")


# --------------------------------------------------------------------------
# the routed op must actually drive behaviour
# --------------------------------------------------------------------------
def test_qualitative_branch_follows_the_routed_op():
    """The old harness computed `op` then built the prompt from
    `sample.query_type`, so routing was a no-op on 2 of 9 types."""
    seen = {}

    class Spy(MockLLM):
        def complete(self, sp, up):
            if M_CLASSIFY in up:
                return "describe"          # deliberately route to describe
            seen["prompt"] = up
            return super().complete(sp, up)

    sysobj = HarnessTSFMSystem("h", Spy("bb", strength=0.9), MockTSFM(), "bb")
    # the ITEM is an event_explanation, but the router said "describe"
    res = sysobj.answer(make_view("event_explanation", "qualitative"))
    assert res.extra["route_op"] == "describe"
    assert "[task:describe]" in seen["prompt"], (
        "the qualitative prompt ignored the routed op and used the label")
    assert "[task:explain]" not in seen["prompt"]
