"""C2 scoring: multiple-choice parsing, and rules that must match the code.

`models/prompts.py` cites this file. It did not exist when that citation was
written - review #2, claim 7 - so the citation is now true.

PARSER ROBUSTNESS IS A BIAS QUESTION, NOT A TIDINESS ONE
--------------------------------------------------------
A parser that reads a weak model's formatting as "no answer" biases exactly
the axis H3 measures. Review #2 found four such cases, all reproduced:

    '**trend**: increasing'  -> absent   (markdown bold)
    '1. trend: stable'       -> absent   (enumeration)
    'event - spike'          -> absent   (dash separator)
    'event: no spike'        -> spike    <-- negation read as its opposite

The last is cycle 1's negation defect reappearing inside the scorer that was
supposed to have removed it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.qualitative import (extract_slot_mc,          # noqa: E402
                                    score_qualitative_mc)
from src.ground_truth import (QUALITATIVE_SLOTS,              # noqa: E402
                              SLOTS_FOR_TYPE_C2, QualitativeTruth)


# ---------------------------------------------------------------------------
# the four reported failures
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,slot,expected", [
    ("**trend**: increasing", "trend", "increasing"),
    ("1. trend: stable", "trend", "stable"),
    ("event - spike", "event", "spike"),
    # negation must NOT be read as the option it negates
    ("event: no spike", "event", None),
    ("trend: not increasing", "trend", None),
])
def test_review2_parser_cases(text, slot, expected):
    assert extract_slot_mc(text, slot).predicted == expected


@pytest.mark.parametrize("text,slot,expected", [
    ("trend: increasing", "trend", "increasing"),
    ("trend:increasing", "trend", "increasing"),
    ("trend = stable", "trend", "stable"),
    ("Trend: Increasing", "trend", "increasing"),
    ("- seasonal: no", "seasonal", "no"),
    ("- **Trend** — Increasing.", "trend", "increasing"),
    ("2) event: abrupt change", "event", "abrupt change"),
    ("> position: late", "position", "late"),
    ("  * seasonal: yes  ", "seasonal", "yes"),
    # `no major event` is itself a class and contains a negation word: the
    # exact-value match must win before any negation handling runs
    ("event: no major event", "event", "no major event"),
    # review #3 reads a denied two-class slot as the other class, and that
    # is the better reading: on yes/no, denying the slot IS "no". On a
    # three-class slot it stays `absent`, because rejecting one option does
    # not name another.
    ("seasonal: not seasonal", "seasonal", "no"),
    ("seasonal: none", "seasonal", "no"),
    ("event: none", "event", None),
    # review #3, claim 3: the remaining misreads
    ("position: not early, late", "position", "late"),
    ("event: abrupt-change", "event", "abrupt change"),
    ("event: abrupt_change", "event", "abrupt change"),
    ("`trend`: stable", "trend", "stable"),
])
def test_formatting_variants(text, slot, expected):
    assert extract_slot_mc(text, slot).predicted == expected


def test_negation_is_scoped_to_its_clause():
    """`not early, late` must answer `late`.

    Review #3, claim 3: scoping the negation to the whole value let a `not`
    anywhere earlier cancel every option after it, so a line that named the
    right answer read as no answer at all.
    """
    assert extract_slot_mc("position: not early, late",
                           "position").predicted == "late"
    assert extract_slot_mc("trend: not increasing; decreasing",
                           "trend").predicted == "decreasing"


def test_two_lines_naming_different_options_is_ambiguous():
    """Reading the first line silently would reward hedging across lines."""
    p = extract_slot_mc("trend: increasing\ntrend: decreasing", "trend")
    assert p.status == "ambiguous" and p.predicted is None
    # the same option twice is not ambiguity
    assert extract_slot_mc("trend: stable\ntrend: stable",
                           "trend").predicted == "stable"


def test_class_names_may_be_hyphenated_or_underscored():
    for v in ("abrupt change", "abrupt-change", "abrupt_change"):
        assert extract_slot_mc(f"event: {v}", "event").predicted ==             "abrupt change"


def test_negation_only_resolves_a_two_class_slot():
    """"not yes" determines a yes/no slot; "no spike" does not determine event.

    With two classes, rejecting one names the other. With three, rejecting one
    leaves two, which is not an answer - recorded `absent`, never guessed.
    """
    assert extract_slot_mc("seasonal: not yes", "seasonal").predicted == "no"
    p = extract_slot_mc("event: not spike", "event")
    assert p.predicted is None and p.status == "absent"


def test_naming_two_options_is_ambiguous_not_a_guess():
    p = extract_slot_mc("trend: increasing or decreasing", "trend")
    assert p.status == "ambiguous" and p.predicted is None


def test_missing_line_is_absent_not_wrong():
    """`absent` and wrong must stay distinguishable - cycle 1 could not."""
    p = extract_slot_mc("The series goes up a bit.", "trend")
    assert p.status == "absent"


# ---------------------------------------------------------------------------
# review #4: the four misreads that were still open
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,slot,expected", [
    ("seasonal: non-seasonal", "seasonal", "no"),
    ("event: no major events", "event", "no major event"),
    # `or` does not end a negation's scope
    ("trend: stable, not increasing or decreasing", "trend", "stable"),
    # on a yes/no slot the leading option settles it
    ("seasonal: yes, there is no doubt", "seasonal", "yes"),
    # the question tells models to write this when there is no event
    ("position: none", "position", None),
])
def test_review4_parser_cases(text, slot, expected):
    assert extract_slot_mc(text, slot).predicted == expected


# ---------------------------------------------------------------------------
# pre-run self-check (2026-11-11): replies a real chat model plausibly gives
# ---------------------------------------------------------------------------
# The mock always answers in the clean two-line form, so nothing before this
# exercised the ways a real model deviates from it. Ten of these forty-odd
# readings were wrong (all read as "no answer") until parser v5: quotes
# copied from the question, and both statements on one line. Left unfixed,
# formatting would have been scored as ignorance, hardest on weak models.
_T, _S, _E, _P = "trend", "seasonal", "event", "position"


@pytest.mark.parametrize("text,want", [
    ("'trend: increasing'\n'seasonal: yes'", {_T: "increasing", _S: "yes"}),
    ('"trend: decreasing"\n"seasonal: no"', {_T: "decreasing", _S: "no"}),
    ("trend: <stable>\nseasonal: <no>", {_T: "stable", _S: "no"}),
    ("trend: increasing and seasonal: yes", {_T: "increasing", _S: "yes"}),
    ("trend: stable, seasonal: no", {_T: "stable", _S: "no"}),
    ("'trend: stable' and 'seasonal: no'", {_T: "stable", _S: "no"}),
    ("event: abrupt change; position: middle",
     {_E: "abrupt change", _P: "middle"}),
    ("event: spike, position: early", {_E: "spike", _P: "early"}),
    ("Here is my answer:\ntrend: decreasing\nseasonal: yes",
     {_T: "decreasing", _S: "yes"}),
    ("Based on the series:\n\n**trend:** stable\n**seasonal:** yes",
     {_T: "stable", _S: "yes"}),
    ("```\ntrend: increasing\nseasonal: no\n```",
     {_T: "increasing", _S: "no"}),
    ("trend: increasing (slope t = 5.2)\nseasonal: no (acf 0.21)",
     {_T: "increasing", _S: "no"}),
    ("event: abrupt change (a level shift of about 12)\nposition: late",
     {_E: "abrupt change", _P: "late"}),
    # the question's template echoed back is NOT an answer ...
    ("trend: <increasing | decreasing | stable>\nseasonal: <yes | no>",
     {_T: None, _S: None}),
    # ... but a model that restates it and THEN answers is read by its answer
    ("Format: trend: <increasing | decreasing | stable>\ntrend: stable",
     {_T: "stable"}),
    # prose that merely mentions a slot word must not be read as a statement
    ("There is no clear seasonal pattern: the series looks noisy.",
     {_S: None}),
    ("A trend-following series with an event - driven jump.",
     {_T: None, _E: None}),
    ("I cannot determine this from the data.", {_T: None, _S: None}),
    ("event: no major event.\nposition: N/A",
     {_E: "no major event", _P: None}),
])
def test_realistic_model_replies(text, want):
    for slot, expected in want.items():
        assert extract_slot_mc(text, slot).predicted == expected, (text, slot)


# ---------------------------------------------------------------------------
# the options OFFERED in the question must be the options SCORED
# ---------------------------------------------------------------------------
def test_question_offers_exactly_the_scored_options():
    """src/samples.py duplicates the slot classes (ground_truth imports it,
    so it cannot import them back). The two copies must never diverge, or a
    model would be offered an option that can never be right."""
    from src import samples as SM
    assert SM.C2_ASKED_SLOTS == dict(SLOTS_FOR_TYPE_C2)
    for slot, options in SM.C2_SLOT_OPTIONS.items():
        assert tuple(options) == tuple(QUALITATIVE_SLOTS[slot])


def test_a_compliant_answer_to_the_real_question_parses():
    """Follow the format the question states, literally, and be read."""
    assert extract_slot_mc("trend: decreasing\nseasonal: yes",
                           "trend").predicted == "decreasing"
    assert extract_slot_mc("event: abrupt change\nposition: middle",
                           "event").predicted == "abrupt change"
    assert extract_slot_mc("event: no major event\nposition: none",
                           "event").predicted == "no major event"


# ---------------------------------------------------------------------------
# the scorer's contract
# ---------------------------------------------------------------------------
def test_scores_only_the_asked_slots():
    t = QualitativeTruth(trend="increasing", seasonal="no",
                         event="spike", position="late")
    r = score_qualitative_mc("event: spike\nposition: late", t,
                             "event_explanation")
    assert r["n_slots"] == 2 and r["score"] == 1.0


def test_absent_and_wrong_are_counted_separately():
    t = QualitativeTruth(trend="increasing", seasonal="no")
    r = score_qualitative_mc("trend: decreasing", t, "trend_description")
    assert r["n_correct"] == 0
    assert r["n_absent"] == 1          # seasonal line missing
    assert r["score"] == 0.0


def test_position_is_not_scored_when_there_is_no_event():
    """Where nothing happened has no right answer, so it is not counted."""
    t = QualitativeTruth(trend="stable", seasonal="no",
                         event="no major event", position=None)
    for answer in ("event: no major event\nposition: none",
                   "event: no major event\nposition: late",
                   "event: no major event"):
        r = score_qualitative_mc(answer, t, "event_explanation")
        assert r["n_slots"] == 1 and r["score"] == 1.0, answer
    r = score_qualitative_mc("event: spike\nposition: late", t,
                             "event_explanation")
    assert r["n_slots"] == 1 and r["score"] == 0.0
