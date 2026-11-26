"""Slot-based scoring for the two open-ended query types.

WHY THIS REPLACES KEYWORD COVERAGE
----------------------------------
The previous metric was

    coverage = |{reference keyphrases present in the answer}| / |reference|

a recall statistic with NO precision penalty, over a vocabulary containing
mutually exclusive labels. Measured on the real n100 pool, the single
constant string

    "The series is increasing, decreasing, stable, seasonal, spike,
     abrupt change, no major event, early, middle, late."

scored mean coverage 1.0000 (min 1.0000) on BOTH qualitative cells without
reading the series at all. Reported system scores were 0.6-0.8, i.e. below a
contradictory constant. The metric had no construct validity.

THE REPLACEMENT
---------------
The reference is already slotted (trend, seasonality, event, position), so
score it as what it is: a small set of mutually exclusive classification
decisions. For each applicable slot we extract which classes the answer
asserts:

    exactly one  -> that is the prediction; correct or wrong
    none         -> "absent"   (scored 0, counted separately)
    more than one-> "ambiguous"(scored 0, counted separately)

Asserting everything now yields "ambiguous" in every exclusive slot, which
scores 0. The attack string above drops from 1.000 to below 0.25 (it retains
only the chance accuracy of the one non-exclusive slot, seasonality).

KNOWN LIMITATION, REPORT IT
---------------------------
`seasonal` is a presence/absence slot, not an exclusive one: "no" is the
absence of a positive marker. A contradictory answer therefore still reads as
"yes" there and collects chance accuracy. Negation is handled for the common
phrasings ("not seasonal", "no clear seasonality"), but a presence slot
cannot express self-contradiction. This residual is why the slot scorer still
needs human validation (see PREREGISTRATION.md) rather than being trusted
outright.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.ground_truth import (QUALITATIVE_SLOTS, SLOTS_FOR_TYPE,
                              SLOTS_FOR_TYPE_C2, QualitativeTruth,
                              c2_scored_slots)

#: slot -> ordered list of (class, compiled pattern).
#: Order matters only where `NEGATION_FIRST` marks the slot: there the first
#: matching class wins outright instead of counting towards ambiguity.
_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "trend": [
        ("increasing",
         r"\b(increas\w*|ris\w*|upward|up[\s-]*trend|grow\w*|climb\w*|"
         r"ascend\w*|trending\s+up)\b"),
        ("decreasing",
         r"\b(decreas\w*|declin\w*|fall\w*|downward|down[\s-]*trend|"
         r"drop\w*|descend\w*|shrink\w*|trending\s+down)\b"),
        ("stable",
         r"\b(stable|flat|steady|unchanged|constant|sideways|"
         r"no\s+(clear\s+)?trend|little\s+change|no\s+overall\s+change)\b"),
    ],
    "seasonal": [
        # negation first: "not seasonal" also contains "seasonal"
        ("no",
         r"\b(not\s+seasonal\w*|non[\s-]*seasonal\w*|no\s+(clear\s+)?"
         r"season\w*|aperiodic|not\s+periodic|no\s+(clear\s+)?periodic\w*|"
         r"no\s+(clear\s+)?cycl\w*|without\s+season\w*)\b"),
        ("yes",
         r"\b(seasonal\w*|periodic\w*|cyclic\w*|cycle[sd]?|"
         r"repeating\s+pattern)\b"),
    ],
    "event": [
        ("no major event",
         r"\b(no\s+(major|significant|notable|clear)\s+event\w*|"
         r"nothing\s+notable|no\s+events?\b|uneventful|"
         r"no\s+(major|significant|notable)\s+(change|anomal\w*))\b"),
        ("spike", r"\b(spike[sd]?|spiking|outlier[s]?|sudden\s+peak|"
                  r"transient\s+peak)\b"),
        ("abrupt change",
         r"\b(abrupt\w*|level\s+shift|step\s+change|regime\s+(change|shift)|"
         r"structural\s+break|sudden\s+shift|jump[sed]*)\b"),
    ],
    "position": [
        ("early", r"\b(early|beginning|start|outset|initial|first\s+third)\b"),
        ("middle",
         r"\b(middle|midpoint|mid[\s-]*series|halfway|cent(re|er)|"
         r"second\s+third)\b"),
        ("late", r"\b(late|later|latter|end|final\s+third|last\s+third|"
                 r"recently|towards?\s+the\s+end)\b"),
    ],
}

#: slots where the first matching class wins instead of signalling ambiguity
NEGATION_FIRST = frozenset({"seasonal"})

_COMPILED = {slot: [(name, re.compile(pat, re.I))
                    for name, pat in entries]
             for slot, entries in _PATTERNS.items()}

# every declared class must have a pattern, and vice versa
for _slot, _classes in QUALITATIVE_SLOTS.items():
    _have = {name for name, _ in _PATTERNS[_slot]}
    if _have != set(_classes):
        raise RuntimeError(
            f"qualitative slot {_slot!r}: patterns {sorted(_have)} do not "
            f"match declared classes {sorted(_classes)}")


@dataclass(frozen=True)
class SlotPrediction:
    slot: str
    predicted: str | None       # a class name, or None
    status: str                 # "one" | "absent" | "ambiguous"
    matched: tuple[str, ...]


def extract_slot(answer: str, slot: str) -> SlotPrediction:
    """Which class of `slot` does this answer assert?"""
    text = answer or ""
    matched = tuple(name for name, pat in _COMPILED[slot]
                    if pat.search(text))
    if not matched:
        return SlotPrediction(slot, None, "absent", matched)
    if len(matched) == 1:
        return SlotPrediction(slot, matched[0], "one", matched)
    if slot in NEGATION_FIRST:
        # ordered: the earliest declared class wins (negation beats presence)
        for name, _ in _COMPILED[slot]:
            if name in matched:
                return SlotPrediction(slot, name, "one", matched)
    return SlotPrediction(slot, None, "ambiguous", matched)


#: slots where naming two classes is a self-contradiction (everything except
#: the presence/absence slot)
EXCLUSIVE_SLOTS = frozenset(QUALITATIVE_SLOTS) - NEGATION_FIRST


def score_qualitative(answer: str, truth: QualitativeTruth,
                      query_type: str) -> dict:
    """Macro-averaged per-slot accuracy, plus a full diagnostic breakdown.

    `incoherent` flags an answer that names two classes of the same
    exclusive slot. It is REPORTED, not used to zero the whole item: "rose
    then fell" is a genuine description that happens to be unscoreable in a
    single-class trend slot, and zeroing its other slots too would punish
    the phrasing rather than the content.
    """
    expected = truth.slots(query_type)
    per_slot: dict[str, dict] = {}
    n_correct = n_ambiguous = n_absent = 0
    incoherent = False
    for slot in SLOTS_FOR_TYPE[query_type]:
        pred = extract_slot(answer, slot)
        correct = int(pred.status == "one" and pred.predicted == expected[slot])
        n_correct += correct
        n_ambiguous += int(pred.status == "ambiguous")
        n_absent += int(pred.status == "absent")
        if pred.status == "ambiguous" and slot in EXCLUSIVE_SLOTS:
            incoherent = True
        per_slot[slot] = dict(expected=expected[slot],
                              predicted=pred.predicted,
                              status=pred.status, correct=correct,
                              matched="|".join(pred.matched))
    n_slots = len(expected)
    return dict(score=n_correct / n_slots, n_slots=n_slots,
                n_correct=n_correct, n_ambiguous=n_ambiguous,
                n_absent=n_absent, incoherent=int(incoherent),
                per_slot=per_slot, scoring="slot_accuracy")


# --------------------------------------------------------------------------
# C2: closed-set multiple choice  (PREREGISTRATION_C2.md §C2-3, §C2-4)
# --------------------------------------------------------------------------
# The model is asked for one line per slot, `slot: option`. Parsing that is
# unambiguous, which removes three cycle-1 defects at once:
#
#   * the `seasonal` negation regex scored "does not exhibit any seasonal
#     patterns" as `yes`. There is no negation to handle here: the model
#     writes the option it chose.
#   * correct-but-hedged prose matched two classes and scored 0 as
#     `ambiguous` on 24/100 `event_explanation` items.
#   * `absent` (no answer) and wrong were indistinguishable without a judge.
#
# Scoring is still `n_correct / n_slots` over the asked slots only, so the
# primary metric's definition is unchanged - only the extraction is.

#: Leading list/markdown noise: bullets, enumeration, quoting, bold.
#: Weak models emit all of these, and a parser that reads them as "no answer"
#: biases exactly the axis H3 measures.
#: Leading list/markdown noise: bullets, enumeration, quoting, bold, code.
#: Weak models emit all of these, and a parser that reads them as "no answer"
#: biases exactly the axis H3 measures.
_MC_PREFIX = r"[\s>*_#`'\"\-•]*(?:\d+[.)]\s*)?[\s>*_#`'\"\-]*"
#: The slot NAME may itself be wrapped: `trend`: , **trend**: , 'trend':
_MC_WRAP = r"[`*_'\"]*"
#: Separator between the slot name and the option. Cycle-1 answers used
#: `=`; models also use `-`, en/em dashes, and decoration before the colon.
_MC_SEP = r"[\s*_`]*[:=\-–—]\s*"

#: Where a `slot: option` statement may begin.
#:
#:   * at the START OF A LINE, after list/markdown/quote decoration - the
#:     format the question asks for;
#:   * or MID-LINE after whitespace, a comma, a semicolon or a bracket, but
#:     then only with `:` or `=` as the separator. The question asks for two
#:     lines; a model that writes "trend: stable, seasonal: no" on one line,
#:     or copies the question's quotes ('trend: stable' and 'seasonal: no'),
#:     has still answered, and reading that as "no answer" would penalise
#:     formatting, hardest on the weak rungs of H3. A dash is NOT accepted
#:     mid-line, or "trend-following" and "event - driven" would match.
_MC_LINE = {
    slot: re.compile(
        rf"(?:^{_MC_PREFIX}{_MC_WRAP}{slot}{_MC_WRAP}{_MC_SEP}"
        rf"|(?<=[\s,;(]){_MC_WRAP}{slot}{_MC_WRAP}\s*[:=]\s*)"
        rf"(.+)$",
        re.I | re.M)
    for slot in QUALITATIVE_SLOTS
}

#: The start of ANOTHER slot's statement inside a captured value. A one-line
#: answer captures everything to the end of the line, so the value is cut
#: here: "stable, seasonal: no" is `stable` for the trend slot.
_MC_NEXT_SLOT = re.compile(
    r"[\s,;('\"`*_]+(?:and\s+)?['\"`*_]*(?:"
    + "|".join(QUALITATIVE_SLOTS) + r")['\"`*_]*\s*[:=]", re.I)

#: Words that invert the option that follows them. `event: no spike` must not
#: be read as `spike`; under cycle 1 the equivalent failure scored "does not
#: exhibit any seasonal patterns" as `yes`, and repeating it inside the
#: multiple-choice scorer would be the same defect in a new place.
_MC_NEGATION = re.compile(
    r"\b(?:no|not|non|none|without|lacks?|absent|isn'?t|aren'?t|"
    r"doesn'?t|don'?t|never|neither|nor)\b")

#: Clause separators. Negation must NOT leak across them: in
#: "not early, late" the `not` governs `early` only, and scoping it to the
#: whole value cancelled `late` as well, so the line read as no answer.
_MC_CLAUSE = re.compile(
    r"\s*(?:,|;|/|\||\bbut\b|\bthen\b|\band\b)\s*")
# `or` is deliberately NOT a separator: in "not increasing or decreasing" the
# negation governs both options, and splitting there read the second one as
# affirmed (review #4).

#: A value that denies the slot outright instead of naming an option:
#: `seasonal: none`, `seasonal: not seasonal`. On a TWO-class slot that
#: determines the answer; on a three-class slot it does not, and is recorded
#: `absent` rather than guessed.
_MC_DENIES_SLOT = re.compile(
    r"^(?:none|nothing|n/?a|absent|no|not\s+\S+|non[\s-]?\S+)$")


def _normalise_value(v: str) -> str:
    """Lowercase, strip decoration, and unify `abrupt-change`/`abrupt_change`.

    Class names contain spaces (`abrupt change`, `no major event`); models
    write them hyphenated or underscored, which a word-boundary search for
    the spaced form never matches.
    """
    m = _MC_NEXT_SLOT.search(v)
    if m:
        v = v[:m.start()]
    v = v.strip().strip("*_`'\"<>").strip().strip(".,;:").strip("'\"<>")
    v = v.strip().lower()
    return re.sub(r"[-_]+", " ", v).strip()


def _clause_verdicts(value: str, cls: str):
    """(affirmed, negated) for `cls`, judged clause by clause."""
    affirmed = negated = False
    pat = re.compile(rf"\b{re.escape(cls)}s?\b")     # plural tolerated
    for clause in _MC_CLAUSE.split(value):
        m = pat.search(clause)
        if not m:
            continue
        if _MC_NEGATION.search(clause[:m.start()]):
            negated = True
        else:
            affirmed = True
    return affirmed, negated


def extract_slot_mc(answer: str, slot: str) -> SlotPrediction:
    """Read `slot: option` from a formatted answer.

    Order matters. An exact match on the whole value wins first, because
    `no major event` is a declared class that contains a negation word and
    any negation handling would otherwise reject the correct answer. Then
    options are judged per clause, so a negation governs only its own clause.
    Several lines naming different options is ambiguity, not the first one.
    """
    classes = QUALITATIVE_SLOTS[slot]
    lines = _MC_LINE[slot].findall(answer or "")
    if not lines:
        return SlotPrediction(slot, None, "absent", ())

    picks: list[str] = []
    denied = False
    for raw in lines:
        value = _normalise_value(raw)

        exact = next((c for c in classes
                      if value in (c.lower(), c.lower() + "s")), None)
        if exact is not None:
            picks.append(exact)
            continue

        # The question's own option list echoed back ("yes | no",
        # "increasing | decreasing | stable") is not an answer. Checked
        # before the leading-option rule, which would read "yes | no" as yes.
        tokens = [t.strip() for t in
                  re.split(r"\s*(?:\||/|,|\bor\b)\s*", value) if t.strip()]
        if len(tokens) > 1 and set(tokens) == {c.lower() for c in classes}:
            continue

        # On a yes/no slot a leading option settles it: "yes, there is no
        # doubt" is `yes`, and the later "no" is prose, not a second answer.
        if len(classes) == 2:
            lead = next((c for c in classes
                         if re.match(rf"{re.escape(c.lower())}\b", value)),
                        None)
            if lead is not None:
                picks.append(lead)
                continue

        aff, neg = [], []
        for c in classes:
            a, n = _clause_verdicts(value, c.lower())
            if a:
                aff.append(c)
            elif n:
                neg.append(c)

        if len(aff) == 1:
            picks.append(aff[0])
        elif len(aff) == len(classes):
            # Every option named: this is the question's own template
            # echoed back ("trend: <increasing | decreasing | stable>"),
            # not an answer. Skip it, so a model that restates the format
            # and THEN answers is read by its answer.
            continue
        elif len(aff) > 1:
            return SlotPrediction(slot, None, "ambiguous", tuple(aff))
        elif neg and len(classes) == 2:
            other = [c for c in classes if c not in neg]
            if len(other) == 1:
                picks.append(other[0])
        elif _MC_DENIES_SLOT.match(value) and len(classes) == 2:
            # "none" / "not seasonal": denies presence. For yes/no that is
            # `no`; review #3 reads it the same way.
            no_like = next((c for c in classes if c.lower() == "no"), None)
            if no_like is not None:
                picks.append(no_like)
            else:
                denied = True
        elif neg:
            denied = True

    uniq = sorted(set(picks))
    if len(uniq) == 1:
        return SlotPrediction(slot, uniq[0], "one", (uniq[0],))
    if len(uniq) > 1:
        # e.g. `trend: increasing` on one line and `trend: decreasing` on
        # another. Reading the first silently would reward a model that
        # hedges across lines.
        return SlotPrediction(slot, None, "ambiguous", tuple(uniq))
    return SlotPrediction(slot, None, "absent", () if not denied else ("-",))


def score_qualitative_mc(answer: str, truth: QualitativeTruth,
                         query_type: str) -> dict:
    """C2 scorer: multiple-choice extraction over the ASKED slots only.

    Returns the same shape as `score_qualitative`, so every downstream
    consumer (evaluate, the shard columns, the decomposition) works unchanged.
    """
    # `c2_scored_slots` drops `position` when there is no event to locate
    expected = c2_scored_slots(truth, query_type)
    slots = tuple(expected)

    per_slot: dict[str, dict] = {}
    n_correct = n_ambiguous = n_absent = 0
    incoherent = False
    for slot in slots:
        pred = extract_slot_mc(answer, slot)
        correct = int(pred.status == "one" and pred.predicted == expected[slot])
        n_correct += correct
        n_ambiguous += int(pred.status == "ambiguous")
        n_absent += int(pred.status == "absent")
        if pred.status == "ambiguous" and slot in EXCLUSIVE_SLOTS:
            incoherent = True
        per_slot[slot] = dict(expected=expected[slot],
                              predicted=pred.predicted,
                              status=pred.status, correct=correct,
                              matched="|".join(pred.matched))
    n_slots = len(expected)
    return dict(score=n_correct / n_slots, n_slots=n_slots,
                n_correct=n_correct, n_ambiguous=n_ambiguous,
                n_absent=n_absent, incoherent=int(incoherent),
                per_slot=per_slot, scoring="slot_accuracy_mc")


#: the exact string that scored 1.000 under the old metric; kept here so the
#: regression test and the paper quote the same literal
GAMING_PROBE = ("The series is increasing, decreasing, stable, seasonal, "
                "spike, abrupt change, no major event, early, middle, late.")
