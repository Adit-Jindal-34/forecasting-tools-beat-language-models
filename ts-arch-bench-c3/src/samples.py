"""Dataset interface, the sample model, and the inference-time view.

THE CENTRAL INVARIANT OF THIS BUILD
-----------------------------------
A system under test never receives an object that carries ground truth.

The runner holds a full `Sample`, which knows `query_type` (the label),
`seasonal_period` (a generator parameter) and the full series including the
forecast target. A system receives an `InferenceSample`, which carries only
what a deployed system could actually observe:

    sample_id   an opaque handle, for joining results back
    query       the natural-language question
    window      the visible history, length L, as a plain array
    L           the history length
    horizon     how many steps ahead to forecast (stated in the query text)

`InferenceSample` is frozen and has no reference back to its `Sample`.

THE PARAPHRASE BANK, AND WHY IT EXISTS
--------------------------------------
Earlier revisions of this build used ONE fixed question template per query
type. The router sees only question text, so that made every item of a type
produce a byte-identical classify prompt: at temperature 0 the whole
benchmark contained just NINE distinct routing decisions, replicated 100
times each. Routing accuracy per cell could only ever be 0% or 100%, and any
self-versus-oracle test was pseudo-replication.

(The original repo hid this by interpolating `request_id=sample.sample_id`
into the classify prompt. That did not create real variation - it sampled the
model's decision boundary under an irrelevant nonce. The published "73 of 100
localization mis-routes" is therefore one question resampled under a nonce,
not 100 independent items. It needs restating in the paper.)

Each query type now has a bank of `N_VARIANTS = 100` distinct phrasings,
built compositionally from `N_FRAMINGS = 25` task framings and `N_LEADS = 4`
instruction leads:

    variant v  ->  LEADS[v // 25] + FRAMES[qt][v % 25] + CLAUSE[qt]

`framing_id = v % 25` and `lead_id = v // 25` are both recorded on the
Sample. Whether the lead axis moves routing at all is an EMPIRICAL question,
so `src/screening.py::routing_variance_within_framing` measures it and a
pre-registered rule decides the clustering level. Do not assume n = 100.

THE OPERATIONAL CLAUSE IS LOAD-BEARING
--------------------------------------
Paraphrase the framing; never the operation. "A spike is a local maximum
whose value exceeds the median by more than 3 times the median absolute
deviation" defines the answer key - rewording it silently changes what is
being asked. `OPERATIONAL_CLAUSE` is appended verbatim to every framing of a
type, and `REQUIRED_PHRASES` is checked at import by
`_validate_paraphrase_bank()`, so a careless edit fails the build rather
than quietly redefining the task.

Leakage invariants (asserted in tests/test_leakage.py):
  I1  window(L1) is a strict suffix of window(L2) for L1 < L2.
  I2  target() is identical for every L.
  I3  No index >= max_history is ever part of any window.
  I4  Observation ground truths are computed from window(L) only.
  I5  InferenceSample exposes no ground-truth-bearing attribute.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class InferenceSample:
    """What a system under test is allowed to see. Frozen on purpose."""

    sample_id: str
    query: str
    window: np.ndarray
    L: int
    horizon: int

    def __post_init__(self) -> None:
        if len(self.window) != self.L:
            raise ValueError(
                f"InferenceSample {self.sample_id}: window has "
                f"{len(self.window)} points but L={self.L}")


@dataclass
class Sample:
    """The runner-side record. Never handed to a system."""

    sample_id: str
    series: np.ndarray          # length max_history + horizon
    max_history: int            # N
    horizon: int
    query_type: str             # GROUND TRUTH LABEL - runner only
    query: str
    query_kind: str             # forecast | numeric_exact | localization | qualitative
    pattern: str                # generator parameter - runner only
    seasonal_period: int        # generator parameter - runner only
    paraphrase_id: int = 0      # 0..N_VARIANTS-1
    framing_id: int = 0         # paraphrase_id % N_FRAMINGS
    lead_id: int = 0            # paraphrase_id // N_FRAMINGS
    metadata: dict = field(default_factory=dict)

    def window(self, L: int) -> np.ndarray:
        """Visible history for history length L (the last L points)."""
        if L > self.max_history:
            raise ValueError(f"L={L} exceeds max_history={self.max_history}")
        if L < 2:
            raise ValueError(f"L={L} is too short to be a time series")
        return self.series[self.max_history - L:self.max_history]

    def target(self) -> np.ndarray:
        """The fixed future beyond the history end. Identical for all L."""
        return self.series[self.max_history:self.max_history + self.horizon]

    def view(self, L: int) -> InferenceSample:
        """The inference-time projection. This is what `system.answer` gets."""
        return InferenceSample(
            sample_id=self.sample_id, query=self.query,
            window=self.window(L).copy(), L=int(L), horizon=int(self.horizon))


# ===========================================================================
# paraphrase bank
# ===========================================================================
N_FRAMINGS = 25
N_LEADS = 4
N_VARIANTS = N_FRAMINGS * N_LEADS          # 100

#: Instruction leads, shared across types. This is the compositional axis
#: whose effect on routing is measured, not assumed.
PARAPHRASE_LEADS: tuple[str, ...] = (
    "",
    "Consider the time series provided. ",
    "Using only the values given, answer the following. ",
    "Examine the series below before answering. ",
)

#: Appended verbatim to every framing of a type. This text DEFINES the
#: operation and must never be paraphrased.
OPERATIONAL_CLAUSE: dict[str, str] = {
    "forecasting": "",
    "statistics": "",
    "counting": (" A spike is a local maximum whose value exceeds the median "
                 "by more than 3 times the median absolute deviation."),
    "extremum": "",
    "change": "",
    "correlation": (" This is the Pearson correlation between the series and "
                    "itself shifted by one step."),
    "localization": (" Report the 0-based index i of the step from i to i+1."),
    "trend_description": "",
    "event_explanation": "",
}

#: Every rendered query of a type must contain all of these, case-sensitive.
#: Checked at import. This is the mechanical guard on the authoring rule.
REQUIRED_PHRASES: dict[str, tuple[str, ...]] = {
    "forecasting": ("next {h} values",),
    "statistics": ("arithmetic mean",),
    "counting": ("median absolute deviation",),
    "extremum": ("maximum value",),
    "change": ("percent change", "first", "last"),
    "correlation": ("lag-1 autocorrelation", "shifted by one step"),
    "localization": ("largest absolute one-step change", "0-based index"),
    "trend_description": ("trend", "seasonal"),
    "event_explanation": ("early", "middle", "late"),
}

PARAPHRASE_FRAMES: dict[str, tuple[str, ...]] = {
    "forecasting": (
        "Forecast the next {h} values of the series.",
        "What are the next {h} values of this series?",
        "Predict the next {h} values.",
        "Produce a forecast of the next {h} values.",
        "Give the next {h} values that follow this series.",
        "Extend the series by the next {h} values.",
        "Report the next {h} values.",
        "Estimate the next {h} values of the series.",
        "Continue the series and provide the next {h} values.",
        "Project the next {h} values.",
        "I need the next {h} values of this series.",
        "Output the next {h} values.",
        "Compute a forecast for the next {h} values.",
        "Determine the next {h} values of the series.",
        "Return the next {h} values.",
        "Looking forward, what are the next {h} values?",
        "What will the next {h} values be?",
        "Supply the next {h} values of the series.",
        "Generate the next {h} values.",
        "State the next {h} values.",
        "Please forecast the next {h} values.",
        "Work out the next {h} values of this series.",
        "Provide a point forecast for the next {h} values.",
        "Calculate the next {h} values.",
        "Tell me the next {h} values of the series.",
    ),
    "statistics": (
        "What is the arithmetic mean of the series?",
        "Compute the arithmetic mean of the series.",
        "Report the arithmetic mean of these values.",
        "What is the arithmetic mean of this time series?",
        "Calculate the arithmetic mean.",
        "Give the arithmetic mean of the series.",
        "Determine the arithmetic mean of the values.",
        "I need the arithmetic mean of the series.",
        "State the arithmetic mean of this series.",
        "Find the arithmetic mean of the series.",
        "What is the arithmetic mean across all points?",
        "Return the arithmetic mean of the series.",
        "Work out the arithmetic mean.",
        "Averaging the series, what is the arithmetic mean?",
        "Provide the arithmetic mean of these observations.",
        "What is this series' arithmetic mean?",
        "Evaluate the arithmetic mean of the series.",
        "Obtain the arithmetic mean of the data.",
        "Please compute the arithmetic mean.",
        "Summarise the series by its arithmetic mean.",
        "Which value is the arithmetic mean of the series?",
        "Output the arithmetic mean.",
        "Tell me the arithmetic mean of the series.",
        "Produce the arithmetic mean of the values.",
        "Measure the arithmetic mean of this series.",
    ),
    "counting": (
        "How many spike events occur in the series?",
        "Count the spike events in this series.",
        "How many times does the series spike?",
        "Report the number of spike events.",
        "What is the number of spikes in the series?",
        "Determine how many spike events are present.",
        "How many spikes can be found in these values?",
        "Give the count of spike events in the series.",
        "Tally the spike events in this series.",
        "State how many spikes occur.",
        "I need the number of spike events in the series.",
        "Find the number of spikes in the data.",
        "How many spike events does this series contain?",
        "Return the spike count for the series.",
        "Work out how many spikes are present.",
        "Please count the spikes in this series.",
        "What is the total number of spike events?",
        "Output the number of spikes.",
        "Identify how many spike events occur.",
        "Compute the number of spikes in the series.",
        "How many separate spikes appear in the series?",
        "Provide a count of the spike events.",
        "Measure how many spikes the series contains.",
        "Enumerate the spike events in this series.",
        "Tell me the number of spikes in the series.",
    ),
    "extremum": (
        "What is the maximum value of the series?",
        "Report the maximum value in this series.",
        "Find the maximum value of the series.",
        "What is the largest maximum value observed?",
        "Give the maximum value of these observations.",
        "Determine the maximum value of the series.",
        "State the maximum value reached by the series.",
        "I need the maximum value of this series.",
        "Return the maximum value.",
        "Which maximum value does the series attain?",
        "Compute the maximum value of the data.",
        "Output the maximum value of the series.",
        "What maximum value does this series reach?",
        "Identify the maximum value in the series.",
        "Work out the maximum value of these values.",
        "Please report the maximum value.",
        "Provide the maximum value of the series.",
        "Measure the maximum value attained.",
        "Tell me the maximum value of the series.",
        "What is the peak, i.e. the maximum value?",
        "Extract the maximum value from the series.",
        "Locate the maximum value of this series.",
        "Evaluate the maximum value of the series.",
        "Summarise the series by its maximum value.",
        "Obtain the maximum value of these numbers.",
    ),
    "change": (
        "What is the percent change from the first value to the last value?",
        "Compute the percent change between the first and last value.",
        "Report the percent change from the first to the last value.",
        "By what percent change does the series move from first to last?",
        "Give the percent change from the first value to the last.",
        "Determine the percent change between the first and the last value.",
        "State the percent change from the first point to the last point.",
        "I need the percent change from the first value to the last value.",
        "Return the percent change between first and last value.",
        "What percent change separates the first and last value?",
        "Calculate the percent change from the first to last value.",
        "Output the percent change between the first and last value.",
        "Work out the percent change from first value to last value.",
        "Please compute the percent change from the first to the last value.",
        "Provide the percent change between the first and last value.",
        "Measure the percent change from the first value to the last.",
        "Tell me the percent change from the first to the last value.",
        "Express the move from the first to the last value as a percent change.",
        "Find the percent change between the first and last observation.",
        "What is the overall percent change, first value to last value?",
        "Evaluate the percent change from the first to the last value.",
        "Quantify the percent change between the first and last value.",
        "Obtain the percent change from the first value to the last value.",
        "How large is the percent change from the first to the last value?",
        "Summarise the move as a percent change from first to last value.",
    ),
    "correlation": (
        "What is the lag-1 autocorrelation of the series?",
        "Compute the lag-1 autocorrelation of these values.",
        "Report the lag-1 autocorrelation.",
        "Determine the lag-1 autocorrelation of the series.",
        "Give the lag-1 autocorrelation of this series.",
        "State the lag-1 autocorrelation of the data.",
        "I need the lag-1 autocorrelation of the series.",
        "Return the lag-1 autocorrelation.",
        "Calculate the lag-1 autocorrelation of the series.",
        "What value is the lag-1 autocorrelation?",
        "Output the lag-1 autocorrelation of the series.",
        "Work out the lag-1 autocorrelation.",
        "Please compute the lag-1 autocorrelation of the series.",
        "Provide the lag-1 autocorrelation of these observations.",
        "Measure the lag-1 autocorrelation of this series.",
        "Tell me the lag-1 autocorrelation of the series.",
        "Find the lag-1 autocorrelation for the series.",
        "Evaluate the lag-1 autocorrelation of the data.",
        "How strong is the lag-1 autocorrelation of the series?",
        "Quantify the lag-1 autocorrelation.",
        "Obtain the lag-1 autocorrelation of the series.",
        "Estimate the lag-1 autocorrelation of these values.",
        "Summarise the series by its lag-1 autocorrelation.",
        "What is this series' lag-1 autocorrelation?",
        "Derive the lag-1 autocorrelation of the series.",
    ),
    "localization": (
        "At which index does the largest absolute one-step change occur?",
        "Where does the largest absolute one-step change occur?",
        "Find the position of the largest absolute one-step change.",
        "Locate the largest absolute one-step change in the series.",
        "At what point does the largest absolute one-step change happen?",
        "Identify where the largest absolute one-step change occurs.",
        "Which index holds the largest absolute one-step change?",
        "I need the position of the largest absolute one-step change.",
        "Determine where the largest absolute one-step change occurs.",
        "State the position of the largest absolute one-step change.",
        "Return the position of the largest absolute one-step change.",
        "Output where the largest absolute one-step change occurs.",
        "Work out where the largest absolute one-step change is.",
        "Please locate the largest absolute one-step change.",
        "Provide the position of the largest absolute one-step change.",
        "Pinpoint the largest absolute one-step change in this series.",
        "Tell me where the largest absolute one-step change occurs.",
        "Where in the series is the largest absolute one-step change?",
        "Give the position of the largest absolute one-step change.",
        "Which step shows the largest absolute one-step change?",
        "Point to the largest absolute one-step change in the series.",
        "Detect the largest absolute one-step change and give its position.",
        "Report the location of the largest absolute one-step change.",
        "Establish where the largest absolute one-step change occurs.",
        "Search the series for the largest absolute one-step change.",
    ),
    "trend_description": (
        "Describe the overall trend of the series in one sentence, and state "
        "whether it looks seasonal.",
        "In one sentence, what is the overall trend, and does the series look "
        "seasonal?",
        "Summarise the trend of this series in one sentence and say whether it "
        "is seasonal.",
        "Characterise the overall trend in one sentence, noting whether the "
        "series is seasonal.",
        "Give a one-sentence description of the trend and whether the series "
        "appears seasonal.",
        "What is the overall trend here, and is the series seasonal? Answer in "
        "one sentence.",
        "Report the trend of the series in one sentence, including whether it "
        "is seasonal.",
        "Describe in one sentence how the series trends and whether it is "
        "seasonal.",
        "State the overall trend and whether the series is seasonal, in one "
        "sentence.",
        "In a single sentence, describe the trend and any seasonal behaviour.",
        "Explain the overall trend in one sentence and say if it is seasonal.",
        "Tell me the trend of this series and whether it looks seasonal, in "
        "one sentence.",
        "Provide a one-sentence account of the trend and the seasonal "
        "character of the series.",
        "How does this series trend, and is it seasonal? One sentence.",
        "Write one sentence on the overall trend and whether the series is "
        "seasonal.",
        "Assess the trend of the series in one sentence and note if seasonal.",
        "Briefly, in one sentence: what is the trend and is it seasonal?",
        "Outline the overall trend in one sentence, mentioning seasonal "
        "structure.",
        "Give your reading of the trend and the seasonal pattern in one "
        "sentence.",
        "Summarise in one sentence the direction of the trend and whether the "
        "series is seasonal.",
        "Comment in one sentence on the overall trend and seasonal behaviour.",
        "Describe the trend and whether the series is seasonal, using one "
        "sentence.",
        "What overall trend does the series show, and is it seasonal? One "
        "sentence please.",
        "Offer a one-sentence summary of the trend and of any seasonal "
        "pattern.",
        "Judge the overall trend and the seasonal character in one sentence.",
    ),
    "event_explanation": (
        "Explain the most important event in the series in one sentence: what "
        "happened, and whether it occurred early, in the middle, or late.",
        "In one sentence, what is the most important event, and did it happen "
        "early, in the middle, or late?",
        "Describe the main event in one sentence and say whether it was early, "
        "middle, or late.",
        "What is the most notable event here? Answer in one sentence and state "
        "whether it was early, middle, or late.",
        "Summarise the key event in one sentence, noting if it occurred early, "
        "in the middle, or late.",
        "Identify the most important event in one sentence and whether it fell "
        "early, middle, or late.",
        "Report the main event and its timing in one sentence: early, middle, "
        "or late.",
        "In a single sentence, explain the principal event and whether it was "
        "early, middle, or late.",
        "State the most important event and whether it happened early, in the "
        "middle, or late, in one sentence.",
        "Give a one-sentence account of the key event and whether it was "
        "early, middle, or late.",
        "What notable event occurs, and was it early, middle, or late? One "
        "sentence.",
        "Tell me the main event in this series and whether it was early, "
        "middle, or late, in one sentence.",
        "Explain in one sentence the dominant event and its position: early, "
        "middle, or late.",
        "Write one sentence on the most important event and whether it was "
        "early, in the middle, or late.",
        "Characterise the main event in one sentence, including whether it was "
        "early, middle, or late.",
        "Point out the most important event in one sentence and say if it was "
        "early, middle, or late.",
        "Briefly, in one sentence: what happened, and was it early, middle, or "
        "late?",
        "Describe what happened in this series in one sentence and whether it "
        "was early, middle, or late.",
        "Outline the key event in one sentence and locate it as early, middle, "
        "or late.",
        "Provide a one-sentence explanation of the main event and whether it "
        "occurred early, middle, or late.",
        "Comment in one sentence on the most important event and its timing: "
        "early, middle, or late.",
        "Which event matters most here, and was it early, middle, or late? "
        "Answer in one sentence.",
        "Summarise in one sentence what happened and whether it was early, in "
        "the middle, or late.",
        "Give your reading of the main event in one sentence, noting early, "
        "middle, or late.",
        "Account for the most important event in one sentence and state "
        "whether it was early, middle, or late.",
    ),
}


# ===========================================================================
# cycle 2 (PREREGISTRATION_C2.md, amendment A9): the two qualitative types
# ===========================================================================
# Cycle 1's qualitative questions asked for "one sentence" and were scored by
# regex over prose. C2 asks a closed-set question whose ANSWER FORMAT AND
# DEFINITIONS LIVE IN THE QUESTION TEXT, so every arm receives them
# identically. Keying the format on the routed operation instead would hand
# the bare arm the right template for free while the harness had to route to
# it - the `op=` asymmetry cycle 1 removed - and would make `describe` and
# `explain` non-interchangeable (review #2, claim 5).
#
# The seven other query types are unchanged in C2 and keep their cycle-1
# framings and clauses.

#: slot -> options, in the order they are offered. MUST equal
#: `src.ground_truth.QUALITATIVE_SLOTS`; tests/test_c2_pipeline.py asserts
#: it. Duplicated here because ground_truth imports this module.
C2_SLOT_OPTIONS: dict[str, tuple[str, ...]] = {
    "trend": ("increasing", "decreasing", "stable"),
    "seasonal": ("yes", "no"),
    "event": ("spike", "abrupt change", "no major event"),
    "position": ("early", "middle", "late"),
}

#: which slots each C2 question asks for. MUST equal
#: `src.ground_truth.SLOTS_FOR_TYPE_C2` (asserted by the same test).
C2_ASKED_SLOTS: dict[str, tuple[str, ...]] = {
    "trend_description": ("trend", "seasonal"),
    "event_explanation": ("event", "position"),
}

#: Appended verbatim to every C2 framing. Defines the concepts and the answer
#: format. One line, no newline: the question must stay a single line because
#: prompt parsers (and the mock) read `Question: (.+)`.
C2_CLAUSE: dict[str, str] = {
    "trend_description": (
        " Here trend means a sustained drift up or down across the whole "
        "series, not a single jump and not the swings of a repeating cycle; "
        "seasonal means a cycle that repeats at a fixed interval. Reply with "
        "exactly two lines and nothing else: 'trend: <increasing | "
        "decreasing | stable>' and 'seasonal: <yes | no>'."),
    "event_explanation": (
        " Here a spike is a single isolated point far above its neighbours; "
        "an abrupt change is a lasting shift in level; and the position is "
        "the third of the series (early, middle or late) in which the event "
        "happens. Reply with exactly two lines and nothing else: 'event: "
        "<spike | abrupt change | no major event>' and 'position: <early | "
        "middle | late>'. If there is no major event, write 'position: "
        "none'."),
}

C2_REQUIRED_PHRASES: dict[str, tuple[str, ...]] = {
    "trend_description": ("'trend: <", "'seasonal: <", "seasonal"),
    "event_explanation": ("'event: <", "'position: <", "early", "middle",
                          "late"),
}

C2_FRAMES: dict[str, tuple[str, ...]] = {
    "trend_description": (
        "What is the overall trend of the series, and is the series seasonal?",
        "Describe the overall trend and state whether the series is seasonal.",
        "Classify the trend of this series and say whether it is seasonal.",
        "Which way does the series trend overall, and is it seasonal?",
        "Determine the overall trend and whether the series is seasonal.",
        "Report the trend of the series and whether it is seasonal.",
        "Characterise the overall trend and note whether the series is seasonal.",
        "Identify the trend direction and whether the series is seasonal.",
        "Assess the overall trend and whether the series looks seasonal.",
        "State the overall trend and whether the series is seasonal.",
        "Give the trend of this series and say if it is seasonal.",
        "Summarise the trend and whether the series is seasonal.",
        "How does this series trend, and is it seasonal?",
        "Judge the overall trend and whether the series is seasonal.",
        "Tell me the trend of the series and whether it is seasonal.",
        "What overall trend does the series show, and is it seasonal?",
        "Decide the trend direction of the series and whether it is seasonal.",
        "Label the overall trend and the seasonal character of the series.",
        "Evaluate the trend of this series and whether it is seasonal.",
        "Work out the overall trend and whether the series is seasonal.",
        "Provide the trend of the series and whether it is seasonal.",
        "Read off the overall trend and say whether the series is seasonal.",
        "Establish the trend direction and whether the series is seasonal.",
        "Name the overall trend and indicate whether the series is seasonal.",
        "Briefly: what is the trend, and is the series seasonal?",
    ),
    "event_explanation": (
        "What is the most important event in the series, and where does it "
        "occur?",
        "Identify the most important event in the series and where it occurs.",
        "Which major event, if any, does the series contain, and where?",
        "Name the main event in this series and locate it.",
        "Determine the most notable event in the series and its location.",
        "Report the main event in the series and where it happens.",
        "Classify the most important event and say where it occurs.",
        "What notable event occurs in the series, and where?",
        "State the main event of the series and where it falls.",
        "Find the most important event in this series and locate it.",
        "Describe the key event in the series and where it happens.",
        "Tell me the main event in this series and where it occurs.",
        "Which event matters most in the series, and where is it?",
        "Point out the most important event and where it occurs.",
        "Give the principal event of the series and its location.",
        "Decide what the main event in the series is and where it occurs.",
        "Characterise the dominant event in the series and locate it.",
        "Summarise the key event in the series and where it falls.",
        "Work out the most important event and where it happens.",
        "Label the main event of this series and where it occurs.",
        "Establish the most notable event in the series and where it is.",
        "Provide the main event in the series and where it occurs.",
        "Assess what the major event in the series is and where it occurs.",
        "Pick out the most important event in the series and locate it.",
        "Briefly: what happened in the series, and where?",
    ),
}


def build_query(query_type: str, horizon: int, variant: int = 0,
                design: str = "c1") -> str:
    """Render one paraphrase. `variant` in [0, N_VARIANTS).

    `design="c2"` swaps in the closed-set framings and clause for the two
    qualitative types; every other type renders exactly as in cycle 1.
    """
    v = int(variant) % N_VARIANTS
    lead = PARAPHRASE_LEADS[v // N_FRAMINGS]
    if design == "c2" and query_type in C2_FRAMES:
        return lead + C2_FRAMES[query_type][v % N_FRAMINGS] \
            + C2_CLAUSE[query_type]
    frames = PARAPHRASE_FRAMES[query_type]
    body = frames[v % N_FRAMINGS] + OPERATIONAL_CLAUSE[query_type]
    return (lead + body).format(h=horizon)


def framing_of(variant: int) -> int:
    return int(variant) % N_FRAMINGS


def lead_of(variant: int) -> int:
    return (int(variant) % N_VARIANTS) // N_FRAMINGS


#: back-compat: the canonical phrasing of each type is variant 0
QUERY_TEMPLATES: dict[str, str] = {
    qt: PARAPHRASE_FRAMES[qt][0] + OPERATIONAL_CLAUSE[qt]
    for qt in PARAPHRASE_FRAMES
}


def _validate_paraphrase_bank() -> None:
    """Fail the import if the authoring rule was broken.

    Three things must hold for every type:
      * exactly N_FRAMINGS framings, all distinct;
      * every rendered variant contains the type's REQUIRED_PHRASES, so a
        reworded framing cannot silently redefine the operation;
      * all N_VARIANTS renderings are distinct, or the routing collapse this
        bank exists to fix comes straight back.
    """
    if len(PARAPHRASE_LEADS) != N_LEADS:
        raise RuntimeError(f"expected {N_LEADS} leads, got "
                           f"{len(PARAPHRASE_LEADS)}")
    for qt, frames in PARAPHRASE_FRAMES.items():
        if len(frames) != N_FRAMINGS:
            raise RuntimeError(
                f"paraphrase bank for {qt!r} has {len(frames)} framings, "
                f"expected {N_FRAMINGS}")
        if len(set(frames)) != N_FRAMINGS:
            dupes = [f for f in frames if frames.count(f) > 1]
            raise RuntimeError(f"{qt!r} has duplicate framings: {dupes[:2]}")
        if qt not in OPERATIONAL_CLAUSE or qt not in REQUIRED_PHRASES:
            raise RuntimeError(f"{qt!r} is missing a clause or phrase rule")
        rendered = {build_query(qt, 8, v) for v in range(N_VARIANTS)}
        if len(rendered) != N_VARIANTS:
            raise RuntimeError(
                f"{qt!r} renders only {len(rendered)} distinct queries out of "
                f"{N_VARIANTS}; identical prompts collapse routing")
        for v in range(N_VARIANTS):
            text = build_query(qt, 8, v)
            for phrase in REQUIRED_PHRASES[qt]:
                needle = phrase.format(h=8)
                if needle not in text:
                    raise RuntimeError(
                        f"{qt!r} variant {v} is missing the required phrase "
                        f"{needle!r}. The operational clause defines the "
                        f"answer key and must appear verbatim.\n  got: {text}")
    missing = set(PARAPHRASE_FRAMES) ^ set(OPERATIONAL_CLAUSE)
    if missing:
        raise RuntimeError(f"paraphrase bank/clause mismatch: {sorted(missing)}")

    # cycle 2: same three rules, plus the options in the clause must be the
    # declared options, and the question must stay on one line
    for qt, frames in C2_FRAMES.items():
        if len(frames) != N_FRAMINGS or len(set(frames)) != N_FRAMINGS:
            raise RuntimeError(
                f"C2 bank for {qt!r} needs {N_FRAMINGS} distinct framings")
        rendered = [build_query(qt, 8, v, design="c2")
                    for v in range(N_VARIANTS)]
        if len(set(rendered)) != N_VARIANTS:
            raise RuntimeError(f"C2 {qt!r} renders duplicate queries")
        for text in rendered:
            if "\n" in text:
                raise RuntimeError(f"C2 {qt!r} query spans several lines")
            for phrase in C2_REQUIRED_PHRASES[qt]:
                if phrase not in text:
                    raise RuntimeError(
                        f"C2 {qt!r} query lacks required phrase {phrase!r}")
        for slot in C2_ASKED_SLOTS[qt]:
            want = f"'{slot}: <" + " | ".join(C2_SLOT_OPTIONS[slot]) + ">'"
            if want not in C2_CLAUSE[qt]:
                raise RuntimeError(
                    f"C2 {qt!r} clause does not offer exactly {want}")
    # a describe question must not look like an explain question to a
    # keyword router, and vice versa
    for text in (build_query("trend_description", 8, v, design="c2")
                 for v in range(N_VARIANTS)):
        for word in ("early", "middle", "late"):
            if word in text.lower().split():
                raise RuntimeError("C2 trend question mentions a position")


_validate_paraphrase_bank()


class Dataset:
    """Abstract dataset interface."""

    name: str = "base"

    def load(self, n_base_series: int, n_per_query_type: int,
             query_types: list, max_history: int, horizon: int,
             seed: int) -> list:
        raise NotImplementedError
