"""Prompt templates and their dispatch markers. Imports NOTHING from `src`.

WHY THIS IS ITS OWN MODULE
--------------------------
This is module hygiene, not a leakage fix - be precise about that, because
overstating it invites the question of what was actually open. The real
barrier against ground-truth access is `InferenceSample` (see
`src/samples.py`); a system holds an object with no label on it, so there is
nothing to reach.

What this split does buy: `models/llm.py` contains `MockLLM`, which imports
`src.ground_truth` to synthesise answers at a target accuracy. That gave
`systems/*` a transitive import path to the answer-key module purely because
they needed prompt strings. Templates now live here, importing nothing from
`src`, so `systems/` depends only on text.

THE MARKER CONTRACT
-------------------
Markers are module constants shared by the templates and by every
dispatcher. They are never retyped. `assert_markers_dispatch()` runs at
import and raises if any rendered template fails to contain its marker.

That guard exists because of a real defect: the old build dispatched on
`"[task:classify]"` while the template rendered `[task:classify
request=s_0001]`. The substring never matched, the mock's classify branch
was dead, and the oracle fallback therefore fired on 100% of mock samples -
silently, for weeks.

TEMPLATE TEXT IS CACHE-KEY MATERIAL
-----------------------------------
`src.cache.call_key` hashes the rendered prompts, so a single changed
character here invalidates every cached call. `tests/test_prompts.py` pins a
hash of each template: moving or reformatting this file is free, editing it
is not, and the test makes the difference explicit instead of surprising.
Bump `src.PROMPT_VERSION` deliberately when you mean it.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# dispatch markers: ONE definition, used by templates and dispatchers alike
# --------------------------------------------------------------------------
M_ANSWER = "[task:answer"
M_CLASSIFY = "[task:classify]"
M_DESCRIBE = "[task:describe]"
M_EXPLAIN = "[task:explain]"
M_VOICE = "[task:voice]"
M_JUDGE = "[task:judge]"

#: Used by the bare arm for EVERY query type. It carries no operation hint:
#: the old build injected `op={QUERY_OP[query_type]}` - the ground-truth
#: operation - into this prompt on all 900 samples, while the harness had to
#: infer the same thing from question text alone.
GENERIC_PROMPT = (
    M_ANSWER + " history_len={L}]\n"
    "You are given a numeric time series. Answer the question using only the "
    "series values.\n"
    "Series (oldest to newest, {L} values): {series}\n"
    "Question: {question}\n"
    "If the answer is a single number, reply with that number and nothing "
    "else. If it is a sequence of future values, reply with the values "
    "separated by commas and nothing else. Otherwise answer in one short "
    "sentence."
)

#: Contains the question only - no series, and no request id. The absent
#: request id is deliberate: interpolating a per-sample nonce (as the old
#: build did) does not create real routing variation, it samples the model's
#: decision boundary under an irrelevant token. Genuine variation comes from
#: the paraphrase bank in `src/samples.py`.
CLASSIFY_PROMPT = (
    M_CLASSIFY + "\n"
    "Select the single operation that answers this query, from: {ops}\n"
    "Question: {question}\n"
    "Reply with exactly one operation word."
)

QUAL_PROMPT = (
    "[task:{task}]\n"
    "history_len={L}\n"
    "{feature_block}"
    "Question: {question}\n"
    "{series_block}"
    "Answer in one sentence."
)

VOICE_PROMPT = (
    M_VOICE + "\nThe computed result is {value}. State the final answer only.")

JUDGE_PROMPT = (
    M_JUDGE + "\n"
    "Question: {question}\n"
    "Reference answer: {refs}\n"
    "Model answer: {answer}\n"
    "Rate the model answer 0, 0.5 or 1 for agreement with the reference. "
    "Reply with JSON: {{\"score\": <0|0.5|1>}}"
)

#: C2 (PREREGISTRATION_C2.md, amendment A9).
#:
#: The answer format and the definitions of every option live in the QUESTION
#: (src/samples.py, C2_CLAUSE), so every arm receives them identically and
#: these templates carry no per-query-type text at all. An earlier C2 draft
#: put a rules block and an answer template here, keyed by query type; that
#: handed the bare arm the right template for free while the harness had to
#: route to it, and made `describe` and `explain` non-interchangeable.
#:
#: `evidence_block` is the numeric block from systems/tools.py, or "" for
#: the plain bare arm. The series is ALWAYS shown: cycle 1 hid it from the
#: harness's model whenever features were supplied, so the model could only
#: copy them.
GENERIC_PROMPT_C2 = (
    M_ANSWER + " history_len={L}]\n"
    "You are given a numeric time series. Answer the question about it.\n"
    "{evidence_block}"
    "Series (oldest to newest, {L} values): {series}\n"
    "Question: {question}\n"
    "If the question states an answer format, follow it exactly. Otherwise, "
    "if the answer is a single number, reply with that number and nothing "
    "else; if it is a sequence of future values, reply with the values "
    "separated by commas and nothing else."
)

#: The harness's final step for a qualitative query. Identical to the bare
#: arm's information except for the task header, and identical between
#: `describe` and `explain` except for that header - which is what makes the
#: two operations functionally equivalent (PREREGISTRATION_C2.md §C2-8).
QUAL_PROMPT_C2 = (
    "[task:{task}]\n"
    "history_len={L}\n"
    "{evidence_block}"
    "Series (oldest to newest, {L} values): {series}\n"
    "Question: {question}\n"
    "Follow the answer format stated in the question exactly."
)

#: One line per tool, for the classify prompt. Cycle 1 passed bare operation
#: names: at 32B, 99/100 `localization` questions went to `pct_change`, so
#: routing accuracy was partly a measure of how guessable the names were
#: rather than of the model's choice.
OP_DESCRIPTIONS: dict[str, str] = {
    "mean": "arithmetic mean of the values",
    "median": "median of the values",
    "std": "standard deviation of the values",
    "max": "largest value",
    "min": "smallest value",
    "count_spikes": "how many isolated upward spikes occur",
    "pct_change": "percent change from the first value to the last",
    "lag1_autocorr": "correlation between consecutive values",
    "change_point": "index of the largest single step between neighbours",
    "forecast": "predict the next values beyond the end of the series",
    "describe": "describe the overall shape: trend and seasonality",
    "explain": "explain a notable event: what happened and where",
}

CLASSIFY_PROMPT_C2 = (
    M_CLASSIFY + "\n"
    "Select the single operation that answers this query.\n"
    "Operations:\n{op_menu}\n"
    "Question: {question}\n"
    "Reply with exactly one operation name."
)


def op_menu(ops) -> str:
    """Render `ops` as a described menu, in the order given."""
    return "\n".join(f"  {o} - {OP_DESCRIPTIONS.get(o, o)}" for o in ops)


#: template name -> (template, marker). Used by the dispatch guard and by
#: the pinned-hash test.
TEMPLATES: dict[str, tuple[str, str]] = {
    "generic": (GENERIC_PROMPT, M_ANSWER),
    "classify": (CLASSIFY_PROMPT, M_CLASSIFY),
    "qual_describe": (QUAL_PROMPT, M_DESCRIBE),
    "qual_explain": (QUAL_PROMPT, M_EXPLAIN),
    "generic_c2": (GENERIC_PROMPT_C2, M_ANSWER),
    "qual_c2_describe": (QUAL_PROMPT_C2, M_DESCRIBE),
    "qual_c2_explain": (QUAL_PROMPT_C2, M_EXPLAIN),
    "classify_c2": (CLASSIFY_PROMPT_C2, M_CLASSIFY),
    "voice": (VOICE_PROMPT, M_VOICE),
    "judge": (JUDGE_PROMPT, M_JUDGE),
}


def render_samples() -> dict[str, str]:
    """One rendering of each template, for the guard and the hash test."""
    return {
        "generic": GENERIC_PROMPT.format(L=4, series="1, 2, 3, 4",
                                         question="q?"),
        "classify": CLASSIFY_PROMPT.format(ops="mean, max", question="q?"),
        "qual_describe": QUAL_PROMPT.format(task="describe", L=4,
                                            feature_block="", question="q?",
                                            series_block=""),
        "qual_explain": QUAL_PROMPT.format(task="explain", L=4,
                                           feature_block="", question="q?",
                                           series_block=""),
        "generic_c2": GENERIC_PROMPT_C2.format(
            L=4, evidence_block="", series="1, 2, 3, 4", question="q?"),
        "qual_c2_describe": QUAL_PROMPT_C2.format(
            task="describe", L=4, evidence_block="", series="1, 2, 3, 4",
            question="q?"),
        "qual_c2_explain": QUAL_PROMPT_C2.format(
            task="explain", L=4, evidence_block="", series="1, 2, 3, 4",
            question="q?"),
        "classify_c2": CLASSIFY_PROMPT_C2.format(
            op_menu=op_menu(["mean", "max"]), question="q?"),
        "voice": VOICE_PROMPT.format(value="1"),
        "judge": JUDGE_PROMPT.format(question="q", refs="r", answer="a"),
    }


def assert_markers_dispatch() -> None:
    """Every rendered template must contain its dispatch marker."""
    rendered = render_samples()
    for name, (_, marker) in TEMPLATES.items():
        if marker not in rendered[name]:
            raise RuntimeError(
                f"prompt template {name!r} does not contain its dispatch "
                f"marker {marker!r}. This is exactly the defect that "
                "silenced the old build's mock classifier and turned the "
                "oracle fallback into a 100% code path.")


assert_markers_dispatch()
