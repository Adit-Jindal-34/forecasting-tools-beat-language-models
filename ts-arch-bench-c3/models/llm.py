"""LLM adapters and the behavioural mock.

PROMPTS
-------
`GENERIC_PROMPT` is used by the bare arm for EVERY query type. It carries no
operation hint. The old build injected `op={QUERY_OP[query_type]}` - the
ground-truth operation - into the bare arm's prompt on all 900 samples. See
systems/bare_llm.py.

THE MOCK BUG THIS BUILD FIXES
-----------------------------
The old `MockLLM.complete` dispatched on `"[task:classify]" in user_prompt`,
but the template rendered `[task:classify request=s_0001]`. The substring
never matched, the classify branch was dead, every classify call returned the
catch-all "I cannot determine the answer.", and the oracle fallback therefore
fired on 100% of mock samples. The committed cache proves it:

    validation_results/cache/inference.jsonl -> classify source 96/96 fallback

Every mock-backed result in that repo - H1, H2, H3, the crossover analysis,
every smoke run - measured a harness with perfect oracle routing.

Two guards here make that class of bug impossible to reintroduce silently:
  * markers are MODULE CONSTANTS shared by the templates and the dispatcher,
    never retyped;
  * `_assert_markers_dispatch()` runs at import and raises if any rendered
    template fails to hit its branch.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from abc import ABC, abstractmethod

import numpy as np

# Templates and markers now live in models/prompts.py, which imports nothing
# from `src`. Re-exported here so existing call sites keep working.
from .prompts import (  # noqa: F401,E402
    M_ANSWER, M_CLASSIFY, M_DESCRIBE, M_EXPLAIN, M_VOICE, M_JUDGE,
    GENERIC_PROMPT, CLASSIFY_PROMPT, QUAL_PROMPT, VOICE_PROMPT, JUDGE_PROMPT,
    GENERIC_PROMPT_C2, QUAL_PROMPT_C2, CLASSIFY_PROMPT_C2, op_menu,
    assert_markers_dispatch as _assert_markers_dispatch,
)

#: Present in a prompt exactly when it carries the C2 numeric evidence block
#: (systems/tools.py::evidence_to_block). The mock keys "was I assisted?" on
#: it, the way cycle 1's mock keyed on "computed features:".
C2_EVIDENCE_MARKER = "computed evidence"
#: Present in every C2 qualitative QUESTION (src/samples.py, C2_CLAUSE).
C2_FORMAT_MARKER = "Reply with exactly two lines"

_SERIES_LINE = re.compile(r"Series \(oldest to newest, \d+ values\): (.+)")


def series_from_prompt(prompt: str) -> np.ndarray:
    m = _SERIES_LINE.search(prompt)
    if not m:
        return np.array([])
    return np.array([float(v) for v in re.split(r"[,;\s]+", m.group(1).strip())
                     if v], float)


def extract_numbers(text: str) -> list:
    t = (text or "").replace(",", " ")
    return [float(v) for v in
            re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", t)]


def extract_int(text: str) -> int | None:
    m = re.search(r"-?\d+", text or "")
    return int(m.group()) if m else None


class LLMClient(ABC):
    name: str = "llm-base"
    model: str = "unknown"

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Mock LLM - a behavioural placeholder, not a model
# --------------------------------------------------------------------------
#: Question pattern -> operation. The mock infers the op the way a real model
#: must: from the question text alone.
#:
#: These key on the OPERATIONAL CLAUSES that `src.samples.REQUIRED_PHRASES`
#: guarantees in every paraphrase, not on incidental wording. Keying on
#: surface phrasing (the old table matched literals like "describe the
#: overall trend") silently breaks the moment the paraphrase bank varies the
#: framing, and a mock that cannot route makes every mock run meaningless.
#: `tests/test_routing.py::test_mock_infers_the_right_op_for_every_paraphrase`
#: checks all 900 renderings.
#:
#: Order matters: the most specific clause wins.
_OP_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"median absolute deviation", "count_spikes"),
    (r"\bnext \d+ values\b", "forecast"),
    (r"\barithmetic mean\b", "mean"),
    (r"\bmaximum value\b", "max"),
    (r"\bminimum value\b", "min"),
    (r"\bpercent change\b", "pct_change"),
    (r"\blag-1 autocorrelation\b", "lag1_autocorr"),
    (r"\b0-based index\b", "change_point"),
    (r"\b(early|middle|late)\b", "explain"),
    (r"\bseasonal\b", "describe"),
)
_COMPILED_OPS = tuple((re.compile(p, re.I), op) for p, op in _OP_PATTERNS)
_ALL_OPS = tuple(dict.fromkeys(op for _, op in _OP_PATTERNS))


class MockLLM(LLMClient):
    """Deterministic, strength-parameterised stand-in for a real backbone.

    Simulates: approximate in-head arithmetic whose reliability scales with
    `strength`; routing that is mostly right; and - new in this build - a
    configurable `prose_rate` of rambling replies, so the permissive parser
    and the abstention path are actually exercised by tests instead of being
    dead code nobody notices.
    """

    def __init__(self, backbone_name: str, strength: float = 0.6,
                 error_scale: float = 0.25, prose_rate: float = 0.0,
                 length_effect: float = 0.0, seed: int = 0,
                 assist_effect: float = 1.0):
        self.name = f"mock-llm-{backbone_name}"
        self.strength = float(strength)
        self.error_scale = float(error_scale)
        self.prose_rate = float(prose_rate)
        self.length_effect = float(length_effect)
        self._seed = int(seed)
        # `assist_effect` scales HOW MUCH the harness's "computed features:"
        # block helps the mock. 1.0 is the historical behaviour. 0.0 makes
        # the mock arm-blind, which is what a null fixture needs: see
        # `_assist_p` and `configs/null_fixture.yaml`.
        self.assist_effect = float(assist_effect)
        # The call cache keys on this string. It MUST carry every parameter
        # that changes the reply, or a null-fixture run and an ordinary mock
        # run would silently share cached answers.
        self.model = (f"mock-llm@strength={strength}"
                      f",assist={self.assist_effect}"
                      f",err={self.error_scale}"
                      f",prose={self.prose_rate}"
                      f",len={self.length_effect}")

    def _rng(self, prompt: str) -> np.random.Generator:
        h = int(hashlib.sha256(prompt.encode()).hexdigest()[:12], 16)
        return np.random.default_rng(self._seed * 1000003 + h)

    def _p_correct(self, L: int, kind: str) -> float:
        p = 0.5 + 0.48 * self.strength
        if self.length_effect > 0:
            frac = max(0.0, (np.log2(max(L, 32)) - 5.0) / (12.0 - 5.0))
            p -= self.length_effect * 0.5 * frac
        if kind == "forecast":
            p *= 0.85
        if kind == "classify":
            p = 0.85 + 0.14 * self.strength
        return float(np.clip(p, 0.05, 0.995))

    @staticmethod
    def _infer_op(question: str) -> str:
        q = question or ""
        for pat, op in _COMPILED_OPS:
            if pat.search(q):
                return op
        return "describe"

    # -- branches ----------------------------------------------------------
    def _do_classify(self, prompt: str, rng) -> str:
        m = re.search(r"Question: (.+)", prompt)
        correct = self._infer_op(m.group(1) if m else "")
        op = (correct if rng.random() < self._p_correct(128, "classify")
              else _ALL_OPS[rng.integers(len(_ALL_OPS))])
        if rng.random() < self.prose_rate:
            # the failure mode that produced all 19 real parse failures
            return f"To answer this question I would use the {op} operation."
        return op

    def _do_answer(self, prompt: str, rng) -> str:
        """The bare arm's generic prompt: infer the op, then approximate it."""
        from src import ground_truth as gt
        m = re.search(r"history_len=(\d+)", prompt)
        L = int(m.group(1)) if m else 128
        qm = re.search(r"Question: (.+)", prompt)
        question = qm.group(1) if qm else ""
        op = self._infer_op(question)
        x = series_from_prompt(prompt)
        if len(x) == 0:
            return "I cannot read the series."

        if op == "forecast":
            hm = re.search(r"next (\d+) values", question)
            h = int(hm.group(1)) if hm else 8
            vals = _inhead_forecast(x, h)
            if rng.random() >= self._p_correct(L, "forecast"):
                vals = vals * (1 + rng.uniform(-self.error_scale,
                                               self.error_scale))
            return ", ".join(f"{v:.6g}" for v in vals)

        if op in ("describe", "explain"):
            if C2_FORMAT_MARKER in question:
                return _mock_qual_c2(
                    x, question, rng, self.strength,
                    assisted=C2_EVIDENCE_MARKER in prompt,
                    assist_effect=self.assist_effect)
            return _mock_qual_sentence(x, op, rng, self.strength,
                                       assisted=False,
                                       assist_effect=self.assist_effect)

        value = {
            "mean": lambda: float(np.mean(x)),
            "max": lambda: float(np.max(x)),
            "min": lambda: float(np.min(x)),
            "count_spikes": lambda: float(gt.count_spikes(x)),
            "pct_change": lambda: gt.pct_change(x),
            "lag1_autocorr": lambda: gt.lag1_autocorr(x),
            "change_point": lambda: float(gt.change_point_index(x)),
        }[op]()
        is_int = op in ("count_spikes", "change_point")
        if rng.random() < self._p_correct(L, "exact"):
            return _fmt(value, is_int)
        if is_int:
            return str(max(0, int(value + rng.integers(-3, 4))))
        if op == "lag1_autocorr":
            return _fmt(float(np.clip(value + rng.uniform(-0.3, 0.3), -1, 1)))
        return _fmt(value * (1 + rng.uniform(-self.error_scale,
                                             self.error_scale)))

    def _do_qual(self, prompt: str, rng, task: str) -> str:
        qm = re.search(r"Question: (.+)", prompt)
        question = qm.group(1) if qm else ""
        if C2_FORMAT_MARKER in question:
            # C2: the answer format comes from the QUESTION, never from the
            # routed task, so `describe` and `explain` behave identically.
            return _mock_qual_c2(
                series_from_prompt(prompt), question, rng, self.strength,
                assisted=C2_EVIDENCE_MARKER in prompt,
                assist_effect=self.assist_effect)
        assisted = "computed features:" in prompt
        if assisted:
            return _qual_from_features(prompt, rng, self.strength, task,
                                       assist_effect=self.assist_effect)
        x = series_from_prompt(prompt)
        return _mock_qual_sentence(x, task, rng, self.strength, assisted=False,
                                   assist_effect=self.assist_effect)

    # -- API ---------------------------------------------------------------
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        rng = self._rng(user_prompt)
        if M_CLASSIFY in user_prompt:
            return self._do_classify(user_prompt, rng)
        if M_JUDGE in user_prompt:
            return self._do_judge(user_prompt)
        if M_DESCRIBE in user_prompt:
            return self._do_qual(user_prompt, rng, "describe")
        if M_EXPLAIN in user_prompt:
            return self._do_qual(user_prompt, rng, "explain")
        if M_VOICE in user_prompt:
            m = re.search(r"computed result is (-?[\d.eE+-]+)", user_prompt)
            return m.group(1) if m else "unknown"
        if M_ANSWER in user_prompt:
            return self._do_answer(user_prompt, rng)
        raise ValueError(
            "MockLLM received a prompt with no recognised task marker. "
            "This is the bug that made the old build's classifier branch "
            f"dead code. Prompt starts: {user_prompt[:80]!r}")

    @staticmethod
    def _do_judge(prompt: str) -> str:
        refs = re.search(r"Reference answer: (.+)", prompt)
        ans = re.search(r"Model answer: (.+)", prompt)
        if not (refs and ans):
            return json.dumps({"score": 0.0})
        parts = [p.strip().lower() for p in refs.group(1).split("|")
                 if p.strip()]
        a = ans.group(1).lower()
        cov = (sum(1 for p in parts if p in a) / len(parts)) if parts else 0.0
        return json.dumps({"score": 1.0 if cov >= 0.8 else
                           0.5 if cov >= 0.4 else 0.0})


# --------------------------------------------------------------------------
# mock helpers
# --------------------------------------------------------------------------
_TREND_WORD = {"increasing": "increasing", "decreasing": "decreasing",
               "stable": "stable"}


def _sentence(trend: str, seasonal: bool, task: str,
              event: str | None, position: str | None) -> str:
    """One sentence naming exactly one class per applicable slot."""
    season = "seasonal" if seasonal else "not seasonal"
    if task == "describe":
        return f"The series is {_TREND_WORD[trend]} and {season}."
    return (f"The series is {_TREND_WORD[trend]} and {season}; the main "
            f"event is a {event} occurring {position} in the window.")


def _assist_p(strength: float, assisted: bool,
              assist_effect: float = 1.0) -> float:
    """Probability of a correct qualitative answer.

    The two historical constants were `0.75 + 0.20*strength` when the
    harness supplied a `computed features:` block and `0.50 + 0.45*strength`
    when it did not. That is a hand-set, per-arm success rate, and with
    `assist_effect=1.0` it is preserved exactly.

    `assist_effect` interpolates between them, so `0.0` makes the mock
    ARM-BLIND. A fixture needs that to be a null: if the mock is allowed to
    answer better merely because the prompt looks like the harness's, then
    the fixture cannot distinguish a harness that genuinely helps from a
    pipeline that manufactures a difference - which is the one failure the
    benchmark exists to rule out.
    """
    lo = 0.50 + 0.45 * strength
    hi = 0.75 + 0.20 * strength
    if not assisted:
        return lo
    return lo + float(assist_effect) * (hi - lo)


def _mock_qual_sentence(x, task: str, rng, strength: float,
                        assisted: bool, assist_effect: float = 1.0) -> str:
    from src import ground_truth as gt
    from systems.tools import estimate_period
    if len(x) < 8:
        return "The series is stable and not seasonal."
    period = estimate_period(x)
    trend = gt.trend_direction(x)
    seasonal = gt.is_seasonal(x, period)
    idx = gt.change_point_index(x)
    if gt.count_spikes(x) >= 1:
        event = "spike"
    elif gt.has_level_shift(x, idx):
        event = "abrupt change"
    else:
        event = "no major event"
    position = gt.event_position(idx, len(x))
    p = _assist_p(strength, assisted, assist_effect)
    if rng.random() >= p:                      # a confident wrong answer
        trend = str(rng.choice(["increasing", "decreasing", "stable"]))
        seasonal = bool(rng.random() < 0.5)
        event = str(rng.choice(["spike", "abrupt change", "no major event"]))
        position = str(rng.choice(["early", "middle", "late"]))
    return _sentence(trend, seasonal, task, event, position)


def _qual_from_features(prompt: str, rng, strength: float, task: str,
                        assist_effect: float = 1.0) -> str:
    f = re.search(r"computed features: (.+)", prompt)
    block = f.group(1) if f else ""
    trend = ("increasing" if "trend=increasing" in block else
             "decreasing" if "trend=decreasing" in block else "stable")
    seasonal = "seasonal=yes" in block
    event = ("spike" if "spike=yes" in block else
             "abrupt change" if "abrupt_change=yes" in block else
             "no major event")
    position = ("early" if "position=early" in block else
                "late" if "position=late" in block else "middle")
    if rng.random() >= _assist_p(strength, True, assist_effect):
        # FAILURE BREADTH, and a second per-arm channel that is easy to miss.
        # Historically this path corrupted ONLY `trend`, while the unassisted
        # path in `_mock_qual_sentence` corrupts all four slots. So even with
        # the success rates equalised, the assisted arm kept 3 of 4 slots on
        # every error and scored ~0.02 higher for that reason alone. At
        # assist_effect=0.0 the breadth is matched too, so the only remaining
        # difference between the arms is exactly zero.
        if assist_effect == 0.0:
            trend = str(rng.choice(["increasing", "decreasing", "stable"]))
            seasonal = bool(rng.random() < 0.5)
            event = str(rng.choice(["spike", "abrupt change",
                                    "no major event"]))
            position = str(rng.choice(["early", "middle", "late"]))
        else:
            trend = str(rng.choice(["increasing", "decreasing", "stable"]))
    return _sentence(trend, seasonal, task, event, position)


def _mock_qual_c2(x, question: str, rng, strength: float, assisted: bool,
                  assist_effect: float = 1.0) -> str:
    """A C2 multiple-choice reply: two `slot: option` lines.

    The mock's "reading" of the series is the computed reference labels
    (`gt.qualitative_computed_c2`), which agree with the generator-truth key
    often but not always - so, unlike cycle 1's mock, this one is NOT handed
    the answer key. With probability 1 - p it answers at random instead.
    Which two lines it writes follows the question, as a model would.
    """
    from src import ground_truth as gt
    if len(x) < 8:
        guess = dict(trend="stable", seasonal="no", event="no major event",
                     position="middle")
    else:
        guess = gt.qualitative_computed_c2(x)
    if rng.random() >= _assist_p(strength, assisted, assist_effect):
        guess = dict(
            trend=str(rng.choice(["increasing", "decreasing", "stable"])),
            seasonal=str(rng.choice(["yes", "no"])),
            event=str(rng.choice(["spike", "abrupt change",
                                  "no major event"])),
            position=str(rng.choice(["early", "middle", "late"])))
    if "'event: <" in question:
        pos = ("none" if guess["event"] == "no major event"
               else guess["position"])
        return f"event: {guess['event']}\nposition: {pos}"
    return f"trend: {guess['trend']}\nseasonal: {guess['seasonal']}"


def _fmt(v: float, is_int: bool = False) -> str:
    return str(int(round(v))) if is_int else f"{v:.6g}"


def _inhead_forecast(x: np.ndarray, h: int) -> np.ndarray:
    n = len(x)
    if n < 2:
        return np.full(h, x[-1] if n else 0.0)
    last = x[-max(2, n // 4):]
    slope = (float(np.polyfit(np.arange(len(last)), last, 1)[0])
             if len(last) >= 3 else float(x[-1] - x[-2]))
    return np.array([x[-1] + slope * (i + 1) for i in range(h)])
