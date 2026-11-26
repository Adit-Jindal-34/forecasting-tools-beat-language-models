"""Bare LLM: series + question in, answer out. No tools, no delegation.

THE ASYMMETRY THIS REMOVES
--------------------------
The old build built its numeric prompt as

    NUMERIC_PROMPT.format(op=QUERY_OP[sample.query_type], ...)

rendering a header of `[task:numeric op=mean history_len=128]`. That is the
GROUND-TRUTH OPERATION, handed to the bare arm on all 900 samples, while the
harness had to infer the same thing from question text alone. The comparison
was therefore not "architecture A vs architecture B" but "oracle-routed bare
LLM vs self-routed harness", and the routing tax was charged to one side.
It landed hardest on localization - the single cell the bare LLM won.

Here both arms get the same information budget: the question text and the
series, nothing else. One generic prompt is used for every query type, so
inferring what is being asked is part of the task for BOTH arms, as it is in
deployment.

Answer parsing happens on the EVALUATOR side, which legitimately knows what
it asked. Parsing is not leakage; prompting is.
"""
from __future__ import annotations

import numpy as np

from models import llm as LM
from src.samples import InferenceSample

from .base import TimeSeriesSystem, TraceStep
from .tools import series_evidence, evidence_to_block

SYSTEM_PROMPT = ("You are a precise time-series analyst. Answer exactly in "
                 "the requested format, using only the provided series.")


def fmt_series(x: np.ndarray) -> str:
    return ", ".join(f"{v:.6g}" for v in x)


class BareLLMSystem(TimeSeriesSystem):
    """One LLM call per sample. The model reads the numbers itself."""

    def __init__(self, name: str, llm, backbone: str,
                 params: dict | None = None):
        super().__init__()
        self.name = name
        self.backbone = backbone
        self.llm = llm
        self.routing = "self"          # a bare LLM has nothing to route
        self.params = params or {}

    def _answer(self, s: InferenceSample, oracle_op: str | None):
        if self.params.get("design") == "c2":
            return self._answer_c2(s)
        prompt = LM.GENERIC_PROMPT.format(
            L=s.L, series=fmt_series(s.window), question=s.query)
        trace = [TraceStep("prompt", dict(L=s.L, chars=len(prompt),
                                          mode="generic"))]
        text = self._llm_call(self.llm, SYSTEM_PROMPT, prompt)
        trace.append(TraceStep("final", dict(mode="llm")))
        # answered/abstained are decided by the evaluator, which knows the
        # expected answer shape; the system just reports what it said.
        return text, text, True, False, "", trace, dict(route_source="none")

    def _answer_c2(self, s: InferenceSample):
        """Cycle 2. One generic prompt for every query type, as in cycle 1.

        With `params.features` this is the `bare_features` arm
        (PREREGISTRATION_C2.md §C2-5): the SAME numeric evidence block the
        harness's model sees, beside the series, with no routing and no tool
        choice. It separates "does the computation help" from "does the
        harness scaffolding help". The block is computed for every question
        regardless of type - this arm does not know the type either.
        """
        features = bool(self.params.get("features"))
        trace = []
        evidence = ""
        if features:
            e = series_evidence(s.window)
            trace.append(TraceStep("feature", e))
            evidence = evidence_to_block(e) + "\n"
        prompt = LM.GENERIC_PROMPT_C2.format(
            L=s.L, evidence_block=evidence, series=fmt_series(s.window),
            question=s.query)
        trace.append(TraceStep("prompt", dict(
            L=s.L, chars=len(prompt),
            mode="generic-c2+features" if features else "generic-c2")))
        text = self._llm_call(self.llm, SYSTEM_PROMPT, prompt)
        trace.append(TraceStep("final", dict(mode="llm")))
        return text, text, True, False, "", trace, dict(route_source="none")
