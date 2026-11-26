"""Harness: the LLM routes, deterministic tools and a TSFM do the work.

    InferenceSample
        |
    1. classify  -> RouteResult (or abstain)        [one LLM call]
        |           routing='oracle' skips this and takes the given op
    2. dispatch
        |   forecast          -> TSFM
        |   numeric / index   -> deterministic tool (exact)
        |   describe/explain  -> computed features, verbalised by the LLM
        |
    3. final answer

THREE DEFECTS FIXED HERE
------------------------
1. The oracle-leak fallback is gone. A reply that cannot be routed becomes an
   explicit abstention (see systems/routing.py).

2. The qualitative branch now follows the ROUTED op. The old build computed
   `op`, discarded it, and built the prompt from `sample.query_type`:

       task="describe" if sample.query_type == "trend_description" else ...
       qtype=sample.query_type, ...

   so routing was a no-op on 2 of 9 types - which is the real reason
   mis-routing 66/100 items "cost only 0.032". It cost nothing because the
   router's output was never used.

3. The feature extractor estimates the seasonal period instead of being
   handed the generator's true one.
"""
from __future__ import annotations

from models import llm as LM
from src.samples import InferenceSample

from .base import TimeSeriesSystem, TraceStep
from .bare_llm import fmt_series
from .routing import (VALID_OPS, parse_route, oracle_route, RouteResult,
                      STATUS_CLEAN)
from .tools import (run_tool, series_features, features_to_block, TOOL_OPS,
                    series_evidence, evidence_to_block)

SYSTEM_PROMPT = ("You are the routing layer of a time-series agent. You "
                 "classify queries and dispatch them to the right "
                 "specialist; you do not do the arithmetic yourself.")

#: ops the router may choose from, shown in the classify prompt
ROUTABLE_OPS = tuple(sorted(VALID_OPS))


class HarnessTSFMSystem(TimeSeriesSystem):
    def __init__(self, name: str, llm, tsfm, backbone: str,
                 harness_cfg: dict | None = None, routing: str = "self"):
        super().__init__()
        self.name = name
        self.backbone = backbone
        self.llm = llm
        self.tsfm = tsfm
        if routing not in ("self", "oracle"):
            raise ValueError(f"routing must be 'self' or 'oracle', got {routing!r}")
        self.routing = routing
        self.cfg = dict({"numeric_final": "direct",
                         "assist_qualitative": True}, **(harness_cfg or {}))

    # -- step 1 ------------------------------------------------------------
    def _classify(self, s: InferenceSample,
                  oracle_op: str | None) -> tuple[RouteResult, list]:
        if self.routing == "oracle":
            if oracle_op is None:
                raise ValueError(
                    f"{self.name}: routing='oracle' but no oracle_op given")
            route = oracle_route(oracle_op)
            return route, [TraceStep("classify", route.as_detail())]

        if self.cfg.get("design") == "c2":
            # C2-8: the router is told what each tool does. Cycle 1 passed
            # bare names, so routing partly measured how guessable they were.
            prompt = LM.CLASSIFY_PROMPT_C2.format(
                op_menu=LM.op_menu(ROUTABLE_OPS), question=s.query)
        else:
            prompt = LM.CLASSIFY_PROMPT.format(
                ops=", ".join(ROUTABLE_OPS), question=s.query)
        raw = self._llm_call(self.llm, SYSTEM_PROMPT, prompt)
        route = parse_route(raw)
        detail = route.as_detail()
        detail["prompt_chars"] = len(prompt)
        return route, [TraceStep("classify", detail)]

    # -- step 3 ------------------------------------------------------------
    def _final_numeric(self, value, is_int) -> tuple:
        if self.cfg.get("numeric_final") == "llm":
            text = self._llm_call(
                self.llm, SYSTEM_PROMPT,
                LM.VOICE_PROMPT.format(
                    value=int(round(value)) if is_int else f"{value:.6g}"))
            return text, text, [TraceStep("final", dict(mode="llm-voiced"))]
        text = str(int(round(value))) if is_int else f"{value:.6g}"
        parsed = int(round(value)) if is_int else float(value)
        return text, parsed, [TraceStep("final", dict(mode="direct"))]

    # -- main --------------------------------------------------------------
    def _answer(self, s: InferenceSample, oracle_op: str | None):
        x = s.window
        route, trace = self._classify(s, oracle_op)
        extra = dict(route_op=route.op, route_status=route.status,
                     route_source=route.source,
                     op_strict=route.op_strict,
                     op_permissive=route.op_permissive)

        if route.abstained:
            trace.append(TraceStep("final", dict(mode="abstain")))
            return ("", None, False, True, f"route_{route.status}",
                    trace, extra)

        op = route.op
        if op == "forecast":
            vals = self.tsfm.forecast(x, s.horizon)
            trace.append(TraceStep("tsfm", dict(
                model=getattr(self.tsfm, "name", "tsfm"), input_len=s.L,
                horizon=s.horizon, output_n=len(vals))))
            trace.append(TraceStep("final", dict(mode="direct")))
            text = ", ".join(f"{v:.6g}" for v in vals)
            return text, list(vals), True, False, "", trace, extra

        if op in TOOL_OPS:
            value, is_int = run_tool(op, x)
            trace.append(TraceStep("tool", dict(tool=op, input_len=s.L,
                                                output=str(value)[:60],
                                                exact=True)))
            text, parsed, ft = self._final_numeric(value, is_int)
            return text, parsed, True, False, "", trace + ft, extra

        # qualitative: describe | explain -- driven by the ROUTED op
        if self.cfg.get("design") == "c2":
            # Numeric evidence BESIDE the series, never instead of it, and
            # no verdict in the key's vocabulary. The prompt is identical
            # for `describe` and `explain` apart from the task header; the
            # answer format comes from the question.
            e = series_evidence(x)
            trace.append(TraceStep("feature", e))
            prompt = LM.QUAL_PROMPT_C2.format(
                task=op, L=s.L, evidence_block=evidence_to_block(e) + "\n",
                series=fmt_series(x), question=s.query)
            text = self._llm_call(self.llm, SYSTEM_PROMPT, prompt)
            trace.append(TraceStep("final", dict(mode="llm-evidence")))
            return text, text, True, False, "", trace, extra

        f = series_features(x)
        trace.append(TraceStep("feature", f))
        assisted = bool(self.cfg.get("assist_qualitative", True))
        prompt = LM.QUAL_PROMPT.format(
            task=op, L=s.L,
            feature_block=(features_to_block(f) + "\n") if assisted else "",
            question=s.query,
            series_block="" if assisted else
            f"Series (oldest to newest, {s.L} values): {fmt_series(x)}\n")
        text = self._llm_call(self.llm, SYSTEM_PROMPT, prompt)
        trace.append(TraceStep("final", dict(
            mode="llm-assisted" if assisted else "llm-unassisted")))
        return text, text, True, False, "", trace, extra
