"""Common system interface and result container.

Every architecture implements `_answer(InferenceSample, oracle_op) -> tuple`.
The signature is the enforcement point: a system is handed an
`InferenceSample`, which carries no label, no generator parameter and no
future data. There is nothing to read even if a system wanted to.

`oracle_op` is the ONE privileged channel, and it is explicit. It exists so
the oracle-routing ablation can be run without a system ever touching
ground truth itself: the runner decides whether to supply it, based on the
system's configured `routing` mode, and records that decision in the trace.
A system with `routing: self` is always passed None.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.samples import InferenceSample


@dataclass
class TraceStep:
    step: str                 # classify | tool | tsfm | feature | final
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(step=self.step, detail=self.detail)


@dataclass
class SystemResult:
    sample_id: str
    system: str
    backbone: str
    answer_text: str
    parsed: Any
    answered: bool            # produced a usable answer of the right shape
    abstained: bool           # explicitly declined (never a silent failure)
    failure: str              # "" | "unparseable" | "ambiguous_route" | ...
    #: ARCHITECTURAL calls: how many times this design consults a model.
    #: A property of the architecture, so this is the right number for a
    #: "how many model calls does this approach need" claim.
    #:
    #: NOT COMPARABLE ACROSS ARMS. The oracle-routing arm issues zero
    #: classify calls BY DESIGN, so it will always look cheaper than the
    #: self-routing arm. That is the ablation, not a finding.
    n_model_calls: int
    #: BILLED calls: how many actually reached the provider; the rest were
    #: cache hits. Use this for cost, never for architecture.
    n_billed_calls: int
    latency_s: float
    trace: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def trace_dicts(self) -> list:
        return [t.to_dict() if isinstance(t, TraceStep) else t
                for t in self.trace]


class TimeSeriesSystem(ABC):
    """Answer one InferenceSample. Must not reach outside its window."""

    name: str = "base"
    backbone: str = "none"
    routing: str = "self"     # "self" | "oracle"

    def __init__(self) -> None:
        self._calls = 0        # architectural
        self._billed = 0       # actually reached the provider
        # Per-THREAD counters for the answer() in progress. With parallel
        # inference several answer() calls on one system overlap, so the
        # shared totals above cannot be differenced into a per-item count.
        self._tl = threading.local()
        self._lock = threading.Lock()

    @abstractmethod
    def _answer(self, s: InferenceSample, oracle_op: str | None) -> tuple:
        """Return (text, parsed, answered, abstained, failure, trace, extra)."""
        raise NotImplementedError

    def answer(self, s: InferenceSample,
               oracle_op: str | None = None) -> SystemResult:
        if not isinstance(s, InferenceSample):
            raise TypeError(
                f"{type(self).__name__}.answer expects an InferenceSample, "
                f"got {type(s).__name__}. Systems never receive a Sample: "
                "that object carries the ground-truth label.")
        if self.routing != "oracle" and oracle_op is not None:
            raise ValueError(
                f"system {self.name!r} has routing={self.routing!r} but an "
                "oracle_op was supplied; only routing='oracle' systems may "
                "receive one")
        t0 = time.time()
        self._tl.calls = 0
        self._tl.billed = 0
        text, parsed, answered, abstained, failure, trace, extra = \
            self._answer(s, oracle_op)
        return SystemResult(
            sample_id=s.sample_id, system=self.name, backbone=self.backbone,
            answer_text=text, parsed=parsed, answered=bool(answered),
            abstained=bool(abstained), failure=failure,
            n_model_calls=self._tl.calls,
            n_billed_calls=self._tl.billed,
            latency_s=round(time.time() - t0, 4),
            trace=trace, extra=extra)

    def _llm_call(self, llm, system_prompt: str, user_prompt: str) -> str:
        out = llm.complete(system_prompt, user_prompt)
        # A CachedLLM reports whether the call was served from disk (per
        # thread). An unwrapped client has no cache, so every call is billed.
        billed = not getattr(llm, "last_call_cached", False)
        self._tl.calls = getattr(self._tl, "calls", 0) + 1
        self._tl.billed = getattr(self._tl, "billed", 0) + int(billed)
        with self._lock:
            self._calls += 1
            self._billed += int(billed)
        return out
