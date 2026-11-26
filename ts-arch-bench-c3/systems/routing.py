"""Routing: turn a classifier reply into an operation, or abstain.

TWO DEFECTS THIS REPLACES
-------------------------
D1, the oracle leak. The old fallback was

    fallback_op = QUERY_OP.get(sample.query_type, "describe")

a per-sample lookup keyed on the GROUND-TRUTH LABEL, so a parse failure read
the answer key. It fired 19/900 times with the real backbone (all on
localization, all scoring exactly 1.000) and - because the mock classifier
branch was unreachable - 100% of the time in every mock run.

D2, the brittle parser. `raw.strip().lower().split()[0]` took the first
whitespace token, so any prose reply ("To find the index where...") fell
through to that fallback.

THE REPLACEMENT
---------------
No fallback to anything label-derived. The parser reports BOTH readings on
every call so parser recovery is measurable offline, and an unrecoverable
reply becomes an explicit abstention:

    clean        first token is a valid op
    recovered    first token is not, but exactly one valid op appears
    ambiguous    two or more distinct valid ops appear      -> ABSTAIN
    unparseable  no valid op appears                        -> ABSTAIN

An abstention scores zero and is counted in its own column. It is never
folded into the accuracy figure, which is what let the leak hide.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

#: the operations a router may select. Defined here, in the systems package,
#: deliberately: `src.config.QUERY_OP` maps LABEL -> op and must never be
#: importable from inside a system.
VALID_OPS: frozenset[str] = frozenset({
    "mean", "median", "std", "max", "min", "count_spikes", "pct_change",
    "lag1_autocorr", "change_point", "forecast", "describe", "explain",
})

STATUS_CLEAN = "clean"
STATUS_RECOVERED = "recovered"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_UNPARSEABLE = "unparseable"

#: longest-first so `pct_change` is matched before any shorter overlap
_OP_RE = re.compile(
    r"(?<![A-Za-z0-9_])(" +
    "|".join(sorted((re.escape(o) for o in VALID_OPS), key=len, reverse=True))
    + r")(?![A-Za-z0-9_])", re.I)


@dataclass(frozen=True)
class RouteResult:
    op: str | None
    status: str
    source: str               # "llm" | "oracle" | "abstain"
    op_strict: str | None     # first-token reading
    op_permissive: str | None # whole-reply scan reading
    candidates: tuple[str, ...]
    raw: str

    @property
    def abstained(self) -> bool:
        return self.op is None

    def as_detail(self) -> dict:
        d = asdict(self)
        d["candidates"] = "|".join(self.candidates)
        return d


def parse_route(raw: str) -> RouteResult:
    """Parse a classifier reply. Never consults ground truth."""
    text = raw or ""
    tokens = text.strip().lower().split()
    first = tokens[0].strip(".,:;'\"`*") if tokens else ""
    strict = first if first in VALID_OPS else None

    found: list[str] = []
    for m in _OP_RE.finditer(text):
        op = m.group(1).lower()
        if op not in found:
            found.append(op)
    permissive = found[0] if len(found) == 1 else None

    if strict is not None:
        return RouteResult(strict, STATUS_CLEAN, "llm", strict, permissive,
                           tuple(found), text)
    if len(found) == 1:
        return RouteResult(found[0], STATUS_RECOVERED, "llm", None,
                           found[0], tuple(found), text)
    if len(found) > 1:
        return RouteResult(None, STATUS_AMBIGUOUS, "abstain", None, None,
                           tuple(found), text)
    return RouteResult(None, STATUS_UNPARSEABLE, "abstain", None, None,
                       (), text)


def oracle_route(op: str) -> RouteResult:
    """The ablation arm: a correct op supplied by the runner, logged as such."""
    if op not in VALID_OPS:
        raise ValueError(f"oracle_op {op!r} is not a valid operation")
    return RouteResult(op, STATUS_CLEAN, "oracle", op, op, (op,), "")


#: which answer shape each op produces, so the evaluator can type-check
OP_OUTPUT_KIND: dict[str, str] = {
    "mean": "numeric", "median": "numeric", "std": "numeric",
    "max": "numeric", "min": "numeric", "count_spikes": "numeric",
    "pct_change": "numeric", "lag1_autocorr": "numeric",
    "change_point": "index", "forecast": "sequence",
    "describe": "text", "explain": "text",
}
