"""The evaluator: one SystemResult + the answer key -> one scored row.

THREE CHANGES FROM THE OLD BUILD
--------------------------------
1. FAILURES ARE SCORED, NOT DROPPED. The old evaluator set `score = NaN` on a
   parse failure and the analysis then called `.dropna()`, so a system was
   never penalised for failing to answer and every reported mean was silently
   conditional on parsing succeeding. Here every row gets a numeric score -
   zero when the system abstained, produced nothing usable, or returned the
   wrong TYPE - and `answered` is carried alongside so coverage and selective
   risk can both be reported (evaluation/metrics.py::selective_report).

2. OUTPUT TYPE VALIDATION (the old D3). A mis-routed localization answer used
   to be scored as `pred_idx = -125` or `321` on a 128-point series with
   nothing checking the range. Each kind now declares its admissible shape;
   violations are scored zero AND counted as `type_error`, a distinct failure
   mode from "wrong answer".

3. PARSING LIVES HERE, not in the system. The evaluator legitimately knows
   what it asked, so it may parse accordingly. A SYSTEM may not be told -
   that is the `op=` leak. Parsing on this side is what let the bare arm's
   prompt become generic.
"""
from __future__ import annotations

import json

import numpy as np

from models.llm import extract_numbers, extract_int
from src import ground_truth as gt
from src.samples import Sample
from systems.base import SystemResult

from . import metrics as M
from .qualitative import score_qualitative, score_qualitative_mc
from .judge import QualJudge

#: failure taxonomy, reported as its own column
OK = ""
F_ABSTAIN = "abstain"
F_UNPARSEABLE = "unparseable"
F_TYPE_ERROR = "type_error"


def _as_float(value) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)) and \
            not isinstance(value, bool):
        v = float(value)
        return v if np.isfinite(v) else None
    nums = extract_numbers(str(value))
    if not nums:
        return None
    v = float(nums[-1])      # trailing number: "...the mean is 42.1"
    return v if np.isfinite(v) else None


def _as_int(value) -> int | None:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(round(float(value))) if np.isfinite(value) else None
    return extract_int(str(value))


def _as_sequence(value) -> list | None:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, float).reshape(-1)
        return arr.tolist() if arr.size and np.all(np.isfinite(arr)) else None
    nums = extract_numbers(str(value))
    return nums or None


def evaluate(sample: Sample, L: int, res: SystemResult,
             *, scoring: dict | None = None,
             forecast_denominator: float | None = None,
             judge: QualJudge | None = None, design: str = "c1") -> dict:
    """Score one answer. Always returns a finite `score`.

    `design="c2"` changes exactly one thing: qualitative answers are scored
    against the generator-truth key with the multiple-choice parser.
    """
    cfg = {"tol": M.DEFAULT_TOL, "loc_tol": M.DEFAULT_LOC_TOL,
           **(scoring or {})}
    window = sample.window(L)
    kind = sample.query_kind
    c2_qual = design == "c2" and kind == "qualitative"
    truth = (gt.qualitative_truth_c2(sample, L) if c2_qual
             else gt.ground_truth(sample, L))

    row = dict(
        sample_id=sample.sample_id, system=res.system, backbone=res.backbone,
        L=L, query_type=sample.query_type, query_kind=kind,
        pattern=sample.pattern, dataset_op=None,        # filled by the runner
        n_model_calls=res.n_model_calls,
        n_billed_calls=res.n_billed_calls, latency_s=res.latency_s,
        answer_text=(res.answer_text or "")[:500],
        route_op=res.extra.get("route_op"),
        route_status=res.extra.get("route_status"),
        route_source=res.extra.get("route_source"),
        op_strict=res.extra.get("op_strict"),
        op_permissive=res.extra.get("op_permissive"),
        trace_json=_compact_trace(res),
        judge_score=float("nan"),
        answered=False, abstained=bool(res.abstained),
        type_valid=True, failure=OK, score=0.0)

    if res.abstained:
        row.update(failure=F_ABSTAIN, score=0.0)
        return row

    if kind == "forecast":
        _score_forecast(row, res, truth, window, sample, forecast_denominator)
    elif kind == "numeric_exact":
        _score_numeric(row, res, truth, window, sample, cfg)
    elif kind == "localization":
        _score_localization(row, res, truth, L, cfg)
    elif kind == "qualitative":
        _score_qualitative(row, res, truth, sample, judge, mc=c2_qual)
    else:
        raise ValueError(f"unknown query kind {kind!r}")
    return row


def _score_forecast(row, res, truth, window, sample, denominator):
    pred = _as_sequence(res.parsed if res.parsed is not None
                        else res.answer_text)
    target = np.asarray(truth, float)
    if pred is None:
        row.update(failure=F_UNPARSEABLE, score=0.0)
        return
    row["raw_forecast_len"] = len(pred)
    if len(pred) < len(target):
        # too short is a type error: padding would invent data
        row.update(type_valid=False, failure=F_TYPE_ERROR, answered=True,
                   score=0.0)
        return
    row["forecast_truncated"] = int(len(pred) > len(target))
    pred = pred[:len(target)]
    den = (denominator if denominator is not None else
           gt.naive_window_error(window, sample.seasonal_period))
    row.update(answered=True, mase_denominator=float(den),
               **M.score_forecast(pred, target, den))


def _score_numeric(row, res, truth, window, sample, cfg):
    from src.config import QUERY_OP
    op = QUERY_OP[sample.query_type]
    value = _as_float(res.parsed if res.parsed is not None
                      else res.answer_text)
    if value is None:
        row.update(failure=F_UNPARSEABLE, score=0.0)
        return
    row.update(answered=True,
               **M.score_numeric(value, truth, op, window,
                                 tol=float(cfg["tol"])))


def _score_localization(row, res, truth, L, cfg):
    idx = _as_int(res.parsed if res.parsed is not None else res.answer_text)
    if idx is None:
        row.update(failure=F_UNPARSEABLE, score=0.0)
        return
    row["pred_idx_raw"] = idx
    if not M.index_in_range(idx, L):
        # the D3 case: a percentage where an index was expected
        row.update(answered=True, type_valid=False, failure=F_TYPE_ERROR,
                   score=0.0)
        return
    row.update(answered=True,
               **M.score_localization(idx, int(truth), L,
                                      tol=int(cfg["loc_tol"])))


def _score_qualitative(row, res, truth, sample, judge, mc: bool = False):
    text = res.answer_text or ""
    if not text.strip():
        row.update(failure=F_UNPARSEABLE, score=0.0)
        return
    rec = (score_qualitative_mc if mc else score_qualitative)(
        text, truth, sample.query_type)
    per_slot = rec.pop("per_slot")
    row.update(answered=True, **rec)
    for slot, d in per_slot.items():
        row[f"slot_{slot}_expected"] = d["expected"]
        row[f"slot_{slot}_predicted"] = d["predicted"]
        row[f"slot_{slot}_status"] = d["status"]
        row[f"slot_{slot}_correct"] = d["correct"]
    if judge is not None:
        row["judge_score"] = judge.judge(sample.query, truth, text)


def _compact_trace(res: SystemResult) -> str:
    try:
        return json.dumps(res.trace_dicts(), default=str)
    except Exception:
        return "[]"


def forecast_denominators(samples, L_longest: int) -> dict:
    """One MASE denominator per base series, from its LONGEST window.

    Holding the denominator fixed is what makes scores comparable across
    history lengths: a per-L denominator moves the yardstick with the
    condition under test. Carried over from the one piece of the old repo
    that got this right (`history_length_revised.py::fixed_mase`).
    """
    return {s.sample_id: gt.naive_window_error(s.window(L_longest),
                                               s.seasonal_period)
            for s in samples}
