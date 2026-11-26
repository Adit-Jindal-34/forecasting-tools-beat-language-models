"""Orchestration: build the registry, run cached inference, evaluate.

THE ORACLE CHANNEL LIVES HERE, AND ONLY HERE
--------------------------------------------
`run_inference` is the one place that reads `QUERY_OP[sample.query_type]`,
and it does so only for systems configured with `routing: oracle`. The value
travels through the explicit `oracle_op` argument of `system.answer`, which
`systems/base.py` refuses for any system not declared as an oracle arm. A
system can therefore never obtain the label by itself, and the ablation is
visible in the trace rather than hidden in a fallback.

Inference is NOT result-cached. Model CALLS are cached (src/cache.py), so
re-running after a parser, router or scorer change replays at zero API cost
while still recomputing everything downstream of the reply.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from evaluation.evaluate import evaluate, forecast_denominators
from evaluation.judge import QualJudge
from models import build_llm, build_tsfm
from src import synthetic
from src.cache import JsonlCache, CachedLLM, CachedTSFM
from src.config import Config, QUERY_OP
from systems.bare_llm import BareLLMSystem
from systems.harness_tsfm import HarnessTSFMSystem

log = logging.getLogger("tsarch")


class Registry:
    """Every model and system for a run, with caching already wired in."""

    def __init__(self, cfg: Config, cache_dir: str | Path,
                 enabled: bool = True):
        self.cfg = cfg
        self.data_fp = cfg.data_fingerprint()
        cache_dir = Path(cache_dir)
        self.call_cache = JsonlCache(cache_dir / "calls.jsonl", enabled)
        self.tsfm_cache = JsonlCache(cache_dir / "forecasts.jsonl", enabled)

        self.llms: dict = {}
        for name, spec in cfg.backbones.items():
            inner = build_llm(name, spec, cfg.mocks, seed=cfg.seed)
            self.llms[name] = CachedLLM(inner, self.call_cache, self.data_fp,
                                        backbone=name)
        self.tsfm = CachedTSFM(build_tsfm(cfg.tsfm, seed=cfg.seed),
                               self.tsfm_cache, self.data_fp)

        self.systems: dict = {}
        for s in cfg.systems:
            self.systems[s.name] = self._build(s)

        self.judge: QualJudge | None = None
        if cfg.judge.get("enabled"):
            jb = cfg.judge.get("backbone")
            if not jb or jb not in self.llms:
                raise ValueError("judge.enabled but judge.backbone is "
                                 "missing or unknown")
            self.judge = QualJudge(self.llms[jb],
                                   mode=cfg.judge.get("mode", "reference"))

    def _build(self, s):
        # `design` has ONE source - the data block - so the pool, the
        # prompts and the scorer cannot disagree about which cycle this is.
        design = self.cfg.design
        if s.kind == "bare_llm":
            return BareLLMSystem(s.name, self.llms[s.backbone], s.backbone,
                                 dict(s.params or {}, design=design))
        if s.kind == "harness_tsfm":
            return HarnessTSFMSystem(s.name, self.llms[s.backbone], self.tsfm,
                                     s.backbone,
                                     dict(self.cfg.harness, design=design),
                                     routing=s.routing)
        raise ValueError(f"unknown system kind {s.kind!r}")

    def cache_stats(self) -> dict:
        return dict(calls=self.call_cache.stats(),
                    forecasts=self.tsfm_cache.stats(),
                    # billed calls in this process that ran out of tokens;
                    # each is also marked in its calls.jsonl record
                    truncated_calls=sum(getattr(m, "truncated", 0)
                                        for m in self.llms.values()))


def make_samples(cfg: Config, root: str | Path) -> list:
    """Materialise the frozen pool. Honours `data.pool_dir` for EVERY runner."""
    return synthetic.materialize(cfg, pool_dir=cfg.pool_dir(Path(root)))


def run_inference(cfg: Config, samples: list, L: int, systems: dict,
                  registry: Registry | None = None) -> dict:
    """Run every system on every sample at history length L.

    Returns {(sample_id, system_name): SystemResult}, in sample order.

    `cfg.inference.workers` > 1 answers that many SAMPLES at once (each
    sample's systems still run in order). The model server batches the
    concurrent requests, which matters for reasoning models whose replies
    are long. Temperature 0 and the call cache make the result independent
    of the worker count; tests/test_parallel.py checks it is identical.
    """
    workers = int((getattr(cfg, "inference", None) or {}).get("workers", 1))

    def one(sample):
        view = sample.view(L)                 # the ONLY thing a system sees
        res = {}
        for name, system in systems.items():
            oracle_op = None
            if getattr(system, "routing", "self") == "oracle":
                # runner-side privileged lookup, by design and logged
                oracle_op = QUERY_OP[sample.query_type]
            res[(sample.sample_id, name)] = system.answer(
                view, oracle_op=oracle_op)
        return res

    out = {}
    if workers <= 1:
        for sample in samples:
            out.update(one(sample))
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # map() yields in input order, so `out` is ordered as in the
            # sequential path whatever order the requests finish in
            for i, res in enumerate(pool.map(one, samples), 1):
                out.update(res)
                if i % 100 == 0:
                    log.info("L=%d: %d/%d samples done", L, i, len(samples))
    log.info("L=%d: %d system results over %d samples (workers=%d)",
             L, len(out), len(samples), workers)
    return out


def evaluate_run(cfg: Config, samples: list, L: int, results: dict,
                 judge: QualJudge | None = None,
                 denominators: dict | None = None) -> list:
    """Score every result. Evaluation never calls a model; re-run it freely."""
    by_id = {s.sample_id: s for s in samples}
    rows = []
    for (sid, name), res in results.items():
        sample = by_id[sid]
        den = denominators.get(sid) if denominators else None
        row = evaluate(sample, L, res, scoring=cfg.scoring,
                       forecast_denominator=den, judge=judge,
                       design=cfg.design)
        # the runner may annotate the correct op post hoc; a SYSTEM may not
        row["dataset_op"] = QUERY_OP[sample.query_type]
        row["paraphrase_id"] = sample.paraphrase_id
        row["framing_id"] = sample.framing_id
        row["lead_id"] = sample.lead_id

        # Routing accuracy, TWO columns, for the same reason §1.2 scores an
        # unanswered item zero instead of dropping it. The single column
        # went None on an abstention and `.mean()` skipped it, so a router
        # that abstained on everything would report 100% routing accuracy on
        # the handful it did answer. Primary counts an abstention as a
        # routing failure; the selective variant is reported beside it.
        routed = row.get("route_op") is not None
        if row.get("route_source") in (None, "none"):
            # a bare LLM has nothing to route; neither column applies
            row["route_correct"] = None
            row["route_correct_selective"] = None
            row["route_equivalent"] = None
        else:
            correct = int(routed and row["route_op"] == row["dataset_op"])
            row["route_correct"] = correct
            row["route_correct_selective"] = correct if routed else None
            # FUNCTIONAL routing (PREREGISTRATION_C2.md §C2-8). `describe`
            # and `explain` dispatch to the same code with the same inputs
            # and differ only in a task header, so sending an event question
            # to `describe` costs nothing. Cycle 1 counted that as an error,
            # which is how a rung with 97% working routing was reported at
            # 3%. The strict column stays; this one sits beside it.
            row["route_equivalent"] = int(
                routed and route_equivalent(row["route_op"],
                                            row["dataset_op"]))
        rows.append(row)
    return rows


#: Sets of operations that are interchangeable in their effect. Declared in
#: PREREGISTRATION_C2.md §C2-8; no other class is declared.
ROUTE_EQUIVALENCE_CLASSES: tuple[frozenset, ...] = (
    frozenset({"describe", "explain"}),
)


def route_equivalent(route_op, dataset_op) -> bool:
    if route_op == dataset_op:
        return True
    return any(route_op in c and dataset_op in c
               for c in ROUTE_EQUIVALENCE_CLASSES)


def fixed_denominators(cfg: Config, samples: list) -> dict | None:
    """MASE denominators from each series' longest configured window."""
    if not cfg.scoring.get("fixed_mase_denominator", True):
        return None
    return forecast_denominators(samples, max(cfg.history_lengths))


def save_csv(rows: list, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_json(obj, path: str | Path) -> None:
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str),
                          encoding="utf-8")
