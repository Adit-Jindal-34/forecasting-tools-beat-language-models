"""Caching, restructured around WHAT IS ACTUALLY EXPENSIVE.

THE OLD DESIGN AND WHY IT COST A RERUN
--------------------------------------
The previous build cached a whole `SystemResult` under a key containing one
global `config fingerprint`. Two consequences:

  * Changing the parser changed the result, so a one-line parser fix could
    only be applied by invalidating cached inference - i.e. by paying for
    every Gemini and Chronos call again.
  * `fingerprint()` hashed the ENTIRE `backbones` dict, so adding a ninth
    backbone to an H3 config invalidated the other eight. groupA, groupB and
    a unified config each produced a different fingerprint and shared no
    cache at all.
  * `seed` was NOT in the fingerprint, while the data manifest DID record
    it. Sample ids are positional, so reseeding produced the same ids over
    different series and the run silently served cached answers for data
    that no longer existed.

THE DESIGN HERE
---------------
Cache the MODEL CALLS, not the system results.

    CallCache   (backbone fingerprint, system prompt, user prompt) -> text
    TSFMCache   (tsfm fingerprint, series, horizon)                -> values

Systems are then re-executed on every run, which is free: tool calls are
pure Python. A parser change, a routing change, a scorer change or an added
oracle-routing arm all replay against a warm cache and issue ZERO API calls.
Only a prompt change (PROMPT_VERSION) invalidates anything.

Keys are per-backbone, so backbones are independent: adding one never
invalidates another. The data fingerprint - which DOES include the seed -
enters the key so that regenerated data can never collide with stale answers.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np

import src


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


class JsonlCache:
    """Append-only JSONL store with an in-memory index. Resumable for free."""

    def __init__(self, path: str | Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = bool(enabled)
        self._index: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0
        # Parallel inference: get/put from several threads. One lock per
        # cache keeps the index and the append-only file consistent.
        self._lock = threading.RLock()
        if self.enabled and self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._index[rec["key"]] = rec
                    except (json.JSONDecodeError, KeyError):
                        continue

    def get(self, key: str):
        with self._lock:
            if not self.enabled:
                self.misses += 1
                return None
            rec = self._index.get(key)
            if rec is None:
                self.misses += 1
                return None
            self.hits += 1
            return rec["value"]

    def put(self, key: str, value, meta: dict | None = None) -> None:
        with self._lock:
            if not self.enabled or key in self._index:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rec = {"key": key, "value": value}
            if meta:
                rec["meta"] = meta
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            self._index[key] = rec
            self.writes += 1

    def stats(self) -> dict:
        return dict(entries=len(self._index), hits=self.hits,
                    misses=self.misses, writes=self.writes)

    def __len__(self) -> int:
        return len(self._index)


# --------------------------------------------------------------------------
# LLM call cache
# --------------------------------------------------------------------------
def call_key(backbone_fp: dict, system_prompt: str,
             user_prompt: str) -> str:
    """Key on exactly what determines the reply, and nothing else.

    At temperature 0 the reply is a function of (model, generation params,
    prompts). The data fingerprint is deliberately NOT in the key: series
    values are already inside the prompt, while classify prompts contain only
    the question text, so routing replies are legitimately shareable across
    pools and seeds. Recording the pool in `meta` keeps provenance without
    forcing a re-ask.

    Caveat to record in the methods section: with temperature > 0 a cache hit
    reuses one sample rather than redrawing. Every config here pins 0.
    """
    return _sha(_canonical(backbone_fp), str(src.PROMPT_VERSION),
                system_prompt, user_prompt)[:32]


class CachedLLM:
    """Wraps an LLMClient so identical prompts are never paid for twice.

    The wrapped client is only constructed-and-called on a miss, so a replay
    with a warm cache never touches the network even if the parser, router or
    scorer changed underneath.
    """

    def __init__(self, inner, cache: JsonlCache, data_fp: str,
                 backbone: str = ""):
        self.inner = inner
        self.cache = cache
        self.data_fp = data_fp
        self.backbone = backbone
        self.name = getattr(inner, "name", "llm")
        self.model = getattr(inner, "model", "unknown")
        self._fp = (inner.fingerprint() if hasattr(inner, "fingerprint")
                    else dict(name=self.name, model=self.model))
        self._tl = threading.local()
        #: Billed calls in THIS process that hit the token limit
        #: (finish_reason == "length"). Logged at the end of a cell.
        self.truncated = 0
        self._count_lock = threading.Lock()

    def fingerprint(self) -> dict:
        return dict(self._fp)

    @property
    def last_call_cached(self) -> bool:
        """Whether THIS thread's most recent call was served from disk."""
        return bool(getattr(self._tl, "cached", False))

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        key = call_key(self._fp, system_prompt, user_prompt)
        hit = self.cache.get(key)
        if hit is not None:
            self._tl.cached = True
            return hit
        self._tl.cached = False
        out = self.inner.complete(system_prompt, user_prompt)
        meta = dict(backbone=self.backbone, model=self.model,
                    pool=self.data_fp)
        # Server-side facts about the call (finish_reason, token count,
        # whether the answer was empty), when the client reports them.
        # Provenance only - `meta` is never part of a key and is never read
        # back by a system or a scorer - but it is what later tells a reply
        # that ran out of tokens from one that said nothing.
        call = getattr(self.inner, "last_call_meta", None)
        if call:
            meta["call"] = call
            if call.get("finish_reason") == "length":
                with self._count_lock:
                    self.truncated += 1
        self.cache.put(key, out, meta=meta)
        return out


# --------------------------------------------------------------------------
# forecaster cache
# --------------------------------------------------------------------------
def tsfm_key(tsfm_fp: dict, series: np.ndarray, horizon: int) -> str:
    """Keyed on the exact window bytes, so identical windows share a forecast
    regardless of which pool or history length produced them."""
    digest = hashlib.sha256(
        np.asarray(series, np.float64).tobytes()).hexdigest()
    return _sha(_canonical(tsfm_fp), digest, str(horizon))[:32]


class CachedTSFM:
    """Wraps a TSFM. Chronos on CPU is the slowest component; cache it hard."""

    def __init__(self, inner, cache: JsonlCache, data_fp: str):
        self.inner = inner
        self.cache = cache
        self.data_fp = data_fp
        self.name = getattr(inner, "name", "tsfm")
        self._fp = (inner.fingerprint() if hasattr(inner, "fingerprint")
                    else dict(name=self.name))
        # The forecaster is NEVER run concurrently: Chronos seeds torch's
        # global generator before each call (models/tsfm.py), so two
        # overlapping calls would draw from each other's seed and break
        # reproducibility. It is fast; serialising it costs nothing.
        self._lock = threading.Lock()

    def fingerprint(self) -> dict:
        return dict(self._fp)

    def forecast(self, series: np.ndarray, horizon: int) -> np.ndarray:
        key = tsfm_key(self._fp, series, horizon)
        with self._lock:
            hit = self.cache.get(key)
            if hit is not None:
                return np.asarray(hit, float)
            out = np.asarray(self.inner.forecast(series, horizon), float)
            self.cache.put(key, out.tolist(), meta=dict(pool=self.data_fp))
            return out
