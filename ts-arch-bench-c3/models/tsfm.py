"""Time-series foundation models (forecast specialists).

TWO REPRODUCIBILITY FIXES
-------------------------
1. Chronos SAMPLES. The old build called it with `num_samples` and reduced by
   median, but never seeded torch, so forecasts were not reproducible: the
   same config reran to different numbers. Every forecast call here seeds a
   generator derived from (global seed, series content, horizon), so a rerun
   reproduces bit-for-bit and a cache hit is provably the same computation.

2. `num_samples` was 20 in the H2 config and 10 in every H3 config, which
   made forecast cells incomparable between the project's own hypotheses.
   The default is now 20 everywhere and the value is part of the fingerprint,
   so a mismatch invalidates the cache instead of passing silently.

Median reduction is correct for an MAE-family metric: the median minimises
expected absolute error. That part of the old build was right.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class TSFM(ABC):
    name: str = "tsfm-base"

    @abstractmethod
    def forecast(self, series: np.ndarray, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def fingerprint(self) -> dict:
        return dict(name=self.name)


def _series_seed(base_seed: int, x: np.ndarray, horizon: int) -> int:
    """A stable per-call seed: same series + horizon -> same sample draw."""
    h = hashlib.sha256(np.asarray(x, np.float64).tobytes())
    h.update(str(horizon).encode())
    h.update(str(base_seed).encode())
    return int(h.hexdigest()[:8], 16)


class MockTSFM(TSFM):
    """Deterministic drift forecaster. A pipeline placeholder, not a model."""

    def __init__(self, name: str = "mock_tsfm", noise: float = 0.0,
                 seed: int = 0):
        self.name = name
        self.noise = float(noise)
        self.seed = int(seed)

    def fingerprint(self) -> dict:
        return dict(name=self.name, noise=self.noise, seed=self.seed)

    def forecast(self, series: np.ndarray, horizon: int) -> np.ndarray:
        x = np.asarray(series, float)
        if len(x) < 2:
            return np.full(horizon, x[-1] if len(x) else 0.0)
        tail = x[-max(2, len(x) // 4):]
        slope = (float(np.polyfit(np.arange(len(tail)), tail, 1)[0])
                 if len(tail) >= 3 else float(x[-1] - x[-2]))
        out = np.array([x[-1] + slope * (i + 1) for i in range(horizon)])
        if self.noise > 0:
            rng = np.random.default_rng(_series_seed(self.seed, x, horizon))
            out = out * (1 + rng.normal(0, self.noise, horizon))
        return out


class ChronosTSFM(TSFM):
    """Amazon Chronos. Median of `num_samples` sampled paths, seeded."""

    #: Chronos T5 truncates context above this; longer windows silently lose
    #: their head. Runs at L > 512 must flag it (see src/screening.py).
    CONTEXT_LIMIT = 512

    def __init__(self, name: str = "chronos",
                 model_id: str = "amazon/chronos-t5-small",
                 device_map: str = "cpu", torch_dtype: str = "float32",
                 num_samples: int = 20, seed: int = 0, pipeline=None):
        self.name = name
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        if self.num_samples < 1:
            raise ValueError("chronos num_samples must be at least 1")
        self._pipe = pipeline

    def fingerprint(self) -> dict:
        return dict(name=self.name, model_id=self.model_id,
                    num_samples=self.num_samples, seed=self.seed,
                    torch_dtype=self.torch_dtype)

    def _load(self):
        if self._pipe is None:
            import torch
            from chronos import ChronosPipeline
            self._pipe = ChronosPipeline.from_pretrained(
                self.model_id, device_map=self.device_map,
                torch_dtype=getattr(torch, self.torch_dtype))
        return self._pipe

    def forecast(self, series: np.ndarray, horizon: int) -> np.ndarray:
        import torch
        pipe = self._load()
        x = np.asarray(series, float)
        torch.manual_seed(_series_seed(self.seed, x, horizon))
        context = torch.as_tensor(x, dtype=torch.float32)
        samples = pipe.predict(context, prediction_length=int(horizon),
                               num_samples=self.num_samples)
        return _point_forecast(samples, int(horizon))


def _point_forecast(samples, horizon: int) -> np.ndarray:
    """Validate the sampler's output shape and reduce by median."""
    values = np.asarray(getattr(samples, "numpy", lambda: samples)(), float)
    values = np.squeeze(values)
    if values.ndim == 1:
        out = values
    elif values.ndim == 2:
        out = np.median(values, axis=0)
    else:
        raise ValueError(f"unexpected forecast sample shape {values.shape}")
    if len(out) != horizon:
        raise ValueError(
            f"forecaster returned {len(out)} values, expected {horizon}")
    if not np.all(np.isfinite(out)):
        raise ValueError("forecast contains non-finite values")
    return out


def build_tsfm(cfg: dict, seed: int = 0) -> TSFM:
    cfg = dict(cfg or {})
    name = cfg.get("name", "mock_tsfm")
    if name in ("mock", "mock_tsfm"):
        return MockTSFM(name="mock_tsfm", noise=float(cfg.get("noise", 0.0)),
                        seed=seed)
    if name == "chronos":
        return ChronosTSFM(
            model_id=cfg.get("model_id", "amazon/chronos-t5-small"),
            device_map=cfg.get("device_map", "cpu"),
            torch_dtype=cfg.get("torch_dtype", "float32"),
            num_samples=int(cfg.get("num_samples", 20)), seed=seed)
    raise ValueError(f"unknown tsfm {name!r}")
