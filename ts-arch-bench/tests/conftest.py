from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (Config, QueryTypeSpec, BackboneSpec,  # noqa: E402
                        SystemSpec)

ALL_TYPES = [
    ("forecasting", "forecast"), ("statistics", "numeric_exact"),
    ("counting", "numeric_exact"), ("extremum", "numeric_exact"),
    ("change", "numeric_exact"), ("correlation", "numeric_exact"),
    ("localization", "localization"),
    ("trend_description", "qualitative"),
    ("event_explanation", "qualitative"),
]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


def make_config(tmp_path, n_base=14, n_per=14, max_history=256,
                prose_rate=0.0, seed=20261102, oracle=True) -> Config:
    cfg = Config(seed=seed, out_dir=str(tmp_path / "out"))
    cfg.data.update(pool_dir=str(tmp_path / "pool"), max_history=max_history,
                    horizon=8, n_base_series=n_base, n_per_query_type=n_per)
    cfg.history_lengths = [128]
    cfg.query_types = [QueryTypeSpec(n, k) for n, k in ALL_TYPES]
    cfg.backbones = {"mock_mid": BackboneSpec("mock_mid", "mock", "mock-llm",
                                              0.6)}
    cfg.systems = [
        SystemSpec("bare_llm", "bare_llm", "mock_mid"),
        SystemSpec("harness", "harness_tsfm", "mock_mid", routing="self"),
    ]
    if oracle:
        cfg.systems.append(
            SystemSpec("harness_oracle", "harness_tsfm", "mock_mid",
                       routing="oracle"))
    cfg.mocks.update(prose_rate=prose_rate)
    cfg.analysis.update(min_n_per_cell=5, n_boot=200)
    cfg.validate()
    return cfg


@pytest.fixture
def cfg(tmp_path) -> Config:
    return make_config(tmp_path)


@pytest.fixture
def samples(cfg):
    from src.pipeline import make_samples
    return make_samples(cfg, ROOT)


def toy_window(n=128, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 1, n)) + 50.0
