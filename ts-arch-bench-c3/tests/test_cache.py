"""Cache keying: the stale-data bug, backbone independence, free replay."""
from __future__ import annotations

import numpy as np
import pytest

from src.cache import JsonlCache, CachedLLM, call_key, tsfm_key
from conftest import make_config


class CountingLLM:
    name, model = "counting", "counting-1"

    def __init__(self, reply="mean"):
        self.calls = 0
        self.reply = reply

    def fingerprint(self):
        return dict(provider="test", model=self.model)

    def complete(self, sp, up):
        self.calls += 1
        return self.reply


def test_seed_is_in_the_data_fingerprint(tmp_path):
    """THE stale-data bug. Sample ids are positional, so a reseeded config
    produced the same ids over different series while the old fingerprint
    was unchanged - and the run served cached answers for data that no
    longer existed."""
    a = make_config(tmp_path, seed=1)
    b = make_config(tmp_path, seed=2)
    assert a.data_fingerprint() != b.data_fingerprint()


def test_data_fingerprint_ignores_pool_location(tmp_path):
    a = make_config(tmp_path, seed=1)
    b = make_config(tmp_path, seed=1)
    b.data["pool_dir"] = str(tmp_path / "somewhere_else")
    assert a.data_fingerprint() == b.data_fingerprint()


def test_backbones_are_independent(tmp_path):
    """Adding a backbone must not invalidate the others. The old global
    fingerprint hashed the whole backbones dict, so H3 groupA, groupB and a
    unified config shared no cache at all."""
    from src.config import BackboneSpec
    cfg = make_config(tmp_path)
    before = cfg.backbone_fingerprint("mock_mid")
    cfg.backbones["another"] = BackboneSpec("another", "mock", "mock-llm", 0.9)
    assert cfg.backbone_fingerprint("mock_mid") == before


def test_harness_and_scoring_settings_do_not_touch_any_key(tmp_path):
    """Post-inference knobs must never invalidate a paid-for model call."""
    cfg = make_config(tmp_path)
    data_before = cfg.data_fingerprint()
    bb_before = cfg.backbone_fingerprint("mock_mid")
    cfg.harness["numeric_final"] = "llm"
    cfg.scoring["tol"] = 0.5
    cfg.analysis["correction"] = "bh"
    assert cfg.data_fingerprint() == data_before
    assert cfg.backbone_fingerprint("mock_mid") == bb_before


def test_call_cache_prevents_a_second_call(tmp_path):
    cache = JsonlCache(tmp_path / "calls.jsonl")
    inner = CountingLLM()
    llm = CachedLLM(inner, cache, "fp", "bb")
    assert llm.complete("s", "u") == "mean"
    assert llm.complete("s", "u") == "mean"
    assert inner.calls == 1, "identical prompt was paid for twice"


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "calls.jsonl"
    inner = CountingLLM()
    CachedLLM(inner, JsonlCache(path), "fp", "bb").complete("s", "u")
    inner2 = CountingLLM()
    assert CachedLLM(inner2, JsonlCache(path), "fp", "bb").complete("s", "u")
    assert inner2.calls == 0, "a warm cache still issued a call"


def test_different_prompts_do_not_collide(tmp_path):
    cache = JsonlCache(tmp_path / "calls.jsonl")
    inner = CountingLLM()
    llm = CachedLLM(inner, cache, "fp", "bb")
    llm.complete("s", "u1")
    llm.complete("s", "u2")
    llm.complete("s2", "u1")
    assert inner.calls == 3


def test_prompt_version_invalidates_calls(tmp_path, monkeypatch):
    fp = dict(provider="test", model="m")
    k1 = call_key(fp, "s", "u")
    import src
    monkeypatch.setattr(src, "PROMPT_VERSION", src.PROMPT_VERSION + 1)
    assert call_key(fp, "s", "u") != k1


def test_generation_params_change_the_key():
    """`temperature` was silently dropped by the old factory; it must now
    both apply and participate in the key."""
    a = dict(provider="gemini", model="m", temperature=0.0)
    b = dict(provider="gemini", model="m", temperature=0.7)
    assert call_key(a, "s", "u") != call_key(b, "s", "u")


def test_forecast_cache_keyed_on_window_bytes():
    fp = dict(name="chronos", num_samples=20)
    x = np.arange(10.0)
    assert tsfm_key(fp, x, 8) == tsfm_key(fp, x.copy(), 8)
    assert tsfm_key(fp, x, 8) != tsfm_key(fp, x + 1, 8)
    assert tsfm_key(fp, x, 8) != tsfm_key(fp, x, 4)
    assert tsfm_key(fp, x, 8) != tsfm_key(
        dict(name="chronos", num_samples=10), x, 8)


def test_replaying_after_a_parser_change_costs_nothing(tmp_path):
    """The whole point of caching CALLS rather than RESULTS: a routing or
    scoring change replays at zero API cost."""
    from models import MockTSFM
    from systems.harness_tsfm import HarnessTSFMSystem
    from src.samples import Sample, build_query
    from conftest import toy_window

    cache = JsonlCache(tmp_path / "calls.jsonl")
    inner = CountingLLM(reply="mean")
    view = Sample("sid", toy_window(136), 128, 8, "statistics",
                  build_query("statistics", 8), "numeric_exact", "t", 1
                  ).view(128)

    llm = CachedLLM(inner, cache, "fp", "bb")
    HarnessTSFMSystem("h", llm, MockTSFM(), "bb").answer(view)
    assert inner.calls == 1

    # a fresh system object, a fresh cache reader: the replay is free
    inner2 = CountingLLM(reply="mean")
    llm2 = CachedLLM(inner2, JsonlCache(tmp_path / "calls.jsonl"), "fp", "bb")
    res = HarnessTSFMSystem("h", llm2, MockTSFM(), "bb").answer(view)
    assert inner2.calls == 0
    assert res.extra["route_op"] == "mean"


def test_port_override_changes_the_connection_not_the_cache_key(monkeypatch):
    """A job may serve on any free port without invalidating the cache.

    `base_url` is in the fingerprint, so H2 can replay H3's 32B cell only if
    both render the same url. Jobs sharing a node cannot share a port, so
    the port actually used comes from TSARCH_VLLM_PORT and must not leak
    into the key.
    """
    from models.providers import OpenAICompatibleLLM, _connection_base
    url = "http://127.0.0.1:8000/v1"
    monkeypatch.delenv("TSARCH_VLLM_PORT", raising=False)
    plain = OpenAICompatibleLLM("m", base_url=url)
    assert plain._conn == url
    monkeypatch.setenv("TSARCH_VLLM_PORT", "8137")
    moved = OpenAICompatibleLLM("m", base_url=url)
    assert moved._conn == "http://127.0.0.1:8137/v1"
    assert moved.fingerprint() == plain.fingerprint()
    # a remote endpoint is never redirected
    assert _connection_base("https://api.example.com/v1") == \
        "https://api.example.com/v1"
    monkeypatch.setenv("TSARCH_VLLM_PORT", "not-a-port")
    import pytest
    with pytest.raises(ValueError):
        _connection_base(url)
