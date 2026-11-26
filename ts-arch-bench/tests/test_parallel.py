"""Parallel inference and reasoning-model support (amendment A16).

The extra-model run adds two things that could silently change results:

  1. `inference.workers` > 1 answers several items at once. It must give
     EXACTLY the rows a sequential run gives.
  2. Reasoning models return a chain of thought plus an answer. Only the
     answer may be scored, whichever way the server returns it.

The end-to-end tests here start a small local HTTP server that speaks the
OpenAI chat-completions protocol the way vLLM does for a reasoning model -
answer in `content`, chain of thought in `reasoning_content` - and answers
with the deterministic mock model. The real client, cache, systems,
evaluator and parallel path then run against it unchanged.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import make_config                            # noqa: E402
from models.llm import MockLLM                              # noqa: E402
from models.providers import (OpenAICompatibleLLM,          # noqa: E402
                              final_answer_text)
from src.cache import JsonlCache                            # noqa: E402
from src.config import BackboneSpec, SystemSpec, load_config  # noqa: E402
from src.pipeline import (Registry, make_samples, run_inference,  # noqa: E402
                          evaluate_run, fixed_denominators)

L = 128
VOLATILE = ("latency_s", "trace_json")      # wall-clock, not results


# ---------------------------------------------------------------------------
# a fake vLLM serving a "reasoning model" backed by the mock
# ---------------------------------------------------------------------------
class _FakeVLLM:
    def __init__(self, model_id: str, style: str = "separate"):
        self.model_id = model_id
        self.style = style                  # "separate" | "harmony_inline"
        self.mock = MockLLM("fake", strength=0.6, seed=20261102)
        self.requests = []
        self.lock = threading.Lock()
        #: user prompt -> True when this reply should be cut off at the
        #: token limit (finish_reason "length", no content)
        self.cut_off = lambda user_prompt: False
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):      # keep pytest output clean
                pass

            def _send(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send({"data": [{"id": outer.model_id}]})

            def do_POST(self):
                n = int(self.headers["Content-Length"])
                req = json.loads(self.rfile.read(n))
                with outer.lock:
                    outer.requests.append(req)
                sys_p = req["messages"][0]["content"]
                usr_p = req["messages"][1]["content"]
                answer = outer.mock.complete(sys_p, usr_p)
                if outer.cut_off(usr_p):
                    # reasoning used the whole budget: no answer at all
                    self._send({"choices": [{
                        "message": {"role": "assistant", "content": None,
                                    "reasoning_content": "thinking " * 99},
                        "finish_reason": "length"}],
                        "usage": {"completion_tokens": req["max_tokens"]}})
                    return
                thought = "Let me look at the numbers carefully. " * 5
                if outer.style == "harmony_inline":
                    msg = {"role": "assistant", "content": (
                        "<|channel|>analysis<|message|>" + thought
                        + "<|end|><|start|>assistant<|channel|>final"
                        "<|message|>" + answer + "<|return|>")}
                else:
                    msg = {"role": "assistant", "content": answer,
                           "reasoning_content": thought}
                self._send({"choices": [{"message": msg,
                                         "finish_reason": "stop"}],
                            "usage": {"completion_tokens": 50}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever,
                         daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake_server():
    srv = _FakeVLLM("openai/fake-reasoner")
    yield srv
    srv.close()


def _cfg(tmp_path, port, workers):
    cfg = make_config(tmp_path, n_base=36, n_per=36, max_history=256)
    cfg.data.update(design="c2", event_zone=128)
    cfg.backbones = {"fake": BackboneSpec(
        "fake", "openai_compatible", "openai/fake-reasoner", 0.5,
        params=dict(base_url=f"http://127.0.0.1:{port}/v1", api_key="none",
                    temperature=0.0, max_tokens=8192,
                    reasoning_effort="medium"))}
    cfg.systems = [SystemSpec("bare_llm", "bare_llm", "fake"),
                   SystemSpec("bare_features", "bare_llm", "fake",
                              params=dict(features=True)),
                   SystemSpec("harness", "harness_tsfm", "fake"),
                   SystemSpec("harness_oracle", "harness_tsfm", "fake",
                              routing="oracle")]
    cfg.inference = dict(workers=workers)
    cfg.validate()
    return cfg


def _rows(cfg, cache_dir):
    samples = make_samples(cfg, ROOT)
    reg = Registry(cfg, cache_dir=cache_dir)
    res = run_inference(cfg, samples, L, reg.systems, reg)
    rows = evaluate_run(cfg, samples, L, res,
                        denominators=fixed_denominators(cfg, samples))
    return pd.DataFrame(rows).drop(columns=list(VOLATILE)), reg


# ---------------------------------------------------------------------------
# 1. parallel == sequential, through the real client and a real HTTP server
# ---------------------------------------------------------------------------
def test_parallel_gives_identical_rows(tmp_path, fake_server):
    seq, _ = _rows(_cfg(tmp_path / "a", fake_server.port, 1),
                   tmp_path / "a" / "cache")
    par, reg = _rows(_cfg(tmp_path / "b", fake_server.port, 8),
                     tmp_path / "b" / "cache")
    assert len(seq) == len(par) == 36 * 9 * 4
    pd.testing.assert_frame_equal(seq.reset_index(drop=True),
                                  par.reset_index(drop=True))
    # every call is in the cache exactly once, and the file agrees with it
    cache = reg.call_cache
    lines = [ln for ln in Path(cache.path).read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == len(cache) == len({json.loads(x)["key"]
                                            for x in lines})


def test_per_item_call_counts_survive_parallelism(tmp_path, fake_server):
    """n_model_calls is per ITEM; with shared counters it would be garbage."""
    par, _ = _rows(_cfg(tmp_path, fake_server.port, 8), tmp_path / "cache")
    bare = par[par.system == "bare_llm"]
    assert (bare.n_model_calls == 1).all()
    oracle_num = par[(par.system == "harness_oracle")
                     & (par.query_kind == "numeric_exact")]
    assert (oracle_num.n_model_calls == 0).all()
    # routed to a describe/explain tool: classify + answer = 2 calls
    self_qual = par[(par.system == "harness")
                    & (par.query_kind == "qualitative")
                    & par.route_op.isin(["describe", "explain"])]
    assert len(self_qual) and (self_qual.n_model_calls == 2).all()
    # routed to a numeric tool or the forecaster: classify only = 1 call
    misrouted = par[(par.system == "harness")
                    & (par.query_kind == "qualitative")
                    & par.route_op.notna()
                    & ~par.route_op.isin(["describe", "explain"])]
    assert (misrouted.n_model_calls == 1).all()


def test_a_warm_replay_bills_nothing(tmp_path, fake_server):
    cfg = _cfg(tmp_path, fake_server.port, 8)
    _rows(cfg, tmp_path / "cache")
    n_before = len(fake_server.requests)
    again, _ = _rows(cfg, tmp_path / "cache")
    assert len(fake_server.requests) == n_before
    assert (again.n_billed_calls == 0).all()


def test_reasoning_effort_is_sent_and_only_the_answer_is_scored(
        tmp_path, fake_server):
    rows, _ = _rows(_cfg(tmp_path, fake_server.port, 4), tmp_path / "cache")
    assert all(r.get("reasoning_effort") == "medium"
               for r in fake_server.requests)
    assert all(r.get("max_tokens") == 8192 for r in fake_server.requests)
    assert not rows.answer_text.str.contains("Let me look").any()
    # arms that always reach the model's answer (no routing step)
    q = rows[(rows.query_type == "trend_description")
             & rows.system.isin(["bare_llm", "bare_features", "harness_oracle"])]
    assert len(q) and (q.slot_trend_status == "one").all()


def test_inline_harmony_output_is_reduced_to_the_answer(tmp_path):
    srv = _FakeVLLM("openai/fake-reasoner", style="harmony_inline")
    try:
        rows, _ = _rows(_cfg(tmp_path, srv.port, 4), tmp_path / "cache")
    finally:
        srv.close()
    assert not rows.answer_text.str.contains(r"<\|").any()
    q = rows[(rows.query_type == "trend_description")
             & rows.system.isin(["bare_llm", "bare_features", "harness_oracle"])]
    assert len(q) and (q.slot_trend_status == "one").all()


# ---------------------------------------------------------------------------
# 2. the answer extractor on its own
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("message,expected", [
    ({"content": "trend: stable\nseasonal: no"}, "trend: stable\nseasonal: no"),
    ({"content": "  42 ", "reasoning_content": "think"}, "42"),
    ({"content": None, "reasoning_content": "ran out of tokens"}, ""),
    ({"content": "<|channel|>analysis<|message|>hmm<|end|><|start|>assistant"
                 "<|channel|>final<|message|>event: spike\nposition: late"
                 "<|return|>"}, "event: spike\nposition: late"),
    # analysis only, never reached a final answer
    ({"content": "<|channel|>analysis<|message|>still thinking"}, ""),
])
def test_final_answer_text(message, expected):
    assert final_answer_text(message) == expected


# ---------------------------------------------------------------------------
# 3. no existing cache key moves
# ---------------------------------------------------------------------------
def test_fingerprint_unchanged_for_ordinary_models():
    """Every Qwen call already cached must still be found."""
    fp = OpenAICompatibleLLM("Qwen/Qwen2.5-32B-Instruct",
                             base_url="http://127.0.0.1:8000/v1",
                             max_tokens=1024).fingerprint()
    assert fp == dict(provider="openai_compatible",
                      model="Qwen/Qwen2.5-32B-Instruct",
                      base_url="http://127.0.0.1:8000/v1",
                      temperature=0.0, max_tokens=1024)
    r = OpenAICompatibleLLM("openai/gpt-oss-20b", reasoning_effort="high")
    assert r.fingerprint()["reasoning_effort"] == "high"


def test_cache_is_safe_under_concurrent_writes(tmp_path):
    cache = JsonlCache(tmp_path / "c.jsonl")
    keys = [f"k{i}" for i in range(400)]

    def write(chunk):
        for k in chunk:
            cache.put(k, k.upper())
            cache.put(k, "duplicate ignored")

    ts = [threading.Thread(target=write, args=(keys[i::8],))
          for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    lines = (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 400
    assert {json.loads(x)["key"] for x in lines} == set(keys)
    reloaded = JsonlCache(tmp_path / "c.jsonl")
    assert all(reloaded.get(k) == k.upper() for k in keys)


# ---------------------------------------------------------------------------
# 4. the extra-model configs
# ---------------------------------------------------------------------------
EXTRA = sorted(p.stem for p in (ROOT / "configs").glob("c2_h2x_*.yaml"))


def test_there_are_extra_model_configs():
    assert len(EXTRA) >= 5


@pytest.mark.parametrize("name", EXTRA)
def test_extra_config_is_h2_with_another_model(name):
    import yaml
    base = load_config(ROOT / "configs" / "c2_h2.yaml")
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    # identical items, arms, scoring and analysis
    assert cfg.data_fingerprint() == base.data_fingerprint()
    assert cfg.seed == base.seed
    assert [s.name for s in cfg.systems] == [s.name for s in base.systems]
    assert [s.routing for s in cfg.systems] == \
        [s.routing for s in base.systems]
    assert cfg.scoring == base.scoring
    for k in ("confirmatory_family", "confirmatory_contrast", "n_boot",
              "min_effect", "alpha", "extra_contrasts"):
        assert cfg.analysis[k] == base.analysis[k], k
    assert cfg.tsfm == base.tsfm
    # a real, served model on its own output directory
    (bb, spec), = cfg.backbones.items()
    assert spec.provider == "openai_compatible"
    assert spec.params["temperature"] == 0.0
    assert spec.params["base_url"] == "http://127.0.0.1:8000/v1"
    assert cfg.out_dir == f"results/c2_h2x_{bb}"
    assert all(s.backbone == bb for s in cfg.systems)
    serve = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(
        encoding="utf-8"))["backbones"][bb]["serve"]
    assert int(serve["max_model_len"]) >= 4096
    # reasoning models get room to finish their reasoning
    if spec.params.get("reasoning_effort"):
        assert spec.params["max_tokens"] >= 8192
        assert int(serve["max_model_len"]) > spec.params["max_tokens"] + 2000


# ---------------------------------------------------------------------------
# 5. the pre-run probe (scripts/probe_server.py)
# ---------------------------------------------------------------------------
def _probe(tmp_path, port, monkeypatch):
    import yaml
    raw = yaml.safe_load((ROOT / "configs" / "c2_h2x_gpt_oss_20b.yaml")
                         .read_text(encoding="utf-8"))
    (bb, spec), = raw["backbones"].items()
    spec["model"] = "openai/fake-reasoner"
    raw["data"]["pool_dir"] = str(tmp_path / "pool")
    raw["data"]["n_base_series"] = raw["data"]["n_per_query_type"] = 36
    raw["data"]["max_history"] = 256
    p = tmp_path / "probe.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("TSARCH_VLLM_PORT", str(port))
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    import probe_server
    importlib.reload(probe_server)
    monkeypatch.setattr(sys, "argv", ["probe", "--config", str(p)])
    return probe_server.main()


def test_probe_passes_a_well_behaved_reasoning_server(tmp_path, fake_server,
                                                      monkeypatch):
    assert _probe(tmp_path, fake_server.port, monkeypatch) == 0


def test_probe_allows_one_reply_cut_off_at_the_token_limit(
        tmp_path, monkeypatch, capsys):
    """A19. One long forecasting question running out of tokens is an
    outcome the run scores (unanswered), not a format failure. This is the
    case that stopped gpt-oss-120b at high effort on 2026-11-14."""
    srv = _FakeVLLM("openai/fake-reasoner")
    srv.cut_off = lambda u: "next 8 values" in u          # forecasting only
    try:
        assert _probe(tmp_path, srv.port, monkeypatch) == 0
    finally:
        srv.close()
    out = capsys.readouterr().out
    assert "WARNING forecasting: cut off at the token limit" in out
    assert "1 of 3 replies cut off" in out


def test_probe_fails_when_every_reply_is_cut_off(tmp_path, monkeypatch,
                                                 capsys):
    srv = _FakeVLLM("openai/fake-reasoner")
    srv.cut_off = lambda u: True
    try:
        assert _probe(tmp_path, srv.port, monkeypatch) == 1
    finally:
        srv.close()
    assert "every probed reply was cut off" in capsys.readouterr().out


def test_probe_fails_when_the_answer_never_arrives(tmp_path, monkeypatch):
    """An empty answer that was NOT cut off (finish_reason "stop") is a
    format problem: the answer is in a field we do not read. Must stop."""
    srv = _FakeVLLM("openai/fake-reasoner")
    srv.mock.complete = lambda s, u: ""          # every answer empty
    try:
        assert _probe(tmp_path, srv.port, monkeypatch) == 1
    finally:
        srv.close()


# ---------------------------------------------------------------------------
# 6. transient server errors are retried; truncation is recorded
# ---------------------------------------------------------------------------
class _ScriptedServer:
    """Answers each POST with the next (status, body) from a script; the
    last entry repeats."""

    def __init__(self, script):
        self.script = list(script)
        self.posts = 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                i = min(outer.posts, len(outer.script) - 1)
                outer.posts += 1
                status, body = outer.script[i]
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever,
                         daemon=True).start()

    def client(self, **kw):
        return OpenAICompatibleLLM(
            "m", base_url=f"http://127.0.0.1:{self.port}/v1",
            retry_wait_s=0.0, **kw)

    def close(self):
        self.server.shutdown()


def _ok(content, finish="stop", **extra):
    return 200, {"choices": [{"message": dict(role="assistant",
                                               content=content, **extra),
                              "finish_reason": finish}],
                 "usage": {"completion_tokens": 7}}


def test_a_transient_server_error_is_retried():
    srv = _ScriptedServer([(500, {"error": "boom"}), (503, {"error": "busy"}),
                           _ok("trend: stable")])
    try:
        assert srv.client().complete("s", "u") == "trend: stable"
        assert srv.posts == 3
    finally:
        srv.close()


def test_retries_are_bounded():
    srv = _ScriptedServer([(500, {"error": "boom"})])
    try:
        with pytest.raises(RuntimeError, match="HTTP 500.*3 attempt"):
            srv.client(max_retries=2).complete("s", "u")
        assert srv.posts == 3
    finally:
        srv.close()


def test_a_client_error_is_not_retried():
    """A 400 (prompt over the context window, say) is deterministic."""
    srv = _ScriptedServer([(400, {"error": "context length exceeded"}),
                           _ok("never reached")])
    try:
        with pytest.raises(RuntimeError, match="HTTP 400.*context length"):
            srv.client().complete("s", "u")
        assert srv.posts == 1
    finally:
        srv.close()


def test_a_dead_server_fails_after_the_retries():
    client = OpenAICompatibleLLM("m", base_url="http://127.0.0.1:9/v1",
                                 max_retries=1, retry_wait_s=0.0,
                                 timeout_s=2)
    with pytest.raises(RuntimeError, match="2 attempt"):
        client.complete("s", "u")


def test_retry_settings_are_not_in_the_fingerprint():
    a = OpenAICompatibleLLM("m", base_url="http://127.0.0.1:8000/v1")
    b = OpenAICompatibleLLM("m", base_url="http://127.0.0.1:8000/v1",
                            max_retries=0, retry_wait_s=99.0, timeout_s=1)
    assert a.fingerprint() == b.fingerprint()


def test_a_truncated_reply_is_marked_in_the_cache(tmp_path):
    """Cut off inside the reasoning: scored text is empty, and the cache
    record says WHY, so it can be told from a model that said nothing."""
    from src.cache import CachedLLM
    srv = _ScriptedServer([
        _ok(None, finish="length", reasoning_content="thinking " * 20),
        _ok("trend: stable", reasoning_content="short")])
    try:
        cache = JsonlCache(tmp_path / "calls.jsonl")
        llm = CachedLLM(srv.client(), cache, "pool", backbone="bb")
        assert llm.complete("s", "first") == ""
        assert llm.complete("s", "second") == "trend: stable"
        assert llm.truncated == 1
        # a replay serves the stored empty answer and does not re-ask
        assert llm.complete("s", "first") == ""
        assert srv.posts == 2
    finally:
        srv.close()
    recs = [json.loads(x) for x in
            (tmp_path / "calls.jsonl").read_text("utf-8").splitlines()]
    cut, fine = recs
    assert cut["value"] == ""
    assert cut["meta"]["call"]["finish_reason"] == "length"
    assert cut["meta"]["call"]["empty_answer"] is True
    assert cut["meta"]["call"]["reasoning_chars"] > 0
    assert fine["meta"]["call"] == dict(finish_reason="stop",
                                        completion_tokens=7,
                                        reasoning_chars=5)
    # provenance never enters the key
    assert set(cut) == {"key", "value", "meta"}
