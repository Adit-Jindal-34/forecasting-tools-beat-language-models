"""Real backbone adapters: Gemini, any OpenAI-compatible server, local HF.

PARAMS ARE NO LONGER DROPPED
----------------------------
The old factory built `GeminiLLM(spec.model)` and discarded `spec.params`
entirely, so a config that said `params: {temperature: 0.0}` had no effect.
It happened to be harmless because the default matched, but a silently
ignored knob is a reproducibility trap: the config snapshot recorded a
setting the run never applied. Every adapter here takes its generation
parameters explicitly and reports them through `fingerprint()`, which feeds
the cache key.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time

from .llm import LLMClient

log = logging.getLogger("tsarch.llm")


def _gemini_error_info(exc, api_key: str) -> tuple[str, float | None]:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
        payload = json.loads(raw) if raw else {}
    except (OSError, ValueError, AttributeError):
        payload = {}
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = error.get("message", "") if isinstance(error, dict) else ""
    retry_delay = None
    diagnostics: list[str] = []

    def visit(value):
        nonlocal retry_delay
        if isinstance(value, dict):
            delay = value.get("retryDelay")
            if isinstance(delay, str):
                m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", delay)
                if m:
                    retry_delay = float(m.group(1))
            for key in ("quotaId", "quotaMetric"):
                item = value.get(key)
                if isinstance(item, str) and item not in diagnostics:
                    diagnostics.append(item)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(error)
    if diagnostics:
        message = f"{message} [{'; '.join(diagnostics)}]"
    message = str(message or exc.reason).replace(api_key, "[redacted]")
    return message[:700], retry_delay


class GeminiLLM(LLMClient):
    """Gemini adapter. GEMINI_API_KEY in env; generation params from config."""

    def __init__(self, model: str, temperature: float = 0.0,
                 max_tokens: int = 1024, request_delay_s: float | None = None,
                 max_retries: int = 8):
        self.model = model
        self.name = f"gemini-{model}"
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_retries = int(max_retries)
        self._key = os.environ.get("GEMINI_API_KEY", "")
        self._min_interval = float(
            os.environ.get("GEMINI_REQUEST_DELAY_SECONDS", "4")
            if request_delay_s is None else request_delay_s)
        if self._min_interval < 0:
            raise ValueError("request delay must be nonnegative")
        self._last_at = 0.0
        if not self._key:
            raise RuntimeError("GEMINI_API_KEY not set for GeminiLLM")

    def fingerprint(self) -> dict:
        return dict(provider="gemini", model=self.model,
                    temperature=self.temperature, max_tokens=self.max_tokens)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import http.client
        import ssl
        import urllib.error
        import urllib.request
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": self.temperature,
                                 "maxOutputTokens": self.max_tokens},
        }).encode()
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self._key}",
            data=body, headers={"Content-Type": "application/json"})
        retryable = {408, 429, 500, 502, 503, 504}
        for attempt in range(self.max_retries + 1):
            wait = self._min_interval - (time.monotonic() - self._last_at)
            if wait > 0:
                time.sleep(wait)
            self._last_at = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    payload = json.load(r)
                break
            except urllib.error.HTTPError as exc:
                detail, provider_delay = _gemini_error_info(exc, self._key)
                low = detail.lower()
                if exc.code == 429 and ("perday" in low or "per day" in low):
                    raise RuntimeError(
                        f"Gemini HTTP {exc.code}: {detail}. This looks like a "
                        "daily quota limit; wait for the reset.") from None
                if exc.code not in retryable or attempt >= self.max_retries:
                    # never echo the URL: it carries the API key
                    raise RuntimeError(
                        f"Gemini HTTP {exc.code} {exc.reason} after "
                        f"{attempt + 1} attempt(s): {detail}") from None
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = float(retry_after) if retry_after else (provider_delay or 0.0)
                except ValueError:
                    delay = 0.0
                if delay > 120:
                    raise RuntimeError(
                        f"Gemini HTTP {exc.code}: {detail}. Provider asks for "
                        f"a {delay:g}s wait; retry after that.") from None
                time.sleep(delay if delay > 0 else
                           min(2 ** attempt + random.random(), 60.0))
            except (TimeoutError, urllib.error.URLError, ConnectionError,
                    OSError, http.client.HTTPException, ssl.SSLError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Gemini request failed after {attempt + 1} "
                        f"attempt(s): {type(exc).__name__}: "
                        f"{str(exc)[:300]}") from None
                time.sleep(min(2 ** attempt + random.random(), 60.0))
        c = payload.get("candidates", [{}])[0]
        return "".join(p.get("text", "")
                       for p in c.get("content", {}).get("parts", []))


#: gpt-oss "harmony" channel markers. A server that does not split the
#: reasoning from the answer returns them inline in `content`.
_HARMONY_FINAL = "<|channel|>final<|message|>"
_HARMONY_TAIL = ("<|return|>", "<|end|>", "<|call|>")


def final_answer_text(message: dict) -> str:
    """The text to score from one chat-completion message.

    Reasoning models produce a chain of thought and a final answer. Only the
    final answer is scored; the reasoning is never read by the evaluator.

      * vLLM normally returns the reasoning separately (`reasoning_content`
        or `reasoning`) and the answer in `content`: use `content`.
      * If the raw harmony stream comes back inline, keep what follows the
        LAST final-channel marker.
      * If generation was cut off inside the reasoning, `content` is empty
        and the item is scored as unanswered - which is correct, and is why
        the configs give reasoning models a large `max_tokens`.
    """
    text = message.get("content") or ""
    if _HARMONY_FINAL in text:
        text = text.rsplit(_HARMONY_FINAL, 1)[1]
        for t in _HARMONY_TAIL:
            text = text.split(t, 1)[0]
    elif "<|channel|>" in text:
        # an analysis channel and no final one: no answer was produced
        text = ""
    return text.strip()


def _connection_base(base_url: str) -> str:
    """Where to actually CONNECT, given the configured `base_url`.

    `TSARCH_VLLM_PORT`, when set, replaces the port - and only the port - of
    a loopback `base_url`. The fingerprint keeps the CONFIGURED url, so the
    cache key does not change with the port a job happened to get.

    Why this exists: every job serves its model on 127.0.0.1, and loopback
    is shared by everyone on a node. With one fixed port, two jobs on the
    same node (two of ours, or ours and a stranger's) collide: the second
    server cannot bind, and its client then talks to the FIRST job's server
    - a different model, or the same model that disappears when the other
    job ends. The port could not simply be moved in the config, because
    `base_url` is part of the cache fingerprint and H2 is a replay of H3's
    32B cell. So the job picks a free port (slurm/c2_run.sbatch), serves on
    it, and tells the client through this variable.
    """
    port = os.environ.get("TSARCH_VLLM_PORT", "").strip()
    if not port:
        return base_url
    if not port.isdigit():
        raise ValueError(f"TSARCH_VLLM_PORT must be a port number, got "
                         f"{port!r}")
    from urllib.parse import urlsplit, urlunsplit
    u = urlsplit(base_url)
    if u.hostname not in ("127.0.0.1", "localhost", "::1"):
        # never redirect a remote endpoint
        return base_url
    host = f"[{u.hostname}]" if ":" in u.hostname else u.hostname
    return urlunsplit((u.scheme, f"{host}:{port}", u.path, u.query,
                       u.fragment)).rstrip("/")


class OpenAICompatibleLLM(LLMClient):
    """Any /v1/chat/completions server (vLLM, Ollama, LM Studio, OpenAI)."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1",
                 api_key: str = "", temperature: float = 0.0,
                 max_tokens: int = 1024, reasoning_effort: str | None = None,
                 timeout_s: float = 300.0, max_retries: int = 4,
                 retry_wait_s: float = 5.0):
        #: Execution only (never in the fingerprint): how often a request
        #: that failed for a TRANSIENT reason is sent again, and the base of
        #: the exponential wait between attempts. See `complete`.
        self.max_retries = int(max_retries)
        self.retry_wait_s = float(retry_wait_s)
        self._tl = threading.local()
        self.model = model
        self.name = f"openai-compat-{model}"
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        #: For reasoning models (gpt-oss): "low" | "medium" | "high". None
        #: for ordinary models, which then send exactly what cycle 1/2 sent.
        self.reasoning_effort = reasoning_effort
        self.timeout_s = float(timeout_s)
        self._base = base_url.rstrip("/")
        self._key = api_key or os.environ.get("OPENAI_API_KEY", "none")
        self._conn = _connection_base(self._base)

    def fingerprint(self) -> dict:
        fp = dict(provider="openai_compatible", model=self.model,
                  base_url=self._base, temperature=self.temperature,
                  max_tokens=self.max_tokens)
        # Added ONLY when set, so every existing cache key is unchanged.
        if self.reasoning_effort:
            fp["reasoning_effort"] = self.reasoning_effort
        return fp

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import urllib.request
        self._tl.meta = None
        msg = {
            "model": self.model, "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
        }
        if self.reasoning_effort:
            msg["reasoning_effort"] = self.reasoning_effort
        body = json.dumps(msg).encode()
        req = urllib.request.Request(
            f"{self._conn}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._key}"})
        payload = self._post(req)
        choice = payload["choices"][0]
        message = choice["message"]
        text = final_answer_text(message)
        # What happened to THIS call, for the cache record (src/cache.py).
        # Without it a reply cut off inside the reasoning is stored as an
        # empty string and cannot later be told from a model that answered
        # nothing. Provenance only: it never reaches a system or a scorer.
        usage = payload.get("usage") or {}
        reasoning = (message.get("reasoning_content")
                     or message.get("reasoning") or "")
        meta = dict(finish_reason=choice.get("finish_reason"))
        if usage.get("completion_tokens") is not None:
            meta["completion_tokens"] = usage["completion_tokens"]
        if reasoning:
            meta["reasoning_chars"] = len(reasoning)
        if not text:
            meta["empty_answer"] = True
        self._tl.meta = meta
        return text

    @property
    def last_call_meta(self) -> dict:
        """Server-side facts about THIS thread's most recent call."""
        return dict(getattr(self._tl, "meta", None) or {})

    def _post(self, req) -> dict:
        """Send one request, retrying TRANSIENT failures only.

        A run is ~3,000 requests, and reasoning models hold a connection
        for minutes with many requests in flight. Without a retry, one
        timeout or one 5xx ends the whole cell: nothing is mis-scored (the
        cell is simply not written and the cache resumes it), but the
        re-submission waits in the queue again.

        Retried: timeouts, dropped connections, HTTP 408/429/5xx.
        NOT retried: every other 4xx. Those are deterministic - a prompt
        over the context window, an unknown model name - and sending them
        again cannot help; failing at once keeps the cause visible.

        A retry re-asks the same question at temperature 0, so it changes
        nothing about what is measured.
        """
        import http.client
        import urllib.error
        import urllib.request
        retryable = {408, 429, 500, 502, 503, 504}
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    return json.load(r)
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", "replace")[:500]
                except Exception:                 # body already consumed
                    detail = ""
                if exc.code not in retryable or attempt >= self.max_retries:
                    raise RuntimeError(
                        f"{self.model}: HTTP {exc.code} {exc.reason} after "
                        f"{attempt + 1} attempt(s): {detail}") from None
                why = f"HTTP {exc.code}"
            except (TimeoutError, urllib.error.URLError, ConnectionError,
                    http.client.HTTPException, json.JSONDecodeError,
                    OSError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"{self.model}: request failed after "
                        f"{attempt + 1} attempt(s): {type(exc).__name__}: "
                        f"{str(exc)[:300]}") from None
                why = f"{type(exc).__name__}: {str(exc)[:120]}"
            wait = min(self.retry_wait_s * 2 ** attempt, 120.0)
            wait += random.random() * min(1.0, self.retry_wait_s)
            log.warning("%s: %s; retry %d/%d in %.0fs", self.model, why,
                        attempt + 1, self.max_retries, wait)
            time.sleep(wait)
        raise AssertionError("unreachable")       # the loop returns or raises


class TransformersLLM(LLMClient):
    """Local Hugging Face model, greedy decoding, seeded for reproducibility."""

    def __init__(self, model: str, load_in_4bit: bool = True,
                 max_new_tokens: int = 160, revision: str | None = None,
                 seed: int = 0):
        try:
            import torch
            from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                      BitsAndBytesConfig)
        except ImportError as exc:
            raise RuntimeError(
                "TransformersLLM needs torch, transformers, accelerate and "
                "bitsandbytes in the active environment.") from exc
        self.model = model
        self.name = f"transformers-{model}"
        self.max_new_tokens = int(max_new_tokens)
        self.load_in_4bit = bool(load_in_4bit)
        self.revision = revision
        self.seed = int(seed)
        self._torch = torch
        torch.manual_seed(self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        kwargs = dict(device_map="auto", dtype=torch.float16,
                      low_cpu_mem_usage=True, revision=revision)
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)
        self._model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self._model.eval()

    def fingerprint(self) -> dict:
        return dict(provider="transformers", model=self.model,
                    load_in_4bit=self.load_in_4bit,
                    max_new_tokens=self.max_new_tokens,
                    revision=self.revision or "", seed=self.seed)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        messages = ([{"role": "system", "content": system_prompt}]
                    if system_prompt else [])
        messages.append({"role": "user", "content": user_prompt})
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        n_in = inputs["input_ids"].shape[-1]
        with self._torch.inference_mode():
            out = self._model.generate(
                **inputs, do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, n_in:],
                                     skip_special_tokens=True).strip()
