"""Model factories."""
from __future__ import annotations

from .llm import LLMClient, MockLLM
from .tsfm import TSFM, MockTSFM, ChronosTSFM, build_tsfm

__all__ = ["LLMClient", "MockLLM", "TSFM", "MockTSFM", "ChronosTSFM",
           "build_tsfm", "build_llm"]


def build_llm(name: str, spec, mocks: dict, seed: int) -> LLMClient:
    """Config backbone spec -> client. Generation params are PASSED THROUGH.

    The old factory called `GeminiLLM(spec.model)` and dropped `spec.params`,
    so configured temperature and token limits were silently ignored.
    """
    p = dict(spec.params or {})
    if spec.provider == "mock":
        return MockLLM(name, strength=spec.strength,
                       error_scale=float(mocks.get("error_scale", 0.25)),
                       prose_rate=float(mocks.get("prose_rate", 0.0)),
                       length_effect=float(mocks.get("length_effect", 0.0)),
                       # 1.0 reproduces the historical per-arm success rates;
                       # 0.0 makes the mock arm-blind for the null fixture.
                       assist_effect=float(mocks.get("assist_effect", 1.0)),
                       seed=seed)
    if spec.provider == "gemini":
        from .providers import GeminiLLM
        return GeminiLLM(spec.model,
                         temperature=float(p.get("temperature", 0.0)),
                         max_tokens=int(p.get("max_tokens", 1024)),
                         request_delay_s=p.get("request_delay_s"),
                         max_retries=int(p.get("max_retries", 8)))
    if spec.provider == "openai_compatible":
        from .providers import OpenAICompatibleLLM
        return OpenAICompatibleLLM(
            spec.model, base_url=p.get("base_url",
                                       "http://localhost:11434/v1"),
            api_key=p.get("api_key", ""),
            temperature=float(p.get("temperature", 0.0)),
            max_tokens=int(p.get("max_tokens", 1024)),
            reasoning_effort=p.get("reasoning_effort"),
            timeout_s=float(p.get("timeout_s", 300.0)),
            # execution only; not in the fingerprint or any cache key
            max_retries=int(p.get("max_retries", 4)),
            retry_wait_s=float(p.get("retry_wait_s", 5.0)))
    if spec.provider == "transformers":
        from .providers import TransformersLLM
        return TransformersLLM(
            spec.model, load_in_4bit=bool(p.get("load_in_4bit", True)),
            max_new_tokens=int(p.get("max_new_tokens", 160)),
            revision=p.get("revision"), seed=seed)
    raise ValueError(f"unknown provider {spec.provider!r}")
