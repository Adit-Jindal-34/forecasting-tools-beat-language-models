"""Download the extra H2 models (amendment A16) into $HF_HOME, once.

Run on the LOGIN node (compute jobs then read only from the local cache):

    source /scratch/$USER/claude-ts-arch-bench/env.sh
    nohup python scripts/stage_extra_models.py > logs/stage_extra.log 2>&1 &

Each repo ships files the run never reads; skipping them saves ~200 GB:
  * gpt-oss: `original/` (a second copy of the weights) and `metal/`
    (Apple-silicon build)
  * Mistral-Small-3.2: two full copies of the weights, the Hugging Face
    shards and Mistral's own `consolidated.safetensors`. vLLM is run in
    Mistral's own format (`--tokenizer-mode mistral ...`, as the model card
    recommends), so only `consolidated.safetensors`, `params.json` and
    `tekken.json` are fetched.
Re-running is safe: finished files are not downloaded again.
"""
from __future__ import annotations

import sys
import time

from huggingface_hub import snapshot_download

MODELS: list[tuple[str, dict]] = [
    ("openai/gpt-oss-20b", dict(ignore_patterns=["original/*", "metal/*"])),
    ("ibm-granite/granite-3.3-8b-instruct", {}),
    ("microsoft/phi-4", {}),
    ("mistralai/Mistral-Small-3.2-24B-Instruct-2506",
     dict(allow_patterns=["consolidated.safetensors", "params.json",
                          "tekken.json", "*.md", "*.txt",
                          "generation_config.json"])),
    ("allenai/OLMo-2-0325-32B-Instruct", {}),
    ("openai/gpt-oss-120b", dict(ignore_patterns=["original/*", "metal/*"])),
]

failed = []
for repo, kw in MODELS:
    t0 = time.time()
    print(f"=== {repo}", flush=True)
    try:
        path = snapshot_download(repo, max_workers=8, **kw)
        print(f"    ok in {time.time() - t0:.0f}s -> {path}", flush=True)
    except Exception as exc:                      # keep going; report at end
        print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
        failed.append(repo)

print("=== done; failed:", failed or "none", flush=True)
sys.exit(1 if failed else 0)
