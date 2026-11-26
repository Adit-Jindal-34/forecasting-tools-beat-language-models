"""Check a freshly served model BEFORE spending GPU time on a full run.

    python scripts/probe_server.py --config configs/c2_h2x_gpt_oss_20b.yaml

Sends three real benchmark prompts through exactly the client the run will
use (same params, same port override) and fails unless:

  * the server answers and serves the configured model;
  * the scored text (`final_answer_text`) is non-empty and contains no
    reasoning-channel markup;
  * not EVERY reply was cut off at the token limit (see below);
  * a qualitative reply is read by the multiple-choice parser on BOTH asked
    slots;
  * a forecasting reply contains at least the eight requested numbers.

The raw response (including any separate reasoning field) is printed for
the log, so a format problem is visible without re-running anything.
Nothing here is cached: probe replies never enter the run's cache.

Why it exists: a reasoning model whose answer comes back in an unexpected
field would score 0 on every item, and that would only show up after the
whole run. This costs seconds.

A CUT-OFF REPLY IS NOT A FORMAT PROBLEM (amendment A19)
-------------------------------------------------------
A reasoning model can spend its whole `max_tokens` thinking and never reach
the answer (`finish_reason == "length"`, empty content). The run's declared
rule for that is "scored as unanswered and counted in the coverage column"
(A16) - it is an outcome to be measured, not a reason to refuse the model.
The first version of this probe treated it as a failure, so ONE long
forecasting question stopped gpt-oss-120b at high effort from running at
all, while the medium-effort run, where 6% of replies were cut off the same
way, went ahead.

So: a cut-off reply is printed as a WARNING. The probe still fails if every
probed reply was cut off, because then nothing shows that an answer can be
read at all and the run would very likely be hours of unanswered items.
An empty reply that was NOT cut off is still a format failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.qualitative import extract_slot_mc        # noqa: E402
from models import build_llm                              # noqa: E402
from models import llm as LM                              # noqa: E402
from models.llm import extract_numbers                    # noqa: E402
from models.providers import _connection_base, final_answer_text  # noqa: E402
from src.config import load_config                        # noqa: E402
from src.pipeline import make_samples                     # noqa: E402
from systems.bare_llm import SYSTEM_PROMPT, fmt_series    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    name = next(iter(cfg.backbones))
    spec = cfg.backbones[name]
    client = build_llm(name, spec, cfg.mocks, cfg.seed)
    base = _connection_base(spec.params.get("base_url", ""))

    with urllib.request.urlopen(f"{base}/models", timeout=30) as r:
        served = [m["id"] for m in json.load(r)["data"]]
    print(f"[probe] served: {served}")
    if spec.model not in served:
        print(f"[probe] FAIL: {spec.model} is not served here")
        return 1

    samples = make_samples(cfg, ROOT)
    pick = {}
    for s in samples:
        pick.setdefault(s.query_type, s)
    problems = []
    cut_off = []
    probed = ("trend_description", "event_explanation", "forecasting")
    for qt in ("trend_description", "event_explanation", "forecasting"):
        v = pick[qt].view(128)
        prompt = LM.GENERIC_PROMPT_C2.format(
            L=v.L, evidence_block="", series=fmt_series(v.window),
            question=v.query)
        body = {"model": spec.model, "temperature": client.temperature,
                "max_tokens": client.max_tokens,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": prompt}]}
        if client.reasoning_effort:
            body["reasoning_effort"] = client.reasoning_effort
        req = urllib.request.Request(
            f"{base}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer none"})
        with urllib.request.urlopen(req, timeout=client.timeout_s) as r:
            payload = json.load(r)
        choice = payload["choices"][0]
        msg = choice["message"]
        text = final_answer_text(msg)
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        usage = payload.get("usage", {})
        print(f"\n[probe] --- {qt}")
        print(f"[probe] finish_reason={choice.get('finish_reason')} "
              f"usage={usage}")
        print(f"[probe] reasoning field: {len(reasoning)} chars")
        print(f"[probe] raw content: {(msg.get('content') or '')[:400]!r}")
        print(f"[probe] scored text: {text[:300]!r}")
        if not text and choice.get("finish_reason") == "length":
            used = usage.get("completion_tokens", "?")
            cut_off.append(qt)
            print(f"[probe] WARNING {qt}: cut off at the token limit "
                  f"({used} of max_tokens={client.max_tokens} used, no "
                  "answer). Scored as unanswered in the run.")
            continue
        if not text:
            problems.append(f"{qt}: empty answer "
                            f"(finish_reason={choice.get('finish_reason')})")
            continue
        if "<|" in text:
            problems.append(f"{qt}: reasoning markup left in the answer")
        if qt == "forecasting":
            n = len(extract_numbers(text))
            if n < v.horizon:
                problems.append(f"{qt}: {n} numbers, need {v.horizon}")
        else:
            slots = (("trend", "seasonal") if qt == "trend_description"
                     else ("event", "position"))
            for slot in slots:
                p = extract_slot_mc(text, slot)
                print(f"[probe] slot {slot}: {p.predicted} [{p.status}]")
            # the first slot must always be read; `position: none` is a
            # legitimate absent answer for the second
            if extract_slot_mc(text, slots[0]).status != "one":
                problems.append(f"{qt}: slot {slots[0]!r} not read")

    if len(cut_off) == len(probed):
        problems.append(
            f"every probed reply was cut off at max_tokens="
            f"{client.max_tokens} before an answer; raise max_tokens (and "
            "serve.max_model_len) in the config under a dated amendment")
    if problems:
        print("\n[probe] FAIL:\n  " + "\n  ".join(problems))
        return 1
    if cut_off:
        print(f"\n[probe] OK, with {len(cut_off)} of {len(probed)} replies "
              f"cut off ({', '.join(cut_off)}): the format is readable; "
              "expect unanswered items and watch the coverage column")
    else:
        print("\n[probe] OK: replies are in the expected format")
    return 0


if __name__ == "__main__":
    sys.exit(main())
