"""Optional LLM judge for the qualitative cells.

READ THIS BEFORE ENABLING IT
----------------------------
The old build's judge was handed the reference keyphrases and asked how well
the answer covered them - i.e. it graded the SAME construct as the automatic
metric, only fuzzily. Turning it on could not have revealed that the
automatic metric was blind, because it shared the metric's definition of
"right". It is not an independent validation.

Neither is the version here. `mode="reference"` renders the slot truth and
asks for agreement; it is a softer reader of the same rubric, useful for
catching phrasing the slot regexes miss, and nothing more. `mode="blind"`
withholds the reference entirely and asks only whether the description is
supported by the series, which is independent but much noisier.

THE ONLY INDEPENDENT VALIDATION IS HUMAN LABELS. See PREREGISTRATION.md:
50-100 stratified items, three annotators, Krippendorff's alpha reported,
and the correlation between slot accuracy and the human labels reported
whether or not it is flattering.

Judge failures return NaN and are COUNTED. `judge_score` is never folded
into the primary score.
"""
from __future__ import annotations

import json
import re

from models.llm import LLMClient, JUDGE_PROMPT
from src.ground_truth import QualitativeTruth


def render_reference(truth: QualitativeTruth) -> str:
    bits = [f"trend is {truth.trend}",
            f"series is {'seasonal' if truth.seasonal == 'yes' else 'not seasonal'}"]
    if truth.event is not None:
        bits.append(f"main event is {truth.event}")
    if truth.position is not None:
        bits.append(f"event occurs {truth.position} in the window")
    return "; ".join(bits)


class QualJudge:
    def __init__(self, llm: LLMClient, mode: str = "reference"):
        if mode not in ("reference", "blind"):
            raise ValueError("judge mode must be 'reference' or 'blind'")
        self.llm = llm
        self.mode = mode
        self.name = getattr(llm, "name", "judge")
        self.n_calls = 0
        self.n_failed = 0

    def judge(self, question: str, truth: QualitativeTruth,
              answer: str) -> float:
        refs = (render_reference(truth) if self.mode == "reference"
                else "(withheld: rate only whether the claim is plausible "
                     "for a time series)")
        prompt = JUDGE_PROMPT.format(question=question, refs=refs,
                                     answer=(answer or "")[:800])
        self.n_calls += 1
        try:
            raw = self.llm.complete(
                "You are a strict evaluator. Output JSON only.", prompt)
        except Exception:
            self.n_failed += 1
            return float("nan")
        m = re.search(r"\{.*\}", raw or "", re.S)
        if not m:
            self.n_failed += 1
            return float("nan")
        try:
            return float(json.loads(m.group(0)).get("score", float("nan")))
        except (json.JSONDecodeError, TypeError, ValueError):
            self.n_failed += 1
            return float("nan")

    def stats(self) -> dict:
        return dict(judge=self.name, mode=self.mode, calls=self.n_calls,
                    failed=self.n_failed)
