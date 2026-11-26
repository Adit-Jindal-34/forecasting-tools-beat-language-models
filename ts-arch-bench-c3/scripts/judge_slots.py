"""Validate the slot EXTRACTOR against an independent reader (Gemini).

    # see the cost and the exact prompt without spending a call
    python scripts/judge_slots.py --results results/h2_query_type --dry-run

    # then, with GEMINI_API_KEY set
    python scripts/judge_slots.py --results results/h2_query_type \
        --n-per-stratum 50 --out results/judge_slots

WHAT QUESTION THIS ANSWERS
--------------------------
Two of the three confirmatory cells (`trend_description`, `event_explanation`)
are scored by `evaluation/qualitative.py`, a REGEX slot extractor. PREREG
§4.1 calls that measure "provisionally measured" and D7 fixes the
consequence in advance: if inter-annotator alpha on a shared slot falls
below 0.667, that cell leaves the confirmatory family and **m drops from 3
to 1**. So the scorer's validity is load-bearing for the headline claim.

This script runs D7's EXTRACTION task with a strong independent model in
place of a human annotator: given only the answer text, which class does
each slot take? Agreement with the regex extractor is reported as
Krippendorff's alpha per slot, using the same `evaluation.human_validation`
code the human protocol uses, so the two are directly comparable.

WHAT THIS IS NOT
----------------
**Not a substitute for the human annotation in D7.** An LLM judge and a
regex extractor can share biases - both may read "ticking upward" as
`increasing` for the same wrong reason - so HIGH agreement is only weak
positive evidence. LOW agreement is the informative outcome: it means the
scorer's readings are idiosyncratic, and it would be found here for ~300
calls instead of after paying three annotators.

**Not a metric.** Nothing here re-scores any item, and no verdict is
derived. The pre-registered primary stays `n_correct / n_slots`.

CHOOSING THE JUDGE
------------------
`--provider openai_compatible` (the LOCAL 72B on vLLM) is the default
recommendation, and the reasoning changed during development:

  * The first argument against it was that the 72B is "a system under test".
    That is too strong. H2's answers were generated ENTIRELY by the 32B, so
    the 72B reading them is a different model judging another model's output,
    not self-judging. It does share a training family with the 32B, which
    makes it a weaker independence claim than a closed model - state that as
    a limitation rather than implying full independence.
  * It is free, unlimited, and pinnable to a HF commit hash.
  * Gemini's free tier allowed ~2 calls before silent rate-limited backoff
    when measured on 2026-11-07, which makes a 300-call run impractical.
  * A Gemini version cannot be pinned at all: `gemini-2.5-flash` was retired
    for new users mid-session, which is exactly the reproducibility problem a
    benchmark cannot absorb in a metric.

So: use the local 72B for the real run and report the shared-family caveat.
Use `--provider gemini` only for a small cross-family spot check, where the
one-off, non-reproducible nature is acceptable because it is reported as a
spot check rather than as a metric.

Neither discharges D7's HUMAN annotation.

BLINDING
--------
The judge sees `answer_text` ONLY. Not the ground truth, not the regex
extractor's reading, not the score, and not which arm produced the text -
`answer_text` is the model's reply, so the harness's `computed features:`
block is not in it. Both arms are sampled equally, or the comparison would
confound arm with text style.

COST
----
One call per sampled item, all four slots in that one call. The default
sample is `--n-per-stratum` items per (cell x arm), so 2 cells x 3 arms x 50
= 300 calls. Free-tier limits vary and Google publishes no guaranteed
table - check https://aistudio.google.com/rate-limit for yours. Replies are
cached in the OUTPUT directory, keyed on the prompt, so a re-run costs
nothing. The cache is deliberately NOT the experiment's `cache/calls.jsonl`:
a validation script has no business writing into the archived call record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd                                       # noqa: E402

from evaluation.human_validation import krippendorff_alpha  # noqa: E402
from src.ground_truth import QUALITATIVE_SLOTS, SLOTS_FOR_TYPE  # noqa: E402

SLOT_SCORING = "slot_accuracy"

#: codings outside the class vocabulary. These mirror the extractor's own
#: `status` values so the two coders share one label space: a disagreement
#: between "absent" and a class is a real disagreement, not a type error.
UNCLEAR = "unclear"
ABSENT = "absent"

SYSTEM_PROMPT = (
    "You read a short description of a time series and report which "
    "category each requested attribute falls into, using ONLY the words of "
    "the description. You never guess from plausibility and never use "
    "outside knowledge about time series. Answer with JSON only."
)


def _slot_menu(query_type: str) -> dict[str, list[str]]:
    return {s: list(QUALITATIVE_SLOTS[s]) for s in SLOTS_FOR_TYPE[query_type]}


def build_prompt(answer_text: str, query_type: str) -> str:
    menu = _slot_menu(query_type)
    lines = [
        "Below is a description of a time series, written by someone else.",
        "",
        "--- BEGIN DESCRIPTION ---",
        str(answer_text).strip(),
        "--- END DESCRIPTION ---",
        "",
        "For each attribute, say which category the DESCRIPTION places it "
        "in. Rules:",
        f'  * If the description does not mention it, answer "{ABSENT}".',
        f'  * If it names two or more categories for one attribute, or is '
        f'genuinely unclear, answer "{UNCLEAR}".',
        "  * Do not infer from what seems likely. Report only what is said.",
        "",
        "Attributes and their allowed categories:",
    ]
    for slot, classes in menu.items():
        allowed = ", ".join(f'"{c}"' for c in classes)
        lines.append(f'  {slot}: {allowed}, "{UNCLEAR}", "{ABSENT}"')
    lines += [
        "",
        "Reply with a single JSON object mapping each attribute name to one "
        "category string. No prose, no code fence.",
    ]
    return "\n".join(lines)


def parse_reply(reply: str, query_type: str) -> dict[str, str]:
    """Lenient parse; anything unrecognised becomes `unclear`.

    A judge that fails to answer is NOT silently dropped: it is coded
    `unclear`, which counts as a disagreement with a confident extractor
    reading. Dropping it would quietly restrict the comparison to the items
    the judge found easy, which is the same selection bias
    `machine_vs_human` refuses to hide.
    """
    menu = _slot_menu(query_type)
    text = (reply or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    obj = {}
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            obj = {}
    out = {}
    for slot, classes in menu.items():
        v = obj.get(slot)
        v = str(v).strip().lower() if v is not None else UNCLEAR
        allowed = {str(c).lower(): str(c) for c in classes}
        if v in allowed:
            out[slot] = allowed[v]
        elif v == ABSENT:
            out[slot] = ABSENT
        else:
            out[slot] = UNCLEAR
    return out


def extractor_coding(row: pd.Series, slot: str) -> str | None:
    """What the REGEX extractor read, in the shared label space."""
    status = row.get(f"slot_{slot}_status")
    if not isinstance(status, str):
        return None
    if status == "one":
        pred = row.get(f"slot_{slot}_predicted")
        return None if pd.isna(pred) else str(pred)
    if status == "absent":
        return ABSENT
    if status == "ambiguous":
        return UNCLEAR
    return None


def _load(results: Path) -> pd.DataFrame:
    raw = results / "raw_results.csv"
    if raw.exists():
        return pd.read_csv(raw)
    shards = sorted((results / "shards").glob("*.csv"))
    if not shards:
        sys.exit(f"no raw_results.csv and no shards under {results}")
    return pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)


NL = chr(10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None,
                    help="output dir (default: <results>/judge_slots)")
    ap.add_argument("--n-per-stratum", type=int, default=50,
                    help="items per (query_type x system). 50 -> 300 calls "
                         "on a two-cell three-arm run")
    # gemini-2.5-flash was retired for new users; the API's own 404 names
    # gemini-3.8-flash as the replacement ("This model models/gemini-2.5-flash
    # is no longer available to new users"). Model names move, so this is a
    # default rather than a constant - pass --model to override.
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--provider", default="gemini",
                    choices=["gemini", "openai_compatible"],
                    help="gemini needs GEMINI_API_KEY and is rate-limited on "
                         "the free tier (measured: ~2 calls before silent "
                         "backoff). openai_compatible talks to a local vLLM "
                         "server and is free and unlimited.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                    help="only for --provider openai_compatible")
    ap.add_argument("--by", default="",
                    help="extra DESIGN columns (comma-separated) to add "
                         "to the stratification and to break agreement "
                         "down by, e.g. --by backbone for H3 or --by L "
                         "for H1. Default empty, which reproduces the "
                         "H2 run exactly: same strata, same prompts, so "
                         "the existing judge cache still hits.")
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between calls, to stay inside rate limits")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the sample size and one prompt, call nothing")
    args = ap.parse_args()

    results = Path(args.results)
    out = Path(args.out) if args.out else results / "judge_slots"
    df = _load(results)
    s = df[df.get("scoring") == SLOT_SCORING].copy()
    if s.empty:
        sys.exit("no slot-scored rows; nothing for this script to validate")
    s = s[s["answer_text"].notna()]

    # stratify on DESIGN variables only (cell x arm [x --by]), never on the
    # outcome. `--by backbone` matters for H3: pooling the seven rungs would
    # average a 0.5B model's prose in with a 72B's, and "does the extractor
    # get less reliable on weaker backbones" is exactly the question that
    # decides whether H3's ladder shape is a property of the architecture or
    # of the instrument reading it. Likewise `--by L` for H1, where the
    # answer texts get longer along the sweep.
    extra = [c for c in (x.strip() for x in args.by.split(",")) if c]
    bad = [c for c in extra if c not in s.columns]
    if bad:
        sys.exit("--by names column(s) not in the shards: {} ; available: {}"
                 .format(bad, sorted(s.columns)))
    strat = ["query_type", "system"] + extra
    picked = []
    for _, part in s.groupby(strat):
        take = part.sample(n=min(args.n_per_stratum, len(part)),
                           random_state=args.seed)
        picked.append(take)
    sample = pd.concat(picked, ignore_index=True)

    print(f"slot-scored rows available : {len(s)}")
    print(f"stratified on              : {strat}")
    print(f"strata                     : "
          f"{sample.groupby(strat).size().to_dict()}")
    print(f"items sampled              : {len(sample)}")
    print(f"calls that will be made    : {len(sample)} (one per item)")
    print(f"model                      : {args.model}")

    if args.dry_run:
        r = sample.iloc[0]
        print("\n--- example prompt (item 1) ---")
        print(build_prompt(r["answer_text"], r["query_type"]))
        print("\nDRY RUN: no call made, nothing written.")
        print("Set GEMINI_API_KEY and drop --dry-run to execute.")
        return

    if args.provider == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            sys.exit("GEMINI_API_KEY is not set. Get a key at "
                     "https://aistudio.google.com/apikey , then:\n"
                     "  export GEMINI_API_KEY=...        (bash)\n"
                     "Check your own rate limits at "
                     "https://aistudio.google.com/rate-limit first - the free "
                     "tier allowed ~2 calls when measured on 2026-11-07.\n"
                     "Or use --provider openai_compatible against a local "
                     "vLLM server, which is free and unlimited.")
        from models.providers import GeminiLLM
        llm = GeminiLLM(args.model, temperature=0.0, max_tokens=256,
                        request_delay_s=args.delay, max_retries=8)
    else:
        # Local vLLM. No rate limit, no cost, and pinnable to a commit hash,
        # which a Gemini version is not - as the retirement of
        # gemini-2.5-flash mid-session demonstrated.
        from models.providers import OpenAICompatibleLLM
        llm = OpenAICompatibleLLM(args.model, base_url=args.base_url,
                                  api_key="none", temperature=0.0,
                                  max_tokens=256)
    print(f"provider                   : {args.provider}")

    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "judge_cache.jsonl"
    cache: dict[str, str] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                cache[rec["key"]] = rec["reply"]
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"cache                      : {len(cache)} replies reusable")

    rows, n_called, stopped = [], 0, None
    with cache_path.open("a", encoding="utf-8") as cf:
        for i, r in sample.reset_index(drop=True).iterrows():
            prompt = build_prompt(r["answer_text"], r["query_type"])
            key = hashlib.sha256(
                f"{args.provider}\x00{args.model}\x00{SYSTEM_PROMPT}"
                f"\x00{prompt}".encode()
            ).hexdigest()[:24]
            if key in cache:
                reply = cache[key]
            else:
                try:
                    reply = llm.complete(SYSTEM_PROMPT, prompt)
                except (RuntimeError, OSError) as exc:
                    # Free-tier daily caps are low and undocumented, so a
                    # run hitting one mid-way is the expected case, not an
                    # anomaly. Stop cleanly and still report alpha on what
                    # landed: crashing would discard a usable partial
                    # sample and print nothing at all. Replies already
                    # cached above survive, so a later run resumes rather
                    # than re-paying for them.
                    stopped = f"{type(exc).__name__}: {exc}"
                    break
                cache[key] = reply
                cf.write(json.dumps({"key": key, "reply": reply}) + "\n")
                cf.flush()
                n_called += 1
                # Every 5, not every 25. `GeminiLLM` retries rate limits
                # silently with the provider's own retryDelay, so on a
                # throttled free tier a single call can take minutes with no
                # output at all - indistinguishable from a hang. Frequent
                # progress makes the difference visible.
                if n_called % 5 == 0:
                    print(f"  {n_called} calls made "
                          f"({i + 1}/{len(sample)} items)", flush=True)
                # NO sleep here. `GeminiLLM.complete` already enforces a
                # minimum interval between calls, measured from the previous
                # call's start, so `--delay` is honoured once. Sleeping again
                # here stacked a second delay on top and doubled the run for
                # no extra rate-limit protection.
            judged = parse_reply(reply, r["query_type"])
            for slot, jcode in judged.items():
                ecode = extractor_coding(r, slot)
                # The unit must be unique over the WHOLE sample,
                # including the sweep axis. alpha is computed from a dict
                # keyed on `unit`, so a key that repeats across rungs keeps
                # only the last one silently: before --by existed this was
                # safe (single backbone, single L), but across 7 H3 rungs it
                # discarded six sevenths of the codings and still printed a
                # confident-looking alpha. With extra=[] the string below is
                # byte-identical to the previous one.
                _uparts = ([str(r["query_type"]), str(r["system"]),
                            str(r["sample_id"])]
                           + [str(r[_c]) for _c in extra] + [str(slot)])
                row = dict(
                    unit="|".join(_uparts),
                    query_type=r["query_type"], system=r["system"],
                    sample_id=r["sample_id"], slot=slot,
                    extractor=ecode, judge=jcode)
                for _c in extra:
                    row[_c] = r[_c]
                rows.append(row)

    if stopped:
        print(f"\n!! STOPPED EARLY after {n_called} new calls: {stopped}")
        print("   Replies already fetched are cached, so re-running this "
              "same command resumes\n   from here rather than re-paying for "
              "them. Alpha below is on the PARTIAL\n   sample and its "
              "precision is correspondingly lower.")

    ann = pd.DataFrame(rows)
    if not ann.empty and ann.unit.duplicated().any():
        _d = int(ann.unit.duplicated().sum())
        sys.exit(f"{_d} duplicate annotation unit key(s). alpha is computed "
                 f"from a dict keyed on `unit`, so duplicates would be "
                 f"dropped silently and alpha reported on a subset. Add the "
                 f"distinguishing column to --by.")
    if ann.empty:
        sys.exit("\nno codings collected, so there is nothing to report. "
                 "If the message above\nmentions a model name, pass a "
                 "current one with --model.")
    ann.to_csv(out / "judge_codings.csv", index=False)
    # An ITEM is (query_type, system, sample_id [, --by]) - NOT sample_id
    # alone. The same series is shown to all three arms and under both
    # query types, so sample_id repeats and nunique() under-counts items
    # by that factor: the 2026-11-08 H2 judge run reported 'over 100 of
    # 300 sampled items' for a run that had in fact coded all 300 (900
    # units = 150 trend_description items x 2 slots + 150
    # event_explanation items x 4 slots). A number that makes a complete
    # run look truncated is worse than a merely wrong one: it invites
    # throwing the result away or re-paying for it.
    _item_keys = ["query_type", "system", "sample_id"] + extra
    items_done = int(ann.groupby(_item_keys).ngroups)
    print(f"\nwrote {out / 'judge_codings.csv'} ({len(ann)} slot codings "
          f"over {items_done} of {len(sample)} sampled items), "
          f"{n_called} new calls")
    if items_done < len(sample) and not stopped:
        print(f"  NOTE: {len(sample) - items_done} sampled item(s) "
              f"yielded no parseable coding. They are absent from alpha, "
              f"not counted as agreement or disagreement.")

    # ---- agreement, using the SAME alpha the human protocol uses ----
    report = {"model": args.model, "provider": args.provider,
              "n_items_sampled": int(len(sample)),
              "n_items_judged": int(items_done),
              "n_calls_made": int(n_called), "seed": args.seed,
              "results_dir": str(results),
              "stopped_early": stopped, "slots": {}}
    print("\n=== Krippendorff alpha: regex extractor vs independent judge ===")
    print("D7 decision threshold: alpha < 0.667 on a shared slot removes "
          "that cell from the confirmatory family.\n")
    print(f"{'slot':<12} {'n_units':>8} {'alpha':>8}   reading")
    print("-" * 58)
    for slot, part in ann.groupby("slot"):
        units = {u: [e, j] for u, e, j in
                 zip(part.unit, part.extractor, part.judge)}
        res = krippendorff_alpha(units, level="nominal")
        a = res.get("alpha")
        report["slots"][slot] = res
        shown = "   nan" if a is None or a != a else f"{a:+.3f}"
        print(f"{slot:<12} {res.get('n_units', 0):>8} {shown:>8}   "
              f"{res.get('reading', '')}")

    # Breakdown along each --by axis. A pooled alpha can sit comfortably
    # above threshold while one rung of the sweep sits far below it, and that
    # rung is the one whose cell cannot carry a confirmatory claim.
    if extra:
        report["by"] = {}
        for _col in extra:
            report["by"][_col] = {}
            print(NL + "=== alpha by {} x slot ===".format(_col))
            print("{:<16} {:<12} {:>6} {:>8}".format(_col, "slot", "n",
                                                     "alpha"))
            print("-" * 46)
            for (_lvl, _slot), _part in ann.groupby([_col, "slot"]):
                _units = {u: [e, j] for u, e, j in
                          zip(_part.unit, _part.extractor, _part.judge)}
                _res = krippendorff_alpha(_units, level="nominal")
                _a = _res.get("alpha")
                report["by"][_col].setdefault(str(_lvl), {})[_slot] = _res
                _shown = ("   nan" if _a is None or _a != _a
                          else "{:+.3f}".format(_a))
                print("{:<16} {:<12} {:>6} {:>8}".format(
                    str(_lvl), _slot, _res.get("n_units", 0), _shown))

    (out / "judge_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'judge_report.json'}")
    print("\nREMINDER: high agreement here is WEAK evidence - a regex and an "
          "LLM can share\na misreading. Low agreement is the informative "
          "outcome. This does not\ndischarge D7's human annotation.")


if __name__ == "__main__":
    main()
