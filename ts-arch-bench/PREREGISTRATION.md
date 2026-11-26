# Pre-registration — ts-arch-bench (corrected build)

**Dated 2026-11-02.** Commit this file *before* running anything whose result
will be reported. Any later change to it is a protocol deviation and must be
reported as one, with both the original and revised analysis shown.

This file covers the **framework**. Each hypothesis (H1/H2/H3) gets its own
dated section appended below, written before its runner is executed.

---

## 0. Why pre-registration, here, now

The previous build's decision rule contained a branch (`no_split`) that was
mathematically unreachable before any code ran, and the proposed remedy was
to change the rule after seeing the data. Publishing the planned-versus-actual
diff is the practice an evaluations venue exists to reward; retro-editing the
rule is the thing it exists to catch. So: the rule is fixed here, the screen
that modifies its denominator is fixed here, and both verdicts get reported.

---

## 1. Measurement

### 1.1 Primary score

Per item, in `[0, 1]`, higher better. Computed per **kind**; never averaged
across kinds.

| Kind | Primary | Secondary reported |
|---|---|---|
| numeric (unbounded) | `1/(1+scaled_err)` | `within_tol` at tol = 0.05 |
| numeric (bounded, `lag1_autocorr`) | `1/(1+abs_err)` | `within_tol` at 0.05 |
| counting | exact match | off-by-one rate |
| localization | hit@2 | hit@{0,1,5}, `type_error` rate |
| forecast | `1/(1+MASE)`, denominator fixed across L | `beats_naive` |
| qualitative | macro slot accuracy | per-slot confusion, `incoherent` rate |

`scaled_err = |v − gt| / natural_scale(op, window)`. The scale is a property
of the **window**, never of the target. Table in `evaluation/metrics.py`.

### 1.2 Failure accounting — fixed in advance

Every item receives a finite score. A system that does not answer scores
**zero**, and three numbers are reported together:

- `scored_mean` — failures as zero. **This is the primary.**
- `coverage` — fraction answered.
- `selective_mean` — mean among answered items only.

Failure taxonomy: `abstain`, `unparseable`, `type_error`. Routing taxonomy:
`clean`, `recovered`, `ambiguous`, `unparseable`.

### 1.3 Known measurement limitations, declared up front

1. The `seasonal` slot is presence/absence, so a self-contradictory answer
   still collects chance accuracy there. The slot scorer is **not** validated
   against humans yet (see §4).
2. For `counting`, `localization`, `change` and `correlation` the answer key
   is an **operationalisation**, not a canonical quantity. A system can answer
   the question sensibly and miss the key by using a different threshold.
   Reported as a limitation, not patched away.
3. **(Amended 2026-11-05 — see §5, deviation D1.)** `statistics`,
   `extremum`, `change`, `correlation`, `counting` and `localization` route
   to a deterministic tool that *is* the answer-key function, so a
   correctly-routed harness scores exactly 1.000 on them by construction.
   These cells are flagged **`saturated`**, which is a **separate axis from
   `decidable`**: a cell with one arm pinned at 1.000 and the other at 0.64
   has a real 0.35 gap and is perfectly decidable, but it measures the
   unpinned arm alone rather than architecture. Saturated cells are reported
   with their trivial baseline and excluded from the confirmatory family
   except where §3 names an explicit override.

4. Routing is a property of the **question**, not of the item: the
   classifier never sees the series. Routing statistics are reported at the
   independent unit chosen by the clustering rule in §3, never at the item
   level by default.

---

## 2. Screening, run before any hypothesis test

`src/screening.py` with the thresholds in `DEFAULT_SCREEN`:

```
min_sd      0.01    response SD below this carries no information
ceiling     0.95    every system above this: no headroom
floor       0.05    every system below this: nothing to distinguish
min_effect  0.02    the practical-effect floor
```

A cell is **undecidable** if any of: all systems at ceiling, all at floor, no
system shows response variance, or the largest observed pairwise gap is below
`min_effect`.

**Decidability is screened per contrast, not per cell.** A cell-level maximum
over all pairs lets a third arm — the oracle — make a cell look decidable
when the confirmatory contrast inside it is not. `screening.json` reports
`pair_gaps` and `decidable_pairs` for every pair, and the verdict uses the
`(bare_llm, harness)` entry.

**`saturated` is reported independently of `decidable`** (see §1.3.3). A cell
can be both: decidable because the gap is large, saturated because one arm
carries no information.

Output is written to `screening.json` *before* tests are run, and includes the
**trivial baseline** for every cell — what a constant answer scores. System
means are read against that floor, not against zero.

Both verdicts are always reported:

- **as pre-registered** — over all cells;
- **screened** — over decidable cells only.

Neither is substituted for the other.

---

## 3. Statistics

- Paired, non-parametric, per cell: Wilcoxon signed-rank two-sided on the
  full score vectors (**never** after dropping unanswered items).
- Effect: paired bootstrap 95% CI of the mean difference, 5000 resamples,
  seeded; Cohen's `dz`; and `paired_sign_effect` = `mean(sign(d))` =
  P(d>0) − P(d<0). *(Renamed 2026-11-05 — §5, deviation D4. It was labelled
  "Cliff's delta", which is defined over all cross-pairs of two independent
  samples and is not this quantity.)* Ties dilute it toward zero, so it is
  always read beside `n_zero_diff`.

- **Family, declared by name (amended 2026-11-05 — §5, deviation D2).**

  > Confirmatory family for the ANSWER-ACCURACY contrast = `bare_llm` vs
  > `harness`, over exactly these three query types: **`forecasting`,
  > `trend_description`, `event_explanation`**. **m = 3.**
  >
  > `harness_oracle` is a **pre-registered secondary ablation**: effects and
  > confidence intervals reported in full, *outside* the correction.

  **Revised 2026-11-06 — §5, deviation D6.** The family is now exactly the
  complement of `a_priori_saturated`; no saturated cell is in it, and
  `saturated_cells_kept_in_family` is asserted empty by
  `tests/test_family_and_clustering.py`.

  The earlier version kept `counting` and `localization` by judgement,
  giving m = 5. That is withdrawn in favour of **decomposed metrics**, which
  is what the tool-evaluation literature recommends: separate measures for
  tool invocation, task completion and outcome verification rather than one
  family mixing them (*Benchmarking the Benchmarks: A Validity Audit of
  Tool-Calling Evaluation*, arXiv:2607.02577).

  The argument is arithmetic, not taste. In a saturated cell a
  correctly-routed harness scores exactly 1.000, so its per-item score **is**
  its routing outcome and the gap reduces to
  `route_correct − bare_accuracy` — a quantity this build already reports
  twice over:

  | claim | where it is measured |
  |---|---|
  | tool invocation | `route_correct` / `route_correct_selective`, per cell, all nine types, clustered per this section |
  | cost of mis-routing | the `harness` vs `harness_oracle` ablation, secondary, outside correction |
  | task completion | the m = 3 answer-accuracy family above |

  So `localization` is not demoted: the quantity it was kept for is measured
  better in the routing columns — at n = 100 with genuine paraphrase variance
  after D5 — than as an answer-score contrast. And `counting`'s gap would
  have measured in-context MAD arithmetic against tool access, which is a
  statement about the bare arm rather than about architecture.

  What remains is the set of cells where the harness is **not** pinned by
  construction: `forecasting` (a TSFM against in-context extrapolation) and
  the two qualitative cells (feature-assisted generation against raw
  generation).

  ⚠ **m drops from 5 to 3, which makes the surviving tests easier to call
  significant** (`alpha/3 > alpha/5`). That is the anti-conservative
  direction D2 warned about. It is declared here in advance, justified on
  construct validity and never on convenience, and `cfg.family_report()`
  prints m on every run.

  Saturation is an **a priori architectural property** — it follows from
  `systems/tools.py` importing `src.ground_truth`, not from observed data —
  so family membership is a design decision fixed before the run, not a
  screen applied after it.

  `m = 3` is passed explicitly to `holm()`. With `m` inferred from the data,
  a degenerate comparison returns NaN and silently leaves the family,
  shrinking `m` and making the survivors *easier* to call significant: the
  same raw `p = 0.02` adjusts to 0.18 at m=9 and to 0.02 at m=1. An
  untestable cell still consumes its share of the family — you planned to
  test it.

- Correction: **Holm**, `alpha = 0.05`.

- **Clustering rule for routing-derived contrasts, fixed in advance
  (2026-11-05 — §5, deviation D3).** The classifier sees only question text,
  so every item sharing a phrasing receives the same routing decision.
  Treating those items as independent counts one decision many times.

  > Let `f` be the mean fraction of framing families (25 per type) showing
  > any within-family variation in `route_op`. If `f <= 0.20`, the
  > **framing** is the independent unit (n = 25/type); otherwise the
  > **item** is (n = 100/type). The self-versus-oracle contrast uses
  > `paired_compare_clustered` at whichever unit this selects.

  The threshold is set before the variance is measured, because choosing the
  clustering level after seeing it is itself a researcher degree of freedom.
  `screening.json` records `f`, the decision and the effective n.
- A win requires all three: Holm `p < .05`, bootstrap CI excluding zero, and
  `|mean diff| >= 0.02`.
- Holm re-ranks the whole family when any p moves, so a projected flip in one
  cell is **re-run**, never arithmetic'd.
- **Sensitivity grid is mandatory**, not optional: the verdict is reported as
  a function of (Holm | BH) × (effect floor 0.01 | 0.02 | 0.05) × (screened |
  unscreened) × (failures-as-zero | dropped). One word is not a result.
- Rank correlations (H3 shape): Spearman rho with bootstrap CI as **primary**;
  exact permutation p reported beside it; `min_detectable_rho(n)` reported
  always. With small `n` a null is written *"this design could not detect a
  relationship of plausible size"*, never *"there is no relationship"*.

---

## 4. Deferred, and pre-registered for the next cycle

Declared now so a reader sees these gaps are known rather than missed.

1. **Human validation of the qualitative metric.** 50–100 stratified items,
   three independent annotators, Krippendorff's alpha reported, and the
   correlation between slot accuracy and human labels reported *whether or not
   it is flattering*. Until this exists, qualitative cells are reported as
   provisionally measured.

   **Protocol, specified 2026-11-06 — §5, deviation D7.** Written before any
   sheet was produced, and now implemented in
   `evaluation/human_validation.py` with `scripts/export_annotation.py` and
   `scripts/analyse_annotation.py`.

   *Two tasks, not one.* A slot score is `extracted == expected`, so a
   disagreement has two possible sources with different remedies, and one
   pooled number would hide which is at fault:

   | task | annotator sees | validates | a failure means |
   |---|---|---|---|
   | **A** extraction | the answer text, no series | the regexes in `evaluation/qualitative.py` | fixable in the extractor; `PARSER_VERSION` covers it, so re-scoring is free |
   | **B** answer key | the series, no answer text | `qualitative_truth` | **not** fixable in the extractor; the key disagrees with readers of the same data |

   Task B is the one that can embarrass the benchmark. `is_seasonal` uses the
   generator's **true** period, which neither an annotator nor a deployed
   system can observe. Systematic human/key disagreement on `seasonal` would
   be evidence the answer key encodes information absent from the data.

   *Sampling.* Proportional stratification on **design variables only** —
   `query_type × system` for Task A, `query_type × pattern` for Task B.
   Deliberately **not** on the machine's own verdict: enriching for
   disagreement would be far more informative per annotation but conditions
   the sample on the quantity being validated, so the resulting agreement
   rate would not estimate the population rate. Sampling weights are
   recorded on every row so an enriched supplementary sample can be added
   later and reweighted.

   *Blinding.* Annotators never see the machine labels (`_key.json` stays
   with the experimenter), never see each other's sheets, and receive
   per-coder shuffled row orders. Task A withholds the series and Task B
   withholds the answer text, so neither judgement can be anchored by the
   other.

   *Reporting.* Per slot per task: inter-annotator Krippendorff's alpha
   (nominal) as the **ceiling**, then machine-versus-consensus agreement
   *and* its alpha, then the confusion matrix. Raw agreement is never
   reported alone: on a skewed slot it can be near 1.0 while alpha is near
   0, which is agreement carrying no information. Items with no human
   consensus are counted, never dropped — a metric that looks good only on
   the items humans found easy is not validated. Ties are not broken.

   *Decision rule, fixed in advance.* Per slot, on the inter-annotator alpha,
   taking the **lower** of Task A and Task B, and applying to a cell the
   worst result among the slots that cell uses:

   | alpha | consequence for the qualitative cells |
   |---|---|
   | ≥ 0.800 | reported as measured; the "provisionally measured" caveat is withdrawn |
   | 0.667–0.800 | caveat retained; conclusions stated as tentative |
   | < 0.667 | **the metric cannot carry a confirmatory claim.** The affected cell leaves the confirmatory family and is reported descriptively |

   The last branch has teeth and is declared knowingly: `trend_description`
   and `event_explanation` are two of the three cells in the §3 family, so
   failing validation on a shared slot such as `trend` or `seasonal` would
   reduce **m from 3 to 1** — `forecasting` alone. A one-cell confirmatory
   family is a thin result, and it is the honest one; a three-cell family
   resting on a metric that failed its own validation is not. Fixing the
   threshold here rather than after seeing alpha is the entire point.
2. **New query-type families** that a one-tool-per-question architecture
   cannot trivially win: compositional (two chained operations),
   underspecified (routing *is* the task), no-tool (where abstention becomes
   measurable), context-dependent.
3. **Item-level IRT.** A 2PL fit over items once enough systems exist for the
   parameters to mean anything. With a handful of systems, discrimination is
   reported descriptively and never used as a gate.

---

## 5. Protocol deviations

Append dated entries here.

**D1–D7 were written before any run against a real backbone.** At the time
those were written no real model call had been made from this build — every
cache file in the tree was `mock-llm` — so none of them could have been
informed by a result.

**That is no longer true of entries from D8 onward, and they say so.** On
2026-11-07 this build ran against self-hosted vLLM on hpc01: H2 complete
and six of seven H3 rungs, ~12,400 served completions. Any amendment dated
2026-11-07 or later was therefore written with results in hand and is
**post-hoc by construction**. It is recorded as such, with what was known
at the time, rather than presented alongside the pre-run entries as though
it carried the same weight. An amendment that changes a hypothesis test
after seeing data is not admissible as confirmatory evidence; an amendment
that adds a *check* is, provided the check's outcome is reported whichever
way it falls. D8 and D10 are both of the second kind, and the
outcome of each went against the thing that prompted it — D8 against the
suspicion that the H2 effect was a scoring artifact, D10 against the
extractor it was built to vindicate.

| # | Date | What changed | Why | Effect on reported results |
|---|---|---|---|---|
| **D1** | 2026-11-05 | §1.3.3: tool-saturated cells are flagged on a **separate `saturated` axis** instead of being predicted "undecidable". The saturated set is also corrected from four types to six. | The original text claimed `statistics`, `extremum`, `change` and `correlation` would screen out as undecidable. The screening code never would have marked them so: a cell with the harness pinned at 1.000 and the bare arm at 0.64 trips none of the four undecidability criteria. The prose and the code contradicted each other, and the committed mock `screening.json` already showed `statistics` and `extremum` returning **decidable**. | None yet — no run has been scored. Had this been found after a run, adjusting the screen to match would have been exactly the retro-editing §0 forbids. |
| **D2** | 2026-11-05 | §3: the Holm family is now declared **by name** (5 named cells, `bare_llm` vs `harness`, m = 5) rather than as "every (system pair × query type) test in the run". `harness_oracle` moves to a secondary ablation outside the correction. | The original wording implied 27 tests with three arms, and `holm()` counted only finite p-values, so the realised family depended on how many comparisons turned out degenerate. The bias is anti-conservative: fewer surviving cells → smaller m → easier significance. | None yet. **Superseded by D6** (2026-11-06): the family is now the three non-saturated cells, m = 3. D2's substance — a family declared by name with `m` passed explicitly — stands. |
| **D3** | 2026-11-05 | §3: added a pre-registered clustering rule (threshold `f <= 0.20`) selecting framing vs item as the independent unit for routing-derived contrasts. | The paraphrase bank is compositional (25 framings × 4 leads). Whether the lead axis moves routing is empirical; without a rule fixed in advance, the clustering level would be chosen after seeing the variance. | None yet. The measured `f` and the resulting unit are written to `screening.json` on every run. |
| **D4** | 2026-11-05 | §3: `cliffs_delta` renamed `paired_sign_effect` throughout. | `mean(sign(d))` is P(d>0) − P(d<0) for paired data. Cliff's delta is defined over all cross-pairs of two independent samples. The name was simply wrong. | None — identical statistic, corrected label. |
| **D5** | 2026-11-05 | Question text changed from one fixed template per query type to a 100-variant paraphrase bank. | With one template per type the classifier received a byte-identical prompt for every item of a type, so the benchmark contained **nine** distinct routing decisions replicated 100 times each. Routing accuracy per cell could only be 0% or 100%. | None yet, but it **invalidates comparison with the original repo's figures**. The published "73 of 100 localization mis-routes" came from interpolating `request_id` into the prompt, i.e. one question resampled under a nonce, not 100 independent items. Restate it as such in the paper. |

| **D6** | 2026-11-06 | §3: the confirmatory family drops from five named cells to **three** — exactly the non-saturated set (`forecasting`, `trend_description`, `event_explanation`). `counting` and `localization`, previously kept by judgement, move to descriptive reporting. m = 3. | Decomposed metrics. In a tool-saturated cell a correctly-routed harness scores exactly 1.000, so its answer score *is* its routing outcome and the bare-vs-harness gap reduces to `route_correct − bare_accuracy` — already reported by the routing columns and the oracle ablation. Including it in the answer-accuracy family double-counts one quantity and lets a cell be won by construction. Follows the tool-evaluation literature's recommendation of separate measures for tool invocation, task completion and outcome verification (arXiv:2607.02577); the independent-ceiling-dimension treatment in arXiv:2605.30504 and its exclusion of constant-response items before fitting support the same split. | None yet — no run has been scored. ⚠ Note the direction: m = 3 makes surviving tests **easier** to call significant than m = 5 (`alpha/3 > alpha/5`), the same anti-conservative direction D2 flagged. Declared in advance and justified on construct validity. The six saturated cells are still run and still reported. |

| **D9** ⚠ POST-HOC | 2026-11-07 | §6 (H3): `analysis.robustness_backbones` is emptied from `[llama3_1_8b, llama3_1_70b]` to `[]`. The cross-family robustness axis is **dropped**. The backbone definitions, their HF ids and their declared MMLU-Pro strengths stay in the config, unused. The `ladder` is untouched, so the confirmatory DiD and its two end rungs are unchanged. | Both Llama models are gated on Hugging Face. No token was present on hpc01 (`~/.hf_token` absent; `env.sh` reads it unconditionally), so neither rung was ever submitted — array indices 7 and 8 never ran. Because they were still *declared*, `load_shards` refused the grid and H3 yielded **no verdict at all**, despite all seven Qwen rungs completing: analysis job 437039 exited 1 with "the analysis stage refuses to run on a partial grid". The choice is therefore between leaving H3 permanently unanalysable, obtaining a token and queueing two more rungs, or dropping a secondary axis. Dropping it is the only option that costs nothing in the confirmatory family, which is where every claim in the paper lives. | **H3 becomes analysable on the seven Qwen rungs already on disk; no new inference.** What is lost is the ability to say the result holds across model families — a real limitation that the paper must state plainly rather than omit, since a single-family ladder cannot separate "advantage varies with strength" from "advantage varies with something idiosyncratic to Qwen's recipe." ⚠ Direction of bias: unknown rather than conservative. Nothing about the Qwen ladder was used to make this choice — the trigger was a missing credential, not a result — but it is post-hoc and the ladder had been seen, so it is recorded as such. Restoring the axis later is a one-line config change and would be a further deviation. |
| **D10** ⚠ POST-HOC | 2026-11-08 | §4.1: adds an **LLM-reader pre-check** of the slot extractor (`scripts/judge_slots.py`, `slurm/judge_slots.sbatch`) ahead of D7's human annotation. It runs D7's **Task A** only — the reader sees the answer text, never the series, the answer key, the score or the arm — and reports Krippendorff's alpha against the regex extractor per slot. Nothing in the scoring code, the answer key, the family or the tests is changed by it. | After D6 and D9, two of the three confirmatory cells (`trend_description`, `event_explanation`) are scored by the regex slot extractor that §4.1 itself calls "provisionally measured", and **nothing had tested it**. D7 specified the consequence in advance but needs three human annotators, which are not available; waiting for them left the headline resting on an unmeasured instrument indefinitely. An LLM reader is a weaker instrument than a human but a far stronger one than none, and it was obtainable the same day. The check was specified and committed before it was run, and its outcome was reported whichever way it fell. | **Outcome, job 437521 on 2026-11-08 (H2, 300 items, 900 units, Qwen2.5-72B-Instruct served locally): `trend` α=+0.835 and `position` α=+0.954 clear the bar; `seasonal` α=+0.379 and `event` α=+0.441 fall FAR below D7's 0.667.** `seasonal` is a slot in **both** qualitative cells (`SLOTS_FOR_TYPE`), i.e. exactly the "shared slot" D7 names, and `event` sits in the other. So the configuration D7 pre-declared as cutting **m from 3 to 1** is the realised one. ⚠ This does **not** fire D7: D7's alpha is inter-annotator agreement among humans, and an LLM reader is not an annotator — substituting one is itself a deviation. The result is therefore reported as a **sensitivity analysis**, with the pre-registered m=3 family as primary and the m=1 family (`forecasting` alone) alongside it. **Both are reported in full because the m=1 reading is more favourable to two of the three hypotheses, and an author must not be the one who picks.** At m=1: H2 `forecasting` p=7.05e-10 wins (1/1); H1 `forecasting` p=0.0151 wins (1/1, and monotone across all five L); H3 `forecasting` p=0.5504 does not (0/1). No forecasting verdict changes between m=3 and m=1 — the family size shifts the two qualitative cells only. Note also that `seasonal` is the one slot in `NEGATION_FIRST` and the one §4.1 predicted in advance would be the problem; the pre-check hit precisely where the pre-registration said to look. The remedy is a `PARSER_VERSION` bump and free re-scoring, which is **deliberately NOT being done now**: fixing a scorer after seeing which way it moved the results is the degree of freedom §0 forbids. It stays logged for the human annotation to settle. **Extended 2026-11-09 (job 438435, 30m57s, one 72B load): the same check along both sweep axes — H1 by history length (1500 items, 4500 units), H3 by backbone (2100 items, 6300 units).** Two findings that change how the earlier cells should be read. (a) **H1: extractor reliability RISES with L** (`seasonal` +0.344→+0.558, `event` +0.372→+0.509; `trend`/`position` flat). Random coding pushes both arms toward a common chance score, so a less reliable rung has its gap compressed more, and `Delta = gap(512) - gap(32)` is biased **upward — toward H1's own prediction**. Hence H1's `trend_description` +0.0450 (ns) is consistent with instrument drift alone, while its `event_explanation` −0.0800 is significantly negative *despite* that upward bias and is therefore the **conservative** finding. `forecasting` is untouched (MASE, no slots). (b) **H3: the two qualitative cells are not equally affected.** At H3's DiD endpoints (0.5B vs 72B) `trend_description` clears the bar (worst slot `seasonal` +0.720) while `event_explanation` fails (`event` +0.493) — so the cell saying the gap SHRINKS with capability is the unreliable one, and the cell saying it GROWS survives (Holm m=2: p_adj=1.99e-07). That sign is **suggestive only**: reliability also rises from the weak to the strong end, biasing `Delta = gap(weak) - gap(strong)` downward, i.e. in the direction it reports. H3 is better understood, **not resolved**. ⚠ Reading D7's slot-level pooled rule at the endpoint level, with an LLM reader, is an elaboration of D7 and not D7 itself; it is reported as a direction-of-bias sensitivity analysis, and the pre-registered m=3 result stays primary. Full tables: `results-archive/20261108-slot-extractor-validation/WHAT_IS_THIS.md`. |
| **D8** ⚠ POST-HOC | 2026-11-07 | Two additions, **both written after H2 and six H3 rungs had been scored** (see the §5 preamble). (a) `MockLLM` gains `assist_effect`; at `0.0` the mock no longer varies its answer quality by sniffing the harness's `computed features:` marker out of the prompt, which makes a **null fixture** expressible for the first time (`configs/null_fixture.yaml`, `tests/test_null_fixture.py`). (b) `evaluation/decomposition.py` adds a **declared secondary** three-way attribution of zero-scored slot items — `nothing_extracted` / `ambiguous` / `extracted_but_all_wrong` — reported per cell and arm. The pre-registered primary metric `n_correct / n_slots` is **unchanged**; nothing is re-scored and no verdict is derived from the decomposition. | (a) Every pre-existing fixture varied *backbone strength*; none varied the *harness effect*. So the suite could confirm that the pipeline reports an effect it was built to report, and could not detect the one failure the benchmark exists to rule out — a harness measuring higher for reasons unrelated to answering better. Worse, the mock could not express a null even in principle: it selected between two hand-set per-arm success rates (`0.75+0.20·s` assisted vs `0.50+0.45·s` bare) **and** two failure breadths (assisted corrupted only `trend`; bare corrupted all four slots), so it manufactured roughly +0.02 for the harness by construction. (b) `score = n_correct/n_slots` collapses an unscoreable item, a self-contradictory one and a cleanly-wrong one into the same 0.0. If the arms differ in that mix, the contrast is part capability and part instrument in unknown proportion. | **The check ran and did not support the suspicion that motivated it.** Null fixture: 0/2 wins, contrasts −0.048 and −0.018, both CIs covering zero; positive control at `assist_effect=1.0` fires (`event_explanation` +0.063, p_adj 0.0004), so the detector is live rather than vacuously green. Measured sensitivity floor ≈ +0.10. Decomposition on real H2: `trend_description` 0.350 → **0.2986** under the reading most generous to the bare arm, so the instrument explains **at most ~15%**; `event_explanation` 0.245 → **0.2663**, i.e. the contrast *grows*, because the harness has 8 unscoreable items against the bare arm's 4 and the instrument runs **against** the harness there. Both confirmatory wins stand. Two incidental defects surfaced and are **not** fixed here: the harness emits unscoreable qualitative output on ~5–8% of items, and the bare arm is scored 0 on 16/100 `trend_description` items for naming two classes of one exclusive slot ("rose then fell") — a genuine scorer-fairness question worth a sensitivity line, not a reinterpretation of the verdict. A null on the *numeric* path remains unbuilt: the harness's tool path is exact by design, so equivalence there needs a tool with error injected to match the mock's arithmetic, which does not exist yet. |
| **D7** | 2026-11-06 | §4.1: the deferred human-validation item gains a full protocol — two separately-blinded tasks (extraction vs answer key), stratification on design variables only, and a decision rule mapping inter-annotator alpha to a consequence for the confirmatory family. | The item was pre-registered but unspecified, so every methodological choice in it was still free: which agreement statistic, what to stratify on, and above all what a low alpha would *mean*. D6 made this urgent rather than tidy — two of the three remaining confirmatory cells are scored by this metric, so after D6 the family rests on a measure §4.1 itself calls provisionally measured. Specifying the rule before any sheet exists is what stops alpha being interpreted after the fact. | None yet; no annotation has been collected. Note the declared downside: alpha < 0.667 on a shared slot would cut **m from 3 to 1**. That consequence is accepted in advance. |

**Standing note on the deferred items in §4.** Item 1 has a specified protocol
and working tooling (D7) and, since 2026-11-08, an **LLM-reader pre-check of
Task A** whose result is recorded in D10 — but still **no human annotation**,
which is the only thing that can fire D7's decision rule. Items 2 and 3 have
not been started. They are pre-registered, not claimed.

⚠ **The pre-check's result is adverse and is not resolved.** Two slots, one of
them shared by both qualitative cells, sit far below D7's threshold. Until
human annotation settles it, every claim resting on `trend_description` or
`event_explanation` is provisional in a way the `forecasting` claims are not,
and the m=1 sensitivity analysis in D10 is the honest companion to the
headline rather than a footnote to it.

---

## 6. Per-hypothesis sections

Each section below was written **before** its runner was first executed, as
§0 requires. Common to all three:

* **Contrast.** `d_i = score(harness)_i - score(bare_llm)_i`, per item.
  Positive `d` favours the harness. `harness_oracle` is the secondary
  ablation of §3 throughout, reported with effects and CIs **outside** the
  correction.
* **Family size.** Every hypothesis declares **m = 3**: one confirmatory test
  per cell of the §3 answer-accuracy family (`forecasting`,
  `trend_description`, `event_explanation`). `m` is passed explicitly to
  `holm()`. The six tool-saturated cells are still run and still reported,
  **descriptively**, and routing claims about them live in the routing
  columns and the oracle ablation (§3, deviation D6) — not in any family.
* **Why not a rank test over the sweep axis.** H1 sweeps five history
  lengths and H3 a seven-rung ladder. A Spearman over five or seven points
  has `min_detectable_rho` close to 1, so it could only ever return "not
  detected". Where a sweep axis is short, the confirmatory test is therefore
  a **paired contrast at the item level** between the two ends of the axis,
  and the intermediate points are reported as the shape of the curve. The
  runner prints `min_detectable_rho(n)` beside every rank statistic it
  reports, and a null is written *"this design could not detect a
  relationship of plausible size"*.
* **Win rule**, unchanged from §3: Holm-adjusted `p < .05` **and** bootstrap
  CI excluding zero **and** `|mean diff| >= 0.02`.

---

### H1 — history length *(dated 2026-11-06)*

**Question.** Does the architectural gap depend on how much history is
visible? The motivating expectation is that the bare arm degrades as the
series it must read in-context grows, while the harness is indifferent
because a tool consumes the array rather than the prompt.

**Design.** `L ∈ {32, 64, 128, 256, 512}` × 9 query types × 100 items × 3
arms, one backbone — **Qwen2.5-32B-Instruct**, the same rung H2 uses, so the
two hypotheses share one cached L = 128 cell. The pool is **fixed across L**: `Sample.window(L)` is a
suffix of `window(L')` for `L < L'` (invariant I1) and `target()` is
identical at every L (I2), so the same item is tracked across the sweep
rather than resampled.

**Confirmatory test — difference-in-differences, per declared cell.**

> For item `i` in cell `c`:
> `Δ_i = d_i(L=512) − d_i(L=32)`.
> Wilcoxon signed-rank two-sided on `Δ` over the 100 items of the cell.
> Three cells, **m = 3**, Holm.

`Δ > 0` means the harness's advantage *widens* with history length. This is
an item-level paired test with n = 100, not a 5-point trend test, for the
reason given above.

**Reported descriptively:** `d` at each of the five L values per cell, with
bootstrap CIs — the dose-response curve, which is where a crossover would be
visible; coverage and the failure taxonomy per L, since the bare arm is
expected to fail *differently* (truncation, unparseable) as prompts grow.

**Declared limitation, specific to H1.** For the qualitative and localization
cells the answer key is computed from `window(L)`, so the question's correct
answer genuinely changes with L. `d` at each L remains a valid paired
contrast between two arms answering the same question, but `Δ` compares gaps
on two different questions about the same series. It is a test of whether the
architectural advantage grows, not of a fixed quantity measured twice. The
forecast cell is exempt: its MASE denominator is pinned across L by
`scoring.fixed_mase_denominator`.

---

### H2 — query type *(dated 2026-11-06)*

**Question.** Does the gap depend on what is being asked? This is the base
case and the hypothesis the §3 family was declared for.

**Design.** One backbone — **Qwen2.5-32B-Instruct**, rung 6 of the H3
ladder — at `L = 128`, 9 query types × 100 items × 3 arms. A mid-to-strong
backbone is chosen deliberately: a harness benefit demonstrated only on a
small model invites the reading that it compensates for model weakness, and
H3 already covers the capability dimension on purpose.

**Confirmatory test.** Wilcoxon signed-rank two-sided on `d` within each of
the three declared cells. **m = 3**, Holm. This is §3 applied directly.

**Reported descriptively:** the six tool-saturated non-family cells
(`statistics`, `counting`, `extremum`, `change`, `correlation`,
`localization`), each with its `saturated` flag and trivial baseline, so a
reader sees why a 0.35 gap there is not evidence about architecture; per-cell routing accuracy in both the `route_correct` and
`route_correct_selective` forms.

**Secondary ablation.** `harness_oracle` vs `harness`, per cell, using
`paired_compare_clustered` at the unit chosen by the §3 clustering rule. The
measured `f`, the selected unit and the effective n are written to
`screening.json` before any test is read.

---

### H3 — backbone strength *(dated 2026-11-06)*

**Question.** Does the architectural gap shrink as the backbone gets
stronger? A harness that only compensates for a weak model is a different
claim from one that helps regardless.

**Design.** `L = 128`, 9 query types × 100 items × 3 arms, repeated across a
**single-family ladder** of seven rungs:

| rung | model |
|---|---|
| 1 | Qwen2.5-0.5B-Instruct |
| 2 | Qwen2.5-1.5B-Instruct |
| 3 | Qwen2.5-3B-Instruct |
| 4 | Qwen2.5-7B-Instruct |
| 5 | Qwen2.5-14B-Instruct |
| 6 | Qwen2.5-32B-Instruct |
| 7 | Qwen2.5-72B-Instruct |

One family is deliberate: mixing families would confound capability with
tokenizer and training recipe, and the x-axis is supposed to be capability
alone. Llama-3.1-8B/70B-Instruct are run as a **cross-family robustness
check**, reported separately and **not** in the confirmatory family.

**The strength axis, declared 2026-11-06 before any rung was run.**
`BackboneSpec.strength` is an *external* capability score, never anything
measured in this benchmark — using in-benchmark accuracy as the x-axis would
regress a quantity on itself.

> **Primary axis: MMLU-Pro accuracy / 100**, vendor self-reported.
> Qwen rungs from the Qwen2.5-LLM release tables; the Llama robustness pair
> from Meta's model card, row `MMLU-Pro (CoT)`, 5-shot,
> `micro_avg/acc_char`. Retrieved 2026-11-06. Sources recorded verbatim in
> `configs/h3_backbone_strength.yaml`.
>
> | rung | model | MMLU-Pro | MMLU-redux |
> |---|---|---|---|
> | 1 | Qwen2.5-0.5B-Instruct | 15.0 | 24.1 |
> | 2 | Qwen2.5-1.5B-Instruct | 32.4 | 50.7 |
> | 3 | Qwen2.5-3B-Instruct | 43.7 | 64.4 |
> | 4 | Qwen2.5-7B-Instruct | 56.3 | 75.4 |
> | 5 | Qwen2.5-14B-Instruct | 63.7 | 80.0 |
> | 6 | Qwen2.5-32B-Instruct | 69.0 | 83.9 |
> | 7 | Qwen2.5-72B-Instruct | 71.1 | 86.8 |
> | — | Llama-3.1-8B-Instruct | 48.3 | n/a |
> | — | Llama-3.1-70B-Instruct | 66.4 | n/a |
>
> **Secondary axis: MMLU-redux / 100** (Qwen rungs only), declared now and
> computed by the runner as a robustness re-plot of the same curve.

**Why MMLU-Pro and not a numeric-reasoning score.** An earlier draft of this
section proposed GSM8K, on the reasoning that "weaker at arithmetic" is
closer to what this benchmark asks than general knowledge is. The published
per-size numbers rule it out: of the metrics Qwen reports across all seven
rungs, only MMLU-Pro and MMLU-redux are **strictly monotone**.

```
GSM8K : 49.6 73.2 86.7 91.6 94.8 95.9 95.8   32B > 72B; top four span 4.3 pts
MATH  : 34.4 55.2 65.9 75.5 80.0 83.1 83.1   ties at the top
GPQA  : 29.8 29.8 30.3 36.4 45.5 49.5 49.0   ties at the bottom, then drops
```

A non-monotone axis would order the ladder in a way its own vendor's numbers
contradict. MMLU-Pro is monotone and has the widest range (15.0 → 71.1), so
it discriminates across the whole ladder instead of compressing at one end.

**Two limitations of this axis, declared.** (i) The numbers are *vendor
self-reported* and the two vendors use different harnesses, which is one
reason the Llama pair stays outside the confirmatory family and is never
plotted on the Qwen ladder's axis. Third-party MMLU-Pro figures for
Llama-3.1-8B-Instruct disagree with Meta's (44.0–44.3 against 48.3), so the
cross-family comparison is qualitative. (ii) MMLU-Pro is a general-knowledge
measure, not a measure of reading numbers out of a prompt, so it is a proxy
for the capability this benchmark actually stresses.

Changing the metric or source after seeing results is a protocol deviation
under §5.

**Confirmatory test — gap at the ends of the ladder, per declared cell.**

> For item `i` in cell `c`:
> `Δ_i = d_i(rung 1) − d_i(rung 7)`.
> Wilcoxon signed-rank two-sided on `Δ` over the 100 items of the cell.
> Three cells, **m = 3**, Holm.

`Δ > 0` means the harness helps the weakest backbone more than the
strongest — the shrinking-gap claim. Again item-level with n = 100, not a
7-point rank test.

**Reported as primary shape, not as a test:** Spearman rho between declared
strength and per-cell mean `d` across the seven rungs, with bootstrap CI,
exact permutation p, and `min_detectable_rho(7)` printed beside it. With
seven rungs this is descriptive; the §3 wording for small-n nulls applies
verbatim.

**Declared limitation.** Seven rungs of one family is a short, correlated
axis: parameter count, training-token count and published capability score
move together, so a relationship found here cannot be attributed to
capability alone over the others.

---

### Execution environment *(dated 2026-11-06)*

Declared because it bears on reproducibility.

Runs execute on a SLURM cluster (`hpc01`, Rocky Linux 8.10) on A100 / H100 /
H200 / V100 nodes, serving each backbone through vLLM behind the
`openai_compatible` provider at `temperature = 0`.

**Batching is not bit-deterministic.** vLLM's continuous batching can return
different tokens for an identical prompt depending on what else occupies the
batch, so `call_key`'s assumption that a reply is a function of
`(model, params, prompt)` holds for *replay* but not for *regeneration*: the
cache makes the recorded reply authoritative and every downstream
re-analysis exact, while a cold re-run on a different batch schedule may
differ. This is the same class of caveat already recorded for
`temperature > 0` in `src/cache.py`, and it is the reason `calls.jsonl` is
treated as a primary artifact and archived off `/scratch` rather than
regenerated.

**`/scratch` is not durable.** Cluster policy deletes files 15 days after last modification, and
$SCRATCH is not backed up at all. The sample pool (`data/<pool>/`), the call cache and all
`results/` artifacts are archived to permanent storage at the end of every
run by `slurm/archive.sh`. A pool that is lost cannot be regenerated
identically across a framework change, so its loss would break the paired
design, not merely cost compute.
