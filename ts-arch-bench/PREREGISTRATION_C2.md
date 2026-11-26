# Pre-registration — cycle 2 (C2)

**Written 2026-11-10, before any C2 code existed and before any C2 run.**
Cycle 1 is frozen: its results stay in `../results-archive/` exactly as
produced, and nothing here re-scores them. This is a **new cycle with a new
design**, which is the only admissible response to the defects in §1 —
§0 of [`PREREGISTRATION.md`](PREREGISTRATION.md) forbids repairing a scorer
after seeing which way it moved results.

Cycle 1's protocol, deviations D1–D10 and verdicts remain valid as a record
of what cycle 1 measured. C2 does not supersede them; it measures something
cycle 1 could not.

---

## 1. Why there is a cycle 2

An independent review on 2026-11-10
([`REVIEW_2026-11-10.md`](REVIEW_2026-11-10.md)) found defects that were then
reproduced from the archive and the source. The decisive one is not a bug but
an **inconsistency between a declared criterion and its implementation**.

`src/config.py` declares:

```python
#: Operations whose tool IS the answer-key function, so a correctly-routed
#: harness scores exactly 1.000 by construction. This is an A PRIORI property
#: of the architecture - it follows from `systems/tools.py` importing
#: `src.ground_truth` ...
TOOL_SATURATED_OPS = frozenset({
    "mean", "median", "std", "max", "min", "count_spikes", "pct_change",
    "lag1_autocorr", "change_point",
})
```

`describe` and `explain` are absent from that set. But
`systems/tools.py::series_features` calls `gt.trend_direction`,
`gt.count_spikes`, `gt.has_level_shift` and `gt.event_position` — the same
functions, on the same window, that `src/ground_truth.py::qualitative_truth`
uses to build the key — and in assisted mode (`assist_qualitative: true`, the
default) `systems/harness_tsfm.py` passes `series_block=""`, so the model
never sees the series. **The harness is handed the answer key for the
`trend`, `event` and `position` slots and scored on whether its prose repeats
it.** Measured on cycle 1's archive:

| trend slot, `trend_description`, 32B | correct / 100 |
|---|---|
| `bare_llm` | 23 |
| `harness` | **100** |
| `harness_oracle` | **100** |

Oracle `trend` accuracy is 93–100% at every rung of the H3 ladder; `position`
is 100% from 7B up. By the criterion `TOOL_SATURATED_OPS` states, `describe`
and `explain` belong in it, and the two qualitative cells belong with the six
already excluded. Only `seasonal` escapes, because the tool estimates the
period while the key uses the generator's true one.

Five further defects, all reproduced:

| # | defect | evidence |
|---|---|---|
| 1.2 | the `seasonal` negation regex misses common negations, so the slot scores phrasing | "does not exhibit any seasonal patterns" → scored **yes**; harness scored "yes" on 99/100 items where the key is "no" on 77 |
| 1.3 | **a constant answer beats every arm** on both qualitative cells | `trend_description` 0.630 vs harness 0.620; `event_explanation` 0.5875 vs 0.520 — computed every run, never reported |
| 1.4 | `event_explanation` scores slots the question never asks | question asks event + position; score also counts trend + seasonal, and an unmentioned slot is 0 |
| 1.5 | the `trend` key is mostly noise | `trend_direction` uses two points; for the always-upward `trend` pattern the key says *stable* 8/15 and *decreasing* 1/15 |
| 1.6 | strict `route_correct` counts functionally identical routing as an error | `describe` and `explain` build a byte-identical feature block; at 7B, 94 items → `describe` (0.564), 3 → `explain` (0.667), 3 → `change_point` (0.000) |

Cycle 1's forecasting results, its statistics code and its leakage controls
were checked and stand.

## 2. What C2 changes, and why each change is forced

Every decision below is fixed now, before any C2 data exists.

### C2-1 — The tool supplies EVIDENCE, never a verdict in the key's vocabulary

`series_features` (C2 mode) returns only numbers:

| field | what |
|---|---|
| `slope_per_step`, `slope_se`, `slope_t` | OLS fit over the whole window |
| `best_lag`, `acf_at_best_lag`, `acf_at_lag_1` | autocorrelation evidence, **detrended** (A2) |
| `max_abs_step`, `max_abs_step_index`, `step_over_noise` | largest one-step change, where it is, and its size against the noise floor |
| `max_abs_step_frac` | that index as a fraction of the window |
| `max_local_peak_robust_z` | robust z of the largest interior local maximum, the set `count_spikes` scans |
| `n_points`, `mean`, `std`, `noise_scale` | scale context |

The authoritative list is `systems/tools.C2_EVIDENCE_FIELDS` (15 fields);
`tests/test_saturation_guard.py` asserts the emitted dict matches it exactly.

No field is a member of any slot's class set. The model must apply a
threshold and a decision to reach `increasing` / `yes` / `spike` / `early`.
**That inference step is the thing the benchmark is supposed to measure**, and
cycle 1 performed it inside the tool.

The model also keeps the series: C2 sets `assist_qualitative` so that the
feature block is added **beside** the series, never instead of it. A harness
whose model cannot see the data is not an architecture under test.

### C2-2 — The answer key's DECISION RULE is not in the tool

- **`trend`**: OLS slope over the **whole window** with a two-sided t-test at
  α = 0.05 → `increasing` / `decreasing` / `stable`. This replaces the
  two-point rule of §1.5. The tool reports `slope_per_step`, `slope_se`
  **and `slope_t`** as evidence; the key's DECISION — the comparison
  against `TREND_T_CRIT` — exists only in `src/ground_truth.py`.
  *(Corrected by A5: an earlier draft said the tool does not apply the
  test, which read as though it does not report `t` either. It does,
  and §7/A4 records that one comparison is a weak inference step.)*
- **`event`**, **`position`**, **`seasonal`**: see amendment **A9**, which
  replaces the computed keys described in this section with the generator's
  record. The tool reports `level_shift_over_noise`,
  `max_local_peak_robust_z`, `max_abs_step_frac` and the detrended
  autocorrelation as evidence; none of them is the key.

  ⚠ **Superseded by A9 (2026-11-10, pre-data).** This section's computed
  keys were shown by reviews #2-#4 to be reproducible from the tool's own
  numbers on 100% of items. The text above is kept as the record of what was
  first declared.

### C2-3 — Closed-set multiple choice, not regex over free text

The question states the options for each asked slot and requires one line per
slot in a fixed format (`trend: increasing`). Scoring parses that format
only. This removes the negation regex (§1.2) and the ambiguity penalty that
punished hedged prose (§1.4) in one move, and it makes "did not answer"
distinguishable from "answered wrongly" without an LLM judge.

Free-text prose is still collected and stored, so the cycle-1 style of
extraction can be reported as a **secondary**, descriptive comparison.

### C2-4 — Score only what the question asks

`SLOTS_FOR_TYPE` in C2: `trend_description` → (`trend`, `seasonal`);
`event_explanation` → (`event`, `position`). The question text and the scored
slots are generated from one source, and a test asserts they agree.

### C2-5 — A fourth arm: `bare_features`

Series **plus** the same numeric evidence block, no routing, no tool
selection. This separates two mechanisms cycle 1 could not: *does the
computation help*, versus *does the harness scaffolding help*. Declared as a
**secondary** contrast, outside the Holm family.

### C2-6 — Classical baselines are first-class, not a footnote

Forecasting reports, on the same items and the same MASE denominator:
`last_value`, `window_mean`, `seasonal_naive_estimated_period`,
`seasonal_naive_true_period`. Cycle 1 computed none of these and the review
had to add them; they showed the bare LLM is **indistinguishable from
seasonal naive** (0.511 vs 0.550, p = 0.24) while the harness beats it
(p ≈ 3e-7). That is the real forecasting result and it must be in the table.

### C2-7 — The constant-answer baseline is reported in every qualitative table

`best_constant_score` is already computed into `screening.json` every run.
C2 prints it beside every qualitative contrast.

**New decision rule, declared now:** a qualitative cell in which the best
constant answer scores at least as high as both arms is reported as
**UNDECIDABLE** and leaves the confirmatory family. Had this rule existed in
cycle 1, both qualitative cells would have left it (§1.3).

### C2-8 — Routing is measured twice, and the router is told what the tools do

- `route_correct` (strict, unchanged) and `route_equivalent` (functional) are
  reported side by side. The equivalence classes are declared here:
  **{`describe`, `explain`}** are interchangeable for both qualitative types,
  because they build a byte-identical feature block and differ only in a
  `[task:]` header. No other class is declared.
- The classify prompt gains one-line tool descriptions. Cycle 1 gave bare op
  names, and at 32B 99/100 `localization` questions went to `pct_change` —
  routing accuracy was partly a measure of how guessable the names were.

### C2-9 — Spikes stay inside the history window

`_inject_spikes` ran over history **and** horizon, adding unpredictable error
to both arms' forecast targets. In C2 injection is confined to the history
window. This changes the generated data, so C2 uses a **new pool** and C2
numbers are not comparable to cycle 1's item by item.

### C2-10 — An automated saturation guard

`tests/test_saturation_guard.py` fails the build if, for any cell in the
confirmatory family, the harness's feature block can reproduce the key's slot
label at or above **0.95** on a generated pool. This is the test that would
have caught §1.1 before a single GPU hour was spent. It sits beside the
existing `tests/test_leakage.py` import check. It does **not** extend that
file: the no-key-decision check lives in the saturation guard, because
cycle 1's `series_features` legitimately calls those functions and a blanket
import rule would have to exempt it. The guard asserts
that no module under `systems/` calls a key **decision** function.

## 3. Hypotheses and predictions for C2

Same three axes. Confirmatory contrast `bare_llm` vs `harness`, two-sided
Wilcoxon signed-rank on item-level pairs, bootstrap CIs, effect floor 0.02,
Holm with **m declared by name** as the number of cells surviving §C2-7.

**The family is declared as the three cells of cycle 1** — `forecasting`,
`trend_description`, `event_explanation` — *conditional on* each surviving the
constant-answer rule. If only `forecasting` survives, m = 1 and that is the
result, not a fallback.

| | prediction, stated so it can fail |
|---|---|
| **H2-C2** | The harness beats the bare arm on `forecasting`. On the qualitative cells, with verdicts removed from the tool, the gap **shrinks substantially** against cycle 1's +0.350 / +0.245 and may vanish. A gap that survives is evidence for the architecture; one that does not is evidence cycle 1 measured the key. |
| **H1-C2** | *(Run on 300 series, three cells - amendment A15.)* The gap widens with history length. **Mechanism declared in advance:** cycle 1's widening came from the forecaster improving with context (+0.0498, p = 1.9e-06), not the bare arm degrading (−0.0233, p = 0.87). C2 predicts the same asymmetry and reports the two arms' changes **separately**, so "the gap widens" cannot be claimed without saying which arm moved. |
| **H3-C2** | The gap shrinks as the backbone strengthens. **Endpoints declared now: 3B and 72B.** 0.5B is excluded as a weak endpoint because it cannot operate the architecture (routes `trend_description` correctly on 7.2% of items and loses to the bare arm). This is declared before the run, not chosen after. |

Additional declared reporting, not tests: `route_correct` beside
`route_equivalent`; each arm's absolute score beside every gap; the constant
and classical baselines in every table.

## 4. What C2 does NOT change

- The statistics (`holm` with explicit m, `paired_compare`, `item_gap`) —
  checked and correct.
- The leakage architecture (`InferenceSample`, the explicit `oracle_op`
  channel, `QUERY_OP` runner-side only).
- The call cache, the fixed pool discipline (invariants I1/I2), the fixed
  MASE denominator, and both DiD sign conventions.
- Cycle 1's archived results and verdicts.

## 5. Declared limitations C2 does not remove

1. **Synthetic data only.** The forecasting cell is portable (MASE against a
   naive denominator); the qualitative keys are not, because they are defined
   by generator-adjacent predicates. A real-data forecasting cell (M4 / ETT)
   is the largest remaining upgrade and is **not** in C2.
2. **One forecaster.** `amazon/chronos-t5-small` alone stands in for "the
   harness" in the forecasting cell. A second forecaster is needed before any
   claim about forecasters in general.
3. **One model family.** Qwen2.5 only, pending a Hugging Face token (cycle
   1 deviation D9). C2 cannot separate capability from Qwen's recipe either.
4. **No human annotation.** Cycle 1's D7 is still uncollected. C2's
   multiple-choice scoring removes the extraction ambiguity that made D7
   urgent, but not the question of whether the *keys* are right (D7 Task B),
   which §1.5 showed was a real failure on `trend`.
5. **C2 is not a replication of cycle 1.** New pool, new keys, new answer
   format. It measures the same hypotheses with an instrument that cycle 1's
   own criteria say is admissible; it does not re-measure cycle 1.

## 6. Status of this document

Written before any C2 code. Any later amendment is appended below with a
date, and an amendment made after C2 data exists is marked ⚠ POST-HOC, as
cycle 1's §5 requires.

⚠ **A1, A2 and A4 were restated on 2026-11-10 after review #3.** Their
original figures were computed on `series[:L]` — the FIRST L points — while
`Sample.window(L)` returns `series[max_history - L : max_history]`, the LAST L.
The direction of every finding survived, but the numbers did not, and the
figures below are the corrected ones.

A1–A15 were written before any C2 model call, so none of them can have been
informed by a C2 result. Most respond to independent reviews
([`REVIEW2_2026-11-10.md`](REVIEW2_2026-11-10.md) and two later ones) whose
claims were reproduced against the archived pool before anything was changed.

A16 is different and says so: it was written **after** the Qwen run of
2026-11-13, with its results known (they are in `FINDINGS_C2_2026-11-13.md`),
but **before any call to the six models it adds**. It changes nothing in the
Qwen analysis; its predictions about the new models are therefore informed
by the Qwen results, which is stated in the row itself.

A19 is **post-hoc**: it was written after the six-model run of 2026-11-14,
with those results known, and changes one setting that produced no result
under A16. The row says what was known and how the setting must be labelled.

| # | date | change | why | effect |
|---|---|---|---|---|
| **A1** | 2026-11-10 | §C2-3: the stated `event` rule is **withdrawn as written**. It said "the largest one-step change exceeds 4x the noise scale"; the key compares the mean of the 4 points after the step with the 4 before (`level_shift`). | A model following the instruction exactly is marked wrong. Measured on the archived pool with the correct window, of the items the key calls `abrupt change` the stated rule agrees on **0/44** (L=32), **0/32** (L=64), **0/20** (L=128), **4/18** (L=256), **10/12** (L=512) — i.e. zero agreement at three of the five lengths. | **Resolved on 2026-11-10, the way review #3 asked for.** The evidence block now reports `level_shift_over_noise` — the key's own quantity, |mean(4 after) − mean(4 before)| against the noise scale — and the prompt states that rule. Agreement with the key goes from 0/44, 0/32, 0/20, 4/18, 10/12 to **100% at every length**. The one-step rule is withdrawn and `tests/test_saturation_guard.py` now asserts it has not come back. ⚠ Side effect: the `event` slot is now fully reproducible from the evidence too, so A4's saturation applies to all four slots rather than three. Making the metric honest made the saturation complete — which is the §7 argument, not a counterargument to it. |
| **A2** | 2026-11-10 | §C2-2: the `seasonal` test and the ACF evidence are computed on the **detrended** window (`detrend`, `best_lag_acf_detrended`). | The raw ACF of a trending series is high at every lag, so the slot measured "is there a trend". Un-detrended, the key called **28/30** `trend` series and **28/28** `abrupt_change` series seasonal at L=512, and the contamination grew with window length — so the slot changed meaning along H1's own sweep axis. | **A real gain.** Scored against the generator's own `seasonal_period > 1`, F1 rises from 0.70 to **0.84** at L=512 and from 0.84 to 0.88 at L=32; `trend` and `noise` contamination goes to 0/30 and 0/28 at every length. **A level shift still reads as seasonal (22/28 at L=512)** — linear detrending cannot remove a step, and no autocorrelation rule can, because a step genuinely is autocorrelated at long lags. Irreducible for a computed key; see §7. |
| **A3** | 2026-11-10 | §C2-3: the multiple-choice parser accepts markdown, enumeration and dash separators, and **rejects negated options**. | Review #2 claim 6, all reproduced: `**trend**: increasing`, `1. trend: stable` and `event - spike` read as no answer, and **`event: no spike` read as `spike`** — cycle 1's negation defect reappearing inside the scorer meant to remove it. Reading weak models' formatting as "no answer" biases H3 specifically. | Fixed and tested (`tests/test_c2_scoring.py`, 24 tests). A negated option on a two-class slot resolves to the other class; on a three-class slot it is `absent`, never guessed. |
| **A4** | 2026-11-10 | §C2-10 is now **implemented as declared** (`tests/test_c2_saturation_rate.py`) and **it does not pass**. | The guard in `tests/test_saturation_guard.py` checks that the evidence block contains no scoring vocabulary, which is weaker than the declared 0.95 reproduction threshold. Review #2 claim 4 predicted the declared version would fail. | It fails, and after A1 it fails on **every slot**: a rule-applier reading only the evidence and the stated rules reproduces the key on **trend, seasonal, position and event at 100%** at L=32, 128 and 512, on both the generator-built pool in the test and the archived pool. (Before A1 the `event` slot sat at 64–83% only because the stated rule was the wrong one.) Marked `xfail(strict=True, raises=AssertionError)` so it flips to a hard failure the moment a redesign fixes it. **C2's qualitative cells are saturated in effect and are not admissible as declared.** |
| **A6** | 2026-11-10 | A seasonality **peak** requirement was implemented, measured and **rejected**; the key stays on the detrended maximum. `seasonal_peak` and `acf_profile` remain in the module as evidence and as the record. | Review #3 observed that a lag-2 maximum is persistence, not a cycle: 18/82 `yes` labels at L=32 and 22/98 at L=512 sat at lag 2. The observation is correct. | **The remedy is worse than the defect.** Against `seasonal_period > 1`, requiring a local maximum takes precision to 1.00 at L≤128 but recall to 0.59–0.69, so F1 falls at every length (0.88→0.78 at L=32, 0.84→0.57 at L=512). Reverted, with the measurement recorded rather than the change kept. |
| **A7** | 2026-11-10 | Parser v3: the slot name may be wrapped in backticks/asterisks; `abrupt-change` and `abrupt_change` normalise to `abrupt change`; negation is scoped to its **clause**; a denied two-class slot resolves to the other class; two lines naming different options is `ambiguous` rather than the first one. | Review #3 claim 3, six further misreads, all reproduced. `position: not early, late` read as no answer because a negation anywhere earlier cancelled every later option. | Fixed; 33 tests in `tests/test_c2_scoring.py`. Review #3's reading of `seasonal: not seasonal` as `no` is adopted over the earlier `absent`: on a two-class slot, denying the slot IS the other class. |
| **A8** | 2026-11-10 | Recorded, NOT fixed: the `trend` key leaves almost no room at short lengths. | Review #3 claim 4, reproduced: the key says `stable` for **93%** of items at L=32 and **92%** at L=64, so the best constant answer scores **0.760** and **0.790** on `trend_description`. | By §C2-7's own rule that cell is **undecidable at L=32 and L=64**. Fixing it needs steeper or better-mixed generated trends, i.e. a new pool, which belongs with the §7 decision rather than ahead of it. |
| **A9** | 2026-11-10 | **§7 decided: route (A), generator-truth keys, with events placed relative to the END of the history.** The qualitative key is now the generator's record of what it injected (`src/synthetic.py::generate_base_series_c2`, `src/ground_truth.py::qualitative_truth_c2`): `trend` = injected drift direction or `stable`; `seasonal` = whether a cycle was injected; `event` = the injected spike / level shift if window(L) contains it, else `no major event`; `position` = the third of the window holding it, **scored only when there is an event**. The pool is a balanced factorial (3 trends x 2 seasonal x 3 events x 2 noise types, shuffled); one event at most, drawn inside the last 128 points of history; seasonal periods 6/8/12 so even L=32 holds 2.6 cycles; nothing injected into the forecast horizon (C2-9). The answer format and the definition of every option are in the **question text**, identical for every arm; the first draft's per-type rules block and answer template (`QUAL_RULES`, `QUAL_MC_PROMPT`) are removed. The computed rules survive only as the reference rule-applier (`qualitative_computed_c2`). | Decision taken by the project owner on 2026-11-10 after reviews #2-#4, all of which reproduced the same defect: a key computed from the window is reproduced by any tool that computes the same statistics (100% on trend, seasonal, position; 100% on event once A1 unified the rule). Review #4 also showed cycle 1's level shifts sat at index 157-360 of 512, i.e. inside the window on 0/14 series at L=32, 64 **and 128**. | **C2-10 now passes without an xfail** (`tests/test_c2_saturation_rate.py`). The same rule-applier scored against the new key, 600 generated series at L=128: trend 0.73, seasonal 0.77, event 0.74; cell score 0.75 (`trend_description`) and 0.82 (`event_explanation`), against 1.00 before. On the real pool (`data/c2_n100`, seed 20261111) the best constant answer scores 0.43 and 0.46 at L=128, and classes are balanced (trend 33/33/34, seasonal 48/52, event 36/31/33). **Three things are declared with it, before any C2 run:** (1) the C2-10 gate is the rule-applier's **cell score** and the `trend`, `seasonal`, `event` slots, each < 0.95; `position` is **reported, not gated** - it is a localisation sub-question that the largest step answers ~95% of the time once an event exists, half of one cell, scored only on event items. (2) **H1's confirmatory family is `forecasting` alone (m = 1)**: the qualitative key depends on whether window(L) contains the event (at L=32 only ~20% of events are inside it and a constant answer scores 0.91 on `event_explanation`), and the injected trend is below detectability at L=32, so those cells are run at every L and reported descriptively. H2 and H3 run at L=128, where every event is inside the window. (3) The saturated numeric cells are unchanged, but `counting` has at most one injected spike per series in this pool; it is descriptive and was never in the family. |
| **A10** | 2026-11-10 | Parser v4 and the remaining review-#4 items. | Four misreads were still open: `seasonal: non-seasonal`, `event: no major events`, `trend: stable, not increasing or decreasing`, `seasonal: yes, there is no doubt`. | Fixed and tested: `or` no longer ends a negation's scope; plurals are tolerated; on a yes/no slot a leading option settles it. `position: none` (what the question tells a model to write when there is no event) is read as no answer and is not scored. The C2 clause shown to every arm is hash-pinned in `tests/test_prompts.py` in place of the removed `QUAL_RULES` pin. |
| **A11** | 2026-11-10 | The declared C2 pieces are now **wired**: `data.design: c2` in the config selects the generator, the prompts, the scorer and the analysis together. Four arms (`bare_llm`, `bare_features`, `harness`, `harness_oracle`); described tool menu for the router (C2-8); `route_equivalent` beside `route_correct`; classical forecasting baselines on the same items and denominator (C2-6); the constant-answer rule deciding the family, with m = the number of surviving cells (C2-7); each arm's own change printed beside every H1/H3 contrast; H3's endpoints read from `analysis.did_endpoints` = 3B and 72B. Configs: `configs/c2_h1.yaml`, `c2_h2.yaml`, `c2_h3.yaml`. | Nothing could run before this. | Verified on the mock backbone only (plumbing, no result): all three runners complete end to end; 244 tests pass; cycle 1 is untouched - its templates are hash-frozen, its manifests carry no `design` key so no archived pool is rebuilt, and `scripts/verify_findings.py` still reports 205 checks, 0 failures. **No C2 model call has been made.** |
| **A12** | 2026-11-11 | Pre-run self-check of the wired code, before any C2 model call. Parser v5: a `slot: option` statement is also read mid-line (after whitespace, comma, semicolon or bracket, with `:` or `=` only) and with quotes or angle brackets copied from the question; the question's own option list echoed back is skipped rather than scored. Job scripts: eight `slurm/*.sh|sbatch` files had CRLF line endings in the working copy and were restored to LF; `slurm/submit_all.sh` takes `TSARCH_H3_CONFIG`, gives the 72B 140G, and skips rungs the config does not declare. | The mock always replies in the clean two-line form, so nothing had exercised how a real model deviates. Of 47 plausible readings, 10 were wrong - every quoted or one-line answer read as "no answer" - which would have scored formatting as ignorance, hardest on the weak rungs H3 depends on. Separately, the documented `--array` submission would have put the 72B on one GPU. | Fixed and tested (`tests/test_c2_scoring.py::test_realistic_model_replies`). A null and a positive control were run through the full C2 pipeline at n=100 on the mock: with the evidence block made useless the qualitative contrasts are +0.040 and -0.050, both CIs covering zero, no win; with it made useful they are +0.115 and +0.110, both wins after Holm. **So the detection floor of the qualitative cells at n=100 is roughly 0.10**: a true harness advantage much smaller than that will read as "not detected", which must be written as such and never as "no effect". |
| **A13** | 2026-11-11 | Recorded, not fixed: the `event` class is balanced only for **L ≥ 128**. `c2_event_in_window` correctly relabels an event that falls before the window start as `no major event`, but with `event_zone: 128` the containment RATE moves with L, so the class distribution moves with it. | Measured on 360 freshly generated C2 series at `max_history=512`: events are inside the window 57/240 at L=32, 120/240 at L=64, and 240/240 from L=128 up. The `event` slot is therefore 6% / 9% / **84%** (spike / abrupt change / no major event) at L=32 and 17% / 16% / 67% at L=64, against a flat 33% / 33% / 33% at L ≥ 128. | **H2 and H3 are unaffected — both run at L=128, where the design is exactly balanced.** For H1, the best constant answer on the `event` slot is **0.84 at L=32** and **0.67 at L=64**, so by §C2-7's own rule that cell would be undecidable there. H1's qualitative cells are already **descriptive only** (A9), so no confirmatory claim is affected, but a descriptive `event_explanation` row read across L would partly track the containment rate rather than the arms. **The containment rate and the per-L constant baseline must be printed beside that row**, and the cell is not to be described as improving or degrading with L. Raising `event_zone` to 32 would balance every length but would force every event into the last 32 points, flattening `position` to `late` at long L — a worse trade, so it is not taken. |
| **A14** | 2026-11-11 | Recorded, not fixed: the `trend` slot is **unanswerable at L=32 and L=64**, for the same structural reason as A13 and on the other cell. | The C2 generator specifies the drift as a rise of 1–3 noise sd per `event_zone` (128) points, so a 32-point window contains only 0.25–0.75 sd of total rise — below what a slope test on 32 noisy points can resolve. Measured on 360 generated series with the strongest simple detector (a fitted-slope t test, which is also what the evidence block reports): accuracy **0.36 at L=32** and **0.45 at L=64** against a best-constant line of **0.33**, rising to 0.71 at L=128, 0.79 at L=256 and 0.86 at L=512. At L=32 it returns `stable` on 77% of items regardless of what was injected. | **H2 and H3 are unaffected** — both run at L=128. For H1 this means **both** qualitative cells, not just `event_explanation` (A13), are uninformative at L=32 and L=64: an apparent improvement across L in `trend_description` would be the slot becoming answerable, not the arms getting better. H1's qualitative cells are already descriptive only (A9), so no confirmatory claim is affected, but **the per-L constant baseline must be printed beside both descriptive rows, and neither cell may be described as improving or degrading with L.** Not fixed because the pool is fixed across L (invariants I1/I2), so the drift cannot be scaled per window; steepening it enough to resolve at L=32 would need 16–48 sd over the full series, which would swamp the seasonal and event components the other slots depend on. |
| **A15** | 2026-11-13 | **Forecasting headroom.** (1) Generator: seasonal periods **12 / 24 / 48** (were 6 / 8 / 12) and cycle amplitude **1.5-5 noise sd** (was 1-3); `C2_GENERATOR_VERSION = 2` is written into the pool manifest so an older pool is rebuilt, never reused. (2) **H1 runs on its own pool of 300 series** (`data/c2_h1_n300`, same seed and generator) with only the three family-type cells - `forecasting`, `trend_description`, `event_explanation`; the six tool-saturated numeric cells are dropped from H1 only and still run in H2 and H3. H2 and H3 are unchanged at 100 series, nine cells, one shared pool. | A13 and A14 measured, per slot and per L, whether the window shows what the key says. Nobody had asked that of `forecasting`, the one cell carrying a confirmatory claim in all three hypotheses. Measured on the first C2 pool with a forecaster told the true period and event location (a ceiling, not a competitor): last value 0.430, true-period seasonal naive 0.497, ceiling **0.559** at L=128 - the score is `1/(1+MASE)` against a seasonal-naive denominator, so the whole usable range is narrow. Worse for H1: with every cycle fitting inside 32 points the ceiling rose only **+0.033** from L=32 to L=512 (dz 0.41), which bounds H1's single confirmatory contrast. The cause was A9's own choice of short periods, made so the seasonal label would not depend on L. | On the A15 generator (360 series): ceiling 0.493 / 0.562 / 0.573 / 0.575 / 0.576 at L = 32 / 64 / 128 / 256 / 512, so the L=32 to L=512 change is **+0.083** (dz 0.55); room at L=128 is +0.095 over estimated-period seasonal naive and +0.130 over last value. At n=300 H1's paired minimum detectable effect is dz 0.16, against 0.28 at n=100. Pinned by `tests/test_c2_forecast_headroom.py`. **What it costs, declared:** (a) `seasonal` joins `trend` and `event` in being fully answerable only from L=128 up - a period-48 cycle is detected 53% of the time at L=32, 97% from L=64 - so all of H1's qualitative rows are descriptive, as A9 already ruled; (b) H1 no longer replays H3's L=128 cell from the cache and is about 19,500 calls; (c) **the ceiling's gain is front-loaded: +0.07 of the +0.08 arrives between L=32 and L=64.** H1's contrast is therefore mostly "32 points versus more", and the whole curve must be read, not only its ends; (d) even so, a real forecaster will gain less than the ceiling, so a null on H1 is to be written "this design could not detect an effect of plausible size". **Figures restated for this generator** (they supersede the ones quoted in A9, A13 and A14, whose rows are left as written): rule-applier against the key at L=128 - trend 0.66, seasonal 0.82, event 0.69, position 0.94; cell score 0.74 and 0.78 (C2-10 passes). Event containment 57/240, 120/240, 240/240 at L = 32 / 64 / 128 (unchanged). Fitted-slope trend accuracy 0.36 / 0.44 / 0.66 / 0.78 / 0.87. Best constant answer at L=128: 0.43 and 0.46 on `data/c2_n100`, 0.42 and 0.44 on `data/c2_h1_n300`. 275 tests pass; no C2 model call has been made. |
| **A16** | 2026-11-14 | **H2 replicated on six more models, five vendors, declared before any of them has been called.** Models: `openai/gpt-oss-20b` and `openai/gpt-oss-120b` (reasoning models; reasoning effort **medium**, plus `gpt-oss-120b` at **high** as a declared second setting), `ibm-granite/granite-3.3-8b-instruct`, `microsoft/phi-4`, `mistralai/Mistral-Small-3.2-24B-Instruct-2506`, `allenai/OLMo-2-0325-32B-Instruct`. Configs `configs/c2_h2x_<model>.yaml` copy `c2_h2.yaml` byte for byte except the backbone, output directory and execution settings: same 100 items per cell, same seed, four arms, scoring, analysis and C2-7 rule. Temperature 0; `max_tokens` 1024 for ordinary models, 8192 (medium) and 16384 (high) for gpt-oss so reasoning is not cut off; only the final answer is scored (`models/providers.py::final_answer_text`). Execution only, no effect on any result: items are answered 8-16 at a time (`inference.workers`; `tests/test_parallel.py` shows the rows are identical to a sequential run), and each model is probed with three real prompts before its run (`scripts/probe_server.py`); a model whose replies are not in the expected format is not run. | The Qwen-only result could be a property of one vendor's training recipe (limitation §5.3, and cycle-1 deviation D9). The candidates were every non-gated, permissively licensed instruct model that fits one H200, from distinct vendors; Gemma-3 and Llama-3.3 were excluded only because their downloads need an access request. gpt-oss is included because it reasons before answering, so it can compute statistics the other models must estimate - the strongest available test of whether a bare LLM can match the tool. | **How it is reported.** Each model is a separate replication of H2 with its own Holm family (m = 3, after C2-7), reported beside - never pooled into - the Qwen result. `scripts/c2_cross_model.py` lines all models up descriptively, without correction across models. **Predictions, stated so they can fail** (informed by the Qwen results, so they are tests of replication, not of discovery): (1) on `forecasting` the harness beats the bare model for every model; (2) for every non-reasoning model the bare model does not beat estimated-period seasonal naive; for gpt-oss this is an open question, not a prediction; (3) on the two descriptive cells `harness` and `bare_features` are within 0.05 of each other (the routing step adds nothing beyond the evidence); (4) with the evidence block, models over-answer `abrupt change` on events, as Qwen did. A model on which the bare arm beats the harness on forecasting counts against the main claim and will be reported as such. Not included: the evidence-block ablation (FINDINGS_C2 §10, item 1), which remains a separate, later amendment. |
| **A17** | 2026-11-14 | **Human baseline on the descriptive cells, declared before anyone grades.** Three graders (the three project members, A/B/C) each answer **all 200** `trend_description` and `event_explanation` items of the H2/H3 pool at L=128 - the exact items every model answered - from a **plot** of the same 128 values and the **same question text, word for word**, choosing one option per asked slot (`position` may be `none`, as the question allows). Kit: `annotation_c2/`, built by `scripts/c2_make_annotation_kit.py` (kit hash `31682a166bd6edbb`); each grader has their own page and their own item order; the pages contain no key, model answer or generator setting. Answers are scored by the **same scorer as the models** (`score_qualitative_mc`) and analysed by `scripts/c2_analyse_annotation.py`. | Reviews and FINDINGS_C2 §9 left open how well a person does on the same items, and whether every item is answerable: the key is what the generator injected, which may not be visible (A13-A15 measured visibility with detectors only). Without a human figure, 0.6 for a model cannot be called good or bad. Cycle 1's D7 specified three annotators and was never run. | **Reported, in this order:** (1) agreement between graders per slot, Krippendorff's alpha (primary) beside raw agreement; (2) each grader's and the majority vote's score; (3) humans beside every model arm on the same items, paired, with bootstrap CIs and raw p, uncorrected; (4) items the majority got wrong, by key class and noise type, and model scores on those items. **Predictions:** (1) every grader beats the constant answer (0.43 trend, 0.46 events); (2) the majority vote scores at least as high as the best model arm on each cell; (3) alpha >= 0.667 on `trend` and `event` (D7's threshold). **Consequence fixed now:** if alpha < 0.667 on a slot, the definitions in the question are not reliably understood by people, and every model result on that slot is reported as resting on an ambiguous question. **Declared limitations:** the graders are the authors - they know the design and have seen aggregate model results, though not the key for any item; they must not open `_private/`, the data, or the archive while grading (honour system, stated in `INSTRUCTIONS.md`). They see a plot, models saw digits; this is the natural presentation for each and is not equalised. |
| **A18** | 2026-11-14 | Execution-only changes to the A16 run, recorded before it starts. (1) `models/providers.py` retries a request that failed for a temporary reason - time-out, dropped connection, HTTP 408/429/5xx - up to 4 times with growing waits; any other error fails at once. (2) Every billed call's record in `calls.jsonl` carries `meta.call` (`finish_reason`, token count, `empty_answer`), and a cell's log reports how many calls hit `max_tokens`. (3) `slurm/c2_run.sbatch` starts each new unit's forecast cache from the Qwen run's stored forecasts (`$TSARCH/cache/c2_mid/forecasts.jsonl`). | (1) One failed request among ~3,000 ended a model's whole step and sent it back to the queue. (2) A reply cut off inside the reasoning is scored as unanswered, which is correct, but without the record it could not be told apart from a model that answered nothing. (3) The harness's forecast is the forecaster's output, so it must be the same numbers for every model; recomputing on a different GPU type could change the last digits. | None on any result. A retry re-asks the same question at temperature 0; none of the three enters a cache key or a score; a stored forecast is keyed on the exact window and forecaster settings, so it can only be reused for the identical series. A16 adds one reporting rule: for the gpt-oss models, the number of truncated replies is reported beside the coverage column. |
| **A19** | 2026-11-14 | **POST-HOC, and labelled so. `gpt-oss-120b` at reasoning effort high: `max_tokens` 16384 -> 32768** (`serve.max_model_len` 24576 -> 40960; execution only: `timeout_s` 1800 -> 3600, `inference.workers` 16 -> 8). **And the pre-run probe no longer treats a reply cut off at the token limit as a format failure:** it prints a warning, and stops the run only if *every* probed reply was cut off or a reply is malformed (`scripts/probe_server.py`). Nothing else changes: same items, arms, scorer, analysis, and the other six configurations are untouched and are not re-run. | Under A16 this setting never ran. On the probe's forecasting prompt the model used all 16,384 output tokens reasoning and returned no answer, and the probe, as first written, stopped the run after 4 minutes. That conflated two things A16 keeps apart: a reply *not in the expected format* (the model is not run) and a reply *cut off before its answer* (scored as unanswered, counted in coverage). The medium-effort runs went ahead with 6% and 11% of replies cut off in exactly that way, so refusing the high-effort setting over one such reply was inconsistent with the declared rule. | **What was known when this was written:** the results of the six finished A16 replications, and the single probe reply above. **No scored item exists for this setting**, so the change cannot have been chosen to move a result of its own. It is still post-hoc with respect to the other models and must be reported as such: 'declared second setting, not runnable at the declared 16,384-token limit (A16); run at 32,768 under A19'. `max_tokens` is part of the cache key, so nothing from the 16,384 attempt is replayed. A16's predictions apply unchanged. If the probe stops the run again at 32,768 (all three replies cut off), the setting is reported as not runnable and the limit is **not** raised a second time. Replies cut off at 32,768 are scored as unanswered and their count is reported beside the coverage column, as for the other gpt-oss runs. |
| **A5** | 2026-11-10 | Documentation corrections. | Review #2 claim 7: `models/prompts.py` cited a `tests/test_c2_scoring.py` that did not exist; §C2-1's field list did not match `C2_EVIDENCE_FIELDS`; §C2-2 said the tool "does not apply the test" while the evidence reports `slope_t`; the stated-rules decision was in the code comments but not in this document. | The test file now exists. The field list below is generated from the code. §C2-2 is corrected: the tool reports `t` as **evidence**, and the key's decision is the comparison against `TREND_T_CRIT`, which lives only in `src/ground_truth.py`. The stated-rules decision is recorded as part of §C2-3 above. |

---

## 7. The qualitative keys - DECIDED 2026-11-10: route (A)

**Decision: generator-truth keys, with events placed relative to the end of
the history.** Recorded as amendment **A9**, before any C2 data exists.

Why this was a decision at all. C2 as first declared did not work for the
qualitative cells: the evidence block plus mechanical rules reproduced the
key exactly, so the cells measured "compare a number to a threshold" for the
harness against mental arithmetic for the bare arm - the same thing that got
six cells excluded in cycle 1. Patching showed diminishing returns (A2 fixed
trend contamination of the seasonal key and left step contamination behind
it), because any key computed from the window is reproduced by a tool that
computes the same thing.

The route **not** taken, (B), was to keep computed keys, make the stated rule
and the key the same function, and report the qualitative cells as
descriptive only. It was smaller, and conceded the confirmatory claim.

What route (A) costs, stated plainly:

1. **H1's qualitative cells are descriptive** (A9, point 2). The key depends
   on which window is shown.
2. **The key is a fact about the generator, not about the visible numbers.**
   A weak injected trend under autocorrelated noise can be genuinely hard to
   see; an item both arms get wrong is a hard item, not a scoring error. The
   rule-applier's 0.73-0.77 is a rough floor for what careful use of the
   evidence achieves, not a ceiling.
3. **Still synthetic.** §5.1 stands: these cells do not port to real data.
