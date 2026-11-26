# Pre-registration — cycle 3: real data

*Written 2026-11-14, before any cycle-3 model call. Amends
[`PREREGISTRATION_C2.md`](PREREGISTRATION_C2.md), which amends
[`PREREGISTRATION.md`](PREREGISTRATION.md). Everything those documents fix
and this one does not reopen stays fixed.*

---

## 0. What cycle 3 is and is not

Cycles 1 and 2 ran entirely on synthetic series. That bought two things —
a known answer key for qualitative questions, and immunity from training-set
contamination on both arms — at the price of the first question any reviewer
asks: *is this just synthetic data?*

Cycle 3 answers it for the one cell where real data has a legitimate answer
key: **forecasting**, where the label is the held-out future.

It is **not** a port of the whole benchmark. Two of cycle 2's three
confirmatory cells cannot move to real data at all, and six more could move
but should not. §2 states which and why. The declared family therefore drops
from m = 3 to **m = 1**.

### What this cycle can overturn

Cycle 2's headline was that the harness beats the bare LLM on forecasting
(0.549 against a seasonal-naive floor of ~0.497). On real data the trivial
baseline is far stronger: declared-period seasonal naive scores **0.5688**
on the cycle-3 pool at the primary horizon. Chronos-t5-small must clear
that, not 0.50. **It may not.** C3-7 pre-registers that outcome as a
possible result.

---

## 1. Numbering

Cycle-1 clauses are `§n`, cycle-2 clauses `C2-n`, cycle-3 clauses `C3-n`.
Where a C3 clause supersedes an earlier one it says so by name.

---

## 2. C3-1 — the declared family is `forecasting` alone, m = 1

| cell | cycle 2 | cycle 3 | why |
|---|---|---|---|
| `forecasting` | in family | **in family** | the label is the held-out future; `1/(1+MASE)` and the classical baselines carry over unchanged |
| `trend_description` | in family | **refused** | see below |
| `event_explanation` | in family | **refused** | see below |
| `statistics`, `counting`, `extremum`, `change`, `correlation`, `localization` | descriptive, a priori saturated | **refused** | see below |

### Why the two qualitative cells are refused, not adapted

Their cycle-2 key is *what the generator injected* (amendment A9). Real data
has no generator. The tempting substitute — derive the key from statistics
(STL, PELT, or the evidence block's own fields) — **recreates cycle 1's
retracted defect exactly**: the harness's tool and the answer key would
compute the same family of quantities, and the cell would saturate by
construction. `TemporalBench` does use rule-based labelling with changepoint
detectors; that is sound for a benchmark whose tool is unrelated to the
labeller, and wrong here.

A real qualitative cell needs multiple human annotators per item —
MMTS-Bench used 10 experts at Fleiss κ = 0.73; the changepoint literature
uses 5 per series. That is a separate project.

### Why the six deterministic cells are refused

They would *work*: their keys are deterministic functions of the visible
window, so they are well defined on any series. But they are a priori
tool-saturated (`src.config.TOOL_SATURATED_OPS`), so the harness scores
1.000 by construction and the contrast measures the bare arm's in-context
arithmetic. Running them on real data costs calls and answers nothing new.

### How the refusal is enforced

Not by convention. `src/real_data.py` raises on any query type outside
`ALLOWED_QUERY_TYPES`, with the reason in the message, and
`Config.validate()` raises before a SLURM array books a GPU.
`tests/test_real_data.py` pins both.

---

## 3. C3-2 — horizon: 24 primary, 8 sensitivity

Cycle 2 used `horizon: 8`. On period-24 hourly data that is **a third of a
cycle**, which is seasonal naive's easiest regime and is why its bar sits so
high there.

**Primary: h = 24** — exactly one declared seasonal cycle for the pool's
hourly series.

*Wording corrected 2026-11-14, before any cycle-3 model call. The first
draft said "the hourly majority of the pool". That is true of
`pool_real_n300` (225 of 300 hourly) but not of the primary pool,
`pool_clean_n300`: 75 hourly (period 24), 150 daily (period 7) and 75
monthly (period 12). For the daily and monthly series h = 24 is about 3.4
and 2 declared cycles. The horizon itself is unchanged; only the sentence
describing it was wrong.*
**Sensitivity: h = 8** — [`configs/c3_sensitivity_h8.yaml`](configs/c3_sensitivity_h8.yaml),
L = 128 only, 32B only. It exists so the horizon choice can be shown not to
drive the result, and to give exactly one point of contact with cycle 2's
numbers.

Keeping h = 8 as primary "for comparability" would be a false economy:
changing the data already breaks comparability. The pool stores 48 points of
target, so both fit; the archive declares 48 for M4 hourly and 56 for NN5
daily, and 56 does not fit, which is why neither archive horizon is used.

The sensitivity arm is a **separate run, not a re-scoring**: the question
text states the horizon, so the arms' prompts and therefore their calls
differ.

---

## 4. C3-3 — clustering: `sample_id` primary, `dataset_id` reported beside it

The pool takes **one window per series**, so an item is a distinct physical
series and `sample_id` is a defensible unit — and it is the unit every
cycle-1 and cycle-2 number was computed under, which is what keeps the
synthetic-versus-real comparison in C3-5 like for like.

But 300 windows drawn from 4 archives are not 300 fully independent
observations: series inside one archive share a sensor network, a sampling
convention and a seasonal structure.

**Declared:** `analysis.cluster_unit: sample_id`,
`analysis.cluster_unit_sensitivity: [dataset_id]`. Both are computed and
printed by `runners/_common.py::clustering_sensitivity`; neither is
substituted for the other.

⚠ With four archives the clustered test has k = 4, below
`paired_compare_clustered`'s `min_n = 10`, so its **p-value is reported as
NaN by construction** and its CI is roughly three times wider. That is not a
failure — it is the honest statement that four archives cannot support a
p-value. If the two units disagree, the write-up must say that the result
depends on treating four archives as four observations.

---

## 5. C3-4 — the pool is content-addressed and declared, not discovered

`data.pool_sha256` is **required** for `data.name: real` and is checked
against the sha256 of `windows.npz` at load. It is also part of
`data_fingerprint()`, so a rebuilt pool cannot collide with cached answers
for the old one.

A paired design across five history lengths and seven backbones is valid
only if every cell saw the same series. A silently regenerated pool would
break that with no error anywhere. Updating the declared hash to match a
rebuilt pool **discards every shard already written under the old one**; the
error message says so.

Declared hashes:

| pool | n | sha256 (first 16) |
|---|---|---|
| `pool_clean_n300` | 300 | `5e7fade71450e8ae` |
| `pool_real_n300` | 300 | `7322149e173d4615` |
| `pool_h2_strata_n240` | 240 | `7d6a2e80d15636bc` |

---

## 6. C3-5 — the 32B rung is included so synthetic ↔ real is paired

Both [`c3_h1_clean.yaml`](configs/c3_h1_clean.yaml) and
[`c3_h3_clean.yaml`](configs/c3_h3_clean.yaml) carry `qwen2_5_32b`, the
backbone cycle 2's H1 and H2 ran on, with `design: c2` so the prompt family
is identical. If real results differ from synthetic ones, that difference is
attributable to the data and not to a newly added model family.

`design: c2` is **required** for a real pool (`Config._validate_real_data`)
for this reason, and because `bare_features` exists only on the C2 path.

---

## 7. C3-6 — H2 is reframed onto series character

On real data every item is a forecasting item, so cycle 2's H2 — "does the
gap depend on what is asked?" — has no arms left to contrast.

What replaces it is the question cycle 2 could only answer post hoc: **does
the gap depend on the series?** At 32B the harness gained **+0.170 on
seasonal series against +0.014 on non-seasonal ones**. That was the largest
conditional effect of the cycle and it was never a declared test. Here it
is one.

```
within stratum s:   d_i = score(harness)_i - score(bare_llm)_i      (paired)
declared contrast:  mean d over WEAK  -  mean d over STRONG     (two-sample)
```

**Strata are fixed in advance** at `STRATUM_CUTS = (0.30, 0.60)` on
`seasonal_strength`, the detrended ACF at the archive-declared period. The
constant lives in `src/real_data.py`, is imported by the builder and the
diagnostics rather than restated, and `tests/test_real_data.py` fails the
build if the copies drift.

⚠ **The contrast is two-sample, not paired, and that is not a detail.** H1
and H3 sweep axes along which the same items appear at both ends
(invariants I1/I2), so a per-item Δ exists. A weak-seasonality series and a
strong-seasonality one are *different series*. `confirmatory_did` raises on
`axis="stratum"`; `runners/h2_series_character.py` uses
`src.stats.two_sample_compare` (bootstrap CI without an equal-variance
assumption, Mann-Whitney p, Cliff's δ).

### ⚠ The confound, and the rule that handles it

Weak-seasonality series long enough for the L = 512 geometry come
overwhelmingly from one archive. Realised composition of
`pool_h2_strata_n240`:

| dataset | weak | strong |
|---|---|---|
| `weather` | 106 | 30 |
| `traffic_hourly` | 7 | 30 |
| `fred_md` | 5 | 30 |
| `nn5_daily` | 2 | 30 |

So the pooled contrast could be reporting a difference between **archives**
rather than between **series characters**.

**Declared rule, fixed before the run:** the runner computes the
within-dataset contrast for every dataset supplying at least
`min_items_per_stratum_for_within_dataset: 20` items in both strata — here
`weather` alone — and prints it beside the pooled number. **Where the two
disagree, the within-dataset version carries the claim.** If the pooled
contrast moves and no within-dataset contrast does, the result is reported
as an archive effect, not a series-character effect. The runner prints
exactly that sentence.

---

## 8. C3-7 — the falsifiable prediction

Measured before any model call (`python data-real/verify_pool.py`,
`pool_clean_n300`, h = 24), mean score `1/(1+MASE)`:

| L | last_value | window_mean | snaive (**declared**) | snaive (estimated) |
|---|---|---|---|---|
| 32 | 0.4724 | 0.4752 | 0.5688 | 0.4943 |
| 64 | 0.4724 | 0.4610 | 0.5688 | 0.4946 |
| 128 | 0.4724 | 0.4269 | 0.5688 | 0.5272 |
| 256 | 0.4724 | 0.3732 | 0.5688 | 0.5609 |
| 512 | 0.4724 | 0.3601 | 0.5688 | **0.5767** |
| **change** | +0.0000 | −0.1151 | **+0.0000** | **+0.0824** |

> **PREDICTION (at risk).** The `harness` arm beats declared-period seasonal
> naive — mean score above **0.5688** at L = 512, h = 24, with a 95% CI on
> the paired difference excluding zero.
>
> **If it fails, cycle 2's headline does not generalise to real data.** That
> is a publishable result and must be reported as the primary finding, not
> explained away. Both outcomes were written down here first.

Two readings that must travel with any H1 number:

* The **declared-period** baseline is **flat in L** — it needs only
  `period` points. So H1 on real data asks: *can the forecaster and the LLM
  each find the period that seasonal naive is simply told?*
* On the **contaminated** pool the estimating baseline gains **+0.2163**
  from L = 32 to L = 512 (0.3299 → 0.5463) — roughly ten times cycle 2's
  entire H1 effect of +0.023. Amendment A15 diagnosed the synthetic pool's
  flat ceiling as the reason H1 had nothing to measure; on real data the
  ceiling moves.

---

## 9. C3-8 — contamination, with the dataset list fixed by name

Synthetic data made both arms immune to training-set overlap. Real public
benchmark data removes that immunity on both sides.

### What was verified

Chronos's own Table 2 (arXiv:2403.07815, Appendix B) partitions its datasets
into pretraining-only, **Benchmark I** (15 datasets, *in* the training
corpus) and **Benchmark II** (27 datasets, *not* in it). Each dataset below
was matched to that table **by series count and series length**, not by
name alone:

| dataset | series × length | Chronos split |
|---|---|---|
| `m4_hourly` | 414 × 748–1008 | **Benchmark I — in training** |
| `electricity_hourly` | 321 × 26304 | **Benchmark I — in training** |
| `traffic_hourly` | 862 × 17544 | Benchmark II — zero-shot |
| `nn5_daily` | 111 × 791 | Benchmark II — zero-shot |
| `weather` | 3010 × 1332–65981 | Benchmark II — zero-shot |
| `fred_md` | 107 × 728 | Benchmark II — zero-shot |

⚠ **This corrects the handoff and the original `MANIFEST.json`**, which
guessed before the tables were read and had it backwards: `traffic_hourly`
was recorded as a documented overlap (it is zero-shot) and `m4_hourly` as
unverified (it is in-domain). `MANIFEST.json` now carries the split, the
evidence, and a note that it was corrected.

### What follows

* **Primary pool: `pool_clean_n300`** — `traffic_hourly`, `nn5_daily`,
  `weather`, `fred_md`, 75 each. Every cross-arm claim runs here.
* **`pool_real_n300` becomes a declared sensitivity arm, not a liability.**
  [`c3_h1_contaminated.yaml`](configs/c3_h1_contaminated.yaml) is
  byte-identical to the clean H1 config except for the pool. H1 is the right
  place to ask, because the contrast is within-forecaster across L and
  contamination hits all five lengths equally. **If H1's answer is the same
  on both pools, memorisation is not driving it.** This consumes no share of
  the Holm family.

### ⚠ What is still NOT controlled

Benchmark II settles contamination for **Chronos**, i.e. the harness arm. It
says nothing about the **bare arm**: these archives are public text and
Qwen2.5 may carry the literal values from pretraining.

The asymmetry runs **against the harness**. If the bare arm has memorised
anything, the measured harness advantage is an *underestimate*. So a
positive result is conservative; a null one may not be read as "the tool
does not help". Only data published after the backbone's training cutoff
would settle it, and that is out of scope for cycle 3. This paragraph must
appear in the write-up.

---

## 10. C3-9 — arms are declared per cell, and the asymmetry is on the record

| run | `bare_llm` | `bare_features` | `harness` | `harness_oracle` |
|---|---|---|---|---|
| H1 L-sweep (32B) | ✓ | **✓** | ✓ | ✓ |
| H1 contaminated (32B) | ✓ | **✓** | ✓ | ✓ |
| H2 strata (32B) | ✓ | **✓** | ✓ | ✓ |
| H3 ladder (7 rungs) | ✓ | **✗** | ✓ | ✓ |
| h = 8 sensitivity (32B) | ✓ | **✓** | ✓ | ✓ |

`bare_features` is dropped from the wide ladder only. On synthetic
forecasting it measured three independent nulls (−0.0093, −0.0173, +0.0033)
and was actively worse at 72B, so paying for it at seven rungs buys little.
**Counter-argument preserved:** those nulls were measured on synthetic
series; real series have genuine structure, so computed features are
plausibly more useful there. Keeping the arm on one backbone is what keeps
that question answerable. `c3_h3_clean.yaml` declares
`extra_contrasts: []` explicitly rather than omitting it, so the asymmetry
is visible in the config.

`harness_oracle` is kept everywhere because it is nearly **free** on a
forecasting-only cell: it is handed the op so issues no routing call, and
its Chronos forecast is bit-identical to the harness's and hits the TSFM
cache. It is also the only thing separating "the forecaster is worse here"
from "the router failed here" — cycle-2 forecasting routing was 0.83 at
0.5B.

### Cost

Billable calls per item on a forecasting-only cell: `bare_llm` 1 +
`bare_features` 1 + `harness` 1 routing + `harness_oracle` 0.

| run | items | cells | billable/item | calls |
|---|---|---|---|---|
| `c3_h1_clean` | 300 | 5 lengths | 3 | 4,500 |
| `c3_h1_contaminated` | 300 | 5 lengths | 3 | 4,500 |
| `c3_h3_clean` | 300 | 7 rungs | 2 | 4,200 |
| `c3_h2_strata` | 240 | 1 | 3 | 720 |
| `c3_sensitivity_h8` | 300 | 1 | 3 | 900 |
| **total** | | | | **≈ 14,800** |

Point H1 and H3 at **one** `--cache-dir`: the call key carries neither the
experiment nor L, so H1's L = 128 cell and H3's 32B rung render identical
prompts and are paid for once.

**Expect a full-price run.** A new pool means a new data fingerprint, so
nothing replays from the cycle-2 cache.

---

## 11. C3-10 — the exclusion rule, fixed before results

`data-real/build_pool.py::degenerate_at_no_L` rejects a series if **any** L
in (32, 64, 128, 256, 512) has `std ≤ 1e-6` or `MAD ≤ 1e-6`.

**Why it is needed.** Real sensor feeds have flat stretches — a traffic loop
reads 0 all night. `gt.local_noise_scale` then bottoms out at
`gt.REL_EPS = 1e-9` and the division explodes. Measured on real Monash data,
`traffic_hourly-T12` at L = 32 rendered
`largest_local_peak_robust_z=35800000.00` into the prompt. It raised
nothing, failed no test and appeared in no coverage column — the item simply
scored, silently, on garbage evidence. The synthetic generator always
injects noise, so this could not fire in cycle 1 or 2.

| | `robust_z` max | `noise_scale` min |
|---|---|---|
| before | 3.6e+07 | 1e-09 |
| after | **104.2** | **1.7e-03** |

**General lesson, recorded for the next sweep design: validate any windowing
invariant at EVERY L, not just the longest.** A shorter history is a
*suffix*, so it can be degenerate while the full window is fine. The first
version of this rule checked L = 512 only.

Cost on the original pool: **82 of 1,708 candidates rejected (4.8%), 74 of
them traffic.** Stated here because a rule fixed before results is
legitimate and the same rule applied after seeing which items looked bad
would be a researcher degree of freedom.

### The defensive floor guard (was open; now implemented)

`series_evidence` emits `robust_scale_usable` (1 or 0), and
`evidence_to_block` refuses to put a scale-divided number in a prompt when
it is 0, printing `unavailable_flat_window` and a one-line explanation
instead.

⚠ `evidence_to_block` output is **inside the call cache key**, so this had
to land before the run and had to be invisible otherwise. The degenerate
branch is purely additive: when `robust_scale_usable` is 1 the rendered
string is byte-for-byte what cycle 2 produced. Verified by replaying the
archived cycle-2 pools — **2,000 (series × L) windows checked, 0 would
render differently** — and pinned by
`tests/test_real_data.py::test_floor_guard_is_invisible_on_an_ordinary_window`.

---

## 12. C3-11 — what is unchanged

Explicitly **not** reopened by this cycle, and inherited as-is:

* the leakage architecture and invariants I1–I5 (I1/I2 now hold by pool
  geometry rather than by generator construction; `tests/test_real_data.py`
  re-pins them);
* `1/(1+MASE)` scoring, the Holm/Wilcoxon/bootstrap layer, the mandatory
  sensitivity grid, the screened-versus-declared double verdict;
* the paraphrase bank and its striding assignment, so routing variance stays
  measurable and `framing_id` stays a usable clustering unit;
* `bare_llm` serialisation — real integer values serialise to ~1,400 tokens
  mean and ~1,830 worst at L = 512, *shorter* than synthetic's ~1,640;
* the a-priori-saturation argument and `DEFAULT_CONFIRMATORY_FAMILY` for
  synthetic runs.

### Config fields that must be ABSENT on a real pool

`data.event_zone`, `data.patterns`, `data.level_range`,
`data.spike_lambda_range`. They do not change the series, but
`data_fingerprint()` hashes them, so a leftover key would define a cache
namespace that means nothing. `Config._validate_real_data` raises on any of
them.

---

## 13. Artifacts and reproduction

The three pools ship with the code and are the ones declared by hash in §5.
**A run rebuilds nothing.**

```bash
python data-real/check_pools.py     # every real config against its pool
python data-real/verify_pool.py --horizon 24     # the tables in §8

python -m pytest tests/ -q          # 335 tests

# plumbing smoke, mock backbone, no GPU - NOT a result
python -m runners.h2_query_type      --config configs/smoke_c3_real.yaml
python -m runners.h2_series_character --config configs/smoke_c3_strata.yaml
```

### ⚠ The pool hash is not reproducible across machines (found 2026-11-14)

`pool_real_n300` was rebuilt from the same raw files on a second machine
before any model call. `windows`, `mase_denominator` and `period` came out
**identical**. `seasonal_strength` and `trend_t` differed in the last
decimal places (at most 6.7e-16 and 1.7e-13: `np.dot` and `np.polyfit` go
through the local BLAS). Those two columns sit in the same `windows.npz`,
so the file hash changed (`58e1e12c…` against the declared `7322149e…`) and
the loader would have refused the rebuilt pool.

Consequences, fixed before the run:

* the builders refuse to overwrite an existing pool
  (`build_pool.py::refuse_overwrite`; `--force` to make a NEW pool, which
  then needs new hashes here and in the configs);
* "reproducible" for these pools means: same series, same windows, same
  denominators, strata equal to ~1e-13 — not the same file hash;
* no item's stratum can change from a difference of that size unless its
  strength sits within 1e-12 of a cut, and none does.

To rebuild from scratch elsewhere (not needed for a run):
`python data-real/fetch.py`, then the two builders with `--force` in an
empty copy of `data-real/`.

### Execution settings (not part of the design)

Three things changed on 2026-11-14 for running on the cluster. None is in a
prompt, a fingerprint, a cache key or the scorer:

* a request that fails for a transient reason (timeout, dropped connection,
  HTTP 408/429/5xx) is sent again, up to four times
  (`models/providers.py`);
* `inference.workers: 8` in the five `c3_*.yaml` configs: eight items are
  sent to the model server at once. `tests/test_parallel.py` checks the
  scored output equals the one-at-a-time output. Cycle 2's Qwen runs were
  one at a time; the six-model H2 replication (C2 amendment A16) used
  sixteen. Set it to 1 to remove the difference;
* each job serves its model on a port of its own (`slurm/pick_port.sh`).

Every run writes `config_used.json` (with `_data_fingerprint`),
`screening.json` **before** any test is read, `analysis.json`,
`sensitivity_grid.csv`, `desc_*.csv` and `raw_results.csv`. On a real pool
`analysis.json` additionally carries `real_pool` (the dataset and stratum
composition, so the per-dataset contamination argument travels with the
numbers) and `clustering_sensitivity`.

`data-real/raw/` is gitignored (re-fetchable, hash-pinned); the pools
themselves **are** tracked, because a silently changed pool breaks the
paired design.

---

## 14. Attribution (required by CC BY 4.0)

> Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., Montero-Manso, P.
> (2021). *Monash Time Series Forecasting Archive.* NeurIPS Datasets and
> Benchmarks Track. arXiv:2105.06643.

Per-dataset citations are in `data-real/MANIFEST.json` (M4 → Makridakis et
al.; NN5 → Ben Taieb et al.; electricity → UCI; traffic → Caltrans PEMS;
FRED-MD → McCracken & Ng).

Chronos split: Ansari et al. (2024), *Chronos: Learning the Language of Time
Series*, arXiv:2403.07815, Table 2.
