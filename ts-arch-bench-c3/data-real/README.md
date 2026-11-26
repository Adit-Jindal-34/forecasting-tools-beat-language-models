# `data-real/` — real time series for the cycle-3 forecasting cell

Length-filtered, stratified pools of real series from the Monash Time
Series Forecasting Archive, built to the same invariants as the synthetic
pools (I1/I2, fixed MASE denominator, one item tracked along the L sweep).

**They cover the forecasting cell only.** Real data supplies a forecasting
label — the held-out future — and no qualitative label at all. See
[§6](#6-what-this-pool-does-not-give-you).

The design that consumes these pools is
[`PREREGISTRATION_C3.md`](../PREREGISTRATION_C3.md).

---

## 0. Three pools, and which to use

| pool | n | datasets | Chronos contamination | use |
|---|---|---|---|---|
| **`pool_clean_n300`** | 300 | traffic, nn5, weather, fred_md | **controlled** — all Benchmark II | **everything cross-arm**: H3, the ride-along, the headline |
| `pool_real_n300` | 300 | m4, nn5, electricity, traffic | **mixed** — m4 and electricity are Benchmark I | H1 only, as the declared contamination-*sensitivity* arm |
| `pool_h2_strata_n240` | 240 | traffic, nn5, weather, fred_md | controlled | the reframed H2: 120 weak vs 120 strong seasonality |

⚠ The split is from Chronos arXiv:2403.07815 Table 2, verified by matching
series count and length per file. It **inverts** the flags the original
handoff guessed: `traffic_hourly` is zero-shot, `m4_hourly` is in-domain.
See `MANIFEST.json`.

⚠ Benchmark II controls contamination for **Chronos**, i.e. the harness arm.
The **bare LLM arm is not controlled** — these archives are public text.
The asymmetry runs against the harness, so a positive result is
conservative and a null one is not interpretable.

---

## 1. What is here

```
data-real/
  README.md            <- this file
  MANIFEST.json        <- provenance, licences, citations, Chronos split
  fetch.py             <- re-download from Zenodo, verify sha256 (idempotent)
  tsf.py               <- Monash .tsf reader (numpy only)
  build_pool.py        <- pool_real_n300          (seed 20261114)
  build_pool_c3.py     <- pool_clean_n300 + pool_h2_strata_n240
  verify_pool.py       <- baseline / headroom / strata diagnostics
                          (--pool NAME --horizon N; default: every pool)
  raw/                 <- the six downloaded .tsf files (~350 MB, gitignored)
  pool_*/
    windows.npz        <- windows (n, 560) + denominators + strata
    meta.json          <- per-item records and the dataset report
```

The loader that turns these into `Sample` objects is
[`src/real_data.py`](../src/real_data.py); `src.pipeline.make_samples`
dispatches to it on `data.name: real`.

## 2. The four datasets, and why these four

Every series in the archive had to clear `L_MAX + H_MAX = 560` points to
support the full L = 32…512 sweep. That immediately rules out the famous
competition sets — M1 (15–150 points), M3 (20–144), Tourism (11–333) and
M4's yearly/quarterly slices are **too short for this design**. What is left
is sensor-and-infrastructure data plus M4's hourly slice:

| dataset | series | eligible | taken | freq | period | length |
|---|---|---|---|---|---|---|
| `m4_hourly` | 414 | 414 | 75 | hourly | 24 | 748–1,008 |
| `nn5_daily` | 111 | 111 | 75 | daily | 7 | 791 |
| `electricity_hourly` | 321 | 313 | 75 | hourly | 24 | 26,304 |
| `traffic_hourly` | 862 | 788 | 75 | hourly | 24 | 17,544 |

All **CC BY 4.0**. **82 of 1,708 series (4.8%) were rejected** by the
exclusion rule in `build_pool.py` — 74 of them traffic sensors. See
[§4.4](#44-the-exclusion-rule-has-to-hold-at-every-l-not-just-at-l_max) for
why that rule is stricter than it first looks.

`75` per dataset so that **no dataset exceeds 25% of the pool**, and n = 300
to match H1's cycle-2 sample size — chosen there (amendment A15) because the
forecasting score has a narrow range and 100 items could not resolve the
contrast.

## 3. Layout and how to read a window

One window per series. Two windows cut from the same series are not
independent, and the paired tests assume items are.

```python
import numpy as np, json
z    = np.load("data-real/pool_real_n300/windows.npz")
meta = json.load(open("data-real/pool_real_n300/meta.json"))

W, LMAX = z["windows"], meta["L_MAX"]          # (300, 560), 512
history  = W[i, LMAX - L : LMAX]               # history at any L in 32..512
target   = W[i, LMAX : LMAX + h]               # target for any h <= 48
denom    = z["mase_denominator"][i]            # fixed; identical at every L
score    = 1.0 / (1.0 + np.mean(np.abs(pred - target)) / denom)
```

Windows at different L are **nested suffixes ending at the same cut**, so the
forecast target never changes along the sweep — invariant I1/I2.

The **MASE denominator** is the mean absolute seasonal difference over the
L=512 history using the period the *archive declares* for the frequency, not
an estimated one. It is public, fixed, and identical at every L, which is
what `scoring.fixed_mase_denominator: true` requires. The arms are still not
told the period — estimating it stays part of the task.

## 4. Diagnostics — run before spending any GPU

`python data-real/verify_pool.py`, horizon 8:

| L | last_value | window_mean | snaive (declared) | snaive (**estimated**) |
|---|---|---|---|---|
| 32 | 0.3762 | 0.3072 | 0.6670 | 0.3835 |
| 64 | 0.3762 | 0.3071 | 0.6670 | 0.3826 |
| 128 | 0.3762 | 0.3353 | 0.6670 | 0.4340 |
| 256 | 0.3762 | 0.3188 | 0.6670 | 0.5544 |
| 512 | 0.3762 | 0.3156 | 0.6670 | **0.6133** |
| **change** | +0.0000 | +0.0083 | +0.0000 | **+0.2298** |

### 4.1 Good news: H1 finally has something to measure

A seasonal-naive baseline that must **estimate** its own period improves
**+0.230** from L=32 to L=512 — it cannot find a period of 24 inside a
32-point window and can inside a 512-point one. That is roughly **ten times
the entire cycle-2 H1 effect (+0.023)**.

Amendment A15 diagnosed exactly this as the synthetic pool's failing: *"with
short periods the forecasting ceiling barely moved with history, leaving H1
nothing to measure."* On this pool the ceiling moves a great deal. H1 is the
hypothesis that gains most from real data.

### 4.2 ⚠ Warning: the trivial baseline here is strong, and it may reverse your headline

`seasonal_naive` with the **declared** period scores **0.6604**. For
comparison, in cycle 2 the harness scored 0.549 against a seasonal-naive
baseline of ~0.497.

So on this pool Chronos-t5-small has to beat **0.66**, not 0.50. It may not.
If it does not, the cycle-2 headline — *"routing to a forecaster beats the
LLM and beats seasonal naive"* — **does not survive real data**, at least for
this forecaster at this horizon.

That is a legitimate and publishable outcome either way, but it must be
pre-registered as a possibility rather than discovered and explained away.
State in advance what would falsify the claim.

Related: the declared-period baseline is **flat in L** (it needs only
`period` points of history), so a well-specified classical method already
extracts everything history length can buy here. Your H1 contrast is
therefore really *"can the forecaster and the LLM each find the period that
seasonal naive is simply told?"*

### 4.3 ⚠ Problem: the H2 strata are not populated

`seasonal_strength` = detrended ACF at the declared period:

| stratum | items |
|---|---|
| weak `[-1.0, +0.2)` | **3** |
| medium `[+0.2, +0.5)` | 25 |
| strong `[+0.5, +1.0)` | **272** |

Per-dataset medians: `m4_hourly` +0.895, `electricity_hourly` +0.783,
`traffic_hourly` +0.708, `nn5_daily` +0.603.

**The reframed H2 — "does the gap depend on seasonal strength?" — cannot be
run on this pool.** Everything long enough for an L-sweep is sensor data, and
sensor data is strongly seasonal. There is no weak-seasonality arm.

H2 needs a **second pool**, and it has to be built differently: weakly
seasonal series are economic ones (monthly/quarterly), which are short, so
that pool is **L = 128 only** — which is fine, because H2 was always a
single-L hypothesis. Candidates: M4 Monthly / Quarterly, M3 Monthly, Tourism
Monthly, filtered to ≥ 136 points. Not built here.

### 4.4 The exclusion rule has to hold at every L, not just at L_MAX

A shorter history is a **suffix** of the long one, so it can be degenerate
while the full window is fine. Real sensor feeds do this constantly — a
traffic loop reads 0 all night.

The first version of this pool checked `std > 0` on the L=512 history only,
and passed items whose L=32 window was entirely zeros. The consequence was
not a crash. `systems/tools.py::series_evidence` divides by
`gt.local_noise_scale`, which bottoms out at `gt.REL_EPS = 1e-9`, so the
`bare_features` arm would have been handed this in its prompt:

```
  points=32 mean=0.001119 sd=0.006229 noise_scale=1e-09
  largest_local_peak_robust_z=35800000.00
```

Thirty-five million, for a window of zeros. On synthetic data this could
never fire, because the generator always injects noise.

`degenerate_at_no_L()` now rejects any series where **any** L in
`SWEEP_LS = (32, 64, 128, 256, 512)` has `std <= 1e-6` or `MAD <= 1e-6`.
That costs 82 of 1,708 candidate series (4.8%, mostly traffic) and brings
the evidence block back into a sane range:

| | `robust_z` max | `noise_scale` min |
|---|---|---|
| before, L=32 | 3.6e+07 | 1e-09 |
| after, worst L | **104.2** | **1.7e-03** |

(Synthetic cycle 2 ran about 0–7. The residual 104 is genuine real-world
spikiness, not a numerical artifact.)

**Pre-register this rule.** It is an exclusion criterion applied before any
result is seen, which is legitimate; applied afterwards it would be a
researcher degree of freedom.

## 5. ⚠ Contamination — read before pre-registering

Synthetic data made both arms immune to training-set overlap. Real public
benchmark data removes that immunity **on both sides**:

- **Harness arm.** Chronos's own paper splits evaluation into Benchmark I
  (datasets *in* its training corpus) and Benchmark II (*not* in it).
  Reported overlaps with the Monash/M-series collections include **Traffic**,
  Solar, Taxi, Weather, Hospital, COVID deaths, US births, Saugeen,
  temperature-rain and KDD Cup 2018.
- **Bare arm.** These archives are public text; the LLM may have the literal
  numbers in its pretraining data.

Per-dataset status, as recorded in `MANIFEST.json`:

| dataset | status |
|---|---|
| `traffic_hourly` | **DOCUMENTED OVERLAP** — do not use as a zero-shot cell |
| `electricity_hourly` | **LIKELY** — treat as in-domain for Chronos until verified |
| `m4_hourly` | UNVERIFIED |
| `nn5_daily` | UNVERIFIED |

**Nothing here is certified clean.** This pool is built for the *H1 history
sweep*, where the contrast is within-forecaster across L and contamination
affects all five lengths equally — a much weaker threat than for a
cross-arm claim. **For H3, and for any "the tool beats the LLM" headline,
replace or re-verify these datasets against the Chronos Benchmark I/II
tables, or move to the GIFT-Eval non-leaking split (Apache 2.0,
`Salesforce/GiftEval`).**

## 6. What this pool does *not* give you

1. **No qualitative labels.** No trend / seasonal / event / position key
   exists for real data. Do **not** manufacture one from
   `seasonal_strength` or `trend_t` — the harness's evidence block computes
   the same family of statistics, so a key drawn from them recreates the
   cycle-1 defect where the tool and the answer key shared code. Those two
   fields are stratification variables, and `meta.json` says so.
   For a real qualitative cell, use human-labelled event data (UCR Anomaly
   Archive, TSB-UAD) and re-run the saturation guard against the new key.
2. **No H2 strata** — see §4.3.
3. **No contamination certificate** — see §5.

## 7. Attribution (required by CC BY 4.0)

> Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J.,
> Montero-Manso, P. (2021). *Monash Time Series Forecasting Archive.*
> NeurIPS Datasets and Benchmarks Track. arXiv:2105.06643.

Plus the per-dataset citations in `MANIFEST.json` (M4 → Makridakis et al.;
NN5 → Ben Taieb et al.; electricity → UCI; traffic → Caltrans PEMS).

## 8. Reproduce

```bash
python data-real/check_pools.py  # the shipped pools against the configs' hashes
# Rebuilding is NOT needed for a run and the builders refuse to overwrite a
# pool: a rebuild elsewhere gives the same windows but a different file hash
# (strata columns differ ~1e-13). To make a NEW pool in an empty copy:
python data-real/fetch.py        # re-download, verify sha256
python data-real/build_pool.py --force      # seed 20261114
python data-real/build_pool_c3.py --force
python data-real/verify_pool.py  # the tables in §4
```
