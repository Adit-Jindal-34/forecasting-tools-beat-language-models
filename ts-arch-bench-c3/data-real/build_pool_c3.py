"""Build the two CYCLE-3 pools that `pool_real_n300` cannot provide.

`build_pool.py` is untouched and still reproduces `pool_real_n300`
bit-identically. This module imports its helpers rather than copying them,
so the exclusion rule, the MASE denominator and the stratification
variables are the same code in all three pools.

WHY TWO MORE POOLS
------------------
`pool_real_n300` has two problems that are properties of WHICH archives it
drew from, not of how it was built.

1. CONTAMINATION. Chronos's own Table 2 (arXiv:2403.07815, Appendix B)
   partitions its datasets into pretraining-only, in-domain evaluation
   (Benchmark I, 15 datasets, IN the training corpus) and zero-shot
   evaluation (Benchmark II, 27 datasets, NOT in it). Matching on series
   count and length, `pool_real_n300` is half contaminated:

       m4_hourly           414 series, len 748-1008   -> Benchmark I
       electricity_hourly  321 series, len 26304      -> Benchmark I
       traffic_hourly      862 series, len 17544      -> Benchmark II
       nn5_daily           111 series, len 791        -> Benchmark II

   Note this INVERTS the flags in `MANIFEST.json`, which guessed before the
   tables were read: Traffic was recorded as a documented overlap and is in
   fact zero-shot, while M4 Hourly was recorded as unverified and is in
   fact in-domain. The manifest has been corrected.

   A within-forecaster contrast across L (H1) is unharmed - contamination
   hits all five history lengths equally - but a cross-arm "the tool beats
   the LLM" headline is not, so `pool_clean_n300` below draws only on
   Benchmark II.

2. NO WEAK-SEASONALITY ARM. Everything long enough for an L=512 sweep in
   the first pool is sensor data, and sensor data is strongly seasonal
   (weak 3 / medium 61 / strong 236 under `STRATUM_CUTS`). So cycle 2's
   most interesting post-hoc finding - +0.170 on seasonal series against
   +0.014 on non-seasonal ones at 32B - has nothing to be tested on.
   `pool_h2_strata` below is balanced across strata BY CONSTRUCTION.

WHAT IS STILL NOT SETTLED
-------------------------
Benchmark II settles contamination for CHRONOS, i.e. for the harness arm.
It says nothing about the BARE arm: these archives are public text and
Qwen2.5 may carry the literal numbers from pretraining. Only data published
after the backbone's training cutoff settles that, and it is out of scope
here - see PREREGISTRATION_C3.md C3-8, which declares the asymmetry rather
than papering over it.

Run:  python data-real/fetch.py && python data-real/build_pool_c3.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pool import (L_MAX, H_MAX, degenerate_at_no_L, refuse_overwrite,  # noqa: E402
                        seasonal_naive_mase_denominator, seasonal_strength,
                        trend_t)
from tsf import read_tsf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

#: Pre-registered cuts on `seasonal_strength`. MUST equal
#: `src.real_data.STRATUM_CUTS`; `tests/test_real_data.py` asserts it.
STRATUM_CUTS = (0.30, 0.60)

#: Chronos Benchmark II only - "not used in training Chronos models, but
#: only for evaluation" (arXiv:2403.07815, Table 2). Each is also long
#: enough that a 512-point history plus a 48-point target fits.
#:
#: `weather` and `fred_md` are new in cycle 3 and carry the only genuinely
#: weak-seasonality series in reach; `traffic_hourly` and `nn5_daily` are
#: already in `pool_real_n300`, which is deliberate - keeping them in both
#: pools is what makes the contaminated and clean results comparable on a
#: shared subset instead of differing in every respect at once.
CLEAN_SOURCES = {
    "traffic_hourly": "traffic_hourly_dataset.tsf",
    "nn5_daily": "nn5_daily_dataset_without_missing_values.tsf",
    "weather": "weather_dataset.tsf",
    "fred_md": "fred_md_dataset.tsf",
}

CLEAN_SEED = 20261115
CLEAN_PER_DATASET = 75          # -> n = 300, no dataset over 25%

H2_SEED = 20261116
#: Items per (stratum, dataset) cell, filled as far as supply allows and
#: topped up from whichever dataset still has stock. The STRATUM totals are
#: what the design needs balanced; the dataset mix inside each is recorded
#: so the confound is visible rather than implied.
H2_PER_STRATUM = 120
H2_STRATA = ("weak", "strong")


def stratum_of(strength: float) -> str:
    lo, hi = STRATUM_CUTS
    if not np.isfinite(strength):
        return "unknown"
    if strength < lo:
        return "weak"
    if strength < hi:
        return "medium"
    return "strong"


def eligible_items(ds: str, filename: str) -> tuple[list, dict]:
    """Every series of a dataset that survives the cycle-3 exclusion rule.

    Identical to `build_pool.main`'s inner loop, extracted so the three
    pools cannot drift apart. In particular `degenerate_at_no_L` is checked
    at EVERY L in the sweep, not only at L_MAX: a shorter history is a
    suffix, so it can be flat while the full window is fine, and a flat
    window is what made `series_evidence` emit
    `largest_local_peak_robust_z=35800000.00` on real data.
    """
    path = os.path.join(RAW, filename)
    f = read_tsf(path)
    need = L_MAX + H_MAX
    out = []
    n_short = n_nan = n_denom = n_degen = 0
    for i, s in enumerate(f.series):
        if len(s) < need:
            n_short += 1
            continue
        if np.isnan(s).any():
            n_nan += 1
            continue
        w = s[-need:]
        hist = w[:L_MAX]
        denom = seasonal_naive_mase_denominator(hist, f.period)
        if not np.isfinite(denom) or denom <= 0:
            n_denom += 1
            continue
        if degenerate_at_no_L(w):
            n_degen += 1
            continue
        strength = seasonal_strength(hist, f.period)
        out.append(dict(
            item_id=f"{ds}-{f.names[i]}", dataset=ds, series_name=f.names[i],
            frequency=f.frequency, period=f.period,
            archive_horizon=f.horizon, window=w, mase_denominator=denom,
            seasonal_strength=strength, trend_t=trend_t(hist),
            stratum=stratum_of(strength)))
    lengths = f.lengths()
    report = dict(
        dataset=ds, file=filename, n_series=int(len(lengths)),
        frequency=f.frequency, declared_period=f.period,
        archive_horizon=f.horizon,
        len_min=int(lengths.min()), len_median=int(np.median(lengths)),
        len_max=int(lengths.max()),
        n_eligible=len(out),
        excluded=dict(too_short=n_short, has_missing=n_nan,
                      zero_denominator=n_denom, degenerate_at_some_L=n_degen),
        sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
        chronos_split="Benchmark II (zero-shot; not in Chronos training)")
    return out, report


def write_pool(items: list, out_name: str, meta_extra: dict) -> str:
    """Same on-disk layout as `pool_real_n300`, so one loader reads all."""
    out = os.path.join(HERE, out_name)
    refuse_overwrite(out)
    os.makedirs(out, exist_ok=True)
    windows = np.stack([it["window"] for it in items]).astype(np.float64)
    np.savez_compressed(
        os.path.join(out, "windows.npz"),
        windows=windows,
        mase_denominator=np.array([it["mase_denominator"] for it in items]),
        period=np.array([it["period"] for it in items], dtype=int),
        seasonal_strength=np.array([it["seasonal_strength"] for it in items]),
        trend_t=np.array([it["trend_t"] for it in items]),
    )
    meta = dict(
        built="2026-11-14",
        n_items=len(items),
        L_MAX=L_MAX, H_MAX=H_MAX,
        layout=("windows[i] has L_MAX + H_MAX points. History at L is "
                "windows[i][L_MAX-L : L_MAX]; the target for horizon h is "
                "windows[i][L_MAX : L_MAX+h]. All L are nested suffixes "
                "ending at one cut, so the target never changes with L."),
        mase_denominator=("mean |x[t] - x[t-period]| over windows[i][:L_MAX] "
                          "with the archive-declared period; identical at "
                          "every L by construction"),
        strata_warning=("seasonal_strength and trend_t are DESCRIPTIVE "
                        "stratification variables. They must never be used "
                        "as an answer key: the harness's evidence block "
                        "computes the same family of statistics, and a key "
                        "drawn from them reproduces the cycle-1 defect."),
        stratum_cuts=dict(weak_below=STRATUM_CUTS[0],
                          strong_at_or_above=STRATUM_CUTS[1]),
        items=[{k: v for k, v in it.items() if k != "window"}
               for it in items],
        **meta_extra)
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    sha = hashlib.sha256(
        open(os.path.join(out, "windows.npz"), "rb").read()).hexdigest()
    print(f"\n{out_name}: n={len(items)}  windows={windows.shape}")
    print(f"  pool_sha256: {sha}   <- paste into the config's data block")
    return sha


def crosstab(items: list) -> dict:
    """dataset -> stratum -> count. Printed and stored, never inferred."""
    out: dict = {}
    for it in items:
        out.setdefault(it["dataset"], {}).setdefault(it["stratum"], 0)
        out[it["dataset"]][it["stratum"]] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


# --------------------------------------------------------------------------
def build_clean(pools: dict) -> None:
    """Uniform sample within each Benchmark-II dataset. The contamination
    control for every cross-arm claim."""
    rng = np.random.default_rng(CLEAN_SEED)
    items, report = [], []
    for ds, fn in CLEAN_SOURCES.items():
        elig, rep = eligible_items(ds, fn)
        pools[ds] = elig                       # reused by the H2 builder
        take = min(CLEAN_PER_DATASET, len(elig))
        pick = rng.choice(len(elig), size=take, replace=False)
        items.extend(elig[k] for k in sorted(pick))
        rep["n_taken"] = take
        report.append(rep)
        print(f"  {ds:20} eligible {rep['n_eligible']:5}/"
              f"{rep['n_series']:<6} taken {take:3}  "
              f"period={rep['declared_period']}")

    sha = write_pool(items, "pool_clean_n%d" % len(items), dict(
        seed=CLEAN_SEED,
        purpose=("contamination-controlled forecasting pool: every dataset "
                 "is in Chronos Benchmark II, i.e. not in the Chronos "
                 "training corpus (arXiv:2403.07815, Table 2)"),
        contamination=dict(
            chronos="controlled: all four datasets are Benchmark II",
            backbone_llm=("NOT controlled: these archives are public text "
                          "and Qwen2.5 may carry the literal values. Only "
                          "post-cutoff data settles that; see "
                          "PREREGISTRATION_C3.md C3-8")),
        datasets=report,
        dataset_by_stratum=crosstab(items)))
    print("  dataset x stratum:", json.dumps(crosstab(items)))
    pools["_clean_sha"] = sha


def build_h2(pools: dict) -> None:
    """Balanced on seasonality stratum - the arm `pool_real_n300` lacks.

    Drawn from the same Benchmark-II datasets, so H2 is contamination-
    controlled too. Supply is uneven (weak series are mostly `weather` and
    `fred_md`; strong ones are mostly `traffic_hourly`), so the dataset mix
    inside each stratum is RECORDED and the runner reports the within-
    dataset contrast beside the pooled one. A stratum effect that is really
    an archive effect must be visible, not assumed away.
    """
    rng = np.random.default_rng(H2_SEED)
    chosen: list = []
    composition = {}
    for stratum in H2_STRATA:
        supply = {ds: [it for it in elig if it["stratum"] == stratum]
                  for ds, elig in pools.items() if not ds.startswith("_")}
        avail = {ds: len(v) for ds, v in supply.items()}
        # Even split across datasets first, then top up from whatever still
        # has stock. Deterministic given H2_SEED and the supply counts.
        quota = {ds: 0 for ds in supply}
        remaining = H2_PER_STRATUM
        while remaining > 0:
            live = [ds for ds in supply if quota[ds] < avail[ds]]
            if not live:
                break
            share = max(1, remaining // len(live))
            for ds in live:
                room = min(share, avail[ds] - quota[ds], remaining)
                quota[ds] += room
                remaining -= room
                if remaining == 0:
                    break
        picked = []
        for ds, k in quota.items():
            if not k:
                continue
            idx = rng.choice(len(supply[ds]), size=k, replace=False)
            picked.extend(supply[ds][i] for i in sorted(idx))
        composition[stratum] = dict(
            requested=H2_PER_STRATUM, taken=len(picked),
            available=dict(sorted(avail.items())),
            taken_per_dataset=dict(sorted(
                {ds: k for ds, k in quota.items() if k}.items())))
        chosen.extend(picked)
        print(f"  {stratum:7} requested {H2_PER_STRATUM}  taken "
              f"{len(picked):4}  from {composition[stratum]['taken_per_dataset']}")

    write_pool(chosen, "pool_h2_strata_n%d" % len(chosen), dict(
        seed=H2_SEED,
        purpose=("reframed H2: does the architectural gap depend on the "
                 "SERIES rather than on the question? On real data every "
                 "item is a forecasting item, so the cycle-2 question-type "
                 "contrast has no arms left; this pool balances weak against "
                 "strong seasonality instead, which is the cycle-2 post-hoc "
                 "finding (+0.170 seasonal vs +0.014 non-seasonal at 32B) "
                 "promoted to a declared test"),
        design_note=("The two strata are DIFFERENT SERIES, so the contrast "
                     "is two-sample, not paired. runners/h2_series_character"
                     ".py uses the unpaired form; confirmatory_did refuses "
                     "axis='stratum' for exactly this reason."),
        confound_warning=(
            "stratum is partially confounded with dataset: weak-seasonality "
            "series long enough for this geometry come mostly from `weather` "
            "and `fred_md`, strong ones mostly from `traffic_hourly`. The "
            "composition below is the measurement of that confound. The "
            "runner reports the within-dataset contrast beside the pooled "
            "one; where a dataset supplies both strata, that version carries "
            "the claim."),
        composition=composition,
        contamination=dict(
            chronos="controlled: all source datasets are Benchmark II"),
        dataset_by_stratum=crosstab(chosen)))
    print("  dataset x stratum:", json.dumps(crosstab(chosen)))


def main() -> None:
    missing = [fn for fn in CLEAN_SOURCES.values()
               if not os.path.exists(os.path.join(RAW, fn))]
    if missing:
        raise SystemExit(
            "missing raw files: " + ", ".join(missing) +
            "\nRun: python data-real/fetch.py")
    pools: dict = {}
    print("=== pool_clean: Chronos Benchmark II only ===")
    build_clean(pools)
    print("\n=== pool_h2_strata: balanced on seasonality ===")
    build_h2(pools)


if __name__ == "__main__":
    main()
