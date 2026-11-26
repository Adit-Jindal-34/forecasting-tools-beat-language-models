"""Real-data pool loader: `data-real/pool_real_*/` -> `Sample` objects.

WHY THIS MODULE EXISTS AT ALL
-----------------------------
Until cycle 3 the pipeline could only reach data through
`src/synthetic.py`, which builds the pool AND the answer keys in one pass.
That is fine while the key is "what the generator injected" and impossible
once the series come from an archive: a Monash series has no generator, so
there is nothing to inject and nothing to read back.

On real data there is exactly ONE legitimate label - the held-out future -
so this loader emits `forecasting` items and refuses anything else. See
`REFUSED_QUERY_TYPES` below for why the two qualitative cells may not be
ported.

WHAT IS AND IS NOT CARRIED OVER
-------------------------------
Carried over unchanged, because they are properties of the protocol rather
than of the generator:

  * `Sample.window(L)` / `Sample.target()` and therefore invariants I1-I5.
    The pool stores `L_MAX + H_MAX` points per item with a COMMON CUT
    ORIGIN, so the history at every L is a suffix ending at one instant and
    the target never moves (`data-real/build_pool.py`).
  * The paraphrase bank and its striding assignment, so routing variance
    stays measurable and `framing_id` stays a usable clustering unit.
  * `1/(1+MASE)` scoring and the classical baselines in `runners/_c2.py`.

Deliberately NOT carried over:

  * `pattern` as a factorial record. There is no factorial design here, so
    the column carries the DATASET name instead - the conservative
    clustering unit - and `metadata["dataset_id"]` carries it explicitly.
  * `seasonal_period` as a generator secret. Here it is the period the
    ARCHIVE declares for the frequency (`data-real/tsf.py::FREQ_PERIOD`).
    It is public, it is fixed before the run, and it is used only by the
    runner: for the MASE denominator (which the pool precomputes) and for
    the declared-period seasonal-naive baseline. No arm is told it -
    estimating the period stays part of the task.
  * Any qualitative key. Strata (`seasonal_strength`, `trend_t`) are
    DESCRIPTIVE stratification variables and must never become an answer
    key: the harness's evidence block computes the same family of
    statistics, so a key drawn from them would recreate cycle 1's exact
    defect of the tool and the key sharing code.

THE POOL IS CONTENT-ADDRESSED
-----------------------------
`data.pool_sha256` must match the sha256 of `windows.npz`. A paired design
across five history lengths and thirteen backbones is only valid if every
cell saw the same 300 series; a silently regenerated pool would break that
without any error. The hash is also part of `data_fingerprint()`, so a new
pool can never collide with cached answers for the old one.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.samples import (Dataset, Sample, build_query, framing_of, lead_of,
                         N_VARIANTS)

#: The only query type a real-data pool may carry.
#:
#: `statistics`, `counting`, `extremum`, `change`, `correlation` and
#: `localization` would *work* - their keys are deterministic functions of
#: the visible window, so they are well defined on any series - but they are
#: a priori tool-saturated (`src.config.TOOL_SATURATED_OPS`), so the harness
#: scores 1.000 by construction and the cell measures the bare arm's
#: in-context arithmetic, not architecture. Running them on real data would
#: cost calls and add nothing.
ALLOWED_QUERY_TYPES: frozenset[str] = frozenset({"forecasting"})

#: Refused, with the reason reported to the user rather than hidden.
#:
#: The cycle-2 key for these is "what the generator injected" (amendment
#: A9). Real data has no generator, and manufacturing a key from statistics
#: (STL, PELT, or the evidence block's own fields) would recreate cycle 1's
#: retracted defect - the harness's tool and the answer key computing the
#: same quantities. A real qualitative cell needs multiple human annotators
#: per item (MMTS-Bench used 10 experts at Fleiss kappa 0.73); that is a
#: separate project, not a config change.
REFUSED_QUERY_TYPES: dict[str, str] = {
    "trend_description": (
        "the cycle-2 key is the generator's injected trend/seasonal record "
        "(amendment A9); real data has no generator, and deriving the key "
        "from window statistics would make the answer key and the harness's "
        "evidence block the same function - the cycle-1 defect"),
    "event_explanation": (
        "same as trend_description: the key is the injected event and its "
        "index. Labelling real events needs several annotators per item, "
        "not a rule the tool also implements"),
}

#: Pre-registered cuts on `seasonal_strength` (detrended ACF at the declared
#: period). DESCRIPTIVE stratification for the reframed H2 and for
#: by-stratum reporting. Fixed here, before any real-data result, so the
#: boundary cannot be chosen once the numbers are in.
STRATUM_CUTS: tuple[float, float] = (0.30, 0.60)


def seasonality_stratum(strength: float) -> str:
    """`weak` | `medium` | `strong`, or `unknown` for a NaN strength."""
    lo, hi = STRATUM_CUTS
    if not np.isfinite(strength):
        return "unknown"
    if strength < lo:
        return "weak"
    if strength < hi:
        return "medium"
    return "strong"


def pool_sha256(pool_dir: Path) -> str:
    """sha256 of `windows.npz` - the pool's content address."""
    return hashlib.sha256(
        (Path(pool_dir) / "windows.npz").read_bytes()).hexdigest()


class RealDataset(Dataset):
    """A frozen real-data pool. Nothing here generates anything."""

    name = "real"

    def __init__(self, pool_dir: Path):
        self.pool_dir = Path(pool_dir)
        if not (self.pool_dir / "windows.npz").exists():
            raise FileNotFoundError(
                f"no real-data pool at {self.pool_dir}. Build one with\n"
                "  python data-real/fetch.py && python data-real/build_pool.py")
        self.meta = json.loads(
            (self.pool_dir / "meta.json").read_text(encoding="utf-8"))
        z = np.load(self.pool_dir / "windows.npz")
        self.windows = np.asarray(z["windows"], float)
        self.mase_denominator = np.asarray(z["mase_denominator"], float)
        self.period = np.asarray(z["period"], int)
        self.seasonal_strength = np.asarray(z["seasonal_strength"], float)
        self.trend_t = np.asarray(z["trend_t"], float)
        self.L_MAX = int(self.meta["L_MAX"])
        self.H_MAX = int(self.meta["H_MAX"])
        self.items = list(self.meta["items"])
        n = len(self.items)
        if self.windows.shape != (n, self.L_MAX + self.H_MAX):
            raise ValueError(
                f"{self.pool_dir}: windows {self.windows.shape} disagree with "
                f"meta (n={n}, L_MAX={self.L_MAX}, H_MAX={self.H_MAX})")
        bad = np.where(~np.isfinite(self.mase_denominator)
                       | (self.mase_denominator <= 0))[0]
        if len(bad):
            raise ValueError(
                f"{self.pool_dir}: items {bad[:5].tolist()} have a "
                "non-positive MASE denominator; the pool builder must "
                "exclude them, not the scorer")

    # ------------------------------------------------------------------
    def load(self, n_base_series: int, n_per_query_type: int,
             query_types: list, max_history: int, horizon: int,
             seed: int) -> list:
        """`n_base_series` / `n_per_query_type` / `seed` are IGNORED.

        They are generator knobs. The real pool is frozen on disk and its
        size is whatever `data-real/build_pool.py` sampled under its own
        seed, recorded in `meta.json`. Pretending a config could resample it
        would invite a silent partial pool; `src.config` refuses a config
        that asks for a different n instead.
        """
        if max_history > self.L_MAX:
            raise ValueError(
                f"data.max_history={max_history} exceeds the pool's "
                f"L_MAX={self.L_MAX}. Rebuild the pool with a larger L_MAX; "
                "padding a history would invent data.")
        if horizon > self.H_MAX:
            raise ValueError(
                f"data.horizon={horizon} exceeds the pool's "
                f"H_MAX={self.H_MAX}")

        names = [qt.name if hasattr(qt, "name") else str(qt)
                 for qt in query_types]
        refused = [n for n in names if n in REFUSED_QUERY_TYPES]
        if refused:
            raise ValueError(
                "real data cannot carry these query types:\n  " + "\n  ".join(
                    f"{n}: {REFUSED_QUERY_TYPES[n]}" for n in refused))
        unsupported = [n for n in names if n not in ALLOWED_QUERY_TYPES]
        if unsupported:
            raise ValueError(
                f"real data supports {sorted(ALLOWED_QUERY_TYPES)} only; "
                f"got {unsupported}. The deterministic-window cells are a "
                "priori tool-saturated (src.config.TOOL_SATURATED_OPS), so "
                "running them on real data buys nothing.")

        lo = self.L_MAX - int(max_history)
        hi = self.L_MAX + int(horizon)
        n = len(self.items)
        samples: list = []
        for qt in query_types:
            name = qt.name if hasattr(qt, "name") else str(qt)
            kind = getattr(qt, "kind", "forecast")
            for k, it in enumerate(self.items):
                # Spread the paraphrase assignment over BOTH compositional
                # axes at any pool size, exactly as `src/synthetic.py` does:
                # `k % N_VARIANTS` below full size emits only lead_id=0 and
                # makes the within-framing variance check vacuous.
                variant = (k * N_VARIANTS) // max(n, 1)
                samples.append(Sample(
                    # OPAQUE on purpose. `item_id` names its archive
                    # ("traffic_hourly-T12"), and a system can see
                    # `InferenceSample.sample_id`. No system puts it in a
                    # prompt today, but the dataset name is precisely the
                    # contamination-sensitive token a model could recall, so
                    # it does not travel on the inference-side object at
                    # all. Provenance lives in `metadata` (runner-side) and
                    # in the `dataset_id` shard column.
                    sample_id=f"real-{k:05d}-{name}",
                    series=self.windows[k, lo:hi].copy(),
                    max_history=int(max_history), horizon=int(horizon),
                    query_type=name,
                    query=build_query(name, int(horizon), variant,
                                      design="c2"),
                    query_kind=kind,
                    # No factorial record exists here. The DATASET is the
                    # conservative clustering unit, so that is what the
                    # `pattern` column carries; `metadata["dataset_id"]` is
                    # the explicit channel the runners read.
                    pattern=str(it["dataset"]),
                    seasonal_period=int(self.period[k]),
                    paraphrase_id=variant, framing_id=framing_of(variant),
                    lead_id=lead_of(variant),
                    metadata=self._metadata(k, it)))
        return samples

    def _metadata(self, k: int, it: dict) -> dict:
        """Runner-side only. `Sample.view()` copies none of it (I5)."""
        strength = float(self.seasonal_strength[k])
        return dict(
            source="real",
            base_index=int(k),
            item_id=str(it["item_id"]),
            dataset_id=str(it["dataset"]),
            series_name=str(it["series_name"]),
            frequency=str(it["frequency"]),
            # the ARCHIVE-declared period, not an estimate and not a secret
            declared_period=int(self.period[k]),
            archive_horizon=it.get("archive_horizon"),
            # fixed at build time from the L_MAX history; identical at every
            # L, which is what `scoring.fixed_mase_denominator` requires
            mase_denominator=float(self.mase_denominator[k]),
            # DESCRIPTIVE strata. Never an answer key - see the module
            # docstring and `meta.json::strata_warning`.
            seasonal_strength=strength,
            trend_t=float(self.trend_t[k]),
            stratum=seasonality_stratum(strength))


# --------------------------------------------------------------------------
# the entry point `src.pipeline.make_samples` calls
# --------------------------------------------------------------------------
def materialize(cfg, pool_dir: str | Path | None = None,
                root: Path | None = None) -> list:
    """Load the frozen real pool for `cfg`, verifying its content hash.

    Mirrors `src.synthetic.materialize`'s signature so `make_samples` can
    dispatch on `data.name` alone. Nothing is written: a real pool is an
    input, tracked in git, never regenerated by a run.
    """
    directory = (Path(pool_dir) if pool_dir is not None
                 else cfg.pool_dir(root or Path.cwd()))
    ds = RealDataset(directory)

    declared = str(cfg.data.get("pool_sha256") or "")
    actual = pool_sha256(directory)
    if not declared:
        raise ValueError(
            f"data.pool_sha256 is required for a real pool. The pool at "
            f"{directory} currently hashes to:\n  pool_sha256: {actual}\n"
            "Declare it in the config. A paired design across L and "
            "backbones is only valid if every cell saw the same series, and "
            "an undeclared pool can change under the run without any error.")
    if declared != actual:
        raise ValueError(
            f"real pool at {directory} does not match the config.\n"
            f"  declared : {declared}\n  actual   : {actual}\n"
            "Either the pool was rebuilt or the config points at a different "
            "one. Both invalidate every shard already written under this "
            "fingerprint - do not 'fix' it by updating the hash unless you "
            "intend to discard those shards.")

    d = cfg.data
    want_n = int(d.get("n_base_series") or 0)
    if want_n and want_n != len(ds.items):
        raise ValueError(
            f"data.n_base_series={want_n} but the pool at {directory} holds "
            f"{len(ds.items)} items. A real pool cannot be resampled from a "
            "config; rebuild it with data-real/build_pool.py or correct the "
            "config.")

    return ds.load(len(ds.items), len(ds.items), cfg.query_types,
                   int(d["max_history"]), int(d["horizon"]), cfg.seed)


def pool_report(samples: list) -> dict:
    """Composition of the loaded pool, for the run's artifacts.

    Printed and written into `analysis.json` so a reader can see which
    archives the items came from and how the strata fell WITHOUT reopening
    the pool - the contamination argument in PREREGISTRATION_C3.md §6 is
    per-dataset, so the dataset mix has to travel with the results.
    """
    from collections import Counter
    forecasting = [s for s in samples if s.query_type == "forecasting"]
    by_ds = Counter(s.metadata.get("dataset_id") for s in forecasting)
    by_st = Counter(s.metadata.get("stratum") for s in forecasting)
    strength = np.array([s.metadata.get("seasonal_strength", np.nan)
                         for s in forecasting], float)
    return dict(
        n_items=len(forecasting),
        n_datasets=len(by_ds),
        items_per_dataset=dict(sorted(by_ds.items())),
        items_per_stratum=dict(sorted(by_st.items())),
        stratum_cuts=dict(weak_below=STRATUM_CUTS[0],
                          strong_at_or_above=STRATUM_CUTS[1]),
        seasonal_strength_median=(float(np.nanmedian(strength))
                                  if len(strength) else float("nan")),
        declared_periods=dict(sorted(Counter(
            s.metadata.get("declared_period")
            for s in forecasting).items())),
        note=("strata are DESCRIPTIVE. They are never an answer key: the "
              "harness's evidence block computes the same family of "
              "statistics, so a key drawn from them would reproduce the "
              "cycle-1 defect."))
