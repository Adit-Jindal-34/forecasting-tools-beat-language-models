"""Synthetic series generator and the frozen sample pool.

TWO ITEM-DESIGN DEFECTS FIXED
-----------------------------
1. DEGENERATE COUNTING. In the old pool, `count_spikes` ground truth was 0
   in 83 of 100 items at L=128, because only 2 of 7 patterns injected spikes
   at all and the `counting` query type drew from every pattern. Combined
   with a relative-error metric that divides by the truth, the cell silently
   collapsed to "will the model say zero?".

   Here EVERY base series carries its own spike intensity, drawn per series,
   so the realised count distribution spreads across 0..4+ at the configured
   history length while the pool stays shared across all query types (which
   is what makes the cells paired).

2. NEAR-ZERO LEVELS. The old series were centred near zero, so `statistics`
   ground truth had |mean| < 0.1*sd in 46/100 items and the `change` anchor
   x[0] sat near zero too. Every series now gets a positive level offset, so
   the mean is well away from zero and percent change is well conditioned.
   The level does NOT make the task easier under this build's metric, which
   scales error by the series' own dispersion, not by the target magnitude.

The generator is deterministic given (seed, data config), and the realised
ground-truth distribution is checked by src/screening.py and asserted by
tests/test_synthetic.py - so a regression back to a degenerate cell fails
the build instead of reaching a paper.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import src

from src.samples import (Dataset, Sample, build_query, framing_of, lead_of,
                         N_VARIANTS)

PATTERNS = ("trend", "seasonality", "multi_seasonality", "noise",
            "spikes", "abrupt_change", "mixed")


def _robust_scale(x: np.ndarray) -> float:
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med))) or float(np.std(x)) or 1.0


def _add_trend(x, rng, amp):
    x += np.linspace(0, amp * rng.uniform(1.0, 4.0), len(x))


def _add_seasonality(x, period, amp, phase=0.0):
    t = np.arange(len(x))
    x += amp * np.sin(2 * np.pi * t / max(period, 2) + phase)


def _add_noise(x, rng, amp):
    x += rng.normal(0, amp, len(x))


def _add_abrupt_change(x, rng, amp):
    k = int(rng.uniform(0.3, 0.7) * len(x))
    x[k:] += amp * rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 6.0)


def _inject_spikes(x: np.ndarray, rng: np.random.Generator,
                   lam_per_128: float) -> None:
    """Poisson spike process with a per-series intensity.

    Spikes are placed at least 3 apart and given 4-9 robust scales of
    amplitude, which clears the median+3*MAD threshold reliably while
    staying inside a plausible range for a real series.
    """
    n = len(x)
    expected = lam_per_128 * n / 128.0
    k = int(rng.poisson(expected))
    if k <= 0:
        return
    scale = _robust_scale(x)
    chosen: list[int] = []
    for _ in range(k * 6):
        if len(chosen) >= k:
            break
        i = int(rng.integers(2, n - 2))
        if all(abs(i - c) > 3 for c in chosen):
            chosen.append(i)
    for i in chosen:
        x[i] += scale * rng.uniform(4.0, 9.0)


#: Per-pattern spike intensity (expected spikes per 128 points).
#:
#: Tuned against the realised ground-truth distribution, not guessed. Two
#: constraints pull against each other and both are checked by
#: tests/test_synthetic.py:
#:   * `counting` needs spikes spread across many patterns, or its answer is
#:     almost always zero (the 83/100 defect);
#:   * `event_explanation` resolves spike BEFORE abrupt change, so spiking an
#:     abrupt-change series erases the only items that can ever carry the
#:     "abrupt change" label.
#: Hence: abrupt_change gets none, the dedicated spikes pattern gets many,
#: everything else gets a moderate per-series draw.
SPIKE_LAMBDA: dict[str, tuple[float, float]] = {
    "trend": (0.0, 3.0),
    "seasonality": (0.0, 2.0),
    "multi_seasonality": (0.0, 2.0),
    "noise": (0.0, 3.0),
    "spikes": (2.0, 5.0),
    "abrupt_change": (0.0, 0.0),
    "mixed": (0.0, 2.0),
}


def generate_base_series(n: int, pattern: str, rng: np.random.Generator,
                         horizon: int, level_range=(20.0, 120.0),
                         spike_lambda_range=None) -> tuple:
    """One full series (history + target). Returns (series, pattern, period)."""
    total = n + horizon
    x = np.zeros(total, float)
    period = 1
    amp = rng.uniform(0.5, 2.0)

    if pattern == "trend":
        _add_trend(x, rng, amp)
        _add_noise(x, rng, amp * 0.3)
    elif pattern == "seasonality":
        period = int(rng.choice([7, 12, 24]))
        # amplitude well above the noise floor so the ACF at the true period
        # actually clears the 0.5 threshold the answer key uses
        _add_seasonality(x, period, amp * 6.0, rng.uniform(0, 2 * np.pi))
        _add_noise(x, rng, amp * 0.25)
    elif pattern == "multi_seasonality":
        period = int(rng.choice([12, 24]))
        _add_seasonality(x, period, amp * 6.0, rng.uniform(0, 2 * np.pi))
        _add_seasonality(x, period * 2, amp * 2.5, rng.uniform(0, 2 * np.pi))
        _add_noise(x, rng, amp * 0.25)
    elif pattern == "noise":
        _add_noise(x, rng, amp * 2.0)
    elif pattern == "spikes":
        _add_noise(x, rng, amp)
    elif pattern == "abrupt_change":
        _add_noise(x, rng, amp * 0.8)
        _add_abrupt_change(x, rng, amp)
    elif pattern == "mixed":
        period = int(rng.choice([7, 12, 24]))
        rho = rng.uniform(0.3, 0.85)
        innov = rng.normal(0, amp, total)
        ar = np.empty(total)
        ar[0] = innov[0]
        for t in range(1, total):
            ar[t] = rho * ar[t - 1] + innov[t]
        x += ar
        if rng.random() < 0.6:
            _add_seasonality(x, period, amp * 5.0, rng.uniform(0, 2 * np.pi))
        else:
            period = 1
        if rng.random() < 0.4:
            _add_abrupt_change(x, rng, amp)
    else:
        raise ValueError(f"unknown pattern {pattern!r}")

    lo, hi = (spike_lambda_range if spike_lambda_range is not None
              else SPIKE_LAMBDA.get(pattern, (0.0, 2.0)))
    _inject_spikes(x, rng, float(rng.uniform(lo, hi)) if hi > lo else lo)

    # positive level: keeps |mean| and |x[0]| away from zero
    x += float(rng.uniform(*level_range))
    return x, pattern, period


# --------------------------------------------------------------------------
# cycle 2 generator: every component is RECORDED, and that record is the key
# --------------------------------------------------------------------------
# PREREGISTRATION_C2.md amendment A9. Cycle 1 (and the first C2 draft) keyed
# the qualitative slots on statistics COMPUTED from the window, so any tool
# that computed the same statistics reproduced the key - the cells were
# saturated (reviews #1-#3). Here the key is WHAT WAS INJECTED: a trend or
# not, a seasonal cycle or not, one spike / one level shift / nothing, and
# where. A tool's numbers are then imperfect evidence for a fact neither arm
# can read off, so both arms can be wrong.
#
# Three design rules, each answering a measured defect:
#
#   * FACTORIAL AND BALANCED. trend (3) x seasonal (2) x event (3) x noise
#     (2) are crossed, so no slot's majority class is far above chance.
#     Cycle 1's trend key said `stable` for 93% of items at L=32.
#   * EVENTS SIT NEAR THE END. The event index is drawn inside the last
#     `event_zone` points of the history. Cycle 1 put level shifts at
#     30-70% of the full series, so the window at L=32/64/128 never
#     contained one (0/14, 0/14, 0/14 - review #4).
#   * NOTHING IS INJECTED INTO THE FORECAST HORIZON (C2-9). A spike is a
#     history-only event; a level shift persists, as a shift must.
#
# SEASONAL PERIODS AND STRENGTH (amendment A15, 2026-11-13, pre-data).
# The first C2 generator used short periods (6, 8, 12) with a cycle of 1-3
# noise sd, so that even L=32 held 2.6 cycles. That made the seasonal label
# independent of L, and quietly removed the thing H1 exists to measure: with
# the whole pattern visible in 32 points, more history could barely help a
# forecaster. Measured on that pool, a near-perfect forecaster improved by
# only +0.033 from L=32 to L=512, so H1's one confirmatory contrast could
# not have moved by more than that.
#
# Periods are now 12, 24, 48 and the cycle is 1.5-5 noise sd. L=128 (where
# H2 and H3 run) still holds at least 2.6 cycles of the longest period. At
# L=32 and L=64 the longer cycles are only partly visible - which is exactly
# why history helps there - so the `seasonal` slot joins `trend` (A14) and
# `event` (A13) in being fully answerable only from L=128 up. H1's
# qualitative cells were already descriptive for that reason.

C2_TRENDS = ("increasing", "decreasing", "stable")
C2_EVENTS = ("spike", "abrupt change", "no major event")
C2_NOISES = ("white", "ar")
C2_PERIODS = (12, 24, 48)
#: seasonal amplitude, in units of the noise sd
C2_SEASONAL_AMP = (1.5, 5.0)
#: Bump whenever the C2 generator's output changes for a given seed. It is
#: written into the pool manifest, so a pool built by an older generator is
#: regenerated instead of being silently reused under an unchanged config.
#:   1  periods 6/8/12, amplitude 1-3 sd   (never run on a real backbone)
#:   2  periods 12/24/48, amplitude 1.5-5 sd   (amendment A15)
C2_GENERATOR_VERSION = 2
#: default width of the end-of-history zone that holds the event
C2_EVENT_ZONE = 128
#: an event needs this many points on each side inside the zone
C2_EVENT_MARGIN = 4


def c2_design_cells(n: int, rng: np.random.Generator) -> list:
    """`n` (trend, seasonal, event, noise) cells: balanced, then shuffled.

    The 36 crossed cells are repeated to length `n` and permuted, so any pool
    size is as balanced as it can be and the cell an item gets is independent
    of its index (cycle 1 assigned `pattern` cyclically, which is why its
    seed-2 replication was seed-stable but not pattern-mix-stable).
    """
    cells = [(t, s, e, z) for t in C2_TRENDS for s in (True, False)
             for e in C2_EVENTS for z in C2_NOISES]
    order = rng.permutation(len(cells))
    out = [cells[order[i % len(cells)]] for i in range(n)]
    return [out[j] for j in rng.permutation(n)]


def generate_base_series_c2(n: int, cell: tuple, rng: np.random.Generator,
                            horizon: int, level_range=(20.0, 120.0),
                            event_zone: int = C2_EVENT_ZONE) -> tuple:
    """One C2 series. Returns (series, truth) with `truth` the answer record.

    All magnitudes are in units of the noise sd `sigma`, so difficulty does
    not depend on the level or scale of a series.
    """
    trend, seasonal, event, noise = cell
    total = n + horizon
    zone = int(min(event_zone, n))
    sigma = float(rng.uniform(0.5, 2.0))

    if noise == "ar":
        rho = float(rng.uniform(0.5, 0.8))
        innov = rng.normal(0, sigma * np.sqrt(1 - rho * rho), total)
        x = np.empty(total)
        x[0] = rng.normal(0, sigma)
        for t in range(1, total):
            x[t] = rho * x[t - 1] + innov[t]
    else:
        rho = 0.0
        x = rng.normal(0, sigma, total)

    # trend: a rise of 1-3 noise sd per `zone` points, over the WHOLE series
    rise = float(rng.uniform(1.0, 3.0)) * sigma
    if trend != "stable":
        sign = 1.0 if trend == "increasing" else -1.0
        x += sign * rise / zone * np.arange(total)

    period = 1
    amp = 0.0
    if seasonal:
        period = int(rng.choice(C2_PERIODS))
        amp = float(rng.uniform(*C2_SEASONAL_AMP)) * sigma
        _add_seasonality(x, period, amp, rng.uniform(0, 2 * np.pi))

    event_index = None
    size = 0.0
    if event != "no major event":
        lo = n - zone + C2_EVENT_MARGIN
        hi = n - C2_EVENT_MARGIN            # exclusive
        event_index = int(rng.integers(lo, hi))
        if event == "spike":
            size = float(rng.uniform(6.0, 10.0)) * sigma
            x[event_index] += size
        else:
            size = float(rng.uniform(4.0, 8.0)) * sigma \
                * float(rng.choice([-1.0, 1.0]))
            # the step is BETWEEN event_index and event_index+1, matching
            # the "step from i to i+1" convention used everywhere else
            x[event_index + 1:] += size

    x += float(rng.uniform(*level_range))
    truth = dict(trend=trend, seasonal=bool(seasonal), period=period,
                 event=event, event_index=event_index, noise=noise,
                 sigma=sigma, rho=rho, trend_rise=rise if trend != "stable"
                 else 0.0, seasonal_amp=amp, event_size=size)
    return x, truth


def c2_pattern_label(truth: dict) -> str:
    """Compact label for the `pattern` column of a C2 row."""
    return "|".join([truth["trend"],
                     "seasonal" if truth["seasonal"] else "nonseasonal",
                     truth["event"].replace(" ", "_"), truth["noise"]])


class SyntheticDataset(Dataset):
    """Deterministic given (seed, data config)."""

    name = "synthetic"

    def __init__(self, patterns=None, level_range=(20.0, 120.0),
                 spike_lambda_range=None, design: str = "c1",
                 event_zone: int = C2_EVENT_ZONE):
        if design not in ("c1", "c2"):
            raise ValueError(f"unknown data design {design!r}")
        self.design = design
        self.event_zone = int(event_zone)
        self.patterns = tuple(patterns or PATTERNS)
        self.level_range = tuple(level_range)
        #: None => use the per-pattern SPIKE_LAMBDA table (the tuned default).
        #: A explicit (lo, hi) overrides it for every pattern, which is only
        #: useful for ablating the item design itself.
        self.spike_lambda_range = (tuple(spike_lambda_range)
                                   if spike_lambda_range is not None else None)

    def load(self, n_base_series: int, n_per_query_type: int,
             query_types: list, max_history: int, horizon: int,
             seed: int) -> list:
        rng = np.random.default_rng(seed)
        base = []
        if self.design == "c2":
            cells = c2_design_cells(n_base_series, rng)
            for i, cell in enumerate(cells):
                series, truth = generate_base_series_c2(
                    max_history, cell, rng, horizon, self.level_range,
                    self.event_zone)
                base.append(dict(i=i, series=series,
                                 pattern=c2_pattern_label(truth),
                                 period=truth["period"], truth=truth))
        for i in range(n_base_series if self.design == "c1" else 0):
            pattern = self.patterns[i % len(self.patterns)]
            series, pat, period = generate_base_series(
                max_history, pattern, rng, horizon,
                self.level_range, self.spike_lambda_range)
            base.append(dict(i=i, series=series, pattern=pat, period=period))

        samples: list = []
        for qt in query_types:
            name = qt.name if hasattr(qt, "name") else qt
            kind = getattr(qt, "kind", "qualitative")
            offset = _rotation(name)
            for k in range(n_per_query_type):
                b = base[(k + offset) % len(base)]
                # full type name in the id: 4-char prefixes collide the
                # moment someone adds a type (`counting`/`count_events`)
                sid = f"b{b['i']:04d}-{name}-{k:03d}"
                # Balanced, deterministic paraphrase assignment, SPREAD over
                # the whole variant space.
                #
                # `k % N_VARIANTS` would be wrong below full size: with
                # n_per_query_type=14 it only ever emits variants 0..13,
                # which all share lead_id=0, so the lead axis is never
                # exercised and the within-framing variance check reports a
                # vacuous zero. Striding covers both axes at any pool size
                # and reduces to `k` when n_per_query_type == N_VARIANTS.
                variant = (k * N_VARIANTS) // max(n_per_query_type, 1)
                samples.append(Sample(
                    sample_id=sid, series=b["series"],
                    max_history=max_history, horizon=horizon,
                    query_type=name,
                    query=build_query(name, horizon, variant,
                                      design=self.design),
                    query_kind=kind, pattern=b["pattern"],
                    seasonal_period=b["period"],
                    paraphrase_id=variant, framing_id=framing_of(variant),
                    lead_id=lead_of(variant),
                    metadata=_metadata(b, self.design)))
        return samples


def _metadata(b: dict, design: str) -> dict:
    """Runner-side metadata. In C2 it carries the generator's record, which
    IS the qualitative answer key - and which `Sample.view()` never copies
    into the `InferenceSample` a system receives."""
    md = dict(base_index=b["i"])
    if design == "c2":
        md["design"] = "c2"
        md["truth"] = dict(b["truth"])
    return md


def _rotation(name: str) -> int:
    """Stable per-type rotation so each type sees the same pool in a
    different order (keeps cells paired without aligning item difficulty)."""
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) % 997


# --------------------------------------------------------------------------
# frozen pool on disk
# --------------------------------------------------------------------------
def materialize(cfg, pool_dir: str | Path | None = None,
                root: Path | None = None) -> list:
    """Generate or reload the frozen pool for `cfg`.

    The manifest records the DATA FINGERPRINT, which includes the seed, so a
    reseeded config can never be served a stale pool.
    """
    d = cfg.data
    directory = Path(pool_dir) if pool_dir is not None else cfg.pool_dir(
        root or Path.cwd())
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    want = dict(
        fingerprint=cfg.data_fingerprint(), seed=cfg.seed,
        query_version=src.QUERY_VERSION, n_variants=N_VARIANTS,
        n_base_series=int(d["n_base_series"]),
        n_per_query_type=int(d["n_per_query_type"]),
        query_types=[qt.name for qt in cfg.query_types],
        max_history=int(d["max_history"]), horizon=int(d["horizon"]),
        patterns=list(d.get("patterns", PATTERNS)),
        level_range=list(d.get("level_range", [20.0, 120.0])),
        spike_lambda_range=d.get("spike_lambda_range"))
    design = str(d.get("design", "c1"))
    event_zone = int(d.get("event_zone", C2_EVENT_ZONE))
    if design != "c1":
        # Added to the manifest ONLY for a non-default design, so every
        # cycle-1 manifest on disk still compares equal and no archived pool
        # is regenerated.
        want["design"] = design
        want["event_zone"] = event_zone
        want["generator_version"] = C2_GENERATOR_VERSION

    if manifest_path.exists():
        have = json.loads(manifest_path.read_text(encoding="utf-8"))
        if have == want and (directory / "pool.npz").exists():
            return _rehydrate(directory, cfg)

    ds = SyntheticDataset(
        patterns=d.get("patterns", PATTERNS),
        level_range=tuple(d.get("level_range", (20.0, 120.0))),
        spike_lambda_range=d.get("spike_lambda_range"),
        design=design, event_zone=event_zone)
    samples = ds.load(int(d["n_base_series"]), int(d["n_per_query_type"]),
                      cfg.query_types, int(d["max_history"]),
                      int(d["horizon"]), cfg.seed)
    uniq: dict[int, np.ndarray] = {}
    meta = []
    for s in samples:
        bi = s.metadata["base_index"]
        uniq.setdefault(bi, s.series)
        rec = dict(sample_id=s.sample_id, base_index=bi,
                   query_type=s.query_type, query_kind=s.query_kind,
                   pattern=s.pattern, seasonal_period=s.seasonal_period,
                   paraphrase_id=s.paraphrase_id)
        if "truth" in s.metadata:
            rec["truth"] = s.metadata["truth"]
        meta.append(rec)
    np.savez_compressed(directory / "pool.npz",
                        **{f"s{k}": v for k, v in uniq.items()})
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    manifest_path.write_text(json.dumps(want, indent=1), encoding="utf-8")
    return samples


def _rehydrate(directory: Path, cfg) -> list:
    d = cfg.data
    arrays = np.load(directory / "pool.npz")
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    design = str(d.get("design", "c1"))
    out = []
    for m in meta:
        # paraphrase_id MUST come from the manifest, not be recomputed:
        # rebuilding the query from scratch would silently drift from the
        # pool that was actually run if the bank ever changed.
        variant = int(m["paraphrase_id"])
        md = dict(base_index=m["base_index"])
        if "truth" in m:
            md["design"] = design
            md["truth"] = m["truth"]
        out.append(Sample(
            sample_id=m["sample_id"], series=arrays[f"s{m['base_index']}"],
            max_history=int(d["max_history"]), horizon=int(d["horizon"]),
            query_type=m["query_type"],
            query=build_query(m["query_type"], int(d["horizon"]), variant,
                              design=design),
            query_kind=m["query_kind"], pattern=m["pattern"],
            seasonal_period=int(m["seasonal_period"]),
            paraphrase_id=variant, framing_id=framing_of(variant),
            lead_id=lead_of(variant), metadata=md))
    return out
