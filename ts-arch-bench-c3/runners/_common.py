"""Shared machinery for the H1/H2/H3 runners.

WHY THE RUNNERS ARE SHAPED LIKE THIS
------------------------------------
Two constraints from the execution environment (PREREGISTRATION.md §6,
"Execution environment") drive the whole design:

1. SLURM, with a busy queue. One long process that walks the entire grid is
   the worst possible shape: it holds a GPU for hours, dies at the time
   limit, and loses everything. So inference is split into INDEPENDENT CELLS,
   one per (backbone, L), each written to its own shard file and each
   skippable if already complete. An `sbatch --array` then fans the grid
   across the queue, and a re-submit after a timeout resumes rather than
   restarts.

2. `/scratch` deletes files after 30 days. The call cache is therefore a
   PRIMARY ARTIFACT, not a convenience: a cold re-run under vLLM's
   continuous batching is not guaranteed to reproduce the same replies, so
   `calls.jsonl` is what makes a re-analysis exact. It is archived, never
   assumed to be regenerable.

ANALYSIS IS A SEPARATE STAGE, ALWAYS
------------------------------------
Inference writes shards. Analysis reads shards. Nothing in the analysis path
touches a model, so the whole statistical layer can be re-run as often as
you like at zero GPU cost - which is the property that makes a pre-registered
sensitivity grid affordable rather than aspirational.

ONE CONTRAST HELPER, USED BY ALL THREE
--------------------------------------
H2 tests the gap. H1 and H3 test whether the gap MOVES along a short axis,
via the item-level difference-in-differences declared in §6. Both reduce to
`paired_compare` on a per-item vector, so there is one implementation of each
and the runners differ only in what they feed it.
"""
from __future__ import annotations

import copy
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import selective_report          # noqa: E402
from src.config import (Config, load_config,             # noqa: E402
                        save_config_snapshot, a_priori_saturated,
                        VALID_CLUSTER_UNITS as CFG_CLUSTER_UNITS)
from src.pipeline import (Registry, make_samples, run_inference,  # noqa: E402
                          evaluate_run, fixed_denominators)
from src.screening import screen_report                  # noqa: E402
from src.stats import (CORRECTIONS, paired_compare,      # noqa: E402
                       paired_compare_clustered, spearman_with_ci,
                       exact_spearman_p, min_detectable_rho, paired_mde)

log = logging.getLogger("tsarch.runner")

#: every column the analysis stage needs from a shard
SHARD_KEYS = ("sample_id", "system", "backbone", "L", "query_type",
              "query_kind", "pattern", "dataset_op", "paraphrase_id",
              "framing_id", "lead_id", "score", "answered", "abstained",
              "failure", "type_valid", "route_op", "route_status",
              "route_source", "route_correct", "route_correct_selective",
              "route_equivalent",
              # cycle 3: provenance of a REAL item. Empty on a synthetic
              # pool. `dataset_id` is the conservative clustering unit;
              # `stratum` is the reframed H2 axis.
              "dataset_id", "stratum",
              "n_model_calls", "n_billed_calls", "latency_s")

#: Re-exported from `src.config` so the analysis stage and the config
#: validator cannot drift. Declared as `analysis.cluster_unit` and fixed
#: BEFORE the run - the whole point of the pre-registered framing rule in
#: `src/screening.py` is that the clustering level is not chosen after
#: seeing the variance, and a real pool must not be an exception.
VALID_CLUSTER_UNITS = CFG_CLUSTER_UNITS

#: Sweep axis -> the shard column holding it. `L` is H1, `backbone` is H3.
#: `stratum` is cycle 3's reframed H2: on real data every item is a
#: forecasting item, so "does the gap depend on the question?" has no
#: contrast left and the interesting question becomes "does it depend on the
#: SERIES?" - which is the cycle-2 post-hoc finding (+0.170 on seasonal
#: series against +0.014 on non-seasonal ones at 32B) promoted to a declared
#: test. Unlike L and backbone, the two ends hold DIFFERENT ITEMS, so the
#: contrast there is unpaired; `confirmatory_did` refuses it and
#: `runners/h2_series_character.py` runs the two-sample form instead.
AXIS_COLUMN = {"L": "L", "backbone": "backbone", "stratum": "stratum"}


# --------------------------------------------------------------------------
# banners: the mock is an oracle, and the reader must be told in the OUTPUT
# --------------------------------------------------------------------------
def mock_banner(cfg: Config) -> str | None:
    """A loud, unmissable warning when any backbone is the mock.

    `MockLLM._do_answer` imports `src.ground_truth` and synthesises answers
    at a target accuracy. That is the right way to build a
    strength-parameterised simulator and the wrong thing to quote: mock
    numbers validate plumbing, never measurement. A comment in the config is
    not enough, because tables get copied out of stdout.
    """
    mocks = sorted(n for n, b in cfg.backbones.items() if b.provider == "mock")
    if not mocks:
        return None
    bar = "!" * 76
    return "\n".join([
        bar,
        "!! MOCK BACKBONE IN USE: " + ", ".join(mocks),
        "!!",
        "!! MockLLM reads src.ground_truth to generate answers of a target",
        "!! accuracy. Every number below therefore validates PLUMBING ONLY.",
        "!! No table from this run may be quoted as a measurement, in a",
        "!! paper, a slide, an issue or a commit message.",
        bar])


def cost_note() -> str:
    return ("n_model_calls counts architectural calls; n_billed_calls counts "
            "cache misses that actually hit a backend. They are NOT "
            "comparable across arms: the oracle arm skips classification "
            "entirely and issues zero calls on numeric cells by design.")


# --------------------------------------------------------------------------
# config narrowing: one backbone per process
# --------------------------------------------------------------------------
def narrow_config(cfg: Config, backbone: str) -> Config:
    """Return a copy of `cfg` carrying exactly one backbone.

    `Registry` constructs every backbone in the config. With a local
    provider that means loading every model in the H3 ladder into one
    process, which is both impossible and pointless. Narrowing keeps the
    SYSTEM NAMES identical across backbones (`bare_llm`, `harness`,
    `harness_oracle`) so the analysis stage can pair them, while the
    `backbone` column on every row records which model produced it.
    """
    if backbone not in cfg.backbones:
        raise KeyError(
            f"backbone {backbone!r} is not in the config; have: "
            f"{sorted(cfg.backbones)}")
    out = copy.deepcopy(cfg)
    out.backbones = {backbone: cfg.backbones[backbone]}
    for s in out.systems:
        s.backbone = backbone
    out.validate()
    return out


# --------------------------------------------------------------------------
# inference stage: one shard per (backbone, L)
# --------------------------------------------------------------------------
def shard_path(out_dir: Path, backbone: str, L: int) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in backbone)
    return Path(out_dir) / "shards" / f"bb={safe}__L={int(L)}.csv"


def run_cell(cfg: Config, backbone: str, L: int, out_dir: Path,
             resume: bool = True, no_cache: bool = False,
             cache_dir: Path | None = None) -> Path:
    """Run every system on every sample at one (backbone, L). Idempotent.

    `cache_dir` defaults to `<out_dir>/cache` but is worth overriding to a
    single shared location across H1/H2/H3. The call key is
    (backbone fingerprint, PROMPT_VERSION, prompts) and carries neither the
    experiment name nor L, so H2's cells and H3's L=128 rung for the same
    model render byte-identical prompts. One shared cache pays for them once;
    separate caches pay three times for the same GPU work.
    """
    out_dir = Path(out_dir)
    dest = shard_path(out_dir, backbone, L)
    if resume and dest.exists() and dest.stat().st_size > 0:
        log.info("shard exists, skipping: %s", dest.name)
        return dest

    narrow = narrow_config(cfg, backbone)
    samples = make_samples(narrow, ROOT)
    reg = Registry(narrow, cache_dir=Path(cache_dir or (out_dir / "cache")),
                   enabled=not no_cache)
    log.info("cell bb=%s L=%d: %d samples x %d systems",
             backbone, L, len(samples), len(reg.systems))

    results = run_inference(narrow, samples, L, reg.systems, reg)
    rows = evaluate_run(narrow, samples, L, results, judge=reg.judge,
                        denominators=fixed_denominators(narrow, samples))
    df = pd.DataFrame(rows)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    stats = reg.cache_stats()
    log.info("wrote %s (%d rows); cache=%s", dest.name, len(df), stats)
    if stats.get("truncated_calls"):
        log.warning(
            "%d model call(s) hit max_tokens (finish_reason=length). A reply "
            "cut off before its answer is scored as unanswered. They are "
            "marked meta.call.finish_reason in calls.jsonl.",
            stats["truncated_calls"])
    return dest


def load_shards(out_dir: Path, expect_backbones=None,
                expect_lengths=None) -> pd.DataFrame:
    """Concatenate every shard, and say plainly what is missing."""
    shards = sorted((Path(out_dir) / "shards").glob("bb=*__L=*.csv"))
    if not shards:
        raise FileNotFoundError(
            f"no shards under {Path(out_dir) / 'shards'}. Run the inference "
            "stage first (--stage inference, or the sbatch array).")
    df = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)

    missing = []
    for bb in (expect_backbones or []):
        for L in (expect_lengths or []):
            if not shard_path(out_dir, bb, L).exists():
                missing.append(f"bb={bb} L={L}")
    if missing:
        raise RuntimeError(
            "the analysis stage refuses to run on a partial grid; these "
            "cells have no shard:\n  " + "\n  ".join(missing) +
            "\nRe-submit the array; completed cells are skipped.")
    return df


# --------------------------------------------------------------------------
# per-item contrast vectors
# --------------------------------------------------------------------------
def item_gap(df: pd.DataFrame, pair, query_type: str,
             backbone: str | None = None, L: int | None = None,
             strict: bool = True):
    """`d_i = score(pair[1])_i - score(pair[0])_i`, aligned by item.

    Returns (sample_ids, framing_ids, d). Items are aligned on `sample_id`,
    never on row order.

    `strict=True` (the default, and what every PRIMARY analysis uses) makes a
    missing arm a hard error. This matters more than it looks: the aligning
    pivot has to drop an item that is absent from either arm, and a silent
    drop here would be the `.dropna()` bug that PREREGISTRATION.md §1.2
    exists to forbid - every reported mean would quietly become conditional
    on both arms having produced a row. Failures are scored zero upstream, so
    in the primary path NOTHING may go missing, and if something does it is a
    bug to be fixed rather than a row to be skipped.

    `strict=False` is passed by exactly one caller: the `failures="dropped"`
    arm of the pre-registered sensitivity grid, where unanswered items have
    been removed ON PURPOSE to show what that choice would have changed.
    """
    lo, hi = pair
    sub = df[df.query_type == query_type]
    if backbone is not None:
        sub = sub[sub.backbone == backbone]
    if L is not None:
        sub = sub[sub.L == int(L)]
    piv = sub.pivot_table(index="sample_id", columns="system", values="score",
                          aggfunc="first")
    for arm in (lo, hi):
        if arm not in piv.columns:
            raise KeyError(
                f"system {arm!r} absent for query_type={query_type!r} "
                f"backbone={backbone!r} L={L!r}; have {list(piv.columns)}")
    both = piv[[lo, hi]].dropna()
    if len(both) < len(piv):
        lost = [str(i) for i in piv.index.difference(both.index)]
        if strict:
            raise RuntimeError(
                f"{query_type}: {len(lost)} of {len(piv)} items lack a score "
                f"in {lo!r} or {hi!r} (backbone={backbone!r} L={L!r}): "
                f"{lost[:5]}{' ...' if len(lost) > 5 else ''}.\n"
                "Every item is scored - a failure scores ZERO - so in the "
                "primary analysis nothing may be missing. Dropping these "
                "would make the reported mean conditional on both arms "
                "answering, which is the .dropna() defect in "
                "PREREGISTRATION.md §1.2. Fix the shard, do not skip the "
                "items. (The sensitivity grid's failures='dropped' arm is "
                "the one legitimate exception and passes strict=False.)")
        log.debug("%s: %d of %d items dropped by failures='dropped'",
                  query_type, len(lost), len(piv))
    fam = (sub.drop_duplicates("sample_id").set_index("sample_id")
           ["framing_id"].reindex(both.index))
    return both.index.to_numpy(), fam.to_numpy(), (both[hi] - both[lo]).to_numpy()


def cluster_labels(df: pd.DataFrame, sample_ids, unit: str) -> np.ndarray:
    """Cluster label per `sample_id`, read from the shard's own columns.

    `sample_id` reduces `paired_compare_clustered` to `paired_compare`, so
    it is the identity case and needs no column. Anything else must be
    PRESENT and non-null for every item: silently falling back to the item
    when `dataset_id` is missing would turn a clustered test into an
    unclustered one and report a confidence interval that is too narrow -
    the kind of quiet anti-conservatism this build refuses elsewhere
    (see `item_gap`'s strict mode).
    """
    ids = np.asarray(sample_ids)
    if unit == "sample_id":
        return ids
    if unit not in VALID_CLUSTER_UNITS:
        raise ValueError(f"unknown cluster unit {unit!r}; have "
                         f"{list(VALID_CLUSTER_UNITS)}")
    if unit not in df.columns:
        raise KeyError(
            f"cluster_unit={unit!r} but the shards have no {unit!r} column. "
            "Re-run inference: the column is written by "
            "src/pipeline.py::evaluate_run.")
    lut = df.drop_duplicates("sample_id").set_index("sample_id")[unit]
    out = lut.reindex(ids)
    if out.isna().any():
        missing = [str(i) for i in out.index[out.isna()][:5]]
        raise ValueError(
            f"cluster_unit={unit!r} is null for {int(out.isna().sum())} "
            f"items, e.g. {missing}. A synthetic pool leaves these columns "
            "empty; only a real pool can be clustered on them.")
    return out.to_numpy()


def _compare(d, clusters, n_boot: int, seed: int):
    """`paired_compare`, or its cluster bootstrap when clusters are coarser.

    One helper so the confirmatory path cannot drift from the secondary
    one: both reduce to the ordinary paired test when the cluster IS the
    item, which is what keeps every cycle-1 and cycle-2 number reproducible
    bit for bit after this was added.
    """
    zeros = np.zeros_like(d)
    if clusters is None or len(np.unique(clusters)) == len(d):
        return paired_compare(d, zeros, n_boot=n_boot, seed=seed)
    return paired_compare_clustered(d, zeros, clusters, n_boot=n_boot,
                                    seed=seed)


def declared_cluster_unit(cfg: Config) -> str:
    """`analysis.cluster_unit`, defaulting to the item.

    Cycle 1 and 2 declare nothing here and get `sample_id`, which is the
    behaviour every archived result was produced under. A real-data config
    declares `dataset_id`, because 300 windows drawn from 4 archives are not
    300 independent observations (PREREGISTRATION_C3.md C3-3).
    """
    unit = str(cfg.analysis.get("cluster_unit") or "sample_id")
    if unit not in VALID_CLUSTER_UNITS:
        raise ValueError(
            f"analysis.cluster_unit={unit!r} is not one of "
            f"{list(VALID_CLUSTER_UNITS)}")
    return unit


def _restrict(df: pd.DataFrame, drop_unanswered: bool) -> pd.DataFrame:
    """The `failures-as-zero` vs `dropped` axis of the sensitivity grid.

    Dropping is WRONG as a primary (§1.2) - it is the `.dropna()` bug that
    made every old reported mean conditional on parsing succeeding - but the
    grid is required to show what it would have changed.
    """
    if not drop_unanswered:
        return df
    return df[df.answered.astype(bool)]


# --------------------------------------------------------------------------
# confirmatory analyses
# --------------------------------------------------------------------------
def _verdict(res, holm_p: float, alpha: float, min_effect: float) -> dict:
    ci_excludes_zero = bool(
        np.isfinite(res.ci_low) and np.isfinite(res.ci_high)
        and (res.ci_low > 0 or res.ci_high < 0))
    big_enough = bool(np.isfinite(res.mean_diff)
                      and abs(res.mean_diff) >= min_effect)
    significant = bool(np.isfinite(holm_p) and holm_p < alpha)
    return dict(
        p_adj=holm_p, significant=significant,
        ci_excludes_zero=ci_excludes_zero, meets_effect_floor=big_enough,
        win=bool(significant and ci_excludes_zero and big_enough),
        favours=("harness" if res.mean_diff > 0 else
                 "bare_llm" if res.mean_diff < 0 else "tie"))


def confirmatory_single(df: pd.DataFrame, cfg: Config, *,
                        backbone: str | None = None, L: int | None = None,
                        correction: str = "holm", min_effect: float | None = None,
                        cells=None, drop_unanswered: bool = False) -> dict:
    """H2's test: the gap within each declared cell, at one (backbone, L)."""
    an = cfg.analysis
    pair = tuple(an["confirmatory_contrast"])
    cells = list(cells if cells is not None else an["confirmatory_family"])
    m = len(cells)
    alpha = float(an.get("alpha", 0.05))
    min_effect = float(an.get("min_effect", 0.02)
                       if min_effect is None else min_effect)
    work = _restrict(df, drop_unanswered)
    unit = declared_cluster_unit(cfg)

    per_cell, praw = {}, []
    for qt in cells:
        ids, _, d = item_gap(work, pair, qt, backbone=backbone, L=L,
                             strict=not drop_unanswered)
        res = _compare(d, cluster_labels(work, ids, unit),
                       int(an.get("n_boot", 5000)), int(cfg.seed))
        per_cell[qt] = res
        praw.append(res.p_raw)

    adj = CORRECTIONS[correction](praw, m) if correction != "none" else praw
    out = {}
    for qt, padj in zip(cells, adj):
        res = per_cell[qt]
        out[qt] = dict(asdict(res), **_verdict(res, padj, alpha, min_effect))
    return dict(
        test="gap within cell", contrast=list(pair), backbone=backbone, L=L,
        declared_m=m, correction=correction, alpha=alpha,
        min_effect=min_effect, failures="dropped" if drop_unanswered else "zero",
        cluster_unit=unit, cells=out,
        mde_dz=paired_mde(int(np.median([v["n"] for v in out.values()]) or 0),
                          n_tests=m, alpha=alpha))


def confirmatory_did(df: pd.DataFrame, cfg: Config, *, axis: str,
                     first, second, correction: str = "holm",
                     min_effect: float | None = None, cells=None,
                     drop_unanswered: bool = False) -> dict:
    """H1's and H3's test: does the gap MOVE between two ends of an axis?

        Delta_i = d_i(first) - d_i(second)

    declared in §6. `axis` is "L" for H1 and "backbone" for H3. This is an
    item-level paired test with n = items per cell, deliberately NOT a rank
    test over the sweep points: with five or seven points a Spearman's
    `min_detectable_rho` is near 1 and could only ever return "not detected".
    """
    an = cfg.analysis
    pair = tuple(an["confirmatory_contrast"])
    cells = list(cells if cells is not None else an["confirmatory_family"])
    m = len(cells)
    alpha = float(an.get("alpha", 0.05))
    min_effect = float(an.get("min_effect", 0.02)
                       if min_effect is None else min_effect)
    work = _restrict(df, drop_unanswered)
    if axis == "stratum":
        raise ValueError(
            "a difference-in-differences along `stratum` is not valid: L and "
            "backbone hold the SAME items at both ends (invariants I1/I2), "
            "but a weak-seasonality item and a strong-seasonality one are "
            "different series, so Delta_i does not exist. Use "
            "runners/h2_series_character.py, which contrasts the two "
            "per-item gap distributions with a two-sample test.")
    key = AXIS_COLUMN[axis]
    unit = declared_cluster_unit(cfg)
    if unit == key:
        raise ValueError(
            f"cannot cluster on {unit!r} while contrasting along it: the "
            "two ends of the axis would fall in the same cluster and the "
            "bootstrap would resample nothing. Declare a different "
            "analysis.cluster_unit for this hypothesis.")

    per_cell, praw = {}, []
    for qt in cells:
        kw_a = {key: first}
        kw_b = {key: second}
        strict = not drop_unanswered
        ids_a, fam_a, d_a = item_gap(work, pair, qt, strict=strict, **kw_a)
        ids_b, _, d_b = item_gap(work, pair, qt, strict=strict, **kw_b)
        sa = pd.Series(d_a, index=ids_a)
        sb = pd.Series(d_b, index=ids_b)
        joined = pd.concat([sa.rename("a"), sb.rename("b")], axis=1).dropna()
        # Delta_i is one number per ITEM, so the cluster labels are read for
        # the items that survived the join, not for either end separately.
        delta = (joined["a"] - joined["b"]).to_numpy()
        res = _compare(delta,
                       cluster_labels(work, joined.index.to_numpy(), unit),
                       int(an.get("n_boot", 5000)), int(cfg.seed))
        per_cell[qt] = res
        praw.append(res.p_raw)

    adj = CORRECTIONS[correction](praw, m) if correction != "none" else praw
    out = {}
    for qt, padj in zip(cells, adj):
        res = per_cell[qt]
        v = _verdict(res, padj, alpha, min_effect)
        v["favours"] = ("gap larger at %s" % first if res.mean_diff > 0 else
                        "gap larger at %s" % second if res.mean_diff < 0
                        else "no change")
        out[qt] = dict(asdict(res), **v)
    return dict(
        test=f"difference-in-differences along {axis}",
        contrast=list(pair), axis=axis, first=first, second=second,
        declared_m=m, correction=correction, alpha=alpha,
        min_effect=min_effect, cluster_unit=unit,
        failures="dropped" if drop_unanswered else "zero",
        interpretation=(f"Delta_i = d_i({first}) - d_i({second}); "
                        "positive means the harness's advantage is larger at "
                        f"{first}"),
        cells=out)


def secondary_ablation(df: pd.DataFrame, cfg: Config, cluster_unit: str,
                       *, backbone: str | None = None,
                       L: int | None = None) -> dict:
    """`harness_oracle` vs `harness`, OUTSIDE the correction (§3).

    Clustered at the unit the pre-registered rule selected, because the
    quantity driving this contrast is the routing decision and every item
    sharing a phrasing shares one.
    """
    an = cfg.analysis
    pair = tuple(an.get("secondary_contrast") or ["harness", "harness_oracle"])
    out = {}
    for qt in sorted(df.query_type.unique()):
        try:
            ids, fam, d = item_gap(df, pair, qt, backbone=backbone, L=L)
        except KeyError as exc:
            out[qt] = dict(error=str(exc))
            continue
        clusters = (fam if cluster_unit == "framing_id"
                    else cluster_labels(df, ids, cluster_unit))
        res = paired_compare_clustered(
            d, np.zeros_like(d), clusters,
            n_boot=int(an.get("n_boot", 5000)), seed=int(cfg.seed))
        out[qt] = dict(asdict(res), cluster_unit=cluster_unit)
    return dict(
        test="secondary ablation (outside the correction)",
        contrast=list(pair), cluster_unit=cluster_unit,
        note=("Reported with effects and CIs only. No multiplicity "
              "correction is applied and no win/loss verdict is derived: "
              "this is a pre-registered secondary ablation (§3), not a "
              "confirmatory test."),
        cells=out)


def clustering_sensitivity(df: pd.DataFrame, cfg: Config, analyse,
                           cells, **kw) -> dict:
    """Re-run the confirmatory analysis under each declared cluster unit.

    `analysis.cluster_unit` is the PRIMARY, fixed before the run.
    `analysis.cluster_unit_sensitivity` lists the alternatives, and this
    reports them beside it so a reader can see whether the verdict depends
    on the choice. Never substituted for the primary - the same rule §2
    applies to the screened-versus-declared verdicts.

    On a real pool the interesting pair is `sample_id` (one window per
    series, so items are distinct series) against `dataset_id` (series
    inside one archive share a sensor network, so the effective n is the
    number of archives). They can disagree a lot; that disagreement IS the
    finding, so it is printed rather than resolved.
    """
    alts = list(cfg.analysis.get("cluster_unit_sensitivity") or [])
    if not alts:
        return {}
    primary = declared_cluster_unit(cfg)
    out = {}
    for unit in [primary] + [u for u in alts if u != primary]:
        alt_cfg = copy.deepcopy(cfg)
        alt_cfg.analysis = dict(cfg.analysis, cluster_unit=unit)
        try:
            res = analyse(df, alt_cfg, cells=cells, **kw)
        except (KeyError, ValueError) as exc:
            out[unit] = dict(error=str(exc))
            continue
        out[unit] = {qt: dict(
            n=v["n"], n_clusters=v.get("n_clusters"),
            mean_diff=v["mean_diff"], ci_low=v["ci_low"],
            ci_high=v["ci_high"], p_raw=v["p_raw"], p_adj=v["p_adj"],
            win=v["win"]) for qt, v in res["cells"].items()}
    return dict(primary=primary, by_unit=out,
                note=("The primary unit is declared in the config before "
                      "the run. Alternatives are reported, never "
                      "substituted. A verdict that holds under one unit and "
                      "not the other must be reported as exactly that."))


def print_clustering_sensitivity(rep: dict) -> None:
    if not rep:
        return
    print("\n=== clustering sensitivity (primary declared in advance: "
          f"{rep['primary']}) ===")
    hdr = (f"{'unit':<14}{'cell':<18}{'n':>5}{'k':>6}{'mean_d':>10}"
           f"{'95% CI':>19}{'p_adj':>10}{'win':>5}")
    print(hdr)
    print("-" * len(hdr))
    for unit, cells in rep["by_unit"].items():
        if "error" in cells:
            print(f"{unit:<14}{'-':<18}  {str(cells['error'])[:60]}")
            continue
        for qt, v in cells.items():
            ci = f"[{v['ci_low']:+.3f},{v['ci_high']:+.3f}]"
            k = v.get("n_clusters")
            pa = ("nan" if v["p_adj"] != v["p_adj"] else f"{v['p_adj']:.4g}")
            print(f"{unit:<14}{qt:<18}{v['n']:>5}"
                  f"{(k if k is not None else '-'):>6}{v['mean_diff']:>+10.4f}"
                  f"{ci:>19}{pa:>10}{'YES' if v['win'] else '-':>5}")
    print("\n" + rep["note"])


# --------------------------------------------------------------------------
# the mandatory sensitivity grid (§3)
# --------------------------------------------------------------------------
def decidable_cells(screen: dict, declared) -> list:
    """Cells the confirmatory contrast can decide, per the per-pair screen."""
    cs = (screen.get("confirmatory_contrast_screen") or {})
    return [q for q in declared if (cs.get(q) or {}).get("decidable")]


def decidable_at_both_ends(screens: dict, declared) -> list:
    """For H1/H3: a cell counts as screened-in only if it is decidable at
    BOTH ends of the sweep axis.

    The confirmatory test is a difference-in-differences between those two
    ends, so a cell with no headroom at either end cannot show the gap
    moving. Requiring both is the conservative reading and is fixed here
    rather than chosen once the numbers are in.
    """
    sets = [set(decidable_cells(s, declared)) for s in screens.values()]
    keep = set(declared).intersection(*sets) if sets else set()
    return [q for q in declared if q in keep]


def sensitivity_grid(df: pd.DataFrame, cfg: Config, screened: list,
                     analyse, declared=None, **kw) -> list:
    """(Holm | BH) x (0.01 | 0.02 | 0.05) x (screened | unscreened)
    x (failures-as-zero | dropped).

    `analyse` is `confirmatory_single` or `confirmatory_did`, so the grid is
    identical in shape for every hypothesis. "One word is not a result."
    """
    declared = list(declared if declared is not None
                    else cfg.analysis["confirmatory_family"])
    screened = [q for q in screened if q in declared]
    rows = []
    for corr in ("holm", "bh"):
        for floor in (0.01, 0.02, 0.05):
            for scope, cells in (("unscreened", declared),
                                 ("screened", screened)):
                for drop in (False, True):
                    if not cells:
                        rows.append(dict(
                            correction=corr, min_effect=floor, scope=scope,
                            failures="dropped" if drop else "zero",
                            n_cells=0, wins=0, note="no cells in scope"))
                        continue
                    res = analyse(df, cfg, correction=corr, min_effect=floor,
                                  cells=cells, drop_unanswered=drop, **kw)
                    wins = [q for q, v in res["cells"].items() if v["win"]]
                    rows.append(dict(
                        correction=corr, min_effect=floor, scope=scope,
                        failures="dropped" if drop else "zero",
                        n_cells=len(cells), declared_m=res["declared_m"],
                        wins=len(wins), winning_cells="|".join(sorted(wins))))
    return rows


# --------------------------------------------------------------------------
# shape statistics (descriptive: H1's curve, H3's ladder)
# --------------------------------------------------------------------------
def curve(df: pd.DataFrame, cfg: Config, axis: str, points) -> list:
    """Mean gap per (cell, axis point) with bootstrap CI. Descriptive."""
    pair = tuple(cfg.analysis["confirmatory_contrast"])
    key = AXIS_COLUMN[axis]
    rows = []
    for qt in sorted(df.query_type.unique()):
        for pt in points:
            try:
                _, _, d = item_gap(df, pair, qt, **{key: pt})
            except KeyError:
                continue
            if not len(d):
                continue
            res = paired_compare(d, np.zeros_like(d),
                                 n_boot=int(cfg.analysis.get("n_boot", 5000)),
                                 seed=int(cfg.seed))
            rows.append(dict(query_type=qt, **{axis: pt}, n=res.n,
                             mean_gap=res.mean_diff, ci_low=res.ci_low,
                             ci_high=res.ci_high,
                             sign_effect=res.paired_sign_effect,
                             n_zero_diff=res.n_zero_diff))
    return rows


def rank_shape(curve_rows: list, xs: dict, axis: str) -> dict:
    """Spearman of mean gap against a declared x, reported with its own
    detection limit. DESCRIPTIVE - §6 forbids treating this as the test."""
    df = pd.DataFrame(curve_rows)
    out = {}
    for qt, sub in df.groupby("query_type"):
        x = np.array([xs.get(v, np.nan) for v in sub[axis]], float)
        y = sub["mean_gap"].to_numpy(float)
        rho, p, lo, hi, n = spearman_with_ci(x, y)
        out[str(qt)] = dict(
            n_points=n, rho=rho, p_asymptotic=p, ci_low=lo, ci_high=hi,
            p_exact_permutation=exact_spearman_p(x, y),
            min_detectable_rho=min_detectable_rho(n),
            reading=("descriptive only: with n=%d points this design could "
                     "not detect a relationship of plausible size unless "
                     "|rho| exceeds the min_detectable_rho above" % n))
    return out


# --------------------------------------------------------------------------
# artifact writing
# --------------------------------------------------------------------------
def write_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str),
                          encoding="utf-8")


def write_csv(rows, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def descriptive_tables(df: pd.DataFrame) -> dict:
    """Cell means, coverage, failure taxonomy, routing, cost - every arm."""
    out = {}
    out["score_mean"] = df.pivot_table(
        index=["query_type"], columns=["system", "backbone", "L"],
        values="score", aggfunc="mean")
    out["coverage"] = df.pivot_table(
        index=["query_type"], columns=["system", "backbone", "L"],
        values="answered", aggfunc="mean")
    sel = []
    for keys, sub in df.groupby(["system", "backbone", "L", "query_type"]):
        rep = selective_report(sub["score"].tolist(), sub["answered"].tolist())
        sel.append(dict(zip(("system", "backbone", "L", "query_type"), keys),
                        **rep))
    out["selective"] = pd.DataFrame(sel)
    rt = df[df.route_source == "llm"]
    if len(rt):
        agg = dict(route_correct=("route_correct", "mean"),
                   route_correct_selective=("route_correct_selective",
                                            "mean"),
                   n=("route_correct", "size"))
        if "route_equivalent" in rt.columns:
            agg["route_equivalent"] = ("route_equivalent", "mean")
        out["routing"] = (rt.groupby(["system", "backbone", "L", "query_type"])
                          .agg(**agg).reset_index())
    out["cost"] = (df.groupby(["system", "backbone", "L"])
                   .agg(n_model_calls=("n_model_calls", "sum"),
                        n_billed_calls=("n_billed_calls", "sum"),
                        latency_s=("latency_s", "sum")).reset_index())
    return out


def emit(cfg: Config, out: Path, df: pd.DataFrame, screen: dict,
         confirm: dict, secondary: dict, grid: list, extra: dict) -> None:
    """Write every artifact, in the order the pre-registration requires."""
    out = Path(out)
    save_config_snapshot(cfg, out / "config_used.json")
    write_json(screen, out / "screening.json")        # BEFORE any test
    write_json(dict(family=cfg.family_report(),
                    a_priori_saturated=a_priori_saturated(cfg.query_types),
                    confirmatory=confirm, secondary=secondary,
                    cost_note=cost_note(), **extra),
               out / "analysis.json")
    write_csv(grid, out / "sensitivity_grid.csv")
    for name, obj in descriptive_tables(df).items():
        p = out / f"desc_{name}.csv"
        # pivot tables carry a meaningful index; the tidy frames do not
        obj.to_csv(p, index=name in ("score_mean", "coverage"))
    df.to_csv(out / "raw_results.csv", index=False)


def print_header(cfg: Config, title: str) -> None:
    banner = mock_banner(cfg)
    print("=" * 76)
    print(title)
    print("=" * 76)
    if banner:
        print(banner)
    fam = cfg.family_report()
    print(f"confirmatory contrast : {fam['confirmatory_contrast']}")
    print(f"declared family (m={len(fam['confirmatory_family'])}) : "
          f"{fam['confirmatory_family']}")
    print(f"a priori saturated    : {fam['a_priori_saturated']}")
    print(f"saturated kept in fam : {fam['saturated_cells_kept_in_family']}")
    print(f"secondary (no corr.)  : "
          f"{cfg.analysis.get('secondary_contrast')}")
    print("-" * 76)


def print_confirmatory(res: dict, title: str) -> list:
    """Print one confirmatory family and return the winning cells."""
    print(f"\n=== {title} ===")
    print(f"test        : {res['test']}")
    print(f"contrast    : {res['contrast'][1]} - {res['contrast'][0]} "
          f"(positive favours {res['contrast'][1]})")
    print(f"family      : m={res['declared_m']} declared, "
          f"correction={res['correction']}, alpha={res['alpha']}, "
          f"effect floor={res['min_effect']}, failures={res['failures']}")
    hdr = (f"{'cell':<20}{'n':>5}{'mean_d':>10}{'95% CI':>19}"
           f"{'p_raw':>10}{'p_adj':>10}{'win':>5}")
    print(hdr)
    print("-" * len(hdr))
    for qt, v in res["cells"].items():
        ci = f"[{v['ci_low']:+.3f},{v['ci_high']:+.3f}]"
        pr = "nan" if v["p_raw"] != v["p_raw"] else f"{v['p_raw']:.4g}"
        pa = "nan" if v["p_adj"] != v["p_adj"] else f"{v['p_adj']:.4g}"
        print(f"{qt:<20}{v['n']:>5}{v['mean_diff']:>+10.4f}{ci:>19}"
              f"{pr:>10}{pa:>10}{'YES' if v['win'] else '-':>5}")
    wins = sorted(q for q, v in res["cells"].items() if v["win"])
    print(f"\nwins: {len(wins)}/{len(res['cells'])}  {wins}")
    # Printing only. `win` is a DECIDED cell under the two-sided test
    # (significant, CI off zero, above the effect floor) and says nothing
    # about direction; the verdict logic is unchanged. Cycle 3 pre-registers
    # that the harness may LOSE on real data, and a reader must not take
    # "YES" for "the harness won".
    against = sorted(q for q in wins if res["cells"][q]["mean_diff"] < 0)
    if against:
        print(f"NOTE  'win' = decided in EITHER direction. mean_d is NEGATIVE "
              f"for {against}: read the sign of mean_d against the "
              "'contrast' line above before calling it a win for anyone.")
    nan_p = [q for q, v in res["cells"].items() if v["p_raw"] != v["p_raw"]]
    if nan_p:
        print(f"NOTE  p_raw is NaN for {nan_p}: the arms were identical on "
              "every item, or n was below min_n. The cell still consumes "
              f"its share of m={res['declared_m']} - you planned to test it.")
    return wins


def print_secondary(res: dict) -> None:
    print(f"\n=== secondary ablation: {res['contrast'][1]} - "
          f"{res['contrast'][0]} (OUTSIDE the correction) ===")
    print(f"cluster unit: {res['cluster_unit']}")
    hdr = (f"{'cell':<20}{'n':>6}{'k_clust':>9}{'mean_d':>10}"
           f"{'95% CI':>19}{'p_raw':>10}")
    print(hdr)
    print("-" * len(hdr))
    for qt, v in res["cells"].items():
        if "error" in v:
            print(f"{qt:<20}{'-':>6}  {v['error'][:50]}")
            continue
        ci = f"[{v['ci_low']:+.3f},{v['ci_high']:+.3f}]"
        pr = "nan" if v["p_raw"] != v["p_raw"] else f"{v['p_raw']:.4g}"
        print(f"{qt:<20}{v['n']:>6}{v['n_clusters']:>9}"
              f"{v['mean_diff']:>+10.4f}{ci:>19}{pr:>10}")
    print("\n" + res["note"])


def print_verdicts(cfg: Config, wins: list, declared: list,
                   screened: list) -> dict:
    """Both verdicts, side by side, never substituted (§2)."""
    as_prereg = sorted(set(declared)) == sorted(set(wins))
    scr = (bool(screened)
           and set(screened).issubset(set(wins)))
    print("\n=== verdicts (both reported, never substituted) ===")
    print(f"declared family       : {sorted(declared)}")
    print(f"decidable (screened)  : {sorted(screened)}")
    print(f"won                   : {sorted(wins)}")
    print(f"AS PRE-REGISTERED     : "
          f"{'won every declared cell' if as_prereg else 'did NOT win every declared cell'}")
    print(f"SCREENED              : "
          f"{'won every decidable cell' if scr else 'did NOT win every decidable cell'}"
          + ("" if screened else "  (no decidable cells: vacuous)"))
    return dict(declared_family=sorted(declared),
                decidable_cells=sorted(screened), winning_cells=sorted(wins),
                verdict_as_preregistered=bool(as_prereg),
                verdict_screened=bool(scr),
                screened_verdict_is_vacuous=not bool(screened))


def print_curve(rows: list, axis: str) -> None:
    if not rows:
        return
    print(f"\n=== gap by {axis} (descriptive; the shape, not a test) ===")
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="query_type", columns=axis,
                         values="mean_gap", aggfunc="first")
    print(piv.round(4).to_string())


def print_rank_shape(shape: dict, axis_label: str) -> None:
    if not shape:
        return
    print(f"\n=== rank shape vs {axis_label} (DESCRIPTIVE, not the test) ===")
    hdr = (f"{'cell':<20}{'n':>4}{'rho':>8}{'95% CI':>18}"
           f"{'p_exact':>10}{'min|rho|':>10}")
    print(hdr)
    print("-" * len(hdr))
    for qt, v in shape.items():
        ci = f"[{v['ci_low']:+.2f},{v['ci_high']:+.2f}]"
        pe = ("nan" if v["p_exact_permutation"] != v["p_exact_permutation"]
              else f"{v['p_exact_permutation']:.4g}")
        mdr = ("nan" if v["min_detectable_rho"] != v["min_detectable_rho"]
               else f"{v['min_detectable_rho']:.2f}")
        print(f"{qt:<20}{v['n_points']:>4}{v['rho']:>8.3f}{ci:>18}"
              f"{pe:>10}{mdr:>10}")
    print("\nmin|rho| is the smallest correlation this many points could "
          "call significant. A rho below it is NOT evidence of no "
          "relationship - the design could not detect one.")


def add_common_args(ap) -> None:
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=("inference", "analyse", "both"),
                    default="both")
    ap.add_argument("--backbone", default=None,
                    help="inference stage: run only this backbone "
                         "(one per SLURM array task)")
    ap.add_argument("--only-L", type=int, default=None,
                    help="inference stage: run only this history length")
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute shards that already exist")
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass the call cache (never use on a real run)")
    ap.add_argument("--out", default=None, help="override cfg.out_dir")
    ap.add_argument("--cache-dir", default=None,
                    help="shared call cache; point H1/H2/H3 at ONE of these "
                         "so identical prompts are paid for once")


def resolve(args) -> tuple[Config, Path]:
    cfg_path = (Path(args.config) if Path(args.config).is_absolute()
                else ROOT / args.config)
    cfg = load_config(cfg_path)
    out = Path(args.out) if args.out else (
        Path(cfg.out_dir) if Path(cfg.out_dir).is_absolute()
        else ROOT / cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return cfg, out


def pool_note(cfg: Config, samples) -> dict:
    """Composition of a REAL pool, for the artifacts. `{}` on synthetic.

    The contamination argument (PREREGISTRATION_C3.md C3-8) is PER DATASET,
    so the dataset mix has to travel with the results: a reader holding
    `analysis.json` must be able to see which archives produced the numbers
    without reopening the pool.
    """
    if not cfg.is_real:
        return {}
    from src.real_data import pool_report
    return pool_report(samples)


def print_pool_note(rep: dict) -> None:
    if not rep:
        return
    print("\n=== real pool composition (contamination is PER DATASET) ===")
    print(f"  items {rep['n_items']} from {rep['n_datasets']} datasets: "
          f"{rep['items_per_dataset']}")
    print(f"  strata {rep['items_per_stratum']}  (cuts: "
          f"weak < {rep['stratum_cuts']['weak_below']}, strong >= "
          f"{rep['stratum_cuts']['strong_at_or_above']})")
    print(f"  declared periods {rep['declared_periods']}  "
          f"median seasonal strength {rep['seasonal_strength_median']:+.3f}")
    print("  " + rep["note"])


def screen_for(cfg: Config, df: pd.DataFrame, samples, L: int) -> dict:
    return screen_report(
        df.to_dict("records"), samples, L,
        screen=dict(min_effect=float(cfg.analysis.get("min_effect", 0.02))),
        contrast=tuple(cfg.analysis["confirmatory_contrast"]),
        framing_threshold=float(
            cfg.analysis.get("framing_variance_threshold", 0.20)),
        design=cfg.design)
