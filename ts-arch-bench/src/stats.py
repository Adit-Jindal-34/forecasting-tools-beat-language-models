"""Statistics: one implementation of each procedure, shared by every runner.

The old repo had THREE separate Holm implementations (`h2_analysis._holm`,
`history_length_revised.holm`, and inline logic) which could and did drift.
There is one here.

Choices follow Dror et al., "The Hitchhiker's Guide to Testing Statistical
Significance in NLP" (ACL 2018): paired non-parametric tests on paired data,
explicit family definitions, and multiple-comparison correction applied over
a family declared IN ADVANCE rather than discovered afterwards.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import stats as sstats


# --------------------------------------------------------------------------
# multiple comparisons
# --------------------------------------------------------------------------
def holm(pvalues, m: int | None = None) -> list:
    """Holm-Bonferroni adjusted p-values. NaNs pass through as NaN.

    `m` is the DECLARED family size. Pass it.

    Why it matters: with `m=None` the family is `len(finite p-values)`, which
    is data-dependent, because a degenerate comparison (identical arms, every
    paired difference zero) returns NaN and silently leaves the family. The
    bias runs the wrong way - the more cells collapse, the smaller `m`
    becomes and the EASIER it is to call the survivors significant. Measured
    on this codebase: the same raw p = 0.02 adjusts to 0.18 at m=18 and to
    0.10 at m=14.

    Passing the pre-registered `m` keeps the correction fixed however many
    comparisons turn out untestable. An untestable cell still consumes its
    share of the family: you planned to test it.

    Note for interpretation: Holm re-ranks the WHOLE family when any single
    p-value moves, so a change in one cell can cascade into others. That is
    why a projected flip must be re-run, not arithmetic'd.
    """
    p = list(pvalues)
    out = [float("nan")] * len(p)
    valid = sorted((v, i) for i, v in enumerate(p) if np.isfinite(v))
    m_eff = len(valid) if m is None else int(m)
    if m is not None and m_eff < len(valid):
        raise ValueError(
            f"declared family size m={m_eff} is smaller than the "
            f"{len(valid)} testable comparisons supplied; the family was "
            "mis-declared")
    running = 0.0
    for rank, (v, i) in enumerate(valid):
        running = max(running, min(1.0, (m_eff - rank) * v))
        out[i] = running
    return out


def benjamini_hochberg(pvalues, m: int | None = None) -> list:
    """BH adjusted p-values (FDR), with the same declared-`m` contract.

    Reported beside Holm in the sensitivity grid, because which correction
    you choose is itself a researcher degree of freedom worth exposing.
    """
    p = list(pvalues)
    out = [float("nan")] * len(p)
    valid = sorted((v, i) for i, v in enumerate(p) if np.isfinite(v))
    k = len(valid)
    m_eff = k if m is None else int(m)
    if m is not None and m_eff < k:
        raise ValueError(
            f"declared family size m={m_eff} is smaller than the {k} testable "
            "comparisons supplied; the family was mis-declared")
    prev = 1.0
    for rank in range(k - 1, -1, -1):
        v, i = valid[rank]
        prev = min(prev, min(1.0, v * m_eff / (rank + 1)))
        out[i] = prev
    return out


CORRECTIONS = {"holm": holm, "bh": benjamini_hochberg,
               "none": lambda ps, m=None: list(ps)}


# --------------------------------------------------------------------------
# paired comparison
# --------------------------------------------------------------------------
@dataclass
class PairedResult:
    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    p_raw: float
    cohens_dz: float
    #: mean(sign(d)) = P(d>0) - P(d<0). Correct for PAIRED data, but it is
    #: NOT Cliff's delta, which is defined over all cross-pairs of two
    #: independent samples. Ties dilute this toward zero, so read it beside
    #: `n_zero_diff` - a cell where most differences are exactly zero (a
    #: saturated tool cell, or a correctly-routed oracle contrast) will show
    #: a small effect for a structural reason, not a substantive one.
    paired_sign_effect: float
    n_zero_diff: int
    n_clusters: int = 0          # 0 = unclustered (item-level) bootstrap
    cluster_unit: str = "item"


def paired_compare(a, b, n_boot: int = 5000, seed: int = 0,
                   min_n: int = 10) -> PairedResult:
    """Paired comparison of a - b. Inputs must already be aligned by item.

    Callers must NOT drop unanswered items before calling: score them zero
    and pass the full vector, or coverage disappears from the analysis (the
    `.dropna()` bug).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs differ in shape: {a.shape} vs {b.shape}")
    d = a - b
    finite = np.isfinite(d)
    d = d[finite]
    n = len(d)
    if n == 0:
        nan = float("nan")
        return PairedResult(0, nan, nan, nan, nan, nan, nan, 0)
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    p = float("nan")
    if n >= min_n and np.any(d != 0):
        try:
            p = float(sstats.wilcoxon(a[finite], b[finite],
                                      alternative="two-sided",
                                      method="auto").pvalue)
        except ValueError:
            pass
    sd = d.std(ddof=1) if n > 1 else 0.0
    return PairedResult(
        n=n, mean_diff=float(d.mean()), ci_low=lo, ci_high=hi, p_raw=p,
        cohens_dz=float(d.mean() / sd) if sd > 0 else float("nan"),
        paired_sign_effect=float(np.mean(np.sign(d))),
        n_zero_diff=int((d == 0).sum()), n_clusters=0, cluster_unit="item")


def paired_compare_clustered(a, b, clusters, n_boot: int = 5000,
                             seed: int = 0, min_n: int = 10) -> PairedResult:
    """Paired comparison with a CLUSTER bootstrap over `clusters`.

    Use this whenever the quantity driving the difference is shared by
    several items. The motivating case is routing: the classifier sees only
    question text, so every item carrying the same phrasing gets the same
    routing decision. Treating those items as independent counts ONE decision
    many times and produces a spectacular p-value for a sample size of one.

    The bootstrap resamples whole clusters with replacement, so the effective
    n is the number of clusters, not the number of items. The Wilcoxon p is
    computed on CLUSTER MEANS for the same reason.

    `clusters` is any array of labels aligned with `a` and `b` - typically
    `framing_id` within a query type, or `sample_id` when each item has its
    own phrasing (in which case this reduces to `paired_compare`).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    g = np.asarray(clusters)
    if not (a.shape == b.shape == g.shape):
        raise ValueError(
            f"inputs differ in shape: a={a.shape} b={b.shape} "
            f"clusters={g.shape}")
    d = a - b
    finite = np.isfinite(d)
    d, g = d[finite], g[finite]
    if len(d) == 0:
        nan = float("nan")
        return PairedResult(0, nan, nan, nan, nan, nan, nan, 0, 0, "cluster")

    labels = np.unique(g)
    by_cluster = [d[g == lab] for lab in labels]
    cluster_means = np.array([blk.mean() for blk in by_cluster], float)
    k = len(labels)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, float)
    for i in range(n_boot):
        pick = rng.integers(0, k, k)
        boot[i] = float(np.concatenate([by_cluster[j] for j in pick]).mean())
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))

    p = float("nan")
    if k >= min_n and np.any(cluster_means != 0):
        try:
            p = float(sstats.wilcoxon(cluster_means,
                                      alternative="two-sided",
                                      method="auto").pvalue)
        except ValueError:
            pass
    sd = cluster_means.std(ddof=1) if k > 1 else 0.0
    return PairedResult(
        n=int(len(d)), mean_diff=float(d.mean()), ci_low=lo, ci_high=hi,
        p_raw=p,
        cohens_dz=(float(cluster_means.mean() / sd) if sd > 0
                   else float("nan")),
        paired_sign_effect=float(np.mean(np.sign(cluster_means))),
        n_zero_diff=int((cluster_means == 0).sum()),
        n_clusters=k, cluster_unit="cluster")


# --------------------------------------------------------------------------
# rank correlation (H3 shape)
# --------------------------------------------------------------------------
def spearman_with_ci(x, y, n_boot: int = 5000, seed: int = 0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        nan = float("nan")
        return nan, nan, nan, nan, len(x)
    rho, p = sstats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) > 1 and len(np.unique(y[idx])) > 1:
            r = sstats.spearmanr(x[idx], y[idx]).statistic
            if np.isfinite(r):
                boots.append(float(r))
    lo, hi = ((float(np.percentile(boots, 2.5)),
               float(np.percentile(boots, 97.5))) if boots
              else (float("nan"), float("nan")))
    return float(rho), float(p), lo, hi, int(len(x))


def exact_spearman_p(x, y) -> float:
    """Two-sided exact permutation p for Spearman. Exact for n <= 8."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    rx = sstats.rankdata(x)
    observed = abs(float(np.corrcoef(rx, sstats.rankdata(y))[0, 1]))
    if n <= 8:
        hits = total = 0
        for perm in itertools.permutations(y):
            r = abs(float(np.corrcoef(rx, sstats.rankdata(perm))[0, 1]))
            hits += r >= observed - 1e-12
            total += 1
        return hits / total
    rng = np.random.default_rng(20261102)
    n_perm = 20000
    hits = sum(abs(float(np.corrcoef(rx, sstats.rankdata(
        rng.permutation(y)))[0, 1])) >= observed - 1e-12
        for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


def min_detectable_rho(n: int, alpha: float = 0.05) -> float:
    """Smallest |rho| an exact permutation test could call significant at n.

    Report this instead of dichotomising. With n=8 backbones the bar is
    steep, and a null must be written "this design could not detect a
    relationship of plausible size", never "there is no relationship".
    """
    if n < 3:
        return float("nan")
    base = np.arange(float(n))
    for rho_target in np.arange(1.0, 0.0, -0.01):
        y = base.copy()
        k = int(round((1 - rho_target) * n / 2))
        if k > 0:
            y[:k] = y[:k][::-1]
        if exact_spearman_p(base, y) < alpha:
            continue
        return float(round(rho_target + 0.01, 2))
    return 1.0


# --------------------------------------------------------------------------
# prospective power
# --------------------------------------------------------------------------
def paired_mde(n: int, n_tests: int = 1, alpha: float = 0.05,
               power: float = 0.80) -> float:
    """Minimum detectable paired effect (in dz units) after correction.

    Normal approximation using Holm's most stringent first step. Prospective
    sensitivity, NOT a post-hoc power claim about an observed result.
    """
    if n < 2:
        return float("nan")
    a1 = alpha / max(n_tests, 1)
    return float((sstats.norm.ppf(1 - a1 / 2) + sstats.norm.ppf(power))
                 / np.sqrt(n))
