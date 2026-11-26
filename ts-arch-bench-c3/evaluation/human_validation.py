"""Human validation of the qualitative slot metric (PREREGISTRATION.md §4.1).

WHY THIS EXISTS, AND WHY IT MATTERS MORE AFTER D6
-------------------------------------------------
`evaluation/qualitative.py` scores an item as `extracted == expected`. Its
own docstring declines to trust that outright and defers to human
validation. After deviation D6 the confirmatory family is three cells, two of
which - `trend_description` and `event_explanation` - are scored by this
metric. So two thirds of the headline rests on a measure the
pre-registration itself calls provisionally measured, in a study whose
subject is evaluation validity. That is the exact failure mode this project
audits other people for.

TWO VALIDATIONS, NOT ONE
------------------------
A slot score is a conjunction, so a disagreement has two possible sources
and they need separating - the same decomposition argument that produced D6:

  TASK A, extraction validity.  Given the ANSWER TEXT alone, does
    `extract_slot` pick the class a human reads from that text?
    This is where the known `seasonal` defect lives: it is a
    presence/absence slot, so a self-contradictory answer still reads as
    "yes" and collects chance accuracy.

  TASK B, answer-key validity.  Given the SERIES alone, does
    `qualitative_truth` assign the label a human would?
    This is where the privileged-parameter question lives: `is_seasonal`
    uses the generator's TRUE period, which a human - and a deployed
    system - cannot see. Systematic human/key disagreement on `seasonal`
    would be evidence the key encodes information absent from the data.

Reporting one pooled number would hide which half is broken, and they have
different remedies: a regex fix versus an answer-key redefinition.

WHAT IS REPORTED, FLATTERING OR NOT
-----------------------------------
Per slot, per task: Krippendorff's alpha across the three annotators (the
ceiling - machine agreement cannot be trusted beyond the humans' agreement
with each other), the machine-versus-consensus agreement, and the full
confusion matrix. §4.1 requires the correlation be reported whether or not
it is flattering, so nothing here thresholds or hides a result.
"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


# --------------------------------------------------------------------------
# Krippendorff's alpha
# --------------------------------------------------------------------------
def krippendorff_alpha(units: dict | list, level: str = "nominal") -> dict:
    """Krippendorff's alpha for nominal or ordinal data, with missing values.

    `units` maps a unit id -> list of codings (one per coder, `None` for a
    missing one), or is a list of such lists. Units with fewer than two
    codings carry no reliability information and are dropped, which is the
    standard treatment, not a convenience.

    Computed from the coincidence matrix rather than from pairwise agreement,
    so missing data is handled correctly:

        alpha = 1 - (n - 1) * sum_{c!=k} o_ck * d_ck
                    / sum_{c!=k} n_c * n_k * d_ck

    Why alpha and not percentage agreement: alpha corrects for the agreement
    expected by chance given the observed marginals. On a slot whose classes
    are badly unbalanced - `event` is often "no major event" - two coders who
    both always guessed the modal class would show high raw agreement and
    alpha near zero. Reporting raw agreement there would be misleading.
    """
    if isinstance(units, dict):
        items = list(units.values())
    else:
        items = list(units)

    usable = [[v for v in row if v is not None and v == v] for row in items]
    usable = [row for row in usable if len(row) >= 2]
    if not usable:
        return dict(alpha=float("nan"), n_units=0, n_pairable=0,
                    note="no unit had two or more codings")

    classes = sorted({v for row in usable for v in row}, key=str)
    idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)

    # coincidence matrix
    o = np.zeros((K, K), float)
    for row in usable:
        m = len(row)
        for a, b in itertools.permutations(row, 2):
            o[idx[a], idx[b]] += 1.0 / (m - 1)
    n = o.sum()
    marg = o.sum(axis=1)

    if K == 1:
        # every coder gave the same single class to everything. Agreement is
        # perfect but there is no variance to disagree about, so alpha is
        # undefined rather than 1.0 - saying 1.0 would overstate it.
        return dict(alpha=float("nan"), n_units=len(usable),
                    n_pairable=int(n), classes=classes,
                    note="only one class observed: alpha is undefined, not "
                         "1.0. Agreement is trivially perfect and carries no "
                         "reliability information.")

    if level == "ordinal":
        # ordinal metric needs the class order to be meaningful; callers pass
        # classes already in order via `classes` being sorted, so the caller
        # must confirm that is the intended order.
        cum = np.cumsum(marg)
        def delta(i, j):
            lo, hi = (i, j) if i <= j else (j, i)
            inner = cum[hi] - cum[lo] - (marg[hi] + marg[lo]) / 2.0
            return float(inner ** 2)
    else:
        def delta(i, j):
            return 0.0 if i == j else 1.0

    num = den = 0.0
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            d = delta(i, j)
            num += o[i, j] * d
            den += marg[i] * marg[j] * d
    if den == 0:
        return dict(alpha=float("nan"), n_units=len(usable),
                    n_pairable=int(n), classes=classes,
                    note="expected disagreement is zero; alpha undefined")

    alpha = 1.0 - (n - 1.0) * num / den
    return dict(alpha=float(alpha), n_units=len(usable), n_pairable=int(n),
                classes=classes,
                observed_disagreement=float(num / n) if n else float("nan"),
                note=_alpha_reading(alpha))


def _alpha_reading(a: float) -> str:
    """Krippendorff's own thresholds, stated so a number is not left bare."""
    if a != a:
        return "undefined"
    if a >= 0.800:
        return "a >= 0.800: acceptable for drawing conclusions"
    if a >= 0.667:
        return ("0.667 <= a < 0.800: tentative conclusions only (Krippendorff's "
                "lowest conceivable threshold)")
    return ("a < 0.667: NOT reliable. The metric cannot carry a confirmatory "
            "claim at this level of agreement; report it as such rather than "
            "adjusting the threshold.")


# --------------------------------------------------------------------------
# consensus and machine agreement
# --------------------------------------------------------------------------
def consensus(codings: list, min_votes: int = 2):
    """Majority label among annotators, or None where there is no majority.

    A tie is NOT broken. An item the humans cannot agree on is not a
    yardstick for the machine, and silently picking one annotator's label
    would manufacture agreement.
    """
    vals = [v for v in codings if v is not None and v == v]
    if not vals:
        return None, 0
    counts = Counter(vals).most_common()
    top, k = counts[0]
    if k < min_votes:
        return None, k
    if len(counts) > 1 and counts[1][1] == k:
        return None, k          # tie
    return top, k


def machine_vs_human(machine: dict, human: dict) -> dict:
    """Agreement between the machine label and the human consensus.

    Keyed on unit id. Units without a human consensus are reported as
    `n_no_consensus` rather than dropped silently: a metric that only looks
    good on the items humans found easy is not validated.
    """
    shared = [u for u in machine if u in human]
    pairs = []
    no_cons = 0
    for u in shared:
        h = human[u]
        if h is None:
            no_cons += 1
            continue
        pairs.append((machine[u], h))
    if not pairs:
        return dict(n=0, agreement=float("nan"), n_no_consensus=no_cons,
                    note="no unit had both a machine label and a consensus")
    agree = sum(1 for m, h in pairs if m == h) / len(pairs)

    # alpha treating machine and consensus as two coders: corrects for the
    # chance agreement that a skewed class distribution would otherwise buy
    as_units = {f"u{i}": [m, h] for i, (m, h) in enumerate(pairs)}
    k = krippendorff_alpha(as_units, level="nominal")

    conf = Counter((m, h) for m, h in pairs)
    return dict(
        n=len(pairs), agreement=float(agree),
        alpha_machine_vs_consensus=k["alpha"],
        n_no_consensus=no_cons,
        confusion={f"machine={m}|human={h}": c
                   for (m, h), c in sorted(conf.items(), key=str)},
        note=("`agreement` is the raw rate; read it beside "
              "`alpha_machine_vs_consensus`, which corrects for chance given "
              "the observed class marginals. On a skewed slot the raw rate "
              "flatters the metric."))


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------
def stratified_sample(rows, n: int, strata: tuple, seed: int) -> list:
    """Proportional stratified sample over DESIGN variables only.

    `strata` names columns such as ("query_type", "pattern"). It deliberately
    does NOT include the machine's own verdict.

    Stratifying on the verdict would enrich the sample for disagreement and
    make it far more informative per annotation - but it also conditions the
    sample on the very quantity being validated, so the resulting agreement
    rate would not estimate the population rate without reweighting. The
    unbiased design is chosen here, and the sampling weight is recorded on
    every row so a weighted estimate remains possible if an enriched
    supplementary sample is added later.
    """
    import pandas as pd
    df = pd.DataFrame(rows)
    missing = [c for c in strata if c not in df.columns]
    if missing:
        raise ValueError(f"strata columns absent from rows: {missing}")
    rng = np.random.default_rng(seed)

    groups = list(df.groupby(list(strata), dropna=False))
    total = len(df)
    picked = []
    # proportional allocation, at least one per stratum, largest-remainder
    quota = {}
    for key, sub in groups:
        quota[key] = max(1, int(round(n * len(sub) / total)))
    while sum(quota.values()) > n and len(quota) > 1:
        big = max(quota, key=lambda k: quota[k])
        if quota[big] <= 1:
            break
        quota[big] -= 1
    for key, sub in groups:
        take = min(quota[key], len(sub))
        pick = rng.choice(sub.index.to_numpy(), size=take, replace=False)
        for i in pick:
            r = dict(df.loc[i])
            r["_stratum"] = "|".join(str(x) for x in (
                key if isinstance(key, tuple) else (key,)))
            r["_stratum_size"] = int(len(sub))
            r["_sampling_weight"] = float(len(sub) / take)
            picked.append(r)
    return picked
