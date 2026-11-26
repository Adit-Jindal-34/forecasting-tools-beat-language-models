"""Pre-registered item and cell screening.

WHY THIS EXISTS
---------------
In the old suite, 5 of 9 cells had ZERO variance for the harness: the
question was "call np.mean" and the harness was allowed to call np.mean, so
it scored exactly 1.000 on all 100 items. A Wilcoxon test with one side
pinned at a constant is not comparing two systems; it is reporting that the
bare LLM is imperfect at arithmetic read from digit strings. On top of that,
`extremum` was a guaranteed ceiling tie, which made the pre-registered
`no_split` verdict branch MATHEMATICALLY UNREACHABLE before any code ran.

The old plan called this "measurement circularity" and proposed re-deriving
ground truth from the generating process. That is the wrong fix: the mean of
the visible window IS np.mean(window), and substituting the process mean
would make the answer key incorrect. The cells are not circular, they are
TRIVIAL - a property of item design.

WHAT TO DO INSTEAD
------------------
Screen items and cells the way a psychometrician would, and PRE-REGISTER the
screen so it is not a rule change made after seeing results:

  * low-variance screen - drop items/cells whose responses barely vary
    (response SD below `min_sd`, or mean accuracy above `ceiling`);
  * discrimination screen - point-biserial correlation between an item's
    score and the system's total score;
  * trivial-baseline floor - what a constant answer scores, reported for
    every cell so system scores are read against the right zero;
  * decidability - whether the pre-registered effect floor could EVER be
    cleared in this cell.

`no_split` then means "won every DECIDABLE cell" and becomes reachable.
Report both the original and revised verdicts, and never silently swap one
for the other.

Reference practice: TimeSeriesExam fits a 2PL IRT model and drops the least
discriminative items over several rounds; the IRT-auditing literature uses
SD < 1% and mean accuracy > 95% as standard uninformativeness screens. Both
also warn that item parameters are unreliable for small model pools, so with
a handful of systems these numbers are DESCRIPTIVE, never gates.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

#: pre-registered screening thresholds. Changing these after seeing results
#: is a protocol deviation and must be reported as one.
DEFAULT_SCREEN = dict(
    min_sd=0.01,        # response SD below this carries no information
    ceiling=0.95,       # every system above this: no headroom to detect
    floor=0.05,         # every system below this: nothing to distinguish
    min_effect=0.02,    # the pre-registered practical-effect floor
)


@dataclass
class CellScreen:
    query_type: str
    n_items: int
    systems: tuple
    means: dict
    sds: dict
    max_abs_gap: float
    pair_gaps: dict                 # "a|b" -> |mean_a - mean_b|
    decidable_pairs: dict           # "a|b" -> bool, screened per contrast
    at_ceiling: bool
    at_floor: bool
    zero_variance_systems: tuple
    #: SATURATED IS A SEPARATE AXIS FROM DECIDABLE, deliberately.
    #:
    #: A cell where the harness is pinned at 1.000 and the bare arm sits at
    #: 0.65 genuinely CAN show a difference - the gap is 0.35 - so marking it
    #: undecidable would discard a true result. What it cannot do is say
    #: anything about ARCHITECTURE: one side carries no information, so the
    #: test reduces to measuring the other arm alone. Two different facts,
    #: two different fields. The verdict rule chooses which one it uses, and
    #: PREREGISTRATION.md §1.3.3 states that choice in advance.
    saturated: bool
    saturated_systems: tuple
    decidable: bool
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def _pair_key(a: str, b: str) -> str:
    return f"{a}|{b}"


def screen_cells(rows, score_col: str = "score",
                 screen: dict | None = None) -> list:
    """Decidability screen per query type. Run BEFORE any hypothesis test."""
    s = dict(DEFAULT_SCREEN, **(screen or {}))
    df = pd.DataFrame(rows)
    out = []
    for qt, sub in df.groupby("query_type", dropna=False):
        systems = tuple(sorted(sub["system"].unique()))
        means, sds = {}, {}
        for name in systems:
            v = sub.loc[sub.system == name, score_col].astype(float)
            means[name] = float(v.mean()) if len(v) else float("nan")
            sds[name] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        n_items = int(sub.groupby("system").size().max()) if systems else 0

        # per-pair gaps: a cell-level max over ALL pairs lets a third arm
        # (the oracle) make a cell look decidable when the confirmatory
        # contrast inside it is not
        pair_gaps, decidable_pairs = {}, {}
        for i, a in enumerate(systems):
            for b in systems[i + 1:]:
                gap = abs(means[a] - means[b])
                pair_gaps[_pair_key(a, b)] = float(gap)
                decidable_pairs[_pair_key(a, b)] = bool(
                    gap >= s["min_effect"]
                    and not (means[a] >= s["ceiling"] and
                             means[b] >= s["ceiling"])
                    and not (means[a] <= s["floor"] and
                             means[b] <= s["floor"]))
        max_gap = float(max(pair_gaps.values())) if pair_gaps else 0.0

        zero_var = tuple(n for n in systems if sds[n] < s["min_sd"])
        at_ceiling = bool(systems) and all(
            means[n] >= s["ceiling"] for n in systems)
        at_floor = bool(systems) and all(
            means[n] <= s["floor"] for n in systems)

        reasons = []
        if at_ceiling:
            reasons.append("all systems at ceiling: no headroom")
        if at_floor:
            reasons.append("all systems at floor: nothing to distinguish")
        if len(zero_var) == len(systems) and systems:
            reasons.append("no system shows response variance")
        if max_gap < s["min_effect"]:
            reasons.append(
                f"largest observed gap {max_gap:.4f} < effect floor "
                f"{s['min_effect']}")
        decidable = not reasons
        out.append(CellScreen(
            query_type=str(qt), n_items=n_items, systems=systems,
            means=means, sds=sds, max_abs_gap=max_gap,
            pair_gaps=pair_gaps, decidable_pairs=decidable_pairs,
            at_ceiling=at_ceiling, at_floor=at_floor,
            zero_variance_systems=zero_var,
            saturated=bool(zero_var), saturated_systems=zero_var,
            decidable=decidable,
            reason="; ".join(reasons) if reasons else "decidable"))
    return out


def routing_variance_within_framing(rows) -> dict:
    """Does the LEAD axis move routing, or only the framing?

    The paraphrase bank is compositional: 25 framings x 4 leads. If routing
    is constant inside every framing family, the four leads are cosmetic and
    the independent unit is the FRAMING (n=25/type), not the item
    (n=100/type). That decides whether the self-versus-oracle contrast needs
    `paired_compare_clustered`.

    This is measured, never assumed, and the threshold that acts on it is
    pre-registered (`analysis.framing_variance_threshold`) so the clustering
    level is not chosen after seeing the answer.
    """
    df = pd.DataFrame(rows)
    need = {"query_type", "system", "framing_id", "route_op"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"routing variance needs columns {sorted(missing)}")
    df = df[df["route_source"] == "llm"] if "route_source" in df else df
    out: dict[str, dict] = {}
    for (qt, sysname), sub in df.groupby(["query_type", "system"],
                                         dropna=False):
        fams = sub.groupby("framing_id")["route_op"].nunique(dropna=False)
        n_fams = int(len(fams))
        varying = int((fams > 1).sum())
        out[f"{sysname}|{qt}"] = dict(
            system=str(sysname), query_type=str(qt),
            n_framings=n_fams, n_framings_with_variation=varying,
            fraction_varying=(varying / n_fams) if n_fams else float("nan"),
            distinct_route_ops=int(sub["route_op"].nunique(dropna=False)))
    return out


def clustering_decision(variance_report: dict, threshold: float) -> dict:
    """Apply the PRE-REGISTERED rule to the measured variance."""
    fracs = [v["fraction_varying"] for v in variance_report.values()
             if v["fraction_varying"] == v["fraction_varying"]]
    overall = float(np.mean(fracs)) if fracs else float("nan")
    cluster_on_framing = bool(fracs) and overall <= threshold
    return dict(
        mean_fraction_of_framings_with_routing_variation=overall,
        threshold=threshold,
        cluster_unit="framing_id" if cluster_on_framing else "sample_id",
        effective_routing_n_per_type=(25 if cluster_on_framing else 100),
        rule=("pre-registered: if at most `threshold` of framing families "
              "show any within-family routing variation, the framing is the "
              "independent unit"))


def item_discrimination(rows, score_col: str = "score") -> pd.DataFrame:
    """Point-biserial-style discrimination per item, within each system.

    Correlation between an item's score and the system's score on every
    OTHER item (corrected item-total correlation). Near zero means the item
    tells you nothing about the system that the rest of the suite does not.
    """
    df = pd.DataFrame(rows)
    recs = []
    for (qt, name), sub in df.groupby(["query_type", "system"], dropna=False):
        v = sub[score_col].astype(float).to_numpy()
        ids = sub["sample_id"].to_numpy()
        total = float(np.nansum(v))
        n = len(v)
        if n < 3 or np.nanstd(v) < 1e-12:
            for sid in ids:
                recs.append(dict(query_type=qt, system=name, sample_id=sid,
                                 discrimination=float("nan"),
                                 note="degenerate: no variance"))
            continue
        for i, sid in enumerate(ids):
            rest = (total - v[i]) / (n - 1)
            recs.append(dict(query_type=qt, system=name, sample_id=sid,
                             item_score=float(v[i]), rest_mean=float(rest),
                             discrimination=float("nan"), note=""))
        block = [r for r in recs if r["query_type"] == qt
                 and r["system"] == name and r["note"] == ""]
        xs = np.array([r["item_score"] for r in block])
        ys = np.array([r["rest_mean"] for r in block])
        if np.std(xs) > 1e-12 and np.std(ys) > 1e-12:
            r = float(np.corrcoef(xs, ys)[0, 1])
            for rec in block:
                rec["discrimination"] = r
    return pd.DataFrame(recs)


def _c2_constant_baseline(group, L: int, qt: str) -> dict:
    """Best CONSTANT answer for a C2 qualitative cell, scored exactly as an
    item is scored.

    Not the mean of per-slot majority rates: in C2 `position` is scored only
    on items that have an event, so an item's denominator varies and the
    best constant has to be found by scoring every candidate answer.
    """
    import itertools
    from collections import Counter
    from src.ground_truth import (qualitative_truth_c2, c2_scored_slots,
                                  SLOTS_FOR_TYPE_C2, QUALITATIVE_SLOTS)
    slots = SLOTS_FOR_TYPE_C2[qt]
    expected = [c2_scored_slots(qualitative_truth_c2(s, L), qt)
                for s in group]
    slot_rates = {}
    for slot in slots:
        vals = [e[slot] for e in expected if slot in e]
        counts = Counter(vals)
        top = counts.most_common(1)[0] if counts else (None, 0)
        slot_rates[slot] = dict(
            distribution=dict(counts), n_scored=len(vals),
            majority_class=top[0],
            majority_rate=(top[1] / len(vals)) if vals else float("nan"))
    best, best_answer = -1.0, None
    for combo in itertools.product(*(QUALITATIVE_SLOTS[s] for s in slots)):
        answer = dict(zip(slots, combo))
        score = float(np.mean([
            np.mean([answer[k] == v for k, v in e.items()])
            for e in expected]))
        if score > best:
            best, best_answer = score, answer
    return dict(kind="qualitative", slots=slot_rates,
                best_constant_answer=best_answer,
                best_constant_score=best)


def trivial_baselines(samples, L: int, design: str = "c1") -> dict:
    """What a CONSTANT answer scores in each cell. Report it beside systems.

    Without this number a reader cannot tell whether 0.62 on
    `event_explanation` is skill or the majority class. Slot cells in
    particular can have a high floor: with four slots whose majority classes
    are 0.49 / 0.77 / 0.66 / 0.43, an answer that always says the same thing
    already scores about 0.59.
    """
    from collections import Counter
    from src.ground_truth import (ground_truth, qualitative_truth,
                                  SLOTS_FOR_TYPE)
    out: dict[str, dict] = {}
    by_type: dict[str, list] = {}
    for s in samples:
        by_type.setdefault(s.query_type, []).append(s)

    for qt, group in by_type.items():
        if design == "c2" and qt in SLOTS_FOR_TYPE:
            out[qt] = _c2_constant_baseline(group, L, qt)
        elif qt in SLOTS_FOR_TYPE:
            slot_rates = {}
            for slot in SLOTS_FOR_TYPE[qt]:
                counts = Counter(getattr(qualitative_truth(s, L), slot)
                                 for s in group)
                slot_rates[slot] = dict(
                    distribution=dict(counts),
                    majority_class=counts.most_common(1)[0][0],
                    majority_rate=counts.most_common(1)[0][1] / len(group))
            out[qt] = dict(
                kind="qualitative", slots=slot_rates,
                best_constant_score=float(np.mean(
                    [v["majority_rate"] for v in slot_rates.values()])))
        elif qt == "counting":
            counts = Counter(int(ground_truth(s, L)) for s in group)
            top, hits = counts.most_common(1)[0]
            out[qt] = dict(kind="exact_match", modal_answer=top,
                           distribution=dict(sorted(counts.items())),
                           best_constant_score=hits / len(group))
        elif qt == "localization":
            counts = Counter(int(ground_truth(s, L)) for s in group)
            out[qt] = dict(kind="localization",
                           distinct_answers=len(counts),
                           best_constant_score=counts.most_common(1)[0][1]
                           / len(group))
        elif qt == "forecasting":
            # the natural floor is the naive forecast the MASE denominator is
            # built from: MASE = 1 maps to score 1/(1+1) = 0.5
            out[qt] = dict(
                kind="forecast", naive_mase=1.0, best_constant_score=0.5,
                note="score 0.5 corresponds to matching the in-sample naive "
                     "forecast; anything below it is worse than naive")
        else:
            vals = np.array([float(ground_truth(s, L)) for s in group])
            # best constant = the value minimising mean scaled error, which
            # for an absolute-error criterion is the median of the truths
            from evaluation.metrics import natural_scale
            from src.config import QUERY_OP
            op = QUERY_OP[qt]
            med = float(np.median(vals))
            errs = [abs(med - float(ground_truth(s, L)))
                    / natural_scale(op, s.window(L)) for s in group]
            out[qt] = dict(kind="numeric", gt_median=med,
                           gt_sd=float(np.std(vals)),
                           constant_answer=med,
                           best_constant_score=float(
                               np.mean([1.0 / (1.0 + e) for e in errs])),
                           note="the constant that minimises absolute error "
                                "is the median truth; this is the floor a "
                                "system that ignores the series achieves")
    return out


def screen_report(rows, samples, L: int, screen: dict | None = None,
                  contrast: tuple[str, str] | None = None,
                  framing_threshold: float | None = None,
                  design: str = "c1") -> dict:
    """The full pre-test screen, written to disk before any hypothesis test."""
    cells = screen_cells(rows, screen=screen)
    decidable = [c.query_type for c in cells if c.decidable]
    saturated = [c.query_type for c in cells if c.saturated]

    contrast_view = {}
    if contrast:
        key = _pair_key(*sorted(contrast))
        for c in cells:
            contrast_view[c.query_type] = dict(
                gap=c.pair_gaps.get(key),
                decidable=c.decidable_pairs.get(key),
                saturated=c.saturated,
                saturated_systems=list(c.saturated_systems))

    report = dict(
        thresholds=dict(DEFAULT_SCREEN, **(screen or {})),
        n_cells=len(cells), n_decidable=len(decidable),
        decidable_cells=decidable,
        undecidable_cells={c.query_type: c.reason
                           for c in cells if not c.decidable},
        saturated_cells=saturated,
        confirmatory_contrast=list(contrast) if contrast else None,
        confirmatory_contrast_screen=contrast_view,
        cells=[c.as_dict() for c in cells],
        trivial_baselines=trivial_baselines(samples, L, design=design),
        note=("`decidable` and `saturated` are independent axes. A saturated "
              "cell can be perfectly decidable - one arm pinned at 1.000 "
              "against another at 0.65 is a real 0.35 gap - but it measures "
              "the unpinned arm alone, not architecture. Report the "
              "pre-registered verdict and the screened one side by side; "
              "never substitute silently."))

    try:
        variance = routing_variance_within_framing(rows)
        report["routing_variance_within_framing"] = variance
        if framing_threshold is not None:
            report["clustering_decision"] = clustering_decision(
                variance, framing_threshold)
    except ValueError:
        report["routing_variance_within_framing"] = (
            "unavailable: rows lack framing_id / route_op")
    return report
