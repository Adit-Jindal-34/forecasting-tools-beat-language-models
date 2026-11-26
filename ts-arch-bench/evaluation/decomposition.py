"""Why an item scored zero: a three-way taxonomy over slot-scored cells.

WHAT PROBLEM THIS SOLVES
------------------------
`score = n_correct / n_slots` collapses three very different events into the
same 0.0:

  nothing_extracted       the scorer found no class word at all, so the item
                          is unscoreable. A zero here is a MEASUREMENT
                          failure and says nothing about the answer.
  ambiguous               the scorer found class words but the answer named
                          two classes of one exclusive slot ("rose then
                          fell"). A zero here is a SCORER-FAIRNESS question:
                          the description may be accurate and simply not
                          expressible in a single-class slot.
  extracted_but_all_wrong the scorer read a clean, single-class answer and
                          it was wrong. A zero here is a genuine CAPABILITY
                          failure - the thing the benchmark means to measure.

A contrast computed on the collapsed score is a sum over all three. If the
arms differ in their MIX, the headline number is part capability and part
instrument, in unknown proportion. Separating them is the only way to say
which.

This is a REPORTED SECONDARY ANALYSIS, not a replacement metric. The
pre-registered primary is still `n_correct / n_slots` (PREREGISTRATION.md
3). Nothing here re-scores anything or feeds a verdict.

WHY THE DISTINCTION IS NOT COSMETIC
-----------------------------------
Measured on the real H2 run, the mix is strongly arm-dependent and NOT in
the direction a reader would guess: `trend_description`'s bare arm has 54
zeroes of which NONE are unextractable, while `event_explanation`'s harness
arm has 8 zeroes of which ALL are. So "the harness wins because the scorer
can only read its format" is false there, and the unscoreable items
actually belong to the harness. Inspecting the shape of the score
distribution alone invites exactly the opposite conclusion, which is why
this module reads the slot columns instead.

Non-slot cells (forecasting, numeric, localization) have no slots. For
those, `answered` is the only readability signal available and the taxonomy
degenerates to answered / not-answered; they are reported separately rather
than forced into the three-way split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SLOT_SCORING = "slot_accuracy"

#: the three mutually exclusive reasons a slot-scored item can score 0.0
ZERO_REASONS = ("nothing_extracted", "ambiguous", "extracted_but_all_wrong")


def classify_zero(row: pd.Series) -> str:
    """Why this slot-scored item scored zero. Order matters.

    `nothing_extracted` is checked first because an item with every slot
    absent is unscoreable regardless of anything else. `ambiguous` is
    checked before `extracted_but_all_wrong` because an answer naming two
    classes was read but could not be resolved, which is a different claim
    from an answer that was read cleanly and was wrong.
    """
    if row["n_slots"] and row["n_absent"] == row["n_slots"]:
        return "nothing_extracted"
    if row.get("n_ambiguous", 0) > 0:
        return "ambiguous"
    return "extracted_but_all_wrong"


def slot_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Just the slot-scored rows, with the slot columns present and numeric."""
    if "scoring" not in df.columns:
        return df.iloc[0:0]
    out = df[df["scoring"] == SLOT_SCORING].copy()
    for c in ("n_slots", "n_correct", "n_ambiguous", "n_absent"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out


def zero_taxonomy(df: pd.DataFrame,
                  by: tuple[str, ...] = ("query_type", "system")
                  ) -> pd.DataFrame:
    """Counts of each zero reason, per group, plus the non-zero count.

    Returns one row per group with absolute counts, so a reader can see both
    the mix and how much of the cell it covers. Rates are deliberately not
    the only output: 100% of 8 items and 100% of 800 items warrant very
    different attention.
    """
    s = slot_rows(df)
    if s.empty:
        return pd.DataFrame()
    s["_zero"] = s["score"] == 0.0
    z = s[s["_zero"]]
    if len(z):
        z = z.assign(_why=z.apply(classify_zero, axis=1))
        counts = (z.groupby(list(by))["_why"].value_counts()
                  .unstack(fill_value=0))
    else:
        counts = pd.DataFrame()
    for r in ZERO_REASONS:
        if r not in counts.columns:
            counts[r] = 0
    counts = counts[list(ZERO_REASONS)]
    total = s.groupby(list(by)).size().rename("n_items")
    out = counts.join(total, how="right").fillna(0)
    for r in ZERO_REASONS:
        out[r] = out[r].astype(int)
    out["n_items"] = out["n_items"].astype(int)
    out["n_scored_above_zero"] = (
        out["n_items"] - out[list(ZERO_REASONS)].sum(axis=1))
    return out


def decompose(df: pd.DataFrame,
              by: tuple[str, ...] = ("query_type", "system")
              ) -> pd.DataFrame:
    """The reported secondary view: mean score, and the same mean once the
    instrument-driven zeroes are set aside.

    Three columns matter:

      mean_score            the pre-registered primary. Unchanged.
      mean_excl_unscoreable drops `nothing_extracted` items, which carry no
                            information about the answer either way.
      mean_excl_instrument  drops `ambiguous` as well, i.e. the most
                            generous reading available to whichever arm was
                            penalised for phrasing.

    `mean_excl_instrument` is NOT a better metric - dropping items because
    they scored badly is selection on the outcome, and it is reported only
    as an upper bound on how much of a contrast the instrument could
    possibly explain. If a contrast survives it, the instrument is not the
    story; if it vanishes, it might be.
    """
    s = slot_rows(df)
    if s.empty:
        return pd.DataFrame()
    s = s.assign(_why=np.where(
        s["score"] == 0.0,
        s.apply(classify_zero, axis=1) if len(s) else "",
        "scored"))
    g = s.groupby(list(by))
    rows = []
    for key, part in g:
        unscoreable = part["_why"] == "nothing_extracted"
        ambiguous = part["_why"] == "ambiguous"
        keep_a = part[~unscoreable]
        keep_b = part[~(unscoreable | ambiguous)]
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update(
            n_items=len(part),
            mean_score=float(part["score"].mean()),
            n_unscoreable=int(unscoreable.sum()),
            n_ambiguous_zero=int(ambiguous.sum()),
            mean_excl_unscoreable=(float(keep_a["score"].mean())
                                   if len(keep_a) else float("nan")),
            mean_excl_instrument=(float(keep_b["score"].mean())
                                  if len(keep_b) else float("nan")),
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def contrast_table(df: pd.DataFrame, arm_a: str, arm_b: str,
                   cell_col: str = "query_type") -> pd.DataFrame:
    """`arm_b - arm_a` under each of the three readings, per cell.

    The point of putting them side by side: if `delta_score` and
    `delta_excl_instrument` agree, the contrast is not an artifact of how
    zeroes are attributed. If they disagree sharply, the headline number is
    carrying instrument behaviour and the paper has to say so.

    Unpaired means, not the pre-registered paired bootstrap - this is a
    descriptive decomposition, so it deliberately produces no CI and no
    p-value. Significance comes from the confirmatory path only.
    """
    d = decompose(df, by=(cell_col, "system"))
    if d.empty:
        return pd.DataFrame()
    rows = []
    for cell, part in d.groupby(cell_col):
        a = part[part.system == arm_a]
        b = part[part.system == arm_b]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        rows.append(dict(
            **{cell_col: cell},
            delta_score=b.mean_score - a.mean_score,
            delta_excl_unscoreable=(b.mean_excl_unscoreable
                                    - a.mean_excl_unscoreable),
            delta_excl_instrument=(b.mean_excl_instrument
                                   - a.mean_excl_instrument),
            unscoreable_a=a.n_unscoreable, unscoreable_b=b.n_unscoreable,
            ambiguous_a=a.n_ambiguous_zero, ambiguous_b=b.n_ambiguous_zero,
        ))
    return pd.DataFrame(rows)


def non_slot_readability(df: pd.DataFrame) -> pd.DataFrame:
    """For cells with no slots, the only readability signal there is."""
    if "scoring" not in df.columns:
        return pd.DataFrame()
    ns = df[df["scoring"] != SLOT_SCORING]
    if ns.empty:
        return pd.DataFrame()
    return (ns.groupby(["query_type", "system"])
            .agg(n_items=("score", "size"),
                 answered_rate=("answered", "mean"),
                 mean_score=("score", "mean"))
            .reset_index())
