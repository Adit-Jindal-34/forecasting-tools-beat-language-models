"""C2-10: the reproduction-rate guard, against the generator-truth key.

PREREGISTRATION_C2.md §C2-10 declares that the build fails if the harness's
evidence block can reproduce the answer key at or above 0.95 in a
confirmatory cell.

HISTORY, because the numbers below only mean something against it
----------------------------------------------------------------
Cycle 1's tool returned the key's own labels: 100/100.

The first C2 draft returned numbers instead of labels, but the key was still
COMPUTED from those numbers. A mechanical rule-applier reading only the
evidence reproduced trend, seasonal, event and position on 100% of items
(reviews #2-#4), so this file carried a strict `xfail`: the guard was
implemented as declared and it did not pass.

Amendment A9 changed the key to WHAT THE GENERATOR INJECTED. The same
rule-applier is now scored against that key. Measured on 360 generated
series at L=128, on the A15 generator (periods 12/24/48):

    trend 0.66   seasonal 0.82   event 0.69   position 0.94 (event items)
    cell score:  trend_description 0.74   event_explanation 0.78

(On the first C2 generator, periods 6/8/12, the same figures were 0.73 /
0.77 / 0.74 / 0.95 and 0.75 / 0.82.)

So the guard passes, as an ordinary test with no xfail.

WHAT IS GUARDED, EXACTLY (amendment A9 states this before any C2 run)
--------------------------------------------------------------------
  * the CELL score of the rule-applier, for both confirmatory qualitative
    cells, must stay below 0.95 - this is the declared C2-10 quantity;
  * the `trend`, `seasonal` and `event` SLOTS must each stay below 0.95.

`position` is reported, not gated. It is a localisation sub-question -
"which third holds the event" - and once the event exists, the largest step
in the series finds it about 95% of the time. That is a property of the
question, declared as a limitation in A9; it is half of one cell and is
scored only on items that have an event.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import ground_truth as gt                  # noqa: E402
from src import synthetic                           # noqa: E402
from src.ground_truth import SLOTS_FOR_TYPE_C2      # noqa: E402
from systems import tools                           # noqa: E402

#: the declared threshold
C2_10_MAX_REPRODUCTION = 0.95
#: slots gated individually; `position` is reported only (see docstring)
GATED_SLOTS = ("trend", "seasonal", "event")

N_HISTORY = 512
L = 128


def _pool(n: int = 360, seed: int = 20261110):
    """(window, generator record) pairs from the real C2 generator."""
    rng = np.random.default_rng(seed)
    out = []
    for cell in synthetic.c2_design_cells(n, rng):
        x, truth = synthetic.generate_base_series_c2(N_HISTORY, cell, rng,
                                                     horizon=8)
        out.append((np.asarray(x[N_HISTORY - L:N_HISTORY], float), truth))
    return out


def _key(truth: dict) -> dict:
    event, local = gt.c2_event_in_window(truth, N_HISTORY, L)
    return dict(trend=truth["trend"],
                seasonal="yes" if truth["seasonal"] else "no",
                event=event,
                position=(gt.event_position(local, L)
                          if local is not None else None))


def _rule_applier(e: dict) -> dict:
    """Mechanical rules applied to the evidence block ONLY.

    The same thresholds the computed labels use (|t| > 1.96, detrended
    autocorrelation > 0.5, robust z > 3, level shift > 4 noise scales,
    thirds). It stands in for a model that turns the tool's numbers into an
    answer without looking at the series.
    """
    t = e["slope_t"]
    f = e["max_abs_step_frac"]
    return dict(
        trend=("increasing" if t > gt.TREND_T_CRIT else
               "decreasing" if t < -gt.TREND_T_CRIT else "stable"),
        seasonal="yes" if e["acf_at_best_lag"] > 0.5 else "no",
        event=("spike" if e["max_local_peak_robust_z"] > 3.0 else
               "abrupt change" if e["level_shift_over_noise"] > 4.0 else
               "no major event"),
        position=("early" if f < 1 / 3 else "late" if f > 2 / 3 else
                  "middle"))


def reproduction() -> dict:
    """Per-slot and per-cell agreement of the rule-applier with the key."""
    slot_hits = {s: [] for s in ("trend", "seasonal", "event", "position")}
    cell_scores = {qt: [] for qt in SLOTS_FOR_TYPE_C2}
    for x, truth in _pool():
        got, want = _rule_applier(tools.series_evidence(x)), _key(truth)
        for s in slot_hits:
            if want[s] is not None:
                slot_hits[s].append(got[s] == want[s])
        for qt, slots in SLOTS_FOR_TYPE_C2.items():
            scored = [s for s in slots if want[s] is not None]
            cell_scores[qt].append(
                np.mean([got[s] == want[s] for s in scored]))
    return dict(slots={s: float(np.mean(v)) for s, v in slot_hits.items()},
                cells={q: float(np.mean(v)) for q, v in cell_scores.items()})


@pytest.fixture(scope="module")
def rates():
    return reproduction()


@pytest.mark.parametrize("query_type", sorted(SLOTS_FOR_TYPE_C2))
def test_c2_10_cell_reproduction_below_threshold(rates, query_type):
    """The declared guard: evidence + rules must not reproduce the cell."""
    r = rates["cells"][query_type]
    assert r < C2_10_MAX_REPRODUCTION, (
        f"{query_type}: a rule-applier reading only the evidence block "
        f"scores {r:.1%}, at or above the declared "
        f"{C2_10_MAX_REPRODUCTION:.0%}. The cell is saturated: the tool's "
        "numbers determine the key. See PREREGISTRATION_C2.md §C2-10.")


@pytest.mark.parametrize("slot", GATED_SLOTS)
def test_c2_10_slot_reproduction_below_threshold(rates, slot):
    r = rates["slots"][slot]
    assert r < C2_10_MAX_REPRODUCTION, (
        f"slot {slot!r} is reproducible from the evidence block at {r:.1%}")


def test_the_evidence_is_still_evidence(rates):
    """Not saturated must not mean uninformative.

    If the rule-applier fell to chance the block would be noise and the
    harness arm would have nothing to work with. Each gated slot must beat
    its chance rate by a clear margin.
    """
    chance = dict(trend=1 / 3, seasonal=1 / 2, event=1 / 3)
    for slot, c in chance.items():
        assert rates["slots"][slot] > c + 0.15, (
            f"slot {slot!r}: rule-applier at {rates['slots'][slot]:.1%} is "
            f"barely above chance ({c:.0%}); the evidence carries too "
            "little signal to be a fair tool")


def test_the_guard_can_go_red():
    """Positive control: against the COMPUTED labels the same applier scores
    1.000 on every slot - which is exactly the saturation the first C2 draft
    had. If this ever drops, the applier and the computed rules have
    diverged and the guard above is no longer measuring what it claims."""
    for x, _ in _pool(n=72):
        got = _rule_applier(tools.series_evidence(x))
        assert got == gt.qualitative_computed_c2(x)


def test_position_is_reported_not_hidden(rates):
    """`position` is exempt from the gate on purpose; keep it visible."""
    r = rates["slots"]["position"]
    assert 0.5 < r <= 1.0
    print(f"\nposition slot reproduction (reported, not gated): {r:.3f}")
