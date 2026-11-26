"""The null fixture: when the arms are equivalent, the harness must not win.

WHAT THIS TEST IS FOR
---------------------
Every other fixture in this repo varies BACKBONE STRENGTH. None varies the
HARNESS EFFECT. So the smoke suite can confirm the pipeline reports an
effect it was built to report, and it cannot detect the one failure the
benchmark exists to rule out: a harness that measures higher for reasons
unrelated to answering better - a scorer that only parses its output format,
a parser that is stricter on one arm, a prompt that leaks.

`configs/null_fixture.yaml` makes the two arms equivalent by construction
(`mocks.assist_effect: 0.0`, see `models.llm._assist_p`). The correct result
is a contrast indistinguishable from zero and NO declared win.

WHY THE POSITIVE CONTROL MATTERS
--------------------------------
A null test that cannot go red is worth nothing: it would pass just as
happily if the runner silently scored no items at all. So the same fixture
is run a second time with `assist_effect: 1.0`, which restores the mock's
historical per-arm success rates, and the test asserts that this DOES move
the contrast in the harness's favour. Green here means "the detector is
wired up", not merely "nothing happened".

MEASURED SENSITIVITY (n=300 per cell, seed 20261107)
----------------------------------------------------
                       assist=0.0   assist=1.0    swing
  trend_description      -0.0483      +0.0300     +0.078
  event_explanation      -0.0175      +0.0633     +0.081

So the fixture resolves a manufactured advantage of roughly +0.06 in
`event_explanation` and does NOT reach significance for the same swing in
`trend_description`, because that cell's null baseline sits at -0.048 and
the advantage only carries it to +0.030. Honest floor: this detects a
manufactured harness advantage of about +0.10 or more. The real contrasts
it exists to police were +0.245 and +0.350, comfortably above that.

The residual -0.048 is not noise, and `evaluation.decomposition` identifies
its cause: the HARNESS arm emits output the slot scorer cannot read at all
on 15 of 300 `trend_description` items and 12 of 300 `event_explanation`
items, while the bare arm does so on none. Setting those unscoreable items
aside moves the null from -0.0483 to -0.0089 in `trend_description` and
from -0.0175 to +0.0167 in `event_explanation`.

So the fixture is biased slightly AGAINST the harness, which is the safe
direction for a detector - it cannot mask a manufactured win, it can only
mask a small genuine one. The harness's ~5% unscoreable rate on qualitative
cells is a defect in its own right and is tracked separately; see
PREREGISTRATION.md D8.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "null_fixture.yaml"

#: the declared effect floor in the fixture's own analysis block. A null run
#: must not favour the harness by this much in any cell.
EFFECT_FLOOR = 0.02


def _run(tmp: Path, assist_effect: float) -> dict:
    """Run the fixture end to end through the REAL runner CLI.

    Deliberately a subprocess against `runners.h2_query_type` rather than a
    reimplementation of the analysis: a null fixture that exercised a
    parallel code path would not be testing the pipeline that produces the
    paper's numbers.
    """
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["mocks"]["assist_effect"] = assist_effect
    tag = f"assist_{assist_effect}".replace(".", "p")
    cfg["out_dir"] = str(tmp / f"results_{tag}")
    cfg["data"]["pool_dir"] = str(tmp / f"pool_{tag}")
    cfg_path = tmp / f"{tag}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "runners.h2_query_type",
         "--config", str(cfg_path), "--stage", "both"],
        cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, (
        f"the null fixture runner exited {proc.returncode}; a null test that "
        f"cannot run is not a passing null test.\n"
        f"--- stdout tail ---\n{proc.stdout[-2500:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-2500:]}")
    return json.loads(
        (Path(cfg["out_dir"]) / "analysis.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def null_run(tmp_path_factory) -> dict:
    return _run(tmp_path_factory.mktemp("null"), 0.0)


@pytest.fixture(scope="module")
def positive_run(tmp_path_factory) -> dict:
    return _run(tmp_path_factory.mktemp("positive"), 1.0)


# --------------------------------------------------------------------------
# the unit the fixture rests on
# --------------------------------------------------------------------------
def test_assist_p_is_arm_blind_at_zero():
    """This is the whole mechanism. If it regresses, the fixture is not a
    null and every later assertion in this file is vacuous."""
    from models.llm import _assist_p
    for s in (0.0, 0.25, 0.6, 0.9, 1.0):
        assert _assist_p(s, True, 0.0) == _assist_p(s, False, 0.0), (
            f"at strength={s} the mock still answers differently depending "
            f"on whether the prompt looks like the harness's")


def test_assist_p_preserves_the_historical_rates_at_one():
    """`assist_effect` must default to a no-op, or adding it would silently
    change every existing mock config's meaning."""
    from models.llm import _assist_p
    for s in (0.0, 0.6, 1.0):
        assert _assist_p(s, True, 1.0) == pytest.approx(0.75 + 0.20 * s)
        assert _assist_p(s, False, 1.0) == pytest.approx(0.50 + 0.45 * s)


def test_assist_effect_enters_the_model_fingerprint():
    """The call cache keys on the model string. If `assist_effect` were left
    out, a null run and an ordinary mock run would share cached replies and
    the null would silently become a replay of the non-null."""
    from models.llm import MockLLM
    blind = MockLLM("x", strength=0.6, assist_effect=0.0).model
    sighted = MockLLM("x", strength=0.6, assist_effect=1.0).model
    assert blind != sighted, (
        "two mocks that answer differently share one cache key")


# --------------------------------------------------------------------------
# the null itself
# --------------------------------------------------------------------------
def test_null_fixture_declares_no_win(null_run):
    cells = null_run["confirmatory"]["cells"]
    assert cells, "the null fixture produced no cells to test"
    won = sorted(name for name, v in cells.items() if v["win"])
    assert won == [], (
        f"the harness won {won} on a fixture where both arms are equivalent "
        f"by construction. The pipeline is manufacturing a difference: the "
        f"scorer, the parser or the prompt is favouring one arm.")


def test_null_fixture_confidence_intervals_cover_zero(null_run):
    for name, v in null_run["confirmatory"]["cells"].items():
        assert v["ci_low"] <= 0.0 <= v["ci_high"], (
            f"{name}: 95% CI [{v['ci_low']:+.4f},{v['ci_high']:+.4f}] "
            f"excludes zero on a null fixture")


def test_null_fixture_does_not_favour_the_harness_past_the_effect_floor(
        null_run):
    """Direction matters. A null biased AGAINST the harness is safe - it
    cannot hide a manufactured win. One biased FOR it is the defect."""
    for name, v in null_run["confirmatory"]["cells"].items():
        assert v["mean_diff"] < EFFECT_FLOOR, (
            f"{name}: harness - bare_llm = {v['mean_diff']:+.4f}, at or above "
            f"the declared effect floor of {EFFECT_FLOOR} with equivalent arms")


# --------------------------------------------------------------------------
# the positive control: prove the detector can go red
# --------------------------------------------------------------------------
def test_positive_control_produces_a_win(positive_run):
    cells = positive_run["confirmatory"]["cells"]
    won = sorted(name for name, v in cells.items() if v["win"])
    assert won, (
        "restoring the mock's per-arm success rates (assist_effect=1.0) "
        "produced no win in any cell, so the null fixture above cannot "
        "distinguish a working pipeline from one that scores nothing. "
        "The null result is not evidence until this passes.")


def test_arm_sniffing_moves_every_cell_toward_the_harness(
        null_run, positive_run):
    """The sharpest form of the control: it is specifically `assist_effect`
    - the mock reading the harness's marker out of the prompt - that creates
    the advantage, and it should do so in EVERY cell, not just the one that
    reaches significance."""
    blind = null_run["confirmatory"]["cells"]
    sighted = positive_run["confirmatory"]["cells"]
    for name in blind:
        assert sighted[name]["mean_diff"] > blind[name]["mean_diff"], (
            f"{name}: turning arm-sniffing on moved the contrast from "
            f"{blind[name]['mean_diff']:+.4f} to "
            f"{sighted[name]['mean_diff']:+.4f} - the wrong way, or not at "
            f"all. The control is not exercising what it claims to.")
