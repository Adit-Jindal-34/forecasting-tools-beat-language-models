"""Ground-truth leakage: the invariants that define this build.

Every test here corresponds to a defect found in the previous build. If any
of them fails, a published number is invalid - these are not style checks.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

from src.samples import InferenceSample, Sample
from src import ground_truth as gt
from conftest import toy_window

SYSTEMS_DIR = Path(__file__).resolve().parents[1] / "systems"

#: names that carry, or can be used to recover, the ground-truth label.
#: `QUERY_OP` maps label -> correct op, so importing it inside a system is
#: precisely the D1 oracle leak.
FORBIDDEN = {
    "query_type": "the ground-truth label",
    "QUERY_OP": "the label -> correct-operation table (the D1 leak)",
    "dataset_op": "the correct operation under another name",
    "seasonal_period": "a generator parameter the key may use and a system may not",
    "gt.ground_truth": "the answer key itself",
    ".target(": "the forecast target (future data)",
}


def _system_sources():
    return sorted(p for p in SYSTEMS_DIR.glob("*.py")
                  if p.name != "__init__.py")


@pytest.mark.parametrize("path", _system_sources(), ids=lambda p: p.name)
def test_no_forbidden_names_in_systems(path):
    """D1/D3/D5: no module under systems/ may name ground truth."""
    src = path.read_text(encoding="utf-8")
    # strip docstrings and comments: these files DISCUSS the old leak
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                src = src.replace(doc, "")
    src = re.sub(r"#.*", "", src)
    for name, why in FORBIDDEN.items():
        assert name not in src, (
            f"{path.name} references {name!r} ({why}). Systems receive an "
            "InferenceSample and must not be able to reach ground truth.")


@pytest.mark.parametrize("path", _system_sources(), ids=lambda p: p.name)
def test_systems_do_not_import_config(path):
    """src.config holds QUERY_OP; systems/ must not import it at all."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
        elif isinstance(node, ast.Import):
            mod = ",".join(a.name for a in node.names)
        if mod and "src.config" in mod:
            pytest.fail(f"{path.name} imports src.config, which exposes "
                        "QUERY_OP (the oracle-leak table)")


def test_inference_sample_exposes_no_ground_truth():
    """I5: the inference view has no label-bearing attribute."""
    s = Sample("sid", np.arange(40.0), 32, 8, "statistics", "q?",
               "numeric_exact", "trend", 7)
    view = s.view(16)
    assert isinstance(view, InferenceSample)
    for name in ("query_type", "seasonal_period", "pattern", "max_history",
                 "series", "metadata"):
        assert not hasattr(view, name), (
            f"InferenceSample exposes {name!r}; a system could read it")
    assert set(vars(view)) == {"sample_id", "query", "window", "L", "horizon"}


def test_inference_sample_is_frozen_and_detached():
    s = Sample("sid", np.arange(40.0), 32, 8, "statistics", "q?",
               "numeric_exact", "trend", 7)
    view = s.view(16)
    with pytest.raises(Exception):
        view.query = "tampered"
    view.window[0] = -999.0          # a copy: must not corrupt the pool
    assert s.series[32 - 16] != -999.0


def test_window_is_suffix_invariant_I1():
    s = Sample("t", toy_window(140), 128, 8, "statistics", "q",
               "numeric_exact", "trend", 1)
    for small, big in ((32, 64), (64, 128)):
        assert np.array_equal(s.window(small), s.window(big)[-small:])


def test_target_fixed_across_lengths_I2():
    s = Sample("t", toy_window(140), 128, 8, "forecasting", "q", "forecast",
               "trend", 1)
    t0 = s.target()
    for L in (16, 64, 128):
        s.window(L)
        assert np.array_equal(s.target(), t0)


def test_no_future_information_in_window_I3():
    s = Sample("t", toy_window(140), 128, 8, "statistics", "q",
               "numeric_exact", "trend", 1)
    for L in (16, 64, 128):
        assert np.array_equal(s.window(L),
                              s.series[128 - L:128])


def test_observation_gt_uses_window_only_I4():
    base = toy_window(140)
    a = Sample("a", base, 128, 8, "statistics", "q", "numeric_exact",
               "trend", 1)
    altered = base.copy()
    altered[128:] = 9999.0           # different future, same visible history
    b = Sample("b", altered, 128, 8, "statistics", "q", "numeric_exact",
               "trend", 1)
    for qt in ("statistics", "counting", "extremum", "change", "correlation",
               "localization"):
        sa = Sample("a", a.series, 128, 8, qt, "q", "numeric_exact", "t", 1)
        sb = Sample("b", b.series, 128, 8, qt, "q", "numeric_exact", "t", 1)
        for L in (32, 128):
            assert gt.ground_truth(sa, L) == gt.ground_truth(sb, L), (
                f"{qt} ground truth leaked future data at L={L}")


def test_oracle_op_rejected_for_self_routing_system():
    """The privileged channel is refused unless the arm declares itself."""
    from models import MockLLM, MockTSFM
    from systems.harness_tsfm import HarnessTSFMSystem
    s = Sample("sid", toy_window(140), 128, 8, "statistics", "q?",
               "numeric_exact", "trend", 1)
    sysobj = HarnessTSFMSystem("h", MockLLM("bb"), MockTSFM(), "bb",
                               routing="self")
    with pytest.raises(ValueError, match="oracle"):
        sysobj.answer(s.view(128), oracle_op="mean")


def test_system_refuses_a_full_sample():
    """Passing a Sample instead of its view must fail loudly, not silently."""
    from models import MockLLM
    from systems.bare_llm import BareLLMSystem
    s = Sample("sid", toy_window(140), 128, 8, "statistics", "q?",
               "numeric_exact", "trend", 1)
    with pytest.raises(TypeError, match="InferenceSample"):
        BareLLMSystem("b", MockLLM("bb"), "bb").answer(s)
