"""The cycle-2 human-grading kit (amendment A17).

Built on a small generated pool so the test needs no archive. Checks the
three properties the grading depends on:

  * the pages carry no answer information, and each grader's order differs;
  * a grader who answers exactly like the key scores 1.000 under the SAME
    scorer the models are scored with, and three such graders agree
    perfectly;
  * answers from a different kit are refused.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    """A kit built from a small cycle-2 pool written to a temp dir."""
    tmp = tmp_path_factory.mktemp("kit")
    sys.path.insert(0, str(ROOT))
    from src.config import load_config
    from src.pipeline import make_samples
    cfg = load_config(ROOT / "configs" / "c2_h2.yaml")
    cfg.data.update(pool_dir=str(tmp / "pool"), n_base_series=36,
                    n_per_query_type=36)
    make_samples(cfg, ROOT)
    # the kit builder reads the c2_h2 config; point it at the small pool by
    # copying the manifest's settings through a temp config
    import yaml
    raw = yaml.safe_load((ROOT / "configs" / "c2_h2.yaml").read_text(
        encoding="utf-8"))
    raw["data"].update(pool_dir=str(tmp / "pool"), n_base_series=36,
                       n_per_query_type=36)
    out = tmp / "kit"
    code = (
        "import sys, yaml; sys.path.insert(0, r'%s');"
        "import scripts.c2_make_annotation_kit as K;"
        "from src.config import load_config as LC;"
        "import src.config as C;"
        "orig = K.load_config;"
        "K.load_config = lambda p: (lambda c: (c.data.update(%r), c)[1])(orig(p));"
        "sys.argv = ['k', '--out', r'%s'];"
        "sys.exit(K.main())" % (ROOT, dict(n_base_series=36,
                                          n_per_query_type=36), out))
    r = subprocess.run([PY, "-c", code], cwd=ROOT, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return out, tmp


def _page(out, g):
    return (out / f"annotator_{g}.html").read_text(encoding="utf-8")


def test_pages_carry_no_answer_information(kit):
    out, _ = kit
    key = json.loads((out / "_private" / "key.json").read_text(
        encoding="utf-8"))
    for g in "ABC":
        html = _page(out, g)
        for bad in ('"truth"', '"expected"', '"pattern"', "event_index",
                    "seasonal_period", '"noise"', '"sigma"'):
            assert bad not in html, bad
        items = json.loads(re.search(r"const ITEMS = (\[.*?\]);\n", html,
                                     re.S).group(1))
        assert {it["id"] for it in items} == set(key["key"])
        assert all(set(it) == {"id", "type", "question", "slots", "values"}
                   for it in items)
    orders = [re.search(r"const ORDER = (\[.*?\]);", _page(out, g)).group(1)
              for g in "ABC"]
    assert len(set(orders)) == 3, "every grader must get their own order"


def _write_answers(out, folder, kit_hash, fn):
    key = json.loads((out / "_private" / "key.json").read_text(
        encoding="utf-8"))["key"]
    folder.mkdir(parents=True, exist_ok=True)
    for g in "ABC":
        ans = {i: fn(e) for i, e in key.items()}
        (folder / f"answers_{g}.json").write_text(json.dumps(dict(
            annotator=g, kit=kit_hash, version=1, answers=ans)),
            encoding="utf-8")


def test_a_grader_matching_the_key_scores_one_and_agrees(kit):
    out, tmp = kit
    kit_hash = json.loads((out / "_private" / "key.json").read_text(
        encoding="utf-8"))["kit"]

    def perfect(e):
        a = dict(e["expected"])
        if e["type"] == "event_explanation" and "position" not in a:
            a["position"] = "none"
        return a

    _write_answers(out, tmp / "perfect", kit_hash, perfect)
    r = subprocess.run([PY, "scripts/c2_analyse_annotation.py", "--kit",
                        str(out), "--answers", str(tmp / "perfect"),
                        "--out", str(tmp / "res")], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads((tmp / "res" / "report.json").read_text(
        encoding="utf-8"))
    for g in ("A", "B", "C", "majority"):
        for cell, v in rep["accuracy_by_cell"].items():
            assert v[g] == pytest.approx(1.0), (g, cell)
    for slot in ("trend", "event"):
        assert rep["agreement"][slot]["raw_pairwise"] == pytest.approx(1.0)


def test_answers_from_another_kit_are_refused(kit):
    out, tmp = kit
    _write_answers(out, tmp / "other", "not-this-kit", lambda e: {})
    r = subprocess.run([PY, "scripts/c2_analyse_annotation.py", "--kit",
                        str(out), "--answers", str(tmp / "other"),
                        "--out", str(tmp / "res2")], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "different kit" in (r.stdout + r.stderr)
