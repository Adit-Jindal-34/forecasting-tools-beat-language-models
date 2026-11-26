"""Prompt templates: dispatch markers, pinned hashes, import hygiene.

The pinned hashes are the point of this file. `src.cache.call_key` hashes
rendered prompts, so one changed character in `models/prompts.py`
invalidates every cached model call. Moving or reformatting that module is
free; editing a template is not. These tests make the difference explicit
instead of letting it surprise someone mid-run.

If a hash fails and the change was INTENTIONAL: update the constant here AND
bump `src.PROMPT_VERSION`, in the same commit.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from models import prompts as P

ROOT = Path(__file__).resolve().parents[1]

#: sha256 of each rendered template, pinned. See the module docstring.
PINNED: dict[str, str] = {
    "generic": "b705429b3b180cb1245f6ae39d5770897c5f0b4d34f3da90055c495fa1668e58",
    "classify": "815a5c63cce651c076884c75aa0e9f0d4c14e208406239fa03ec5cd4811741dc",
    "qual_describe": "6774344c5193d0657d13f041c1952db0c220605c0add6e6a8c47ddd2dadd1dd6",
    "qual_explain": "4984ef289336d25b9a87d1ee923a860516b4ef8bcfaa96064bb32c0524f2840c",
    "voice": "6610adae8cf54653e917e1aa920cdecd82f574d14e38fc6e3670f23ad33e79ff",
    "judge": "5fbabfc5b9623a1c4eff5a8cc1fad5d09c13fc206c153d1a3c05d05967aeb59f",
    # --- C2 (PREREGISTRATION_C2.md). ADDED 2026-11-10, not edited: every
    # hash above is unchanged, so no cached call was invalidated and
    # src.PROMPT_VERSION is deliberately NOT bumped. See
    # test_cycle1_templates_are_frozen below.
    "generic_c2": "bfd50268ead4e6357df7812f614c68f9b5869f9b61ff5b270a40f326cd9b5e4f",
    "qual_c2_describe": "5557b8b2c0f5a0b257fe33ac78be545a6ae0f3cd653dbc48bf1c02fec3537df5",
    "qual_c2_explain": "0ed8f46a2d08f2eaffd2940e1a91bc01e0bd5871086b40d1b8b5b7dd816aa36e",
    "classify_c2": "e2b3cd2a2d49226924ec898a2d561822ea8422b56904164d1ab7f2a47501b173",
}

#: The templates cycle 1 actually ran. Editing any of these strands
#: every archived cached reply, so they are frozen separately and
#: louder than the additive set.
CYCLE1_TEMPLATES = ("generic", "classify", "qual_describe",
                    "qual_explain", "voice", "judge")

#: The C2 clause carries the DEFINITIONS and the ANSWER FORMAT every arm is
#: shown (src/samples.py). It is part of every C2 qualitative prompt, so it
#: is cache-key material exactly like a template: pinned for the same reason.
PINNED_C2_CLAUSE: dict[str, str] = {
    "event_explanation": "ceee4a363c2acee7c4df7f28696a8b4a72f3b616ae3d0f9cfbf2d9769e387432",
    "trend_description": "13567ce57c747aa064fe5ff2b2166d959e697a9d5b311b280f5e31abb28dbc78",
}


def _hashes() -> dict[str, str]:
    return {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in P.render_samples().items()}


def test_every_template_contains_its_dispatch_marker():
    """The defect that made the old mock's classify branch unreachable."""
    P.assert_markers_dispatch()
    rendered = P.render_samples()
    for name, (_, marker) in P.TEMPLATES.items():
        assert marker in rendered[name], f"{name} lost its marker {marker!r}"


def test_cycle1_templates_are_frozen():
    """Cycle 1's six templates must never change text.

    Separate from the pinned-hash test because the consequences differ. A new
    template costs nothing: nothing is cached against it yet. Editing a
    cycle-1 template makes roughly 12,400 archived completions unreachable
    and silently turns a cache replay into fresh GPU work. If one of these
    must change, it is a new template with a new name, not an edit.
    """
    actual = _hashes()
    for name in CYCLE1_TEMPLATES:
        assert actual[name] == PINNED[name], (
            f"cycle-1 template {name!r} was EDITED. Every archived cached "
            "call keyed on it is now unreachable. Add a new template under a "
            "new name instead, or bump src.PROMPT_VERSION knowing the cache "
            "is being abandoned.")


def test_c2_clause_is_pinned():
    """The definitions and answer format shown to every arm must not drift.

    An edit here changes what was asked. Once C2 has run, that is a protocol
    deviation and strands every cached C2 qualitative call; update the pin
    deliberately and record why in PREREGISTRATION_C2.md §6.
    """
    from src import samples as SM
    actual = {k: hashlib.sha256(v.encode("utf-8")).hexdigest()
              for k, v in SM.C2_CLAUSE.items()}
    assert actual == PINNED_C2_CLAUSE, (
        "C2_CLAUSE changed: " + ", ".join(
            k for k in actual if PINNED_C2_CLAUSE.get(k) != actual[k]))
