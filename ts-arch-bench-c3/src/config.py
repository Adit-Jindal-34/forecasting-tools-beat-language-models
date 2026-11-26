"""Configuration loading and validation.

FINGERPRINTING, CORRECTED
-------------------------
The old build had ONE `fingerprint()` mixing data parameters, every
backbone, the harness settings and the mock knobs, and it omitted `seed`.
That produced two live bugs (see src/cache.py) and one budget bug: adding a
backbone invalidated all the others.

Here there are two, with disjoint jobs:

  data_fingerprint()        what determines the SAMPLE POOL.
                            INCLUDES `seed`. Changing it means different
                            series, so the pool is regenerated.

  backbone_fingerprint(nm)  what determines ONE backbone's replies.
                            Scoped to that backbone alone.

Nothing else enters a cache key. Harness settings, scorer settings and the
analysis block are all post-inference and must never invalidate a model call.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

import src

VALID_KINDS = {"forecast", "numeric_exact", "localization", "qualitative"}
#: `data.name`. Must stay in step with `src.pipeline.POOL_LOADERS`.
VALID_DATA_NAMES = {"synthetic", "real"}
#: Generator knobs that have NO MEANING on a real pool. Leaving one in a
#: real-data config is not harmless: `data_fingerprint()` hashes the whole
#: data block, so a stale `event_zone` would silently define a different
#: cache namespace while having no effect on the series anyone sees.
SYNTHETIC_ONLY_DATA_KEYS = (
    "patterns", "level_range", "spike_lambda_range", "event_zone")

#: Units a contrast may be clustered on, and what each asserts is the
#: independent observation. `analysis.cluster_unit` names the PRIMARY and is
#: fixed before the run; `analysis.cluster_unit_sensitivity` lists
#: alternatives reported beside it. `sample_id` is the identity case and is
#: what every cycle-1 and cycle-2 result was computed under.
VALID_CLUSTER_UNITS = ("sample_id", "framing_id", "dataset_id", "stratum")
VALID_SYSTEM_KINDS = {"bare_llm", "harness_tsfm"}
VALID_PROVIDERS = {"mock", "gemini", "openai_compatible", "transformers"}
VALID_ROUTING = {"self", "oracle"}

#: query type -> the operation that answers it.
#:
#: RUNNER-SIDE ONLY. This maps a GROUND-TRUTH LABEL to the correct operation,
#: so importing it inside `systems/` is exactly the oracle leak that broke
#: the previous build. `tests/test_leakage.py` fails the build if any module
#: under systems/ references it. The oracle-routing ablation reads it HERE,
#: in the runner, and passes the op through the explicit `oracle_op` channel.
QUERY_OP: dict[str, str] = {
    "forecasting": "forecast",
    "statistics": "mean",
    "counting": "count_spikes",
    "extremum": "max",
    "change": "pct_change",
    "correlation": "lag1_autocorr",
    "localization": "change_point",
    "trend_description": "describe",
    "event_explanation": "explain",
}

DEFAULT_KINDS: dict[str, str] = {
    "forecasting": "forecast",
    "localization": "localization",
    "trend_description": "qualitative",
    "event_explanation": "qualitative",
}

#: Operations whose tool IS the answer-key function, so a correctly-routed
#: harness scores exactly 1.000 by construction. This is an A PRIORI property
#: of the architecture - it follows from `systems/tools.py` importing
#: `src.ground_truth`, not from any observed data - which is what makes
#: excluding these cells from the confirmatory family a DESIGN decision
#: rather than a post-hoc screen. See PREREGISTRATION.md §3.
TOOL_SATURATED_OPS: frozenset[str] = frozenset({
    "mean", "median", "std", "max", "min", "count_spikes", "pct_change",
    "lag1_autocorr", "change_point",
})


def a_priori_saturated(query_types) -> list[str]:
    """Query types whose harness arm is pinned at 1.000 by construction."""
    names = [qt.name if hasattr(qt, "name") else str(qt) for qt in query_types]
    return [n for n in names if QUERY_OP.get(n) in TOOL_SATURATED_OPS]


#: Default confirmatory family for the ANSWER-ACCURACY contrast, declared
#: BY NAME. It is exactly the complement of `a_priori_saturated`.
#:
#: SIGNED OFF 2026-11-06 (PREREGISTRATION.md §5, deviation D6).
#:
#: The earlier draft of this constant carried five cells: the three
#: non-saturated ones plus `counting` and `localization` added back by
#: judgement. That has been withdrawn in favour of DECOMPOSED METRICS, which
#: is what the tool-evaluation literature actually recommends - separate
#: measures for tool invocation, task completion and outcome verification
#: (Benchmarking the Benchmarks: A Validity Audit of Tool-Calling
#: Evaluation, arXiv:2607.02577) - rather than one family mixing them.
#:
#: Why the two judgement calls do not survive that principle. In a saturated
#: cell a correctly-routed harness scores exactly 1.000, so its per-item
#: score IS its routing outcome and the bare-vs-harness gap reduces to
#: (route_correct - bare_accuracy). That is arithmetically redundant with
#: metrics this build already reports separately:
#:
#:   tool invocation  -> `route_correct` / `route_correct_selective`, per
#:                       cell, over all nine types, clustered per §3
#:   cost of mis-routing -> the harness vs harness_oracle ablation, already
#:                       pre-registered as secondary and outside correction
#:   task completion  -> the answer-accuracy family defined here
#:
#: So `localization` is not demoted - the quantity it was kept for is
#: measured better elsewhere, at n=100 with real paraphrase variance after
#: D5, instead of as an answer-score contrast. And `counting`'s gap would
#: have measured in-context MAD arithmetic against tool access, which is a
#: statement about the bare arm, not about architecture.
#:
#: What remains here is the set of cells where the harness is NOT pinned by
#: construction and the comparison is therefore genuinely architectural:
#: forecasting (a TSFM against in-context extrapolation) and the two
#: qualitative cells (feature-assisted generation against raw generation).
#:
#: ⚠ m DROPS FROM 5 TO 3, which makes the surviving tests EASIER to call
#: significant (alpha/3 > alpha/5). That is the anti-conservative direction
#: D2 warned about, so it is declared in advance and justified on construct
#: validity, never on convenience. `family_report()` prints it every run.
DEFAULT_CONFIRMATORY_FAMILY: tuple[str, ...] = (
    "forecasting", "trend_description", "event_explanation",
)


@dataclass
class QueryTypeSpec:
    name: str
    kind: str

    @property
    def op(self) -> str:
        return QUERY_OP[self.name]


@dataclass
class BackboneSpec:
    name: str
    provider: str = "mock"
    model: str = "mock-llm"
    strength: float = 0.6          # external capability score, H3 x-axis
    params: dict = field(default_factory=dict)


@dataclass
class SystemSpec:
    name: str
    kind: str
    backbone: str
    routing: str = "self"          # "self" | "oracle" (the ablation arm)
    params: dict = field(default_factory=dict)


@dataclass
class Config:
    seed: int = 42
    out_dir: str = "results"
    data: dict = field(default_factory=lambda: dict(
        name="synthetic", pool_dir="data/default", max_history=512,
        horizon=8, n_base_series=24, n_per_query_type=24, level_range=[20, 120],
        patterns=["trend", "seasonality", "multi_seasonality", "noise",
                  "spikes", "abrupt_change", "mixed"]))
    history_lengths: list = field(default_factory=lambda: [128])
    query_types: list = field(default_factory=list)
    backbones: dict = field(default_factory=dict)
    systems: list = field(default_factory=list)
    tsfm: dict = field(default_factory=lambda: dict(name="mock_tsfm", noise=0.0))
    # `assist_effect` scales the mock's per-arm success-rate gap. 1.0 is the
    # historical behaviour and stays the default so no existing config
    # changes meaning. `configs/null_fixture.yaml` sets 0.0, which is what
    # makes a true null expressible at all. The whole dict enters the call
    # cache fingerprint (see `fingerprint`), so a null run and an ordinary
    # mock run cannot share cached replies.
    mocks: dict = field(default_factory=lambda: dict(
        error_scale=0.25, prose_rate=0.0, length_effect=0.0,
        assist_effect=1.0))
    judge: dict = field(default_factory=lambda: dict(enabled=False))
    scoring: dict = field(default_factory=lambda: dict(
        tol=0.05, loc_tol=2, fixed_mase_denominator=True))
    analysis: dict = field(default_factory=lambda: dict(
        alpha=0.05, min_effect=0.02, min_n_per_cell=50, n_boot=5000,
        correction="holm",
        # The Holm family, declared BY NAME. A family stated as a count is
        # the ambiguity the explicit-family fix exists to remove, and
        # `holm()` counting only finite p-values made the realised family
        # data-dependent AND anti-conservative.
        confirmatory_contrast=["bare_llm", "harness"],
        confirmatory_family=list(DEFAULT_CONFIRMATORY_FAMILY),
        # harness_oracle is a pre-registered SECONDARY ablation: effects and
        # CIs reported, outside the correction.
        secondary_contrast=["harness", "harness_oracle"],
        # Clustering rule for routing-derived contrasts, fixed in advance so
        # the clustering level is not chosen after seeing the variance.
        # If the fraction of framing families showing ANY within-family
        # routing variation is <= this, the FRAMING is the independent unit
        # (n=25/type); otherwise the item is (n=100/type).
        framing_variance_threshold=0.20))
    harness: dict = field(default_factory=lambda: dict(
        numeric_final="direct", assist_qualitative=True))
    #: Execution only - never part of any fingerprint or cache key.
    #: `workers` > 1 sends that many items to the model server at once
    #: (src/pipeline.py). Results are identical to `workers: 1`; only the
    #: wall-clock time changes. tests/test_parallel.py asserts it.
    inference: dict = field(default_factory=lambda: dict(workers=1))

    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.data_name not in VALID_DATA_NAMES:
            raise ValueError(f"data.name must be one of "
                             f"{sorted(VALID_DATA_NAMES)}, got "
                             f"{self.data_name!r}")
        if not self.query_types:
            raise ValueError("config.query_types is empty")
        for qt in self.query_types:
            if qt.kind not in VALID_KINDS:
                raise ValueError(f"query type {qt.name!r}: bad kind {qt.kind!r}")
            if qt.name not in QUERY_OP:
                raise ValueError(f"query type {qt.name!r} has no operation "
                                 "mapping (extend src.config.QUERY_OP)")
        if not self.backbones:
            raise ValueError("config.backbones is empty")
        for bb in self.backbones.values():
            if bb.provider not in VALID_PROVIDERS:
                raise ValueError(f"backbone {bb.name}: bad provider "
                                 f"{bb.provider!r}")
            if not 0.0 <= bb.strength <= 1.0:
                raise ValueError(f"backbone {bb.name}: strength must be in "
                                 f"[0,1], got {bb.strength}")
        if not self.systems:
            raise ValueError("config.systems is empty")
        seen = set()
        for s in self.systems:
            if s.name in seen:
                raise ValueError(f"duplicate system name {s.name!r}")
            seen.add(s.name)
            if s.kind not in VALID_SYSTEM_KINDS:
                raise ValueError(f"system {s.name}: bad kind {s.kind!r}")
            if s.backbone not in self.backbones:
                raise ValueError(f"system {s.name}: unknown backbone "
                                 f"{s.backbone!r}")
            if s.routing not in VALID_ROUTING:
                raise ValueError(f"system {s.name}: bad routing {s.routing!r}")
            if s.routing == "oracle" and s.kind != "harness_tsfm":
                raise ValueError(
                    f"system {s.name}: routing='oracle' only applies to a "
                    "harness; a bare LLM has nothing to route")
        for L in self.history_lengths:
            if L < 8:
                raise ValueError(f"history length {L} is too small")
        if self.design not in ("c1", "c2"):
            raise ValueError(f"data.design must be 'c1' or 'c2', got "
                             f"{self.design!r}")
        if max(self.history_lengths) > int(self.data.get("max_history", 0)):
            raise ValueError("max(history_lengths) must be <= data.max_history")
        if self.is_real:
            self._validate_real_data()
        else:
            n_base = int(self.data.get("n_base_series", 0))
            n_per = int(self.data.get("n_per_query_type", 0))
            if n_per > n_base:
                raise ValueError(
                    f"n_per_query_type ({n_per}) > n_base_series ({n_base}): "
                    "query types would reuse base series unevenly")
        self._validate_family()

    def _validate_real_data(self) -> None:
        """Cycle 3. What a real pool cannot be asked for, checked at load.

        These are refused HERE rather than at pool-load time so that a
        mis-specified config fails before a SLURM array has booked a GPU.
        `src/real_data.py` repeats the query-type check, because the loader
        must also hold for a caller that builds a Config by hand.
        """
        from src.real_data import ALLOWED_QUERY_TYPES, REFUSED_QUERY_TYPES
        bad = [qt.name for qt in self.query_types
               if qt.name not in ALLOWED_QUERY_TYPES]
        if bad:
            why = "\n  ".join(
                f"{n}: {REFUSED_QUERY_TYPES[n]}" for n in bad
                if n in REFUSED_QUERY_TYPES)
            raise ValueError(
                f"data.name='real' supports {sorted(ALLOWED_QUERY_TYPES)} "
                f"only; this config declares {bad}."
                + (f"\n  {why}" if why else
                   "\n  The deterministic-window cells are a priori "
                   "tool-saturated, so a real-data run of them measures the "
                   "bare arm's arithmetic, not architecture."))
        if not self.data.get("pool_sha256"):
            raise ValueError(
                "data.pool_sha256 is required when data.name='real'. The "
                "pool is a tracked input shared by every cell of a paired "
                "design, so its content hash is declared, not discovered.")
        stale = [k for k in SYNTHETIC_ONLY_DATA_KEYS if k in self.data]
        if stale:
            raise ValueError(
                f"data.name='real' but the data block still carries "
                f"generator-only keys {stale}. They do not change the "
                "series, but data_fingerprint() hashes them, so they would "
                "define a cache namespace that means nothing. Remove them.")
        if self.design != "c2":
            # The real cells reuse the C2 prompt family (`GENERIC_PROMPT_C2`
            # plus the evidence block) so that a real result and cycle 2's
            # synthetic result on the same backbone differ by the DATA and
            # nothing else - PREREGISTRATION_C3.md C3-5. `bare_features`
            # also only exists on the C2 path.
            raise ValueError(
                f"data.design must be 'c2' for a real pool, got "
                f"{self.design!r}: the synthetic-vs-real comparison in "
                "PREREGISTRATION_C3.md C3-5 requires an identical prompt "
                "family, and the bare_features arm exists only there.")

    def _validate_family(self) -> None:
        """The declared Holm family must be nameable and resolvable."""
        names = {qt.name for qt in self.query_types}
        system_names = {s.name for s in self.systems}
        fam = list(self.analysis.get("confirmatory_family") or [])
        if not fam:
            raise ValueError(
                "analysis.confirmatory_family is empty. Declare the Holm "
                "family by NAME before running; a family discovered from the "
                "data is anti-conservative (see PREREGISTRATION.md §3).")
        unknown = [q for q in fam if q not in names]
        if unknown:
            raise ValueError(
                f"confirmatory_family names unknown query types: {unknown}")
        for key in ("confirmatory_contrast", "secondary_contrast"):
            pair = list(self.analysis.get(key) or [])
            if not pair:
                continue
            if len(pair) != 2:
                raise ValueError(f"analysis.{key} must name exactly 2 systems")
            missing = [s for s in pair if s not in system_names]
            if missing:
                raise ValueError(f"analysis.{key} names unknown systems: "
                                 f"{missing}")
        units = [self.analysis.get("cluster_unit") or "sample_id"]
        units += list(self.analysis.get("cluster_unit_sensitivity") or [])
        bad = [u for u in units if u not in VALID_CLUSTER_UNITS]
        if bad:
            raise ValueError(
                f"unknown clustering unit(s) {bad}; have "
                f"{list(VALID_CLUSTER_UNITS)}")
        if not self.is_real and any(u in ("dataset_id", "stratum")
                                    for u in units):
            raise ValueError(
                "cluster units 'dataset_id' and 'stratum' only exist on a "
                "real pool; a synthetic run writes those columns empty, so "
                "clustering on them would silently fall back to the item.")

    def family_report(self) -> dict:
        """Planned-vs-actual for the family. Written into every snapshot."""
        fam = list(self.analysis.get("confirmatory_family") or [])
        saturated = a_priori_saturated(self.query_types)
        return dict(
            confirmatory_contrast=list(
                self.analysis.get("confirmatory_contrast") or []),
            confirmatory_family=fam,
            n_tests=len(fam),
            secondary_contrast=list(
                self.analysis.get("secondary_contrast") or []),
            a_priori_saturated=saturated,
            saturated_cells_kept_in_family=[q for q in fam if q in saturated],
            excluded_cells=[qt.name for qt in self.query_types
                            if qt.name not in fam],
            note=("Saturation is an a priori architectural property, not an "
                  "empirical screen, so family membership is a design "
                  "decision fixed before the run. Any cell listed in "
                  "saturated_cells_kept_in_family is a deliberate override "
                  "and must be justified in PREREGISTRATION.md §3."))

    # ------------------------------------------------------------------
    @property
    def data_name(self) -> str:
        """`synthetic` (the default) or `real`. Selects the pool loader."""
        return str(self.data.get("name", "synthetic"))

    @property
    def is_real(self) -> bool:
        return self.data_name == "real"

    @property
    def design(self) -> str:
        """`c1` (cycle 1, the default) or `c2`. Lives in the DATA block.

        One switch, read by the generator, the prompts, the scorer and the
        analysis, because a C2 key cannot score a cycle-1 pool and a C2
        prompt cannot be scored by the cycle-1 regexes. Being part of
        `data`, it is in `data_fingerprint()`, so the two cycles can never
        share a pool.
        """
        return str(self.data.get("design", "c1"))

    def pool_dir(self, root: Path) -> Path:
        """Resolve the sample pool directory.

        Honoured by EVERY runner. In the old build `history_length.py` read
        `data.pool_dir` while `query_type.py` hardcoded `ROOT/'data'`, so all
        configs shared one mutable pool slot and running one experiment
        clobbered another's data.
        """
        p = Path(self.data.get("pool_dir", "data/default"))
        return p if p.is_absolute() else Path(root) / p

    def data_fingerprint(self) -> str:
        """Everything that determines the sample pool.

        INCLUDES `seed` (sample ids are positional, so a reseed otherwise
        serves stale answers for new series) and `QUERY_VERSION` (the
        question text is part of every sample, so editing one paraphrase
        must regenerate the pool rather than reuse it).

        On a REAL pool the series are not a function of this config at all -
        they are a function of the file on disk - so `data.pool_sha256`
        carries the pool's identity into the key. `pool_dir` is still
        dropped: two checkouts may hold the same pool at different paths,
        and a path is a location, not content.
        """
        d = dict(self.data)
        d.pop("pool_dir", None)       # location, not content
        blob = json.dumps(dict(
            data=d, seed=self.seed,
            query_version=src.QUERY_VERSION,
            query_types=[(qt.name, qt.kind) for qt in self.query_types],
        ), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def backbone_fingerprint(self, name: str) -> str:
        """What determines ONE backbone's replies. Scoped to that backbone."""
        bb = self.backbones[name]
        payload = dict(provider=bb.provider, model=bb.model,
                       params=bb.params, prompt_version=src.PROMPT_VERSION)
        if bb.provider == "mock":
            payload["mocks"] = self.mocks
            payload["strength"] = bb.strength
            payload["seed"] = self.seed
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def snapshot(self) -> dict:
        snap = dataclasses.asdict(self)
        snap["_versions"] = dict(prompt=src.PROMPT_VERSION,
                                 query=src.QUERY_VERSION,
                                 parser=src.PARSER_VERSION,
                                 scorer=src.SCORER_VERSION,
                                 framework=src.__version__)
        snap["_data_fingerprint"] = self.data_fingerprint()
        snap["_backbone_fingerprints"] = {
            n: self.backbone_fingerprint(n) for n in self.backbones}
        snap["_family"] = self.family_report()
        return snap


def _coerce_query_types(raw: list) -> list:
    out = []
    for item in raw:
        if isinstance(item, str):
            out.append(QueryTypeSpec(item, DEFAULT_KINDS.get(item,
                                                             "numeric_exact")))
        else:
            out.append(QueryTypeSpec(item["name"], item["kind"]))
    return out


def _merge_data(base: dict, raw: dict) -> dict:
    """Merge the data block, WITHOUT inheriting generator knobs into a real
    pool.

    `Config.data`'s defaults describe the synthetic generator (`patterns`,
    `level_range`, the two n's). Merging them into a `name: real` config
    would put dead keys in `data_fingerprint()` and then trip the staleness
    check in `_validate_real_data`, so a real block starts from the small
    set of fields that still mean something.
    """
    raw = dict(raw or {})
    if str(raw.get("name", base.get("name"))) != "real":
        return {**base, **raw}
    shared = {k: base[k] for k in ("max_history", "horizon") if k in base}
    return {**shared, **raw}


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    base = Config()
    cfg = Config(
        seed=int(raw.get("seed", 42)),
        out_dir=raw.get("out_dir", "results"),
        data=_merge_data(base.data, raw.get("data") or {}),
        history_lengths=[int(x) for x in raw.get("history_lengths", [128])],
        query_types=_coerce_query_types(raw.get("query_types", [])),
        backbones={}, systems=[],
        tsfm={**base.tsfm, **(raw.get("tsfm") or {})},
        mocks={**base.mocks, **(raw.get("mocks") or {})},
        judge={**base.judge, **(raw.get("judge") or {})},
        scoring={**base.scoring, **(raw.get("scoring") or {})},
        analysis={**base.analysis, **(raw.get("analysis") or {})},
        harness={**base.harness, **(raw.get("harness") or {})},
        inference={**base.inference, **(raw.get("inference") or {})},
    )
    for name, bb in (raw.get("backbones") or {}).items():
        bb = dict(bb)
        cfg.backbones[name] = BackboneSpec(
            name=name,
            provider=bb.pop("provider", "mock"),
            model=bb.pop("model", "mock-llm"),
            strength=float(bb.pop("strength", 0.6)),
            params=bb.pop("params", {}) or {})
    for s in raw.get("systems") or []:
        cfg.systems.append(SystemSpec(
            name=s["name"], kind=s["kind"], backbone=s["backbone"],
            routing=s.get("routing", "self"), params=s.get("params", {}) or {}))
    cfg.validate()
    return cfg


def save_config_snapshot(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg.snapshot(), indent=2, default=str),
                          encoding="utf-8")
