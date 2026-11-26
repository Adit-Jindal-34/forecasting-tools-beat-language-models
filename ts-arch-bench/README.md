# ts-arch-bench (corrected build)

A benchmark comparing **two architectures** answering the same questions about
the same time series on the same backbone:

| Arm | System | What it does |
|---|---|---|
| `bare_llm` | bare LLM | the series is pasted into the prompt; the model answers directly |
| `harness` | tool harness | the model routes the question to an operation; a deterministic tool or a TSFM does the work |
| `harness_oracle` | tool harness, oracle routing | identical, but handed the correct operation — the standard pipeline ablation |

Three arms, two architectures. There is **no third architecture**: the old
configs carried a `tslm: mock_tslm` entry but no system had kind
`multimodal_tslm`, so it was constructed and never called. The previous
README's three-architecture framing was wrong.

---

## Why this build exists

This is a rewrite of `D:\ts-arch-bench` after an audit found defects that
invalidated published numbers. The audit is in
[the remediation plan](https://claude.ai/artifact/Q2GWBb59ga2XzmcKu8iq3X).
Every fix below has a regression test; `pytest` fails if one is undone.

| # | Defect in the old build | Fix here | Test |
|---|---|---|---|
| 1 | `keyword_coverage` was recall-only, so one constant self-contradictory string scored **1.000** on every qualitative item | slot-based scoring over mutually exclusive classes | `test_metrics.py::test_constant_contradictory_answer_cannot_score_well` |
| 2 | `MockLLM` classify branch unreachable → **100% oracle fallback** in every mock run | markers are shared constants + import-time dispatch assertion | `test_routing.py::test_mock_actually_routes_rather_than_falling_through` |
| 3 | bare arm got `op={QUERY_OP[query_type]}` in its prompt on all 900 samples | one generic prompt for every type, both arms | `test_e2e.py::test_bare_arm_receives_no_operation_hint` |
| 4 | oracle leak: `fallback_op = QUERY_OP.get(sample.query_type)` | abstention; systems receive an `InferenceSample` with no label | `test_leakage.py`, `test_routing.py::test_unroutable_reply_produces_an_abstention_not_an_oracle_answer` |
| 5 | harness computed `op` then built the qualitative prompt from `query_type` | the routed op drives the prompt | `test_routing.py::test_qualitative_branch_follows_the_routed_op` |
| 6 | `1/(1+rel_err)` divided by the truth → cells incomparable by ~10³ | per-operation natural scale, never the target | `test_metrics.py::test_same_physical_error_costs_the_same_across_numeric_cells` |
| 7 | `counting` ground truth was 0 in **83/100** items | per-series spike intensity; 0.34 zeros, 9 distinct counts | `test_synthetic.py::test_counting_is_not_degenerate` |
| 8 | parse failures scored `NaN` then `.dropna()`'d — never penalised | every row scored; coverage and selective risk reported | `test_metrics.py::test_failures_are_scored_not_dropped` |
| 9 | `seed` absent from the cache fingerprint → stale answers for new data | `data_fingerprint()` includes the seed | `test_cache.py::test_seed_is_in_the_data_fingerprint` |
| 10 | the only classify test asserted the leak was correct behaviour | replaced; the abstention path is now the assertion | `test_routing.py` |
| 11 | brittle first-token parser | permissive scan + `clean/recovered/ambiguous/unparseable` taxonomy | `test_routing.py::test_prose_reply_is_recovered_not_dropped` |
| 12 | no output range validation (`pred_idx = -125` on a 128-point series) | per-kind type validation, `type_error` as its own failure mode | `test_metrics.py::test_index_range_validation_catches_the_d3_case` |
| 13 | H2 ignored `pool_dir`; H1 honoured it → one mutable pool slot | `cfg.pool_dir()` used by every runner | `test_cache.py` |
| 14 | fingerprint hashed all backbones → H3 groups shared no cache | per-backbone fingerprints | `test_cache.py::test_backbones_are_independent` |
| 15 | Chronos unseeded; `num_samples` 20 in H2 vs 10 in H3 | seeded per call; one value, in the fingerprint | `test_cache.py::test_forecast_cache_keyed_on_window_bytes` |
| 16 | `build_llm` dropped `spec.params` (temperature ignored) | params passed through and keyed | `test_cache.py::test_generation_params_change_the_key` |
| 17 | `abrupt change` label unreachable: threshold scaled by a quantity the shift inflates | `local_noise_scale` from median |diff| | `test_synthetic.py::test_level_shift_test_can_actually_fire` |
| 18 | the harness was handed the true `seasonal_period` | `estimate_period()` from the window | `test_leakage.py` |

### Second round — defects found by auditing this build

| # | Defect | Fix here | Test |
|---|---|---|---|
| 19 | one fixed question per type → **9 distinct routing decisions** replicated 100× each; routing accuracy per cell could only be 0% or 100% | 100-variant paraphrase bank (25 framings × 4 leads), operational clause preserved verbatim | `test_paraphrases.py::test_classify_prompts_vary_per_item` |
| 20 | Holm family size inferred from the data; degenerate cells left the family and made survivors *easier* to call significant | family declared by name, `m` passed explicitly | `test_family_and_clustering.py::test_declared_m_makes_holm_independent_of_how_many_cells_collapse` |
| 21 | `route_correct` was `None` on abstention and `.mean()` skipped it — a router that abstained on everything scored 100% | two columns: `route_correct` (abstain = 0, primary) and `route_correct_selective` | `test_e2e.py::test_oracle_arm_routes_perfectly_and_self_arm_does_not_cheat` |
| 22 | cell-level `max_abs_gap` over all pairs let the oracle arm make a cell look decidable when the confirmatory contrast wasn't | per-pair `pair_gaps` / `decidable_pairs`, screened on the declared contrast | `test_family_and_clustering.py::test_per_pair_gaps_do_not_let_the_oracle_rescue_a_cell` |
| 23 | tool-saturated cells returned "decidable", contradicting PREREG §1.3.3 | `saturated` as a **separate axis** from `decidable` | `test_family_and_clustering.py::test_tool_saturated_cell_is_flagged_but_still_decidable` |
| 24 | `mean(sign(d))` mislabelled "Cliff's delta" | renamed `paired_sign_effect`, read beside `n_zero_diff` | `test_family_and_clustering.py::test_paired_sign_effect_is_named_correctly` |
| 25 | pseudo-replication: items sharing a phrasing share one routing decision | `paired_compare_clustered`, with a pre-registered rule choosing the unit | `test_family_and_clustering.py::test_cluster_bootstrap_uses_clusters_as_the_unit` |
| 26 | `n_model_calls` conflated architectural and billed calls; the oracle arm looks cheap *by design* | split into `n_model_calls` / `n_billed_calls`, non-comparability stated in output | `test_e2e.py` |
| 27 | mock numbers looked quotable | loud banner: MockLLM reads `src.ground_truth`; no table may be quoted | `scripts/smoke.py` |
| 28 | prompt templates lived beside `MockLLM`, giving `systems/` a transitive import path to the answer key | `models/prompts.py`, importing nothing from `src`; template hashes pinned | `test_prompts.py` |
| 29 | `variant = k % 100` emitted only variants 0..n−1 below full pool size, all sharing one lead | stride assignment covering both axes at any size | `test_paraphrases.py::test_pool_spreads_variants_over_both_axes` |
| 30 | mock op inference keyed on surface wording the paraphrase bank varies | keys on the guaranteed operational clauses; all 900 renderings checked | `test_paraphrases.py::test_mock_infers_the_right_op_for_every_paraphrase` |

---

## The central invariant

A system under test never receives an object carrying ground truth.

```python
sample.view(L)  ->  InferenceSample(sample_id, query, window, L, horizon)
```

Frozen, detached, five fields. `tests/test_leakage.py` additionally parses
every module under `systems/` and fails the build if it so much as mentions
`query_type`, `QUERY_OP`, `dataset_op`, `seasonal_period`, `gt.ground_truth`
or `.target(` outside a docstring.

The oracle arm is the single privileged channel, and it is explicit:
`run_inference` looks up `QUERY_OP[sample.query_type]` **runner-side** and
passes it as `oracle_op`. `systems/base.py` raises if a system that did not
declare `routing: oracle` is handed one.

---

## Caching: what costs money

```
CallCache   (backbone fingerprint, prompts)  -> reply text
TSFMCache   (tsfm fingerprint, window, h)    -> forecast values
```

Model **calls** are cached, not system results. Systems are re-executed on
every run, which is free. So a change to the parser, the router, the scorer,
or adding the oracle arm **replays at zero API cost**. Only bumping
`PROMPT_VERSION` invalidates anything.

```python
PROMPT_VERSION  # bump -> invalidates model calls (expensive)
PARSER_VERSION  # bump -> free replay
SCORER_VERSION  # bump -> free rescore
```

---

## Metrics

Error is expressed as a fraction of each operation's **natural scale**, which
is bounded away from zero and never derived from the target — the
Hyndman & Koehler argument for MASE, applied to the static cells too.

| Kind | Scale | Score |
|---|---|---|
| `mean` `median` `std` `max` `min` | `std(window)` | `1/(1+scaled_err)`, `within_tol` at 0.05 |
| `pct_change` | `100*std(window)/|x[0]|` | same |
| `lag1_autocorr` | `1.0` (bounded) | absolute error |
| `count_spikes` | — | exact match |
| `change_point` | — | hit@{0,1,2,5}, range-validated |
| forecast | fixed MASE denominator across L | `1/(1+MASE)`, `beats_naive` |
| qualitative | — | macro slot accuracy + `incoherent` flag |

**Never average across kinds.** `score` is comparable within a cell, not
between them.

---

## Routing is a property of the question, not the item

The classifier never sees the series, so with one fixed template per query
type the entire benchmark contained **nine** routing decisions, replicated
100 times each. Each type now draws from a bank of 100 phrasings:

```
variant v  ->  LEADS[v // 25] + FRAMES[qt][v % 25] + OPERATIONAL_CLAUSE[qt]
```

The operational clause — the MAD definition of a spike, the 0-based-index
convention — is appended verbatim and checked at import. Paraphrase the
framing, never the operation.

Whether the lead axis actually moves routing is measured, not assumed:
`routing_variance_within_framing` reports it and a **pre-registered**
threshold picks the independent unit (framing, n=25/type, or item,
n=100/type). The self-versus-oracle contrast then uses
`paired_compare_clustered` at that unit, because items sharing a phrasing
share one routing decision.

## The declared Holm family

```
confirmatory : bare_llm vs harness, over 5 named cells, m = 5
secondary    : harness_oracle, effects + CIs, outside the correction
```

Declared **by name** in `analysis.confirmatory_family` and printed on every
run. `m` is passed explicitly: inferring it from the data lets degenerate
cells leave the family and makes the survivors easier to call significant.

`saturated` and `decidable` are **separate axes**. A cell with the harness
pinned at 1.000 and the bare arm at 0.64 is decidable — the gap is real —
but it measures the bare arm alone, not architecture.

## Screening before testing

`src/screening.py` runs *before* any hypothesis test and writes
`screening.json`: per-cell variance, ceiling/floor flags, decidability
against the pre-registered effect floor, item discrimination, and the
**trivial baseline** — what a constant answer scores in that cell. The smoke
run already shows why this matters: a constant answer scores 0.796 on
`correlation`, so a system scoring 0.965 there has less headroom than it
appears.

---

## Layout

```
src/        samples (Sample / InferenceSample / paraphrase bank),
            ground_truth, synthetic, config, cache, stats, screening, pipeline
systems/    base, routing (parser + abstention), tools, bare_llm, harness_tsfm
evaluation/ metrics, qualitative (slot scorer), evaluate, judge
models/     prompts (templates only, no `src` import), llm (mock),
            providers (Gemini/OpenAI/HF), tsfm (Chronos)
configs/    smoke_mock.yaml, base_gemini.yaml
scripts/    smoke.py
tests/      150 tests, one per defect above
```

Hypothesis runners (H1/H2/H3) are **not** included yet — they are built on
this base once it is agreed.

---

## Running

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python -m pytest                              # 85 tests
python scripts/smoke.py --config configs/smoke_mock.yaml
```

Real runs need `GEMINI_API_KEY` and, for forecasting,
`chronos-forecasting` + `torch`.

See [PREREGISTRATION.md](PREREGISTRATION.md) before running anything whose
result will be reported.
