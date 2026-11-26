# Cycle 3 — the run sheet

*Everything needed to execute cycle 3, in order. The DESIGN and its
justifications are in [`PREREGISTRATION_C3.md`](PREREGISTRATION_C3.md); read
that first and do not deviate from it without recording the deviation.*

---

## 0. What cycle 3 is, in four lines

Move the **forecasting cell** — the one cell where real data has a
legitimate answer key, the held-out future — onto real Monash series.
Declared family **m = 1**. Four runs plus one sensitivity arm, **≈14,800
billable calls**. The headline from cycle 2 is genuinely at risk: the
trivial baseline here scores **0.5688**, not ~0.497.

---

## 1. Put the code on the cluster (nothing to download or build)

The job scripts run from `$TSARCH/ts-arch-bench`. Copy the CONTENTS of this
folder there, over the code that is already in place. This bundle contains
everything the six-model run of 14 Nov used (amendments A16/A17, the probe,
the retry), so nothing is lost by overwriting; `results/`, `logs/` and the
cache are not in the bundle and are left alone.

```bash
# on your machine, from the folder that holds this file
rsync -av --exclude results --exclude logs --exclude '__pycache__' \
    ./ <user>@<login-node>:/scratch/<user>/claude-ts-arch-bench/ts-arch-bench/

# on the cluster
source /scratch/$USER/claude-ts-arch-bench/env.sh
cd $TSARCH/ts-arch-bench
python data-real/check_pools.py
```

`check_pools.py` must end with `all 9 real-data configs match their pools`.
Expected hashes:

| pool | n | sha256 (first 16) |
|---|---|---|
| `pool_clean_n300` | 300 | `5e7fade71450e8ae` |
| `pool_real_n300` | 300 | `7322149e173d4615` |
| `pool_h2_strata_n240` | 240 | `7d6a2e80d15636bc` |

### ⚠ Do NOT rebuild the pools, and do NOT unpack `data-real.zip` over this folder

* **The three pools in `data-real/` are the declared ones. A run needs
  nothing else** - no `fetch.py`, no `build_pool*.py`, no raw files.
* Rebuilding on another machine gives the same windows and denominators
  but a **different file hash** (the two strata columns differ around the
  13th decimal place; measured 14 Nov), and every config then refuses to
  load. The builders now stop if a pool already exists.
* The separate `data-real.zip` that was passed around on 14 Nov is an
  OLDER snapshot: its `pool_real_n300` was built before the flat-window
  exclusion rule (hash `26781a3c...`), its `MANIFEST.json` still has the
  contamination table the wrong way round, and it has no clean pool. Do not
  use it.
* If a hash does not match, **do not edit the config to match the pool.**
  Copy the pool from this bundle again.

Qwen2.5 (seven sizes) and `amazon/chronos-t5-small` are the same models
cycle 2 used, so they are already staged on the cluster.

---

## 2. Check the wiring without a GPU (login node, ~3 min)

```bash
python -m pytest tests/ -q                 # 335 tests, all must pass

# mock backbone - validates PLUMBING ONLY, never quote a number from these
python -m runners.h2_query_type       --config configs/smoke_c3_real.yaml
python -m runners.h2_series_character --config configs/smoke_c3_strata.yaml
```

The smoke runs print a loud banner saying their numbers are meaningless.
`MockLLM` reads `src.ground_truth` to synthesise answers at a target
accuracy, so every table from them validates plumbing and nothing else.

---

## 3. The runs

| unit | run | config | calls | GPU |
|---|---|---|---|---|
| `h3` | H3 ladder + headline | `c3_h3_clean.yaml` | 4,200 | 7 rungs |
| `h1` | H1 L-sweep | `c3_h1_clean.yaml` | 4,500 | 1 (32B) |
| `h2` | H2 reframed | `c3_h2_strata.yaml` | 720 | 1 (32B) |
| `h1c` | H1 contamination arm | `c3_h1_contaminated.yaml` | 4,500 | 1 (32B) |
| `h8` | horizon sensitivity | `c3_sensitivity_h8.yaml` | 900 | 1 (32B) |

### One command

```bash
bash slurm/submit_c3.sh --test    # asks the scheduler; submits nothing
bash slurm/submit_c3.sh           # submits everything
bash slurm/submit_c3.sh h3 h1     # or only some units
```

What it queues:

| job | what | partition | GPUs | mem | time |
|---|---|---|---|---|---|
| `c3-h3-small` | rungs 0.5B, 1.5B, 3B, 7B (array 0-3) | H200 or H100 | 1 | 26G | 2 h |
| `c3-h3-14b` | 14B (array 4) | H200 or H100 | 1 | 90G | 2 h |
| `c3-h3-32b` | 32B (array 5) | H200 | 1 | 90G | 2 h |
| `c3-h3-72b` | 72B (array 6) | H200 | **2** | 128G | 2 h |
| `c3-h1` | five lengths, starts after `c3-h3-32b` | H200 | 1 | 90G | 4 h |
| `c3-h1c` | five lengths, contaminated pool | H200 | 1 | 90G | 4 h |
| `c3-h2` | strata run | H200 | 1 | 90G | 1.5 h |
| `c3-h8` | horizon 8, analyses itself | H200 | 1 | 90G | 1 h |
| `c3-*-an` | CPU analysis after each unit | compute | 0 | 16G | 2 h |

**Why not the single `--array=0-6` of the first draft of this sheet:** it
gave every rung one GPU. The 72B is ~153 GiB of weights and fits on no
single card, so `serve_vllm.sh` would have refused it; the sizes above are
the ones cycle 2 ran with.

**Why H1 waits for the 32B rung:** `c3_h1_clean.yaml` and
`c3_h3_clean.yaml` declare a byte-identical data block (both fingerprint to
`bc48753143b9cb12`), so H3's 32B rung at L=128 and H1's L=128 cell render
the same prompts and the second is a free replay from the shared cache.

**The 32B jobs go to the H200 only.** On an 80 GiB card one GPU leaves
about 3 GiB of KV cache, too little for eight requests at once. If the H200
queue is long:

```bash
C3_PART_32B=gpu_h200_8,gpu_h100_4 C3_GRES_32B=2 bash slurm/submit_c3.sh
```

### What changed in how a job runs (execution only)

* **Eight requests at a time** (`inference.workers: 8` in the `c3_*.yaml`
  configs). The answers are scored exactly as before and
  `tests/test_parallel.py` checks the output equals a one-at-a-time run.
  Cycle 2's Qwen runs were one at a time. To match that, set `workers: 1`
  and ask for more time (`h1`/`h1c` about 8 h, `h2` 1.5 h).
* **A failed request is retried** up to four times (timeouts, dropped
  connections, HTTP 5xx). Before, one such error ended the cell.
* **Each job uses its own port** (`slurm/pick_port.sh`). Before, every job
  served on 8000, and two jobs on the one H200 node would collide.

### If a job fails or times out

Run `bash slurm/submit_c3.sh` again (or with just that unit). Finished
cells are skipped and finished calls replay from the cache. Look at the end
of the job's log first; `[serve]` lines say whether the model fitted.

### Analysis by hand

The `c3-*-an` jobs do this for you; these are the same commands, for
re-running on a CPU node. Analysis touches no model.

```bash
python -m runners.h3_backbone_strength  --config configs/c3_h3_clean.yaml        --stage analyse
python -m runners.h1_history_length     --config configs/c3_h1_clean.yaml        --stage analyse
python -m runners.h2_series_character   --config configs/c3_h2_strata.yaml       --stage analyse
python -m runners.h1_history_length     --config configs/c3_h1_contaminated.yaml --stage analyse
python -m runners.h2_query_type         --config configs/c3_sensitivity_h8.yaml  --stage analyse
```

**Expect a full-price run.** A new pool means a new data fingerprint, so
nothing replays from the cycle-2 cache.

---

## 4. Reading the results — five traps

### 4.0 "win: YES" does not mean the harness won

`win` marks a DECIDED cell under the two-sided test: significant, CI off
zero, above the effect floor - in either direction. A cell where the
harness is significantly WORSE also prints `YES` and counts in `wins: 1/1`.
The tables now print a NOTE when that happens. Always read the sign of
`mean_d` (positive favours the harness).

### 4.1 The bar is 0.5688, not 0.5

`trivial_baselines` reports `best_constant_score: 0.5` for forecasting,
which means "matching the in-sample naive forecast the MASE denominator is
built from". **That is not the bar on real data.** The bar is the
declared-period seasonal naive in the `forecast_baselines` block:

> **Pre-registered prediction (C3-7):** the `harness` arm beats
> declared-period seasonal naive — mean score above **0.5688** at L=512,
> h=24, CI on the paired difference excluding zero.
>
> If it fails, **cycle 2's headline does not generalise to real data.** That
> is the finding, and it is reported as the primary result, not explained
> away.

### 4.2 The clustered p-value is NaN by construction

`clustering_sensitivity` prints both units. `dataset_id` has k = 4, below
`paired_compare_clustered`'s `min_n = 10`, so its p is NaN and its CI is
~3× wider. That is not a bug — it is the honest statement that four
archives cannot support a p-value. If the two units disagree, say in the
write-up that the result depends on treating four archives as four
observations.

### 4.3 H2's pooled contrast may be an archive effect

Weak-seasonality series come overwhelmingly from `weather` (106 of 120).
The runner prints a within-dataset contrast beside the pooled one.
**Pre-registered rule: where they disagree, the within-dataset version
carries the claim.** If pooled moves and within-`weather` does not, report
an archive effect, not a series-character effect. The runner prints that
sentence for you.

### 4.4 Contamination is controlled on ONE side only

Chronos Benchmark II settles it for the **harness** arm. The **bare LLM**
arm is not controlled — these archives are public text and Qwen2.5 may hold
the literal values. The asymmetry runs *against* the harness, so a positive
result is conservative and **a null result may not be read as "the tool does
not help"**. This paragraph must appear in the write-up.

---

## 5. What is in this bundle

```
PREREGISTRATION_C3.md    the design. Read before running.
RUN_CYCLE3.md            this file.
PREREGISTRATION_C2.md    } inherited and still binding except where
PREREGISTRATION.md       } C3 supersedes them by name
FINDINGS_C2_2026-11-13.md   cycle-2 results, for the comparison
PROJECT_CONTEXT.md       the project as a whole
HANDOFF_REAL_DATA_*.md   the workstream handoff. ⚠ carries a SUPERSEDED
                         banner: its contamination table was inverted.

configs/c3_*.yaml        the five runs
configs/smoke_c3_*.yaml  mock-backbone plumbing checks
configs/c2_*.yaml        cycle 2, for the paired synthetic<->real comparison

data-real/               fetcher, .tsf reader, pool builders, diagnostics,
                         and the three built pools (raw/ is NOT included -
                         it is ~350 MB, re-fetchable by fetch.py, and NOT needed)

src/ systems/ models/ evaluation/ runners/ scripts/ tests/
slurm/                   including the new c3_h2_strata.sbatch
requirements.txt pytest.ini
```

Not included: `results-archive/` (162 MB of cycle-1/2 outputs),
`cluster-sync/`, and `data-real/raw/`. None is needed to run cycle 3.

---

## 6. New in cycle 3 — what to review if you are auditing the code

| file | what changed |
|---|---|
| `src/real_data.py` | **new.** The loader. Refuses qualitative and tool-saturated cells by name, verifies the pool hash, emits opaque sample ids |
| `src/pipeline.py` | `make_samples` dispatches on `data.name`; `dataset_id` / `stratum` reach the shards |
| `src/config.py` | `data.name: real` validation; `cluster_unit` + its sensitivity list; real blocks no longer inherit generator defaults |
| `evaluation/evaluate.py` | `forecast_denominators` prefers the pool's fixed denominator |
| `src/stats.py` | **new** `two_sample_compare` (bootstrap CI, Mann-Whitney, Cliff's δ) |
| `runners/_common.py` | clustered confirmatory path, `clustering_sensitivity`, `pool_note`; `confirmatory_did` refuses `axis="stratum"` |
| `runners/h2_series_character.py` | **new.** The reframed H2 |
| `systems/tools.py` | the degenerate-window floor guard (C3-10) |
| `data-real/build_pool_c3.py` | **new.** The two cycle-3 pools |
| `data-real/MANIFEST.json` | contamination flags **corrected** against Chronos Table 2 |
| `tests/test_real_data.py` | **new.** 29 tests |
| `slurm/submit_c3.sh` | **new (14 Nov, cluster fixes).** Submits every unit with the right GPUs, memory and order |
| `slurm/pick_port.sh` | **new.** A private port per job; sourced by every script that starts a server |
| `data-real/check_pools.py` | **new.** Every real config against its pool's hash |
| `data-real/build_pool*.py` | refuse to overwrite an existing pool (`--force` to make a new one) |
| `models/providers.py`, `src/cache.py`, `src/pipeline.py`, `systems/base.py` | the 14 Nov retry / parallel-workers / cut-off-reply code, merged in from the six-model run |
| `configs/c2_h2x_*.yaml`, `scripts/probe_server.py`, `slurm/submit_c2_extra.sh`, `tests/test_parallel.py` | the six-model H2 replication (C2 amendments A16, A17), so there is one code folder, not two |

### Two verifications worth re-running

```bash
# the floor guard must be byte-invisible on cycle-2 data, because
# evidence_to_block output is inside the call cache key
python -m pytest tests/test_real_data.py -k floor_guard -q

# the cycle-2 data fingerprints must be unchanged, or the archived cache
# stops replaying. Expect 25bc2e49e725c4e0 (c2_h1) and 508e6e37e1447485
# (c2_h2, c2_h3).
python -c "
from src.config import load_config
for p in ['configs/c2_h1.yaml','configs/c2_h2.yaml','configs/c2_h3.yaml']:
    print(p, load_config(p).data_fingerprint())"
```
