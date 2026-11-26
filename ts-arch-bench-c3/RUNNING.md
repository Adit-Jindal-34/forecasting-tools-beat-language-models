# Running H1/H2/H3 on hpc01

Read [PREREGISTRATION.md](PREREGISTRATION.md) §6 first. Each hypothesis has a
dated section there describing what it tests and why; those sections were
written before any runner existed, which is the point.

---

## Cycle 2 - run this, not cycle 1

Cycle 1 (sections 0-5 below) is finished and archived. **New runs are cycle
2**: new pool, new answer key, four arms. Read
[PREREGISTRATION_C2.md](PREREGISTRATION_C2.md) §6 (amendments A9-A11) first.
Nothing in cycle 2 replays the cycle-1 cache, so every call is real GPU work.

**Blocked until `TSARCH_BITSID` is set** in
`/scratch/$USER/claude-ts-arch-bench/env.sh` (see "Job naming" below);
`slurm/submit.sh` refuses to submit without it.

```bash
# 0. on the cluster: sync this repo, then prove the plumbing on the mock (CPU)
python -m pytest -q
python -m runners.h2_query_type --config configs/smoke_c2_h2.yaml --stage both
```

**One command submits everything** (H3's seven rungs, H2, H1, and the H3
analysis chained behind them):

```bash
source /scratch/$USER/claude-ts-arch-bench/env.sh
```

```bash
bash slurm/submit_c2.sh
```

`bash slurm/submit_c2.sh --test` asks the scheduler for start estimates
without submitting. Re-running the script is safe at any time: work already
queued is left alone, finished work is skipped.

How it is laid out, and why (measured on 2026-11-13; details in the header of
`slurm/submit_c2.sh`):

| unit | steps | GPUs | memory asked | time asked | where |
|---|---|---|---|---|---|
| `small` | 0.5B, 1.5B, 3B, 7B | 1 | 26G | 2 h | h200 + h100 |
| `mid` | 32B, then H2, then 14B | 1 | 90G | 2 h | h200 + h100 |
| `big` | 72B | 2 | 128G | 2 h | h200 only |
| `h1` | H1, 32B at five lengths, + its analysis | 1 | 90G | 4 h | h200 + h100 |

- **Bundled**, because each job waits in the queue once and the wait is far
  longer than the work.
- **Each 1-GPU unit is queued twice, once per partition.** This account
  cannot name several partitions in one job. The first copy to start takes a
  lock (`$TSARCH/locks/`), cancels the other, and a copy that starts anyway
  exits in seconds.
- **Memory is the scarce resource, not GPUs**, so each unit asks for what
  cycle 1 measured plus a margin. On the day of submission h200 had an idle
  GPU and under 30G of free RAM.
- **Each job serves its model on its own port** and keeps its own cache
  directory (`$TSARCH/cache/c2_<unit>`); H2 runs inside `mid`, straight after
  the 32B rung, so it is still a cache replay.
- a100 (two-month estimate), v100 (no bf16) and the RTX partition (no QOS
  access) are not used.

Logs are `logs/c2_<unit>_<jobid>.out`. A unit that fails or times out
releases its lock; run `bash slurm/submit_c2.sh` again to resume it.

The older per-hypothesis scripts (`submit_all.sh`, `h1_all.sbatch`,
`h2.sbatch`) still work with the `TSARCH_H*_CONFIG` overrides, but they use
the fixed port 8000 and must not be run two to a node.

What the cycle-2 output adds, in the order it prints:

| block | what to read |
|---|---|
| constant-answer rule | a qualitative cell marked UNDECIDABLE has left the family; `m` is what survived |
| confirmatory family | as in cycle 1, over the surviving cells |
| each arm's own change (H1, H3) | which arm moved; never quote the DiD without it |
| declared secondary contrasts | `bare_features - bare_llm` = does the computation help; `harness - bare_features` = does the scaffolding add anything |
| classical baselines | every arm against last-value, window-mean and seasonal-naive on the same items |
| routing, strict vs functional | `route_equivalent` treats `describe` and `explain` as the same tool |

Everything is also in `results/c2_*/analysis.json` under the `c2` key.
Archive with `slurm/archive.sh` as for cycle 1: `/scratch` is not durable.

### H2 on further models (amendment A16)

Six more models, each a separate replication of H2 on the same 100 items per
cell: gpt-oss-20b, gpt-oss-120b (reasoning medium, and high), Granite-3.3-8B,
Phi-4, Mistral-Small-3.2-24B, OLMo-2-32B. Configs are
`configs/c2_h2x_<model>.yaml`.

```bash
# 1. on the LOGIN node, once: download the weights (~250 GB). Wait for
#    "done; failed: none" in the log before submitting anything.
source /scratch/$USER/claude-ts-arch-bench/env.sh
```

```bash
nohup python scripts/stage_extra_models.py > logs/stage_extra.log 2>&1 &
```

```bash
# 2. start estimates only, nothing submitted
bash slurm/submit_c2_extra.sh --test
```

```bash
# 3. submit all five units and the cross-model summary behind them
bash slurm/submit_c2_extra.sh
```

| unit | steps | memory asked | time asked |
|---|---|---|---|
| `oss20` | gpt-oss-20b | 48G | 4 h |
| `oss120m` | gpt-oss-120b, medium | 100G | 6 h |
| `oss120h` | gpt-oss-120b, high | 100G | 10 h |
| `famA` | Granite-3.3-8B, then Phi-4 | 64G | 3 h |
| `famB` | Mistral-Small-3.2, then OLMo-2-32B | 100G | 3 h |

All are 1 GPU, queued on h200 and h100 like the units above.

- **Each model is probed first** (`scripts/probe_server.py`, three real
  prompts). If its replies are not in the expected format the run for that
  model is not started and the log says why (`[probe] FAIL`). Look there
  first when a step fails within a minute of the server coming up.
  A reply cut off at the token limit is a `[probe] WARNING`, not a failure,
  unless all three are cut off (amendment A17).
- **gpt-oss-120b at high effort** was stopped by the first version of the
  probe on 2026-11-14 (one forecasting reply used all 16,384 tokens). Under
  A17 its limit is 32,768. To run only that unit, run the submit script
  again: the four finished units are skipped because their locks are marked
  done, and the cross-model summary is queued again behind it.
- **A transient server error no longer ends a step.** The client retries
  timeouts, dropped connections and HTTP 408/429/5xx (4 times, waiting
  5 s, 10 s, 20 s, 40 s); other 4xx errors fail at once because repeating them cannot
  help. Each retry is a `retry n/4` warning in the log.
- **Replies that ran out of tokens are counted.** A reasoning model cut off
  before its answer is scored as unanswered. The step's log ends with
  `N model call(s) hit max_tokens` when that happened, and each such call
  carries `meta.call.finish_reason = "length"` in `calls.jsonl`. If N is
  large for one model, say so beside its coverage column.
- **The harness's forecasts are the Qwen run's.** A new unit's cache starts
  from `$TSARCH/cache/c2_mid/forecasts.jsonl`, so the tool's forecast is the
  same numbers for every model whichever card the job lands on. The log
  line `forecasts: started from ...` confirms it; if it says none were
  found, restore that file from the archive before trusting a cross-model
  comparison of the harness arm to the last digit.
- A unit that times out resumes from its cache: run the submit script again.

Results: `results/c2_h2x_<model>/` per model and
`results/c2_cross_model/cross_model.csv` for the table across all of them.

---

## Job naming — required; jobs are killed without it

**Since 2026-11-10, every SLURM job on Sharanga (`kudhru`, `jagatsesh`) and
CSIS (`dhruvk`) must be named**

```
name_bitsid_project[_anything]
```

**Anything else is killed immediately on submission.** These are shared
logins, so the job name is the only way to attribute GPU-hours and queue time
to a person rather than to one anonymous account.

Set your identity once, in `/scratch/$USER/claude-ts-arch-bench/env.sh`:

```bash
export TSARCH_NAME=adit             # first name, lowercase, hyphen if needed
export TSARCH_BITSID=2023A7PS1191H  # YOUR BITS ID - no default, on purpose
export TSARCH_PROJECT=tsarch        # short project tag
```

Then **submit through `slurm/submit.sh`, not `sbatch`**:

```bash
slurm/submit.sh slurm/h2.sbatch                     # -> adit_<id>_tsarch_h2
slurm/submit.sh slurm/h1.sbatch '' --array=0-4      # extra args pass through
slurm/submit.sh slurm/judge_slots.sbatch h1h3 --export=ALL,TSARCH_EXP="h1 h3"
```

Arguments are `<script> [tag] [extra sbatch args...]`. The tag is the
optional trailing field the trackers ignore, so use it to tell your own runs
apart in `squeue`; pass `''` to let it default to the script name.

**Why a wrapper rather than editing the `#SBATCH -J` lines.** `#SBATCH -J` is
parsed before any shell runs, so it cannot interpolate `$TSARCH_BITSID`, and
hardcoding an ID would put one person's identity in seven files — all of
them wrong for the next user of a shared login. `slurm/jobname.sh` builds the
name in one place, and `--job-name` on the command line overrides the static
directive.

`TSARCH_BITSID` deliberately has **no default**. A guessed or inherited ID
bills GPU-hours to the wrong person, which is the problem this policy exists
to fix, so `submit.sh` refuses locally — before anything reaches the
scheduler — rather than guessing:

```
ERROR: cannot build a compliant SLURM job name - unset: TSARCH_BITSID
```

The `#SBATCH -J` lines in `slurm/*.sbatch` hold an invalid placeholder on
purpose, so a direct `sbatch slurm/h2.sbatch` fails visibly instead of
running under a name that bills its hours to nobody. Each script also calls
`tsarch_assert_jobname` as a second net; that cannot prevent a kill (the
scheduler decides at submit time) but it puts the reason in the log.

> `STATUS_2026-11-07.md` predates this policy and still shows bare `sbatch`.
> It is a dated record and is left as written — use the forms here.

---

## 0. Before anything

```bash
bash slurm/setup_env.sh          # one time; builds the venv on /scratch
source /scratch/$USER/claude-ts-arch-bench/env.sh
cd $TSARCH/ts-arch-bench
python -m pytest -q              # 151 tests; all must pass
```

Then three things, in this order, before any `sbatch`:

```bash
bash slurm/preflight.sh          # GPU count + VRAM per card, and whether
                                 # Slurm exports SLURM_GPUS_ON_NODE here
```

`sinfo -o "%G"` truncates GRES to ten characters and hides both the GPU
count and the memory, so **which `--gres` each rung needs is unknown until
you run this.** It decides whether the 32B — the H1/H2 backbone — fits one
card or two, and whether `gpu_a100_8` belongs in the partition list at all.

```bash
printf '%s' '<hf-token>' > ~/.hf_token && chmod 600 ~/.hf_token
```

Only needed for the two **gated** `meta-llama/Llama-3.1-*` rungs, which are
the cross-family robustness check and sit outside the confirmatory family —
skip the token and skip those rungs if you prefer. Never put it in a config.

```bash
bash slurm/stage_models.sh --llama    # login node: pre-download ~410 GiB
```

Compute nodes often have no outbound network, in which case a GPU job that
reaches for the hub hangs until its time limit having produced nothing. Once
staged, submit with `TSARCH_OFFLINE=1` so a job cannot silently try.

Everything large lives on `/scratch` because `$HOME` has a 40 GiB quota with
~3.8 GiB free. Everything *irreplaceable* is archived back to `$HOME`,
because `/scratch` deletes files 15 days after last modification. See
`slurm/archive.sh` for which is which and why.

### Both pre-registration decisions are settled

**The §3 family** (deviation D6, 2026-11-06): `m = 3`, exactly the
non-saturated cells — `forecasting`, `trend_description`,
`event_explanation`. A saturated cell's answer score *is* its routing
outcome, so its gap is already carried by `route_correct` and the oracle
ablation; including it would double-count. `family_report()` prints m every
run, and `tests/test_family_and_clustering.py` asserts no saturated cell can
enter the family.

**The H3 strength axis** (§6 H3, declared 2026-11-06): MMLU-Pro / 100,
vendor self-reported, sources and retrieval date recorded verbatim in
`configs/h3_backbone_strength.yaml`. MMLU-redux is the pre-registered
secondary axis and the runner computes it. GSM8K was considered and
**rejected on its own published numbers** — it is non-monotone across the
ladder (32B 95.9 beats 72B 95.8) and compresses the top four rungs into 4.3
points.

Nothing is left `UNDECLARED`, so the runner no longer prints the
strength-axis warning. If you change either decision, log it in §5 first.

---

## 1. Order matters: H3 → H2 → H1

The call cache keys on `(backbone fingerprint, PROMPT_VERSION, prompts)` —
not on the experiment name, and not on `L`. So identical prompts across
hypotheses are paid for **once**, provided every run shares one
`--cache-dir`:

| step | new model calls | why |
|---|---|---|
| H3, 9 rungs at L=128 | the whole spend | nothing cached yet |
| H2, `qwen2_5_32b` at L=128 | **zero** | byte-identical to H3's rung 6 |
| H1, `qwen2_5_32b` at L∈{32,64,256,512} | 4 cells, not 5 | L=128 already cached |

Running them in the other order costs the L=128 cell three times.

```bash
mkdir -p logs

# --- H3: the big spend. One array task per rung. ---
# --gres per rung comes from preflight.sh; these are the 80 GiB-card values.
slurm/submit.sh slurm/h3.sbatch '' --array=0-4,7            # 0.5B..14B + llama-8B
slurm/submit.sh slurm/h3.sbatch '' --gres=gpu:2 --array=5   # 32B, ~69 GiB weights
slurm/submit.sh slurm/h3.sbatch '' --gres=gpu:4 --array=6,8 # 72B / 70B (gpu:2 on H200)
# gpu:2 is NOT enough for a 72B on 80 GiB cards: ~153 GiB of weights against
# 144 GiB of VRAM. serve_vllm.sh aborts with the right number rather than
# OOMing after the download.
# ... wait for the array, then analyse on a CPU node:
python -m runners.h3_backbone_strength \
    --config configs/h3_backbone_strength.yaml --stage analyse

# --- H2: free if H3 is done. No GPU needed. ---
python -m runners.h2_query_type \
    --config configs/h2_query_type.yaml --stage both \
    --cache-dir "$TSARCH_CACHE"

# --- H1: one array task per L. ---
slurm/submit.sh slurm/h1.sbatch '' --array=0-4
python -m runners.h1_history_length \
    --config configs/h1_history_length.yaml --stage analyse

bash slurm/archive.sh h3-h2-h1-$(date +%Y%m%d)
```

Tensor-parallel size is read from `$SLURM_GPUS_ON_NODE`, so it tracks
whatever `--gres` you actually requested instead of a hardcoded table that
can drift.

---

## 2. The two stages

Every runner has `--stage inference` and `--stage analyse`.

**Inference** runs one `(backbone, L)` cell and writes
`results/<exp>/shards/bb=<name>__L=<n>.csv`. It is idempotent: a cell whose
shard already exists is skipped, so re-submitting a timed-out array resumes
rather than restarts.

**Analyse** reads every shard and touches no model. It writes
`screening.json` **before** any test is read (§2), then `analysis.json`,
`sensitivity_grid.csv`, `desc_*.csv` and `raw_results.csv`. Because it is
free, re-run it as often as you like — that is what makes the mandatory
sensitivity grid affordable rather than aspirational.

`--stage analyse` **refuses to run on a partial grid** and names the missing
cells, rather than quietly computing a family over whatever happens to be
on disk.

---

## 3. Shake it down first

Two different shakedowns, catching different things. Neither is an
experiment.

```bash
# (a) my runner logic: ~20 s, no GPU, no queue
python -m runners.h2_query_type --config configs/smoke_h2.yaml --stage both
python -m runners.h1_history_length --config configs/smoke_h1.yaml --stage both
python -m runners.h3_backbone_strength --config configs/smoke_h3.yaml --stage both
```

These use `MockLLM`, which reads `src.ground_truth` to synthesise answers at
a target accuracy. Every runner prints a banner saying so. **No table from a
mock run may be quoted.** What they genuinely verify: pivots, item
alignment, the DiD, the clustering rule, the both-ends screen, the grid, and
— via `prose_rate: 0.15` — the permissive parser and the abstention path.

```bash
# (b) the serving path: ~5 min, 1 GPU. The mock CANNOT catch these.
TSARCH_H2_BACKBONE=qwen2_5_0_5b slurm/submit.sh slurm/h2.sbatch
```

Only a real backbone exercises `OpenAICompatibleLLM`, and these four traps
live there:

- **`base_url` is in the cache fingerprint.** Address vLLM as
  `127.0.0.1:8000` always. `gpunode5:8000` is a *different cache key*, so the
  next job silently re-asks every prompt.
- **`--max-model-len` must be ≥ 8192.** At L=512 the bare arm's prompt is
  ~3.5k tokens. A shorter window makes vLLM *reject* those requests, which
  lands in the results as bare-arm failures rather than as the serving bug it
  is.
- **No bfloat16 on V100** (sm_70). Use **`float16`**, not `float32`: V100 has
  fp16 tensor cores, so fp32 doubles the memory for no benefit and is
  slower. `serve_vllm.sh` detects compute capability < 8.0 and substitutes
  fp16 automatically; override with `VLLM_DTYPE=...` if needed. The same
  applies to `tsfm.torch_dtype` in the configs, which is **not** auto-
  detected — set it to `float16` by hand if you put Chronos on a V100.
- **The served model name must match the config's `model`.** A mismatch 404s
  every request and reads as a wall of unparseable answers. `serve_vllm.sh`
  prints `/v1/models` so you can see it.

---

## 4. Reading the output

`print_header` opens every run with the declared family, which cells are
a priori saturated, and `saturated_cells_kept_in_family`, which after D6
is always empty - an invariant, not a coincidence, asserted by
`tests/test_family_and_clustering.py`.

Then, per hypothesis: the descriptive curve, the confirmatory family table
(`mean_d`, bootstrap CI, raw and Holm-adjusted p, win), **both verdicts side
by side**, the rank shape with `min_detectable_rho` beside it, and the
secondary ablation with its cluster unit.

Things worth not misreading:

- **`p_adj` near 1 everywhere with `m = 3`** is normal on small n. The runner
  notes when `p_raw` is NaN — identical arms, or n below `min_n` — and says
  that the cell still consumes its share of `m`, because you planned to test
  it.
- **`saturated` and `decidable` are different axes.** A cell with the harness
  pinned at 1.000 and the bare arm at 0.65 is a real 0.35 gap and perfectly
  decidable — it just measures the bare arm, not architecture.
- **For H1/H3 a cell is screened-in only if decidable at *both* ends** of the
  sweep. The confirmatory test is a difference between those ends, so no
  headroom at either one means the gap cannot be shown to move. On the mock
  this cut H3 to 2 cells and H1 to 3 — expect real screening, not a
  formality.
- **`min|rho|` above the observed `|rho|` is not a null result.** It means the
  design could not detect a relationship of that size. §3 fixes the wording.
- **`n_model_calls` ≠ `n_billed_calls`**, and neither is comparable across
  arms: the oracle arm skips classification and issues zero calls on numeric
  cells by design.

---

## 5. What is new here, and what is not

Two framework files were modified, both for deviation D6:
`src/config.py` (`DEFAULT_CONFIRMATORY_FAMILY` and its justification) and
`tests/test_family_and_clustering.py` (the `== 5` assertion becomes a
structural invariant: no saturated cell may enter the family). Nothing in
`systems/`, `models/`, `evaluation/` or `scripts/` was touched. `python -m pytest -q` → **151 passed** (150 original + one new family invariant).

Added:

```
runners/_common.py              shared inference + analysis machinery
runners/h{1,2,3}_*.py           one entry point per hypothesis
configs/h{1,2,3}_*.yaml         real runs (vLLM)
configs/smoke_h{1,2,3}.yaml     mock shakedowns
configs/_data_main.yaml         reference copy of the shared data block
slurm/setup_env.sh              venv + every cache redirected to /scratch
slurm/serve_vllm.sh             launch vLLM, block until /health
slurm/h{1,2,3}.sbatch           array jobs
slurm/archive.sh                copy irreplaceable artifacts to $HOME
```

Modified: `PREREGISTRATION.md` §6 only — the three hypothesis sections plus
an execution-environment section, replacing the *"not yet written"* stub.
§1–§5 and deviations D1–D5 are untouched.
