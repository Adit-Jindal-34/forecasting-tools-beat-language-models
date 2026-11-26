#!/usr/bin/env bash
# JOB NAME: supplied by slurm/submit.sh, which passes --job-name and thereby
# overrides the line below. Sharanga and CSIS require
#
#     name_bitsid_project[_anything]
#
# since 2026-11-10 and KILL any job named otherwise, because these are shared
# logins and the name is the only way to attribute GPU-hours to a person.
#
# The placeholder below is INVALID ON PURPOSE - a direct `sbatch` on this
# file must fail visibly rather than bill its hours to nobody. Submit with:
#
#     slurm/submit.sh slurm/c3_h2_strata.sbatch [tag] [extra sbatch args...]
#
#SBATCH --job-name=INVALID-use-slurm-submit-sh
#SBATCH --partition=gpu_h200_8,gpu_h100_4
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=140G
#SBATCH --time=01:30:00
#SBATCH --output=logs/c3h2_%j.out
#SBATCH --error=logs/c3h2_%j.out
#
# CYCLE 3 - the REFRAMED H2 (series character). PREREGISTRATION_C3.md C3-6.
#
# WHY THIS NEEDS ITS OWN SCRIPT AND THE OTHER THREE DO NOT.
# `slurm/h1_all.sbatch`, `slurm/h2.sbatch` and `slurm/h3.sbatch` already read
# their config from an env var, so cycle 3 reuses them unchanged:
#
#   slurm/submit.sh slurm/h1_all.sbatch c3h1 \
#       --export=ALL,TSARCH_H1_CONFIG=configs/c3_h1_clean.yaml
#   slurm/submit.sh slurm/h3.sbatch c3h3 --array=0-6 \
#       --export=ALL,TSARCH_H3_CONFIG=configs/c3_h3_clean.yaml
#
# This one cannot, because the RUNNER is different: the stratum contrast is
# two-sample (a weak-seasonality series and a strong one are different
# series, so there is no per-item Delta), which `confirmatory_did` refuses
# outright. `runners/h2_series_character.py` runs the unpaired form.
#
# SIZING. One backbone (32B, ~69 GiB bf16), one history length, 240 items,
# four arms -> ~720 billable calls. Smallest unit in cycle 3 by a wide
# margin. --gres=gpu:2 because on 80 GiB cards a single GPU leaves only
# ~3 GiB of KV cache and throughput collapses; run slurm/preflight.sh if you
# are unsure which cards the partition has.
#
# CACHE. This run uses `pool_h2_strata_n240`, a DIFFERENT pool from the H1
# and H3 configs, so none of its prompts replay from their cache. Point it
# at the shared $TSARCH_CACHE anyway - it costs nothing and keeps one
# archive per cycle.
set -euo pipefail

source "/scratch/$USER/claude-ts-arch-bench/env.sh"
cd "${TSARCH}/ts-arch-bench"

# Second net for the naming policy. Cannot prevent a kill - the scheduler
# decides at submit time - but it makes the reason visible in the log and
# catches a job that somehow started under the placeholder.
source slurm/jobname.sh
tsarch_assert_jobname

CONFIG="${TSARCH_C3H2_CONFIG:-configs/c3_h2_strata.yaml}"
BB="${TSARCH_C3H2_BACKBONE:-qwen2_5_32b}"

# The config is the single source of truth for the HF id. Reading it here
# rather than duplicating the string means the served name and the name in
# the cache fingerprint cannot disagree.
MODEL="$(python -c "
import sys; sys.path.insert(0, '.')
from src.config import load_config
print(load_config('$CONFIG').backbones['$BB'].model)")"

echo "=== C3 H2-reframed (series character)  backbone=$BB -> $MODEL ==="
python -c "
import sys; sys.path.insert(0, '.')
from src.config import load_config
c = load_config('$CONFIG')
print('pool    :', c.data['pool_dir'])
print('sha256  :', c.data['pool_sha256'][:16], '(checked at load)')
print('strata  :', c.analysis.get('stratum_contrast'))
print('horizon :', c.data['horizon'], ' L:', c.history_lengths)"

# TP is detected from the allocation inside serve_vllm.sh
# a port of this job's own: loopback is shared by every job on
# the node (see slurm/pick_port.sh)
source slurm/pick_port.sh
source slurm/serve_vllm.sh "$MODEL"
trap vllm_stop EXIT

# Inference only. Analysis is CPU-only and must not hold a GPU.
python -m runners.h2_series_character \
  --config "$CONFIG" \
  --stage inference \
  --backbone "$BB" \
  --cache-dir "$TSARCH_CACHE"

echo "=== C3 H2-reframed inference done ==="
# Run the analysis afterwards on a CPU node:
#
#   python -m runners.h2_series_character --config $CONFIG --stage analyse
#
# Read its verdict block carefully: if the POOLED stratum contrast moves and
# the WITHIN-weather one does not, the pre-registered rule (C3-6) says the
# result is an ARCHIVE effect, not a series-character effect.
