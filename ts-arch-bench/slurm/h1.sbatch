#!/usr/bin/env bash
# JOB NAME: supplied by slurm/submit.sh, which passes --job-name and thereby
# overrides the line below. Sharanga and CSIS require
#
#     name_bitsid_project[_anything]
#
# since 2026-11-10 and KILL any job named otherwise, because these are shared
# logins and the name is the only way to attribute GPU-hours to a person.
#
# `#SBATCH -J` is parsed before any shell runs, so it cannot interpolate
# $TSARCH_BITSID; and hardcoding an ID here would put one identity in seven
# files, all of them wrong for the next user of this shared login. Hence the
# placeholder below is INVALID ON PURPOSE - a direct `sbatch` on this file
# must fail visibly rather than bill its hours to nobody. Submit with:
#
#     slurm/submit.sh slurm/h1.sbatch [tag] [extra sbatch args...]
#
#SBATCH --job-name=INVALID-use-slurm-submit-sh
#SBATCH --partition=gpu_a100_8,gpu_h100_4,gpu_h200_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=01:00:00
#SBATCH --output=logs/h1_%A_%a.out
#SBATCH --error=logs/h1_%A_%a.out
#
# MEASURED, not guessed (hpc01, 2026-11-07). End-to-end per rung, including
# model load and scoring 2700 items:
#     0.5B  11 min      32B  25 min      72B  42 min
# so the earlier 8-10 hour limits were 10-20x too generous, and SLURM
# backfill can only slot a job into a gap at least as long as it ASKED for.
# Over-requesting walltime is not free caution; it is self-inflicted queueing.
#
# Memory: the 72B recorded MaxRSS 95.9 GiB against a 96 GiB request - 99.9%
# of its allocation, i.e. it nearly OOM-died. vLLM materialises weights
# through system RAM on the way to the GPU, so memory-mapped safetensors do
# NOT keep RSS low. GPU nodes have ~976 GiB; asking for 140G costs nothing.
# Per-user caps are mem=300G on gpu_h100_4 and gpu_h200_8, so two concurrent
# jobs must stay under that.
#
#
# H1 - history length. ONE ARRAY TASK PER L, because the cost per cell is
# very uneven: L=512 puts ~3.5k tokens of series in every bare-arm prompt,
# roughly an order of magnitude more prefill than L=32. One task per L means
# the cheap cells finish early instead of waiting behind the expensive one,
# and a timeout on L=512 costs only L=512.
#
#   mkdir -p logs
#   slurm/submit.sh slurm/h1.sbatch '' --array=0-4
#
# If H3 has already run, L=128 is already cached and task 2 will finish
# almost instantly - it replays rather than re-asks. Submitting 0-4 anyway is
# correct and free; `run_cell` skips any shard that already exists.
# SIZING: the backbone is Qwen2.5-32B-Instruct, ~69 GiB of bf16 weights.
# On 80 GiB cards --gres=gpu:1 fits but leaves only ~3 GiB of KV cache, so
# throughput is poor; prefer --gres=gpu:2. On 40 GiB cards gpu:1 does not
# fit at all. Run slurm/preflight.sh to find out which you have, and drop
# gpu_a100_8 from --partition above if that node turns out to be 40 GiB.
set -euo pipefail

source "/scratch/$USER/claude-ts-arch-bench/env.sh"
cd "${TSARCH}/ts-arch-bench"

# Second net for the naming policy. Cannot prevent a kill - the scheduler
# decides at submit time - but it makes the reason visible in the log and
# catches a job that somehow started under the placeholder.
source slurm/jobname.sh
tsarch_assert_jobname


LENGTHS=(32 64 128 256 512)
L="${LENGTHS[${SLURM_ARRAY_TASK_ID:-2}]}"
# Overridable so cycle 2 reuses this script unchanged, e.g.
#   --export=ALL,TSARCH_H1_CONFIG=configs/c2_h1.yaml
CONFIG="${TSARCH_H1_CONFIG:-configs/h1_history_length.yaml}"
BB="${TSARCH_H1_BACKBONE:-qwen2_5_32b}"

MODEL="$(python -c "
import sys; sys.path.insert(0, '.')
from src.config import load_config
print(load_config('$CONFIG').backbones['$BB'].model)")"

echo "=== H1 L=$L  backbone=$BB -> $MODEL ==="

# TP is detected from the allocation inside serve_vllm.sh
source slurm/serve_vllm.sh "$MODEL"
trap vllm_stop EXIT

python -m runners.h1_history_length \
  --config "$CONFIG" \
  --stage inference \
  --backbone "$BB" \
  --only-L "$L" \
  --cache-dir "$TSARCH_CACHE"

echo "=== H1 L=$L done ==="
# After the whole array finishes, on a CPU node:
#   python -m runners.h1_history_length --config $CONFIG --stage analyse
