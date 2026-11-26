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
#     slurm/submit.sh slurm/h2.sbatch [tag] [extra sbatch args...]
#
#SBATCH --job-name=INVALID-use-slurm-submit-sh
#SBATCH --partition=gpu_a100_8,gpu_h100_4,gpu_h200_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=01:00:00
#SBATCH --output=logs/h2_%j.out
#SBATCH --error=logs/h2_%j.out
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
# H2 - query type. A single cell: one backbone, L=128.
#
#   mkdir -p logs && slurm/submit.sh slurm/h2.sbatch
#
# IF YOU RAN H3 FIRST, THIS NEEDS NO GPU AT ALL.
# H3's `qwen2_5_32b` rung at L=128 renders byte-identical prompts, so with the
# shared --cache-dir every call is a cache hit. In that case skip this script
# and just run the two stages on a CPU node:
#
#   python -m runners.h2_query_type --config configs/h2_query_type.yaml \
#       --stage both --cache-dir "$TSARCH_CACHE"
#
# The GPU is requested here only to cover the case where H2 runs first.
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


# Overridable so the second-seed robustness replication can reuse this
# script unchanged:
#   slurm/submit.sh slurm/h2.sbatch seed2 \
#          --export=ALL,TSARCH_H2_CONFIG=configs/h2_query_type_seed2.yaml \
#          slurm/h2.sbatch
# That run generates a FRESH pool, so nothing is in the call cache and every
# call is real GPU work (~2,100 calls, ~25 min) - unlike the default config,
# which replays H3's 32B rung for free.
CONFIG="${TSARCH_H2_CONFIG:-configs/h2_query_type.yaml}"
BB="${TSARCH_H2_BACKBONE:-qwen2_5_32b}"

MODEL="$(python -c "
import sys; sys.path.insert(0, '.')
from src.config import load_config
print(load_config('$CONFIG').backbones['$BB'].model)")"

echo "=== H2 backbone=$BB -> $MODEL ==="

# TP is detected from the allocation inside serve_vllm.sh
# a port of this job's own: loopback is shared by every job on
# the node (see slurm/pick_port.sh)
source slurm/pick_port.sh
source slurm/serve_vllm.sh "$MODEL"
trap vllm_stop EXIT

# both stages: this design is a single cell, so analysis needs nothing else
python -m runners.h2_query_type \
  --config "$CONFIG" \
  --stage both \
  --backbone "$BB" \
  --cache-dir "$TSARCH_CACHE"

echo "=== H2 done ==="
