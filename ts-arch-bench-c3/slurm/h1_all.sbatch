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
#     slurm/submit.sh slurm/h1_all.sbatch [tag] [extra sbatch args...]
#
#SBATCH --job-name=INVALID-use-slurm-submit-sh
#SBATCH --partition=gpu_h200_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=140G
#SBATCH --time=10:00:00
#SBATCH --output=logs/h1_all_%j.out
#SBATCH --error=logs/h1_all_%j.out
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
# H1 - history length, ALL five L values in ONE job.
#
#   slurm/submit.sh slurm/h1_all.sbatch
#
# WHY THIS EXISTS ALONGSIDE h1.sbatch
# -----------------------------------
# `h1.sbatch` runs ONE history length per array task, picked from
# SLURM_ARRAY_TASK_ID. That is the right shape when the queue prefers many
# short jobs, but it loads the backbone five times - and with a 32B that is
# roughly five lots of model-load latency for no benefit. Submitted with no
# --array it is worse than useless: the index defaults to 2, i.e. L=128,
# which H3's 32B rung already cached, so the job would do nothing at all.
#
# `runners/h1_history_length.py` loops every configured length when
# `--only-L` is omitted, so one job can do all five with a single load. That
# is what this file does.
#
# Keep both: use h1.sbatch --array=0-4 if the queue is giving short jobs a
# much better deal than long ones, and this when it is not.
#
# COST SHAPE
# ----------
# L=128 is already in the cache from H3's 32B rung, so it replays for free
# and only four lengths cost GPU time. What scales is the bare arm's prompt:
# ~900 tokens at L=128, ~3.5k at L=512, so the long cells dominate. Measured
# reference: the 32B at L=128 took 25 minutes end to end.
#
# --cpus-per-task=4: gpu_h200_8 enforces at most 4 CPUs per GPU and rejects
# the submission outright otherwise ("You can only request up to 4 CPUs per
# H200 GPU"). On gpu_h100_4 or gpu_a100_8 eight is fine.
set -euo pipefail

source "/scratch/$USER/claude-ts-arch-bench/env.sh"
cd "${TSARCH}/ts-arch-bench"

# Second net for the naming policy. Cannot prevent a kill - the scheduler
# decides at submit time - but it makes the reason visible in the log and
# catches a job that somehow started under the placeholder.
source slurm/jobname.sh
tsarch_assert_jobname


# Overridable so cycle 2 reuses this script unchanged, e.g.
#   --export=ALL,TSARCH_H1_CONFIG=configs/c2_h1.yaml
CONFIG="${TSARCH_H1_CONFIG:-configs/h1_history_length.yaml}"
BB="${TSARCH_H1_BACKBONE:-qwen2_5_32b}"

MODEL="$(python -c "
import sys; sys.path.insert(0, '.')
from src.config import load_config
print(load_config('$CONFIG').backbones['$BB'].model)")"

echo "=== H1 all lengths  backbone=$BB -> $MODEL ==="
python -c "
import sys; sys.path.insert(0, '.')
from src.config import load_config
print('lengths:', load_config('$CONFIG').history_lengths)"

# TP is detected from the allocation inside serve_vllm.sh
# a port of this job's own: loopback is shared by every job on
# the node (see slurm/pick_port.sh)
source slurm/pick_port.sh
source slurm/serve_vllm.sh "$MODEL"
trap vllm_stop EXIT

# no --only-L: the runner loops every configured length, one model load
python -m runners.h1_history_length \
  --config "$CONFIG" \
  --stage inference \
  --backbone "$BB" \
  --cache-dir "$TSARCH_CACHE"

echo "=== H1 inference done for all lengths ==="
# Analysis is NOT run here: it is CPU-only and must not hold a GPU. Run it
# afterwards on a compute node:
#   python -m runners.h1_history_length --config $CONFIG --stage analyse
