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
#     slurm/submit.sh slurm/h3.sbatch [tag] [extra sbatch args...]
#
#SBATCH --job-name=INVALID-use-slurm-submit-sh
#SBATCH --partition=gpu_a100_8,gpu_h100_4,gpu_h200_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=02:00:00
#SBATCH --output=logs/h3_%A_%a.out
#SBATCH --error=logs/h3_%A_%a.out
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
# H3 - backbone strength. ONE ARRAY TASK PER RUNG, so a 72B never shares a
# process with a 0.5B and a failure on one rung costs only that rung.
#
#   mkdir -p logs
#   bash slurm/preflight.sh                          # FIRST: settles sizing
#   bash slurm/stage_models.sh --llama               # pre-download weights
#
#   slurm/submit.sh slurm/h3.sbatch '' --array=0-4,7             # 0.5B..14B
#   slurm/submit.sh slurm/h3.sbatch '' --gres=gpu:2 --array=5    # 32B
#   slurm/submit.sh slurm/h3.sbatch '' --gres=gpu:4 --array=6,8  # 72B / 70B
#
# MEASURED ON hpc01, 2026-11-06 (slurm/preflight.sh):
#   gpu_h200_8  gpunode7    8x H200 NVL     141 GiB each   <- best target
#   gpu_a100_8  gpunode4    8x A100-SXM4     80 GiB each
#   gpu_h100_4  gpunode5/6  4x H100 HBM3     80 GiB each
#   gpu_v100_2  gpunode2/3  2x V100-PCIe     32 GiB each   (<=7B only)
#
# On gpu_h200_8 the 32B needs only gpu:1 (~58 GiB of KV cache, against
# ~3 GiB on a single 80 GiB card) and the 72B/70B need gpu:2 rather than
# gpu:4. Prefer H200 for the large rungs: faster, and one fewer GPU per job,
# which schedules better against a busy queue.
# A 72B is ~153 GiB of bf16 weights, so gpu:2 on 80 GiB cards gives 144 GiB
# total and OOMs on the weights alone - it is NOT enough, despite an earlier
# version of this comment saying so. serve_vllm.sh now computes the fit from
# the VRAM actually present and aborts with the right --gres rather than
# OOMing after the download. Run preflight.sh to learn which cards you have;
# %G in sinfo truncates and hides both count and memory.
#
# sbatch CLI flags override the #SBATCH lines above. Tensor-parallel size is
# resolved inside serve_vllm.sh from the allocation (four env vars, then
# nvidia-smi), so it cannot drift out of sync - and it ABORTS rather than
# defaulting to 1 if none of them is set.
#
# Do this experiment FIRST. The call cache keys on prompts only, so once the
# `qwen2_5_32b` rung is done, H2 is a pure cache replay (zero model calls) and
# H1 needs only its four extra history lengths.
set -euo pipefail

source "/scratch/$USER/claude-ts-arch-bench/env.sh"
cd "${TSARCH}/ts-arch-bench"

# Second net for the naming policy. Cannot prevent a kill - the scheduler
# decides at submit time - but it makes the reason visible in the log and
# catches a job that somehow started under the placeholder.
source slurm/jobname.sh
tsarch_assert_jobname


RUNGS=(qwen2_5_0_5b qwen2_5_1_5b qwen2_5_3b qwen2_5_7b qwen2_5_14b \
       qwen2_5_32b qwen2_5_72b llama3_1_8b llama3_1_70b)
BB="${RUNGS[${SLURM_ARRAY_TASK_ID:-0}]}"
# Overridable so cycle 2 reuses this script unchanged, e.g.
#   --export=ALL,TSARCH_H3_CONFIG=configs/c2_h3.yaml
CONFIG="${TSARCH_H3_CONFIG:-configs/h3_backbone_strength.yaml}"

# The config is the single source of truth for the HF id. Reading it here
# rather than duplicating the string means the served name and the name in
# the cache fingerprint cannot disagree.
MODEL="$(python -c "
import sys; sys.path.insert(0, '.')
from src.config import load_config
print(load_config('$CONFIG').backbones['$BB'].model)")"

echo "=== H3 rung ${SLURM_ARRAY_TASK_ID:-0}: $BB -> $MODEL ==="

# TP is detected from the allocation inside serve_vllm.sh
source slurm/serve_vllm.sh "$MODEL"
trap vllm_stop EXIT

python -m runners.h3_backbone_strength \
  --config "$CONFIG" \
  --stage inference \
  --backbone "$BB" \
  --cache-dir "$TSARCH_CACHE"

echo "=== rung $BB done ==="
# Analysis is deliberately NOT run here: it needs every rung's shard. Run it
# once, on a CPU node, after the array completes:
#
#   python -m runners.h3_backbone_strength --config $CONFIG --stage analyse
