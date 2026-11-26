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
#     slurm/submit.sh slurm/judge_slots.sbatch [tag] [extra sbatch args...]
#
#SBATCH --job-name=INVALID-use-slurm-submit-sh
#SBATCH --partition=gpu_h200_8
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=logs/judge_%j.out
#SBATCH --error=logs/judge_%j.out
#
# Validate the slot EXTRACTOR against an independent local reader.
#
#   slurm/submit.sh slurm/judge_slots.sbatch
#   slurm/submit.sh slurm/judge_slots.sbatch h3 --export=ALL,TSARCH_EXP=h3
#
# WHY THIS IS THE PRIORITY REMAINING TEST
# ---------------------------------------
# Two of H2's three confirmatory cells (`trend_description`,
# `event_explanation`) are scored by a REGEX slot extractor that
# PREREGISTRATION.md §4.1 itself calls "provisionally measured". D7 fixes the
# consequence in advance: alpha below 0.667 on a shared slot removes that
# cell from the confirmatory family and cuts m from 3 to 1. So the extractor's
# validity is load-bearing for the headline, and nothing has tested it.
#
# WHY LOCAL RATHER THAN GEMINI
# ----------------------------
# Measured 2026-11-07: the Gemini free tier served ~2 calls before dropping
# into silent rate-limited backoff, and `gemini-2.5-flash` was retired for
# new users mid-session. A model name that can disappear cannot sit under a
# benchmark metric. The local 72B is free, unlimited, and pinnable to a HF
# commit.
#
# THE INDEPENDENCE CAVEAT, STATED PLAINLY
# ---------------------------------------
# H2's answers were generated ENTIRELY by `qwen2_5_32b`, so the 72B reading
# them is a different model judging another model's output - NOT self-
# judging, even though the 72B is the top rung of the H3 ladder. But it
# shares a training family with the 32B, so this is a WEAKER independence
# claim than a cross-family or human judge would give. Report it that way.
# It does not discharge D7's human annotation.
#
# SIZING
# ------
# Qwen2.5-72B-Instruct is ~153 GiB of bf16 weights. On gpu_h200_8 (141 GiB
# cards) gpu:2 fits with room for KV cache. On 80 GiB cards gpu:2 gives only
# 160 GiB and is too tight once KV cache is counted - use gpu:4 there.
# serve_vllm.sh computes the fit from the VRAM actually present and aborts
# with the right --gres rather than OOMing after the load.
#
# --mem=200G: the 72B recorded MaxRSS 95.9 GiB against a 96 GiB request
# during H3, i.e. it nearly OOM-died. vLLM materialises weights through
# system RAM on the way to the GPU. The per-user cap on gpu_h200_8 is 300G.
set -euo pipefail

source "/scratch/$USER/claude-ts-arch-bench/env.sh"
cd "${TSARCH}/ts-arch-bench"

# Second net for the naming policy. Cannot prevent a kill - the scheduler
# decides at submit time - but it makes the reason visible in the log and
# catches a job that somehow started under the placeholder.
source slurm/jobname.sh
tsarch_assert_jobname


# TSARCH_EXP takes a SPACE-SEPARATED list, because the 72B load dominates:
# the 2026-11-08 H2 run spent 630s of its 742s total loading the model and
# only ~112s judging. Two separate jobs would pay that load twice and queue
# twice. One job, one load, several experiments.
#
#   slurm/submit.sh slurm/judge_slots.sbatch h1h3 \
#          --export=ALL,TSARCH_EXP="h1 h3"
#
EXPS="${TSARCH_EXP:-h2}"
JUDGE="${TSARCH_JUDGE:-Qwen/Qwen2.5-72B-Instruct}"
NPS="${TSARCH_NPS:-50}"

# Validate every name BEFORE loading 153 GiB of weights: a typo discovered
# after an 11-minute load wastes the whole allocation.
for EXP in $EXPS; do
  case "$EXP" in
    h1|h2|h3) ;;
    *) echo "unknown TSARCH_EXP entry '$EXP' (want h1, h2 or h3)" >&2
       exit 2 ;;
  esac
done

# TP is detected from the allocation inside serve_vllm.sh
# a port of this job's own: loopback is shared by every job on
# the node (see slurm/pick_port.sh)
source slurm/pick_port.sh
source slurm/serve_vllm.sh "$JUDGE"
trap vllm_stop EXIT

for EXP in $EXPS; do
  # BY is the sweep axis of that experiment. Pooling across it would average
  # a 0.5B model's prose in with a 72B's (H3), or a 32-point answer with a
  # 512-point one (H1), and hide a rung whose agreement is far below
  # threshold inside a comfortable pooled mean. H2 is a single cell, so it
  # has no axis and stays empty - which also keeps its prompts and therefore
  # its judge cache identical to the 2026-11-08 run.
  case "$EXP" in
    h1) RESULTS=results/h1_history_length;     BY=L ;;
    h2) RESULTS=results/h2_query_type;         BY= ;;
    h3) RESULTS=results/h3_backbone_strength;  BY=backbone ;;
  esac
  # single dash, so TSARCH_BY="" can deliberately force pooling
  BY="${TSARCH_BY-$BY}"

  echo
  echo "=================================================================="
  echo "=== judging $EXP slot extraction with $JUDGE   (--by '$BY') ==="
  echo "=================================================================="

  # --out keeps judge artifacts OUT of the archived snapshot: a validation
  # run has no business writing into a frozen result directory.
  python -u scripts/judge_slots.py \
    --results "$RESULTS" \
    --out "${RESULTS}/judge_slots" \
    --provider openai_compatible \
    --base-url http://127.0.0.1:8000/v1 \
    --model "$JUDGE" \
    --by "$BY" \
    --n-per-stratum "$NPS" \
    --delay 0
done

echo
echo "=== judge done for: $EXPS ==="
