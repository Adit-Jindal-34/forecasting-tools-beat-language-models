#!/usr/bin/env bash
# Submit the whole experiment, placed to finish as fast as the QOS allows.
#
#   bash slurm/submit_all.sh            # H3 ladder (+ H1 once 32B is done)
#   bash slurm/submit_all.sh --canary   # just the 0.5B, to prove the stack
#
# WHY NOT ONE sbatch --array
# --------------------------
# An array shares one partition and one --gres. Measured QOS limits for this
# account (sacctmgr, 2026-11-06) make that strictly worse than per-rung jobs:
#
#   partition      my GPU cap   my job cap   max wall   note
#   gpu_v100_2     4            UNLIMITED    3 d        only one with no job cap
#   gpu_a100_8     2            2            5 d
#   gpu_h100_4     1 (!)        2            3 d        one H100 per user
#   gpu_h200_8     2            2            1 d        only place a 72B fits
#   gpu_v100_1     -            -            -          no access (schrodinger)
#
# An array on any single partition would therefore run 2 jobs at a time (or
# 4 on v100_2, GPU-bound). Spreading across four partitions holds ~6-8
# concurrently, because the caps are PER PARTITION.
#
# PLACEMENT, AND WHY EACH
# -----------------------
# Weights are ~2 bytes x params. Our runner issues ONE request at a time, so
# vLLM needs KV cache for a single sequence (~2-3 GiB at 8192 ctx), not for a
# full batch. That is why a 32B fits one 80 GiB card comfortably.
#
#   rung    weights  ->  partition        gres  why
#   0.5-7B  1-17 GiB     gpu_v100_2       1     fits 32 GiB; no job cap here
#   14B     31 GiB       gpu_a100_8       1     too big for one V100
#   32B     69 GiB       gpu_h100_4       1     one H100 is the cap; 80 GiB is enough
#   72B     153 GiB      gpu_h200_8       2     ONLY option: 2x141 GiB.
#                                               4xA100 and 2xH100 both exceed the caps.
#
# The two Llama rungs are gated and skipped unless ~/.hf_token exists; they
# are the cross-family robustness check and sit outside the §3 family.
#
# V100 CAVEAT, WHICH IS WHY --canary EXISTS
# -----------------------------------------
# V100 is sm_70: no bfloat16, and recent vLLM builds may refuse it outright
# or need the xformers attention backend. serve_vllm.sh substitutes fp16
# automatically, but whether this vLLM build runs at all on sm_70 is a fact
# about the installed version, not something to assume. --canary submits the
# 0.5B alone so that costs five minutes instead of four dead jobs.
set -euo pipefail

cd "$(dirname "$0")/.."

# Sharanga/CSIS naming policy: every name_bitsid_project job name
# below is built here, and tsarch_jobname refuses (non-zero, with a
# message) if the identity variables are unset - so this script
# stops before submitting ANY rung rather than part of the ladder.
source "$(dirname "$0")/jobname.sh"
tsarch_jobname >/dev/null   # fail fast, before the first sbatch

ROOT="$PWD"
# Overridable so cycle 2 reuses this placement unchanged:
#   TSARCH_H3_CONFIG=configs/c2_h3.yaml bash slurm/submit_all.sh [--canary]
CFG="${TSARCH_H3_CONFIG:-configs/h3_backbone_strength.yaml}"
[[ -f "$CFG" ]] || { echo "no such config: $CFG" >&2; exit 2; }
# job-name tag, so cycle-1 and cycle-2 rungs are distinguishable in squeue
TAG="h3"; [[ "$CFG" == *c2_* ]] && TAG="c2h3"
echo "config: $CFG"
mkdir -p logs

sub() {   # sub <rung> <partition> <gpus> <walltime> [mem]
  local bb="$1" part="$2" gres="$3" wall="$4" mem="${5:-96G}"
  local jid
  jid=$(sbatch --parsable \
    --job-name="$(tsarch_jobname "$TAG-$bb")" \
    --partition="$part" \
    --gres=gpu:"$gres" \
    --cpus-per-task=8 \
    --mem="$mem" \
    --time="$wall" \
    --output="$ROOT/logs/${TAG}_${bb}_%j.out" \
    --export=ALL,TSARCH="${TSARCH:?source env.sh first}" \
    --wrap "set -euo pipefail
      cd '$ROOT'
      source '$TSARCH/env.sh'
      MODEL=\$(python -c \"
import sys; sys.path.insert(0,'.')
from src.config import load_config
print(load_config('$CFG').backbones['$bb'].model)\")
      echo \"rung $bb -> \$MODEL on $part (gpu:$gres)\"
      source slurm/serve_vllm.sh \"\$MODEL\"
      trap vllm_stop EXIT
      python -m runners.h3_backbone_strength --config '$CFG' \
             --stage inference --backbone '$bb' --cache-dir '$TSARCH/cache'")
  printf "  %-14s %-14s gpu:%s  %-9s  job %s\n" "$bb" "$part" "$gres" "$wall" "$jid"
}

echo "=== submitting H3 ==="
printf "  %-14s %-14s %-7s %-9s  %s\n" RUNG PARTITION GRES WALL JOBID

if [[ "${1:-}" == "--canary" ]]; then
  sub qwen2_5_0_5b gpu_v100_2 1 01:00:00
  echo
  echo "Canary only. If it produces a shard, run without --canary for the rest."
  echo "If vLLM refuses sm_70, move the small rungs to gpu_a100_8 / gpu_h100_4."
  exit 0
fi

# small rungs: the only partition without a job-count cap
sub qwen2_5_0_5b gpu_v100_2 1 01:00:00
sub qwen2_5_1_5b gpu_v100_2 1 01:00:00
sub qwen2_5_3b   gpu_v100_2 1 01:30:00
sub qwen2_5_7b   gpu_v100_2 1 02:00:00
# too big for a single 32 GiB V100
sub qwen2_5_14b  gpu_a100_8 1 02:30:00
# one H100 is this account's cap, and 80 GiB is ample for one sequence
sub qwen2_5_32b  gpu_h100_4 1 04:00:00
# the only placement that exists for a 72B under a 2-GPU cap
# 140G, not 96G: the 72B recorded MaxRSS 95.9 GiB against a 96 GiB request
# in cycle 1 - 99.9% of its allocation, one allocation away from an OOM kill.
sub qwen2_5_72b  gpu_h200_8 2 20:00:00 140G

if ! grep -q "llama3_1_8b" "$CFG"; then
  echo "  (llama rungs skipped: not declared in $CFG)"
elif [[ -s "$HOME/.hf_token" ]]; then
  sub llama3_1_8b  gpu_v100_2 1 02:00:00
  sub llama3_1_70b gpu_h200_8 2 20:00:00
else
  echo "  (llama rungs skipped: no ~/.hf_token; they are outside the family)"
fi

cat <<'NOTE'

Submitted. Notes:
  * Rungs land independently. H3's analyse stage needs all of them, but
    shards accumulate and are safe to inspect as they arrive.
  * Re-running this is safe: run_cell skips any (backbone, L) whose shard
    already exists, so a timed-out rung resumes rather than restarting.
  * H1 is deliberately NOT submitted here. It reuses the 32B, and one job
    looping all five history lengths loads that model once instead of five
    times. Submit it after the 32B rung finishes:
        slurm/submit.sh slurm/h1.sbatch '' \
            --partition=gpu_h100_4 --gres=gpu:1 --time=08:00:00
  * H2 needs no GPU once the 32B rung is done - it is a pure cache replay.
NOTE
echo "watch:  squeue -u \$USER"
