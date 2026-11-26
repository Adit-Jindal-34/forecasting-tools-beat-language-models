#!/usr/bin/env bash
# Submit ALL of cycle 2 in one go: H3 (seven rungs), H2, H1, and the H3
# analysis chained behind them.
#
#   bash slurm/submit_c2.sh --test     # ask SLURM when each job WOULD start
#   bash slurm/submit_c2.sh            # submit everything
#
# FOUR UNITS OF WORK
# ------------------
# The queue wait dwarfs the work here, and every job waits once, so the
# ladder is bundled (slurm/c2_run.sbatch runs several steps per allocation):
#
#   small   0.5B, 1.5B, 3B, 7B           1 GPU    ~70 min
#   mid     32B, then H2, then 14B       1 GPU    ~65 min
#   big     72B                          2 GPUs   ~60 min   h200 only
#   h1      H1, 32B at five lengths      1 GPU    ~2 h
#
# H2 runs inside `mid` right after the 32B rung, while that model is still
# being served: it renders the same prompts, so it is a cache replay.
#
# WHERE, AND WHY EACH UNIT IS SUBMITTED TWICE (measured 2026-11-13)
# ----------------------------------------------------------------
# This account's associations are per partition, so SLURM refuses a
# multi-partition request ("not supported when a partition is set in the
# association"). To still start wherever a GPU frees up first, each 1-GPU
# unit is submitted as SIBLING jobs with the same name, one on gpu_h200_8
# and one on gpu_h100_4. The first sibling to start takes a lock and cancels
# the others (c2_run.sbatch); a sibling that starts anyway sees the lock and
# exits in seconds.
#
#   gpu_h200_8   best queue by far (17 pending, jobs turning over within
#                hours); 2 GPUs per user. The only place the 72B fits.
#   gpu_h100_4   second best (38 pending); 1 H100 per user.
#   gpu_a100_8   NOT used: 49 pending, SLURM's own estimate was two months.
#   gpu_v100_2   NOT used: no bf16 (the forecaster needs it) and vLLM on
#                sm_70 was never proven here.
#   gpu_rtx_pro_6000_*   NOT usable: "Invalid qos specification".
#
# MEMORY IS THE BINDING RESOURCE, NOT GPUS. On 2026-11-13 gpu_h200_8 had two
# idle GPUs and 65 GiB of free RAM: other jobs had reserved the node's
# memory. So each unit asks for what cycle 1 MEASURED (MaxRSS: 7B 20 GiB,
# 14B 35, 32B 71, 72B 96) plus about a quarter, not a round number. A small
# request is what lets a job backfill into a gap.
#
# WALLTIMES are cycle 1's measured runtimes scaled to cycle 2's call counts,
# with under 2x margin - short enough to backfill. A job killed at its limit
# loses nothing: re-run this script and finished work is skipped.
#
# SUBMIT LIMIT. The login may hold 6 jobs per GPU partition, and other
# people share this login. This script uses 4 on h200 and 3 on h100.
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/jobname.sh"
tsarch_jobname >/dev/null          # fail fast if the identity is not set
: "${TSARCH:?source /scratch/\$USER/claude-ts-arch-bench/env.sh first}"
mkdir -p logs "$TSARCH/locks"

TEST=0
[[ "${1:-}" == "--test" ]] && TEST=1

LADDER_JIDS=()      # every job whose end the H3 analysis must wait for

sub() {   # sub <tag> <steps joined by +> <gpus> <time> <mem> <partition>...
  local tag="$1" steps="$2" gres="$3" wall="$4" mem="$5"; shift 5
  local name; name="$(tsarch_jobname "c2-$tag")"
  local lock="$TSARCH/locks/c2_$tag.lock"
  if [[ -e "$lock/done" ]]; then
    printf "  %-6s already finished (%s). Delete it to run again.\n" \
      "$tag" "$lock/done"
    return
  fi
  local existing=""
  if (( ! TEST )); then
    existing="$(squeue -h -u "$USER" -n "$name" -o %i 2>/dev/null || true)"
  fi
  if [[ -n "$existing" ]]; then
    printf "  %-6s already queued or running as %s - not submitted again\n" \
      "$tag" "$name"
    # still make the analysis wait for it
    if [[ "$tag" != h1 ]]; then
      local j
      for j in $existing; do LADDER_JIDS+=("$j"); done
    fi
    return 0
  fi
  local part
  for part in "$@"; do
    local args=(--job-name="$name" --partition="$part" --gres=gpu:"$gres"
                --time="$wall" --mem="$mem" --cpus-per-task=4
                --output="logs/c2_${tag}_%j.out" --error="logs/c2_${tag}_%j.out"
                --export=ALL,TSARCH_C2_STEPS="$steps",TSARCH_C2_TAG="$tag",TSARCH_C2_CACHE="$TSARCH/cache/c2_$tag")
    if (( TEST )); then
      printf "  %-6s %-11s " "$tag" "$part"
      sbatch --test-only "${args[@]}" slurm/c2_run.sbatch 2>&1 | tail -1
      continue
    fi
    # A refused submission (a QOS limit, say) must not abort the rest: the
    # other partition's sibling may well be accepted.
    local jid
    if ! jid="$(sbatch --parsable "${args[@]}" slurm/c2_run.sbatch 2>&1)"; then
      printf "  %-6s NOT submitted on %s: %s\n" "$tag" "$part" \
        "$(printf '%s' "$jid" | tail -1)"
      continue
    fi
    jid="${jid%%;*}"
    if [[ "$tag" != h1 ]]; then LADDER_JIDS+=("$jid"); fi
    printf "  %-6s job %-8s %-11s gpu:%s %-9s %-5s %s\n" \
      "$tag" "$jid" "$part" "$gres" "$wall" "$mem" "${steps//+/ }"
  done
  return 0
}

echo "=== cycle 2: $( ((TEST)) && echo 'estimated start times (nothing submitted)' || echo submitting) ==="

sub small "h3:qwen2_5_0_5b+h3:qwen2_5_1_5b+h3:qwen2_5_3b+h3:qwen2_5_7b" \
    1 02:00:00 26G  gpu_h200_8 gpu_h100_4
sub mid   "h3:qwen2_5_32b+h2+h3:qwen2_5_14b" \
    1 02:00:00 90G  gpu_h200_8 gpu_h100_4
sub big   "h3:qwen2_5_72b" \
    2 02:00:00 128G gpu_h200_8
sub h1    "h1" \
    1 04:00:00 90G  gpu_h200_8 gpu_h100_4

(( TEST )) && exit 0

# H3's analysis needs every rung, so it is chained behind every ladder job,
# siblings included (a cancelled sibling counts as ended). `afterany`, not
# `afterok`: if a rung failed, the analysis should run and REFUSE the partial
# grid loudly rather than sit pending forever.
if (( ${#LADDER_JIDS[@]} )); then
  DEP="afterany$(printf ':%s' "${LADDER_JIDS[@]}")"
  AN="$(tsarch_jobname c2-h3-analyse)"
  AQ="$(squeue -h -u "$USER" -n "$AN" -o %i 2>/dev/null || true)"
  if [[ -n "$AQ" ]]; then
    echo "  analyse already queued as $AN - not submitted again"
  else
    AJ="$(sbatch --parsable --job-name="$AN" --dependency="$DEP" \
            --export=ALL,TSARCH_EXP=c2_h3 slurm/analyse.sbatch)"
    printf "  %-6s job %-8s CPU, runs after the ladder (%s)\n" \
      analyse "${AJ%%;*}" "$DEP"
  fi
fi

cat <<'NOTE'

Submitted. H2 and H1 analyse themselves at the end of their own jobs.
  watch   : squeue -u $USER -o "%.10i %.40j %.11P %.8T %.10M %.20S %R"
  logs    : logs/c2_<tag>_<jobid>.out
  results : results/c2_h3, results/c2_h2, results/c2_h1
Re-running this script is safe: units already queued are left alone, units
already finished are skipped, and inside a unit finished shards are skipped
and finished model calls replay from the cache.
NOTE
