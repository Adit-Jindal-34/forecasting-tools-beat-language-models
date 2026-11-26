#!/usr/bin/env bash
# Submit ALL of cycle 3 (PREREGISTRATION_C3.md) as three bundled GPU jobs plus
# the H3 analysis, sized to the shared login's limits.
#
#   source /scratch/$USER/claude-ts-arch-bench/env.sh
#   cd $TSARCH/ts-arch-bench-c3          # THIS folder, not ts-arch-bench
#   bash slurm/submit_c3_bundled.sh --test   # start estimates, nothing submitted
#   bash slurm/submit_c3_bundled.sh          # submit
#
# WHY NOT slurm/submit_c3.sh. That script queues ~11 GPU jobs, asks for two
# partitions in one job, and its job scripts `cd $TSARCH/ts-arch-bench`. On
# this login (measured 2026-11-14, `sacctmgr show qos`):
#   qos_gpu_h200: 2 running jobs, 6 submitted, 2 GPUs, 300G per user
#   qos_gpu_h100: 2 running jobs, 6 submitted per user
# and the user is SHARED with other people's jobs; a multi-partition request
# is refused outright. Most of those submissions would be rejected, and the
# ones accepted would run in the cycle-2 folder. Same work, bundled the way
# cycle 2 was (slurm/submit_c2.sh):
#
#   small  H3 rungs 0.5B 1.5B 3B 7B 14B          1 GPU   h200 + h100 sibling
#   mid    H3 32B, H1 clean, H2 strata, H8,      1 GPU   h200 + h100 sibling
#          H1 contaminated (analyses itself)             (slower on an H100)
#   big    H3 72B                                2 GPUs  h200 only
#   h3-an  H3 analysis on CPU, after all three
#
# Each unit has a lock in $TSARCH/locks/c3_<tag>.lock and a cache in
# $TSARCH/cache/c3_<tag>. Re-running this script is safe: finished units are
# skipped, queued ones are not submitted twice, failed ones resume.
set -euo pipefail

cd "$(dirname "$0")/.."
CODE="$(pwd)"
source slurm/jobname.sh
tsarch_jobname >/dev/null
: "${TSARCH:?source /scratch/\$USER/claude-ts-arch-bench/env.sh first}"
if [[ "$CODE" == "$TSARCH/ts-arch-bench" ]]; then
  echo "refusing: cycle 3 must run from its own folder, not $CODE" >&2
  exit 1
fi
mkdir -p logs "$TSARCH/locks"

TEST=0
[[ "${1:-}" == "--test" ]] && TEST=1

python data-real/check_pools.py >/dev/null || {
  python data-real/check_pools.py || true
  echo "pool check failed - nothing submitted" >&2; exit 1; }

ALL_JIDS=()

sub() {   # sub <tag> <steps joined by +> <gpus> <time> <mem> <partition>...
  local tag="$1" steps="$2" gres="$3" wall="$4" mem="$5"; shift 5
  local name; name="$(tsarch_jobname "c3-$tag")"
  local lock="$TSARCH/locks/c3_$tag.lock"
  if [[ -e "$lock/done" ]]; then
    printf "  %-6s already finished (%s). Delete it to run again.\n" "$tag" "$lock/done"
    return 0
  fi
  local existing=""
  (( TEST )) || existing="$(squeue -h -u "$USER" -n "$name" -o %i 2>/dev/null || true)"
  if [[ -n "$existing" ]]; then
    printf "  %-6s already queued or running as %s - not submitted again\n" "$tag" "$name"
    local j; for j in $existing; do ALL_JIDS+=("$j"); done
    return 0
  fi
  local part
  for part in "$@"; do
    local args=(--job-name="$name" --partition="$part" --gres=gpu:"$gres"
                --time="$wall" --mem="$mem" --cpus-per-task=4
                --output="logs/c3_${tag}_%j.out" --error="logs/c3_${tag}_%j.out"
                --export=ALL,TSARCH_CODE="$CODE",TSARCH_C3_STEPS="$steps",TSARCH_C3_TAG="$tag",TSARCH_C3_CACHE="$TSARCH/cache/c3_$tag")
    if (( TEST )); then
      printf "  %-6s %-11s " "$tag" "$part"
      sbatch --test-only "${args[@]}" slurm/c3_run.sbatch 2>&1 | tail -1
      continue
    fi
    local jid
    if ! jid="$(sbatch --parsable "${args[@]}" slurm/c3_run.sbatch 2>&1)"; then
      printf "  %-6s NOT submitted on %s: %s\n" "$tag" "$part" "$(printf '%s' "$jid" | tail -1)"
      continue
    fi
    jid="${jid%%;*}"
    ALL_JIDS+=("$jid")
    printf "  %-6s job %-8s %-11s gpu:%s %-9s %-5s %s\n" \
      "$tag" "$jid" "$part" "$gres" "$wall" "$mem" "${steps//+/ }"
  done
  return 0
}

echo "=== cycle 3 (bundled): $( ((TEST)) && echo 'estimated start times (nothing submitted)' || echo submitting) ==="
echo "    code: $CODE"

sub small "h3:qwen2_5_0_5b+h3:qwen2_5_1_5b+h3:qwen2_5_3b+h3:qwen2_5_7b+h3:qwen2_5_14b" \
    1 03:00:00 48G  gpu_h200_8 gpu_h100_4
# mid also gets an H100 sibling: on one 80 GiB card the 32B has little KV
# cache, so fewer requests run at once (slower, same answers), but the H100
# queue was estimated 4 days shorter than the H200's on 2026-11-14.
sub mid   "h3:qwen2_5_32b+h1+h2+h8+h1c" \
    1 08:00:00 90G  gpu_h200_8 gpu_h100_4
sub big   "h3:qwen2_5_72b" \
    2 03:00:00 220G gpu_h200_8

(( TEST )) && exit 0

if (( ${#ALL_JIDS[@]} )); then
  DEP="afterany$(printf ':%s' "${ALL_JIDS[@]}")"
  AN="$(tsarch_jobname c3-h3-an)"
  if [[ -n "$(squeue -h -u "$USER" -n "$AN" -o %i 2>/dev/null || true)" ]]; then
    echo "  h3-an  already queued - not submitted again"
  else
    AJ="$(sbatch --parsable --job-name="$AN" --dependency="$DEP" \
          --partition=compute --cpus-per-task=4 --mem=16G --time=02:00:00 \
          --output=logs/c3_h3_an_%j.out --error=logs/c3_h3_an_%j.out \
          --wrap "source $TSARCH/env.sh && cd $CODE && python -m runners.h3_backbone_strength --config configs/c3_h3_clean.yaml --stage analyse")"
    printf "  %-6s job %-8s CPU, after every unit (%s)\n" h3-an "${AJ%%;*}" "$DEP"
  fi
fi

cat <<NOTE

Submitted from $CODE
  watch   : squeue -u \$USER -o "%.10i %.44j %.11P %.8T %.10M %.20S %R"
  logs    : $CODE/logs/c3_<unit>_<jobid>.out
  results : $CODE/results/c3_*
Re-running this script is safe.
NOTE
