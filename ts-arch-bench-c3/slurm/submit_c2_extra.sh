#!/usr/bin/env bash
# Submit the H2 replication on further models (PREREGISTRATION_C2.md, A16),
# all at once, plus the cross-model summary chained behind them.
#
#   bash slurm/submit_c2_extra.sh --test     # start estimates, nothing submitted
#   bash slurm/submit_c2_extra.sh            # submit everything
#
# Same machinery as slurm/submit_c2.sh (read its header): bundled units,
# each 1-GPU unit queued on h200 AND h100 with the first copy to start
# cancelling the other, a lock per unit, safe to re-run.
#
#   oss20     gpt-oss-20b, reasoning medium                    1 GPU
#   oss120m   gpt-oss-120b, reasoning medium                   1 GPU
#   oss120h   gpt-oss-120b, reasoning high                     1 GPU
#   famA      Granite-3.3-8B, then Phi-4                       1 GPU
#   famB      Mistral-Small-3.2-24B, then OLMo-2-32B           1 GPU
#
# Every unit starts with a probe (scripts/probe_server.py) per model: if a
# model's replies are not in the expected format, that model's run is not
# started and the unit moves on to its next model.
#
# Memory: vLLM reads the weights through system RAM, so each unit asks for a
# little more than its largest model's weights (gpt-oss-120b 65 GB, OLMo-2
# 64 GB, Mistral 48 GB). Walltimes are generous for the reasoning models
# because their replies are long and their speed on this cluster is not yet
# measured; a unit that times out resumes from its cache when re-submitted.
#
# Submissions used: 5 on gpu_h200_8 and 5 on gpu_h100_4, inside the login's
# limit of 6 per partition.
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
  local name; name="$(tsarch_jobname "c2x-$tag")"
  local lock="$TSARCH/locks/c2_$tag.lock"   # tags are distinct from submit_c2.sh units
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
    local j
    for j in $existing; do LADDER_JIDS+=("$j"); done
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
    LADDER_JIDS+=("$jid")
    printf "  %-6s job %-8s %-11s gpu:%s %-9s %-5s %s\n" \
      "$tag" "$jid" "$part" "$gres" "$wall" "$mem" "${steps//+/ }"
  done
  return 0
}

echo "=== cycle 2 extra models: $( ((TEST)) && echo 'estimated start times (nothing submitted)' || echo submitting) ==="

sub oss20   "h2x:gpt_oss_20b"                          1 04:00:00 48G  gpu_h200_8 gpu_h100_4
sub oss120m "h2x:gpt_oss_120b"                         1 06:00:00 100G gpu_h200_8 gpu_h100_4
sub oss120h "h2x:gpt_oss_120b_high"                    1 10:00:00 100G gpu_h200_8 gpu_h100_4
sub famA    "h2x:granite_3_3_8b+h2x:phi_4"             1 03:00:00 64G  gpu_h200_8 gpu_h100_4
sub famB    "h2x:mistral_small_3_2+h2x:olmo_2_32b"     1 03:00:00 100G gpu_h200_8 gpu_h100_4

(( TEST )) && exit 0

# The cross-model table waits for every unit (siblings included).
if (( ${#LADDER_JIDS[@]} )); then
  DEP="afterany$(printf ':%s' "${LADDER_JIDS[@]}")"
  XN="$(tsarch_jobname c2-cross-model)"
  XQ="$(squeue -h -u "$USER" -n "$XN" -o %i 2>/dev/null || true)"
  if [[ -n "$XQ" ]]; then
    echo "  summary already queued as $XN - not submitted again"
  else
    XJ="$(sbatch --parsable --job-name="$XN" --dependency="$DEP"             --partition=compute --cpus-per-task=4 --mem=16G --time=01:00:00             --output=logs/c2_cross_model_%j.out --error=logs/c2_cross_model_%j.out             --wrap "source $TSARCH/env.sh && cd $TSARCH/ts-arch-bench && python scripts/c2_cross_model.py")"
    printf "  %-8s job %-8s CPU, runs after every unit (%s)
"       summary "${XJ%%;*}" "$DEP"
  fi
fi

cat <<'NOTE'

Submitted. Each model's H2 analysis prints at the end of its step.
  watch   : squeue -u $USER -o "%.10i %.44j %.11P %.8T %.10M %.20S %R"
  logs    : logs/c2_<unit>_<jobid>.out
  results : results/c2_h2x_<model>/ , results/c2_cross_model/
Re-running this script is safe.
NOTE
