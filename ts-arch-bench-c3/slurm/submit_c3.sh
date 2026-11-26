#!/usr/bin/env bash
# Submit ALL of cycle 3 (PREREGISTRATION_C3.md) with the right resources.
#
#     source /scratch/$USER/claude-ts-arch-bench/env.sh
#     bash slurm/submit_c3.sh --test     # ask the scheduler, submit nothing
#     bash slurm/submit_c3.sh            # submit
#     bash slurm/submit_c3.sh h3 h1      # only some units (see UNITS below)
#
# WHY A SCRIPT AND NOT FIVE COMMANDS. The first run sheet submitted the H3
# ladder as one array, `--array=0-6`, at the script's default of ONE GPU for
# every rung. The 72B is ~153 GiB of weights and no single card here holds
# it, so slurm/serve_vllm.sh would have refused that rung after it had
# queued. The rungs need three different sizes; they are written down here
# once, copied from what cycle 2 actually ran with (slurm/submit_c2.sh).
#
# UNITS
#   h3    the seven-rung ladder on pool_clean_n300 (three array jobs)
#   h1    H1 L-sweep, clean pool. Starts AFTER the 14B/32B ladder job so its
#         L=128 cell replays H3's 32B rung from the shared cache.
#   h1c   H1 L-sweep, contaminated pool (sensitivity arm)
#   h2    H2 reframed onto series character (its own runner)
#   h8    horizon-8 sensitivity, L=128 (analyses itself at the end)
#
# Every unit except h8 also gets a CPU analysis job that starts when its
# inference jobs end. Analysis touches no model; re-run it freely:
#     slurm/submit.sh slurm/analyse.sbatch c3-h3-an --export=ALL,TSARCH_EXP=c3_h3
#
# RESUBMITTING IS SAFE. Finished cells are skipped (their shard exists) and
# an interrupted cell replays its finished calls from the cache, so a job
# that hit its time limit or lost its node only needs this script run again.
#
# 32B PLACEMENT. The 32B jobs go to the H200 partition only: on an 80 GiB
# card one GPU leaves ~3 GiB for the KV cache, too little for eight
# requests at once. To use the H100s as well, give them two cards:
#     C3_PART_32B=gpu_h200_8,gpu_h100_4 C3_GRES_32B=2 bash slurm/submit_c3.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source slurm/jobname.sh
tsarch_jobname >/dev/null          # fail fast if the identity is not set
: "${TSARCH:?source /scratch/\$USER/claude-ts-arch-bench/env.sh first}"
mkdir -p logs

TEST=0
UNITS=()
for a in "$@"; do
  case "$a" in
    --test) TEST=1 ;;
    h3|h1|h1c|h2|h8) UNITS+=("$a") ;;
    *) echo "unknown argument: $a (want --test, h3, h1, h1c, h2, h8)" >&2; exit 2 ;;
  esac
done
(( ${#UNITS[@]} )) || UNITS=(h3 h1 h1c h2 h8)
want() { local u; for u in "${UNITS[@]}"; do [[ "$u" == "$1" ]] && return 0; done; return 1; }

PART_SMALL="${C3_PART_SMALL:-gpu_h200_8,gpu_h100_4}"
PART_32B="${C3_PART_32B:-gpu_h200_8}"
GRES_32B="${C3_GRES_32B:-1}"
PART_72B="${C3_PART_72B:-gpu_h200_8}"

# The pools are declared by hash; refuse to queue GPU jobs against a pool
# that does not match its config.
python data-real/check_pools.py >/dev/null || {
  python data-real/check_pools.py || true
  echo "pool check failed - nothing submitted" >&2; exit 1; }

LAST_JID=""
# sub <tag> <script> <partition> <gpus> <cpus> <mem> <time> <export> [extra sbatch args...]
sub() {
  local tag="$1" script="$2" part="$3" gres="$4" cpus="$5" mem="$6" wall="$7" exp="$8"
  shift 8
  local name; name="$(tsarch_jobname "c3-$tag")"
  LAST_JID=""
  local args=(--job-name="$name" --partition="$part" --cpus-per-task="$cpus"
              --mem="$mem" --time="$wall" --export="ALL${exp:+,$exp}" "$@")
  [[ "$gres" != 0 ]] && args+=(--gres=gpu:"$gres")
  if (( TEST )); then
    printf "  %-10s " "$tag"
    sbatch --test-only "${args[@]}" "$script" 2>&1 | tail -1
    return 0
  fi
  local existing
  existing="$(squeue -h -u "$USER" -n "$name" -o %i 2>/dev/null | head -1 || true)"
  if [[ -n "$existing" ]]; then
    LAST_JID="${existing%%_*}"
    printf "  %-10s already queued or running as job %s - not submitted again\n" \
      "$tag" "$LAST_JID"
    return 0
  fi
  local jid
  if ! jid="$(sbatch --parsable "${args[@]}" "$script" 2>&1)"; then
    printf "  %-10s NOT submitted: %s\n" "$tag" "$(printf '%s' "$jid" | tail -1)"
    return 0
  fi
  LAST_JID="${jid%%;*}"
  printf "  %-10s job %-9s %-22s gpu:%s %-9s %s\n" \
    "$tag" "$LAST_JID" "$part" "$gres" "$wall" "$mem"
}

# analysis job for one unit, started when the given jobs have ended
analyse() {   # analyse <tag> <TSARCH_EXP> <jid>...
  local tag="$1" exp="$2"; shift 2
  local dep="" j
  for j in "$@"; do [[ -n "$j" ]] && dep+=":$j"; done
  (( TEST )) && return 0
  if [[ -z "$dep" ]]; then
    printf "  %-10s no inference job to wait for - analysis not queued\n" "$tag"
    return 0
  fi
  local name; name="$(tsarch_jobname "c3-$tag")"
  if [[ -n "$(squeue -h -u "$USER" -n "$name" -o %i 2>/dev/null || true)" ]]; then
    printf "  %-10s already queued - not submitted again\n" "$tag"; return 0
  fi
  local jid
  if jid="$(sbatch --parsable --job-name="$name" --dependency="afterany$dep" \
              --export=ALL,TSARCH_EXP="$exp" slurm/analyse.sbatch 2>&1)"; then
    printf "  %-10s job %-9s CPU, after%s\n" "$tag" "${jid%%;*}" "$dep"
  else
    printf "  %-10s NOT submitted: %s\n" "$tag" "$(printf '%s' "$jid" | tail -1)"
  fi
}

echo "=== cycle 3: $( ((TEST)) && echo 'scheduler estimates (nothing submitted)' || echo submitting) ==="

H3X="TSARCH_H3_CONFIG=configs/c3_h3_clean.yaml"
MID_JID=""
if want h3; then
  # rungs: 0 0.5B  1 1.5B  2 3B  3 7B  4 14B  5 32B  6 72B
  sub h3-small slurm/h3.sbatch "$PART_SMALL" 1 4 26G  02:00:00 "$H3X" --array=0-3
  J_SMALL="$LAST_JID"
  sub h3-14b   slurm/h3.sbatch "$PART_SMALL" 1 4 90G  02:00:00 "$H3X" --array=4
  J_14="$LAST_JID"
  sub h3-32b   slurm/h3.sbatch "$PART_32B" "$GRES_32B" 4 90G 02:00:00 "$H3X" --array=5
  MID_JID="$LAST_JID"
  sub h3-72b   slurm/h3.sbatch "$PART_72B" 2 8 128G 02:00:00 "$H3X" --array=6
  J_72="$LAST_JID"
  analyse h3-an c3_h3 "$J_SMALL" "$J_14" "$MID_JID" "$J_72"
fi

if want h1; then
  DEP=()
  # after H3's 32B rung, so the L=128 cell is a free replay
  [[ -n "$MID_JID" ]] && DEP=(--dependency="afterany:$MID_JID")
  sub h1 slurm/h1_all.sbatch "$PART_32B" "$GRES_32B" 4 90G 04:00:00 \
      "TSARCH_H1_CONFIG=configs/c3_h1_clean.yaml" "${DEP[@]}"
  analyse h1-an c3_h1 "$LAST_JID"
fi

if want h1c; then
  sub h1c slurm/h1_all.sbatch "$PART_32B" "$GRES_32B" 4 90G 04:00:00 \
      "TSARCH_H1_CONFIG=configs/c3_h1_contaminated.yaml"
  analyse h1c-an c3_h1c "$LAST_JID"
fi

if want h2; then
  sub h2 slurm/c3_h2_strata.sbatch "$PART_32B" "$GRES_32B" 4 90G 01:30:00 ""
  analyse h2-an c3_h2 "$LAST_JID"
fi

if want h8; then
  sub h8 slurm/h2.sbatch "$PART_32B" "$GRES_32B" 4 90G 01:00:00 \
      "TSARCH_H2_CONFIG=configs/c3_sensitivity_h8.yaml"
fi

(( TEST )) && exit 0
cat <<'NOTE'

Submitted. Watch with:   squeue -u $USER
Logs:                    logs/h3_<job>_<rung>.out  logs/h1_all_<job>.out
                         logs/c3h2_<job>.out       logs/h2_<job>.out
                         logs/analyse_<job>.out
A job that fails or times out: run this script again. Finished cells are
skipped and finished calls replay from the cache.
NOTE
