#!/usr/bin/env bash
# JOB NAME: supplied by slurm/submit_c2.sh via --job-name, which overrides
# the placeholder below. Sharanga and CSIS kill any job not named
#     name_bitsid_project[_anything]
# so the placeholder is INVALID ON PURPOSE: a bare `sbatch` on this file must
# fail visibly. Submit with:
#
#     bash slurm/submit_c2.sh
#
#SBATCH --job-name=INVALID-use-slurm-submit-c2-sh
#SBATCH --partition=gpu_h200_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/c2_%x_%j.out
#SBATCH --error=logs/c2_%x_%j.out
#
# CYCLE 2, ONE JOB = SEVERAL STEPS ON ONE ALLOCATION.
#
# Why bundle. Every job waits in the queue once, however short it is, and on
# this cluster the wait is far longer than the work: a small rung is ~20
# minutes of GPU behind hours of queue. Seven single-rung jobs wait seven
# times and hit the per-user caps (2 jobs on a100, one H100, 2 GPUs on
# h200). Bundled, the ladder is three jobs and H1 a fourth - exactly what
# the caps allow to run at once.
#
# What to run comes from the environment (set by slurm/submit_c2.sh):
#
#   TSARCH_C2_STEPS   space-separated steps, run in order:
#                       h3:<backbone>   one H3-C2 rung (inference)
#                       h2              H2-C2, inference + analysis; a cache
#                                       replay if h3:qwen2_5_32b ran before it
#                       h1              H1-C2, all lengths, + analysis
#
# A failed step does not stop the later ones; the job exits non-zero at the
# end if any step failed, and re-submitting resumes (finished shards are
# skipped, finished calls are cached).
set -uo pipefail

source "/scratch/$USER/claude-ts-arch-bench/env.sh"
cd "${TSARCH}/ts-arch-bench"
mkdir -p logs

source slurm/jobname.sh
tsarch_assert_jobname || exit 1

# steps arrive joined by '+' (commas and spaces are awkward inside --export)
STEPS="${TSARCH_C2_STEPS:?set TSARCH_C2_STEPS, e.g. 'h3:qwen2_5_32b+h2'}"
STEPS="${STEPS//+/ }"

# ---- siblings: the same unit of work is queued on more than one partition --
# This account cannot name several partitions in one job, so submit_c2.sh
# queues SIBLING jobs with one shared name. Whichever starts first takes the
# lock and cancels the rest; a sibling that starts anyway exits here.
#   lock/jobid   the job that owns the work
#   lock/done    the work finished cleanly (later siblings and re-submissions
#                are no-ops until someone deletes it)
TAG="${TSARCH_C2_TAG:-}"
LOCK=""
if [[ -n "$TAG" ]]; then
  mkdir -p "$TSARCH/locks"
  LOCK="$TSARCH/locks/c2_$TAG.lock"
  if ! mkdir "$LOCK" 2>/dev/null; then
    owner="$(cat "$LOCK/jobid" 2>/dev/null || true)"
    if [[ -e "$LOCK/done" ]]; then
      echo "[c2] unit '$TAG' already finished (job ${owner:-?}). Nothing to do."
      exit 0
    fi
    # Captured into a variable, NOT piped into `grep -q`: under `pipefail` a
    # pipeline fails if squeue is cut short by the reader closing early, and
    # a failed check here would be read as "owner is dead" - two jobs would
    # then do the same work at once.
    owner_state=""
    if [[ -n "$owner" && "$owner" != "${SLURM_JOB_ID:-}" ]]; then
      owner_state="$(squeue -h -j "$owner" -t RUNNING,COMPLETING -o %T \
                     2>/dev/null || true)"
    fi
    if [[ -n "$owner_state" ]]; then
      echo "[c2] unit '$TAG' is being run by sibling job $owner. Nothing to do."
      exit 0
    fi
    echo "[c2] stale lock from job ${owner:-?} (not running); taking over"
    rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null || { echo "[c2] lost the race for the lock"; exit 0; }
  fi
  echo "${SLURM_JOB_ID:-local}" > "$LOCK/jobid"
  # Cancel the siblings still waiting. Matched on OUR exact job name, which
  # carries our name, BITS ID and tag, so nobody else's job can match.
  if [[ -n "${SLURM_JOB_NAME:-}" ]]; then
    scancel --user="$USER" --name="$SLURM_JOB_NAME" --state=PENDING \
      2>/dev/null && echo "[c2] cancelled pending siblings of $SLURM_JOB_NAME" \
      || echo "[c2] could not cancel siblings (they will exit on the lock)"
  fi
fi

# One cache directory PER JOB (set by submit_c2.sh). Several jobs appending to
# one calls.jsonl at the same time on a shared filesystem is a risk with no
# benefit: no two jobs serve the same prompts, except H2, which runs inside
# the job that produced them.
CACHE="${TSARCH_C2_CACHE:-$TSARCH_CACHE}"
mkdir -p "$CACHE"
echo "cache: $CACHE"

# The harness's forecast is the forecaster's output, not the model's, so it
# must be THE SAME NUMBERS for every model. A new cache directory would
# recompute it, and on a different card (H100 vs the H200 the Qwen run used)
# bf16 sampling need not reproduce to the last digit. So a cache that has no
# forecasts yet starts from the ones the Qwen run stored. Keys are (forecaster
# settings, exact window bytes, horizon): a copied entry can only ever be
# served for the identical series, and anything not in the file is computed
# as before. Set TSARCH_C2_FORECAST_SEED="" to switch this off.
FSEED="${TSARCH_C2_FORECAST_SEED-$TSARCH/cache/c2_mid/forecasts.jsonl}"
if [[ -n "$FSEED" && ! -s "$CACHE/forecasts.jsonl" ]]; then
  if [[ -s "$FSEED" && "$FSEED" != "$CACHE/forecasts.jsonl" ]]; then
    cp "$FSEED" "$CACHE/forecasts.jsonl.tmp.$$" \
      && mv "$CACHE/forecasts.jsonl.tmp.$$" "$CACHE/forecasts.jsonl" \
      && echo "forecasts: started from $FSEED ($(wc -l < "$CACHE/forecasts.jsonl") entries)"
  else
    echo "forecasts: no stored forecasts at $FSEED; they will be computed here"
  fi
fi
H3_CFG=configs/c2_h3.yaml
H2_CFG=configs/c2_h2.yaml
H1_CFG=configs/c2_h1.yaml

echo "=== c2_run  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date -Is) ==="
echo "steps: $STEPS"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true

model_of() {   # model_of <config> <backbone> -> HF id
  python -c "
import sys; sys.path.insert(0, '.')
from src.config import load_config
print(load_config('$1').backbones['$2'].model)"
}

# A port nobody on this node is listening on. Loopback is shared by every
# job on the node, so the cycle-1 fixed port 8000 collides the moment two
# jobs land together. The client is told the port through TSARCH_VLLM_PORT,
# which changes the connection but NOT the cache key (models/providers.py).
free_port() {
  python - <<'PY'
import os, socket
start = 8000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 1000
for port in list(range(start, start + 200)) + list(range(9000, 9200)):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
PY
}

SERVING=""          # model currently served, so consecutive steps reuse it

serve() {           # serve <hf-model-id> [max_model_len] [extra vllm args]
  local model="$1" maxlen="${2:-8192}" extra="${3:-}"
  # reuse only when the model AND its serving settings are unchanged
  local want="$model|$maxlen|$extra"
  if [[ "$SERVING" == "$want" && -n "${VLLM_PID:-}" ]] \
     && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[c2] reusing the running server for $model"
    return 0
  fi
  stop_server
  local port
  port="$(free_port)"
  [[ -n "$port" ]] || { echo "[c2] no free port found"; return 1; }
  export VLLM_PORT="$port" TSARCH_VLLM_PORT="$port"
  export VLLM_MAX_LEN="$maxlen" VLLM_EXTRA_ARGS="$extra"
  echo "[c2] serving $model on 127.0.0.1:$port (max_model_len=$maxlen${extra:+, $extra})"
  # serve_vllm.sh is written to be sourced; it `return`s non-zero on failure
  source slurm/serve_vllm.sh "$model" || return 1
  # The health check alone cannot tell OUR server from someone else's. Prove
  # that the process we started is alive and that it serves this model.
  [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null     || { echo "[c2] server died"; return 1; }
  curl -sf "http://127.0.0.1:${port}/v1/models" | grep -q "\"$model\"" \
    || { echo "[c2] port $port does not serve $model"; vllm_stop || true
         return 1; }
  SERVING="$want"
}

serve_info() {      # serve_info <config> -> "backbone<TAB>model<TAB>maxlen<TAB>extra"
  python - "$1" <<'PY'
import sys, yaml
raw = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
(name, bb), = raw["backbones"].items()
serve = bb.get("serve") or {}
print("\t".join([name, bb["model"], str(serve.get("max_model_len", 8192)),
                 serve.get("extra_args", "") or ""]))
PY
}

stop_server() {
  if [[ -n "$SERVING" ]]; then
    vllm_stop || true
    SERVING=""
    sleep 20          # let the GPU memory and the port be released
  fi
}
vllm_stop() { :; }    # replaced when serve_vllm.sh is first sourced
trap 'stop_server' EXIT

FAILED=()
for step in $STEPS; do
  echo
  echo "################ step: $step   $(date -Is)"
  t0=$SECONDS
  ok=1
  case "$step" in
    h3:*)
      bb="${step#h3:}"
      model="$(model_of "$H3_CFG" "$bb")" && serve "$model" && \
      python -m runners.h3_backbone_strength --config "$H3_CFG" \
        --stage inference --backbone "$bb" --cache-dir "$CACHE" || ok=0
      ;;
    h2)
      model="$(model_of "$H2_CFG" qwen2_5_32b)" && serve "$model" && \
      python -m runners.h2_query_type --config "$H2_CFG" \
        --stage both --backbone qwen2_5_32b --cache-dir "$CACHE" || ok=0
      ;;
    h1)
      model="$(model_of "$H1_CFG" qwen2_5_32b)" && serve "$model" && \
      python -m runners.h1_history_length --config "$H1_CFG" \
        --stage both --backbone qwen2_5_32b --cache-dir "$CACHE" || ok=0
      ;;
    h2x:*)
      # H2 replication on another model (amendment A16):
      #   h2x:<name>  ->  configs/c2_h2x_<name>.yaml
      # A probe checks the reply format first; if it fails, the full run
      # is NOT started, so a format problem costs seconds, not hours.
      cfgx="configs/c2_h2x_${step#h2x:}.yaml"
      if [[ ! -f "$cfgx" ]]; then
        echo "[c2] no such config: $cfgx"; ok=0
      else
        IFS=$'\t' read -r bbx modelx maxlenx extrax < <(serve_info "$cfgx")
        serve "$modelx" "$maxlenx" "$extrax" && \
        python scripts/probe_server.py --config "$cfgx" && \
        python -m runners.h2_query_type --config "$cfgx" \
          --stage both --backbone "$bbx" --cache-dir "$CACHE" || ok=0
      fi
      ;;
    *)
      echo "[c2] unknown step '$step'"; ok=0
      ;;
  esac
  if (( ok )); then
    echo "################ step $step OK in $(( SECONDS - t0 ))s"
  else
    echo "################ step $step FAILED after $(( SECONDS - t0 ))s"
    FAILED+=("$step")
    # a failed server must not be reused by the next step
    stop_server
  fi
done

stop_server
echo
echo "=== c2_run finished $(date -Is) ==="
if (( ${#FAILED[@]} )); then
  echo "FAILED steps: ${FAILED[*]}"
  echo "Re-submit the same steps: finished shards are skipped and finished"
  echo "calls replay from the cache."
  # release the unit so a re-submission (or a sibling) can pick it up
  [[ -n "$LOCK" ]] && rm -rf "$LOCK"
  exit 1
fi
echo "all steps OK: $STEPS"
[[ -n "$LOCK" ]] && date -Is > "$LOCK/done"
exit 0
