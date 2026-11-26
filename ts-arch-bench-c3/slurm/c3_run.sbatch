#!/usr/bin/env bash
# JOB NAME: supplied by slurm/submit_c3_bundled.sh via --job-name, which
# overrides the placeholder below. Sharanga and CSIS kill any job not named
#     name_bitsid_project[_anything]
# so the placeholder is INVALID ON PURPOSE: a bare `sbatch` on this file must
# fail visibly. Submit with:
#
#     bash slurm/submit_c3_bundled.sh
#
#SBATCH --job-name=INVALID-use-slurm-submit-c3-bundled-sh
#SBATCH --partition=gpu_h200_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --time=02:00:00
#SBATCH --output=logs/c3_%x_%j.out
#SBATCH --error=logs/c3_%x_%j.out
#SBATCH --requeue
#
# CYCLE 3, ONE JOB = SEVERAL STEPS ON ONE ALLOCATION. The same machinery as
# slurm/c2_run.sbatch (read its header), with three differences:
#
#  1. It runs from the folder it was SUBMITTED from (TSARCH_CODE, set by
#     submit_c3_bundled.sh), not from $TSARCH/ts-arch-bench. Cycle 3 lives in
#     its own folder on the cluster so the cycle-2 code - including the
#     gpt-oss runs - stays exactly as it was.
#  2. Locks are $TSARCH/locks/c3_<tag>.lock and caches $TSARCH/cache/c3_<tag>,
#     so nothing is shared with a cycle-2 job.
#  3. No forecast seeding: cycle 3 is a new pool, and models/tsfm.py seeds
#     every Chronos call from the series content, so a recomputed forecast
#     is the same draw.
#
#   TSARCH_C3_STEPS   steps joined by '+', run in order:
#                       h3:<backbone>  one H3 rung, c3_h3_clean.yaml (inference)
#                       h1             H1, clean pool, all L, + analysis
#                       h1c            H1, contaminated pool, all L, + analysis
#                       h2             H2 reframed (series character) + analysis
#                       h8             horizon-8 sensitivity + analysis
#
# h1 after h3:qwen2_5_32b in the SAME job (same cache) replays the L=128 cell
# for free: the two configs declare a byte-identical data block.
#
# A failed step does not stop the later ones; the job exits non-zero at the
# end if any step failed, and re-submitting resumes (finished shards are
# skipped, finished calls are cached).
set -uo pipefail

source "/scratch/$USER/claude-ts-arch-bench/env.sh"
cd "${TSARCH_CODE:?set by slurm/submit_c3_bundled.sh}"
mkdir -p logs

source slurm/jobname.sh
tsarch_assert_jobname || exit 1

STEPS="${TSARCH_C3_STEPS:?set TSARCH_C3_STEPS, e.g. 'h3:qwen2_5_32b+h1'}"
STEPS="${STEPS//+/ }"

# ---- siblings and the per-unit lock (see c2_run.sbatch) --------------------
TAG="${TSARCH_C3_TAG:-}"
LOCK=""
if [[ -n "$TAG" ]]; then
  mkdir -p "$TSARCH/locks"
  LOCK="$TSARCH/locks/c3_$TAG.lock"
  if ! mkdir "$LOCK" 2>/dev/null; then
    owner="$(cat "$LOCK/jobid" 2>/dev/null || true)"
    if [[ -e "$LOCK/done" ]]; then
      echo "[c3] unit '$TAG' already finished (job ${owner:-?}). Nothing to do."
      exit 0
    fi
    owner_state=""
    if [[ -n "$owner" && "$owner" != "${SLURM_JOB_ID:-}" ]]; then
      owner_state="$(squeue -h -j "$owner" -t RUNNING,COMPLETING -o %T \
                     2>/dev/null || true)"
    fi
    if [[ -n "$owner_state" ]]; then
      echo "[c3] unit '$TAG' is being run by sibling job $owner. Nothing to do."
      exit 0
    fi
    echo "[c3] stale lock from job ${owner:-?} (not running); taking over"
    rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null || { echo "[c3] lost the race for the lock"; exit 0; }
  fi
  echo "${SLURM_JOB_ID:-local}" > "$LOCK/jobid"
  if [[ -n "${SLURM_JOB_NAME:-}" ]]; then
    scancel --user="$USER" --name="$SLURM_JOB_NAME" --state=PENDING \
      2>/dev/null && echo "[c3] cancelled pending siblings of $SLURM_JOB_NAME" \
      || echo "[c3] could not cancel siblings (they will exit on the lock)"
  fi
fi

CACHE="${TSARCH_C3_CACHE:?set by slurm/submit_c3_bundled.sh}"
mkdir -p "$CACHE"
echo "code : $(pwd)"
echo "cache: $CACHE"

H3_CFG=configs/c3_h3_clean.yaml
H1_CFG=configs/c3_h1_clean.yaml
H1C_CFG=configs/c3_h1_contaminated.yaml
H2_CFG=configs/c3_h2_strata.yaml
H8_CFG=configs/c3_sensitivity_h8.yaml

echo "=== c3_run  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date -Is) ==="
echo "steps: $STEPS"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true

# ---- GPU health gate (2026-11-21; v3 2026-11-22) --------------------------
# gpunode7 holds H200s that fail a CUDA burn-in while reporting ecc=0 and no
# row remaps, so the counters alone do not catch them. Job 464903 was handed
# such a pair and burned all 8 requeues on it in 27s each: requeueing does not
# help, because Slurm re-assigns the same free cards.
#
# v3 therefore does three things v2 did not:
#   1. tests each GPU in its OWN process, so one card's crash does not hide
#      the others, and names the card that failed
#   2. keeps the burn-in's error text (v2 sent it to /dev/null, leaving us
#      blind for nine attempts)
#   3. SELECTS the healthy cards. The job may ask for more GPUs than it needs
#      (TSARCH_GPU_NEED, default = all allocated); if at least that many pass,
#      CUDA_VISIBLE_DEVICES is narrowed to them and the job proceeds. Only if
#      too few pass do we requeue.
#
# A card is condemned on POSITIVE evidence only: nonzero uncorrectable ECC, a
# pending/failed HBM row remap, or a failed burn-in. "N/A" is unknown and
# passes, so a driver that does not report a counter cannot trap the job.
GPU_BLOCKLIST="${TSARCH:-/tmp}/locks/bad_gpus.txt"
GPU_NEED="${TSARCH_GPU_NEED:-0}"
GPU_BURNIN_S="${TSARCH_GPU_BURNIN_S:-20}"
GPU_ALLOC_UUIDS=""
GPU_GOOD_IDX=""
GPU_BAD_REPORT=""

gpu_gate_note_bad() {   # $1 uuid, $2 reason -> append to the blocklist once
  mkdir -p "$(dirname "$GPU_BLOCKLIST")" 2>/dev/null
  if ! grep -qsF "$1" "$GPU_BLOCKLIST" 2>/dev/null; then
    echo "$1  # $2, job ${SLURM_JOB_ID:-?}, $(date -Is)" >> "$GPU_BLOCKLIST"
  fi
}

gpu_gate_scan() {   # fills GPU_GOOD_IDX / GPU_BAD_REPORT; returns 0 if enough
  local idx uuid vol agg rp rf bad out n_alloc=0
  GPU_ALLOC_UUIDS=""; GPU_GOOD_IDX=""; GPU_BAD_REPORT=""

  while IFS=, read -r idx uuid vol agg rp rf; do
    idx="${idx// /}"; uuid="${uuid// /}"; vol="${vol// /}"
    agg="${agg// /}"; rp="${rp// /}"; rf="${rf// /}"
    [[ -z "$uuid" ]] && continue
    n_alloc=$((n_alloc + 1))
    GPU_ALLOC_UUIDS+="$uuid "
    bad=""

    if [[ -s "$GPU_BLOCKLIST" ]] && grep -qsF "$uuid" "$GPU_BLOCKLIST"; then
      bad="on the blocklist"
    elif [[ "$vol" =~ ^[0-9]+$ ]] && (( vol > 0 )); then
      bad="$vol uncorrectable ECC since boot"
    elif [[ "$agg" =~ ^[0-9]+$ ]] && (( agg > 0 )); then
      bad="$agg lifetime uncorrectable ECC"
    elif [[ "${rf,,}" == "yes" ]]; then
      bad="FAILED HBM row remap"
    elif [[ "${rp,,}" == "yes" ]]; then
      bad="PENDING HBM row remap (needs a reset)"
    else
      # burn-in, in its own process, output KEPT
      out="$(CUDA_VISIBLE_DEVICES="$idx" TSARCH_GPU_BURNIN_S="$GPU_BURNIN_S" \
             python - <<'PYGATE' 2>&1
import os, sys, time, traceback
try:
    import torch
    a = torch.randn(4096, 4096, device="cuda:0", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda:0", dtype=torch.bfloat16)
    budget = float(os.environ.get("TSARCH_GPU_BURNIN_S", "20"))
    t0 = time.time(); it = 0
    while time.time() - t0 < budget:
        c = a @ b; it += 1
    torch.cuda.synchronize()
    if not bool(torch.isfinite(c).all().item()):
        print("non-finite matmul result"); sys.exit(1)
    print("ok %d matmuls" % it)
except SystemExit:
    raise
except Exception as e:
    print("%s: %s" % (type(e).__name__, str(e).replace(chr(10), " ")[:200]))
    sys.exit(1)
PYGATE
)" || bad="burn-in failed: $(echo "$out" | tr '\n' ' ' | tail -c 220)"
    fi

    if [[ -n "$bad" ]]; then
      echo "[gpu-check] gpu$idx $uuid BAD - $bad" >&2
      GPU_BAD_REPORT+="gpu$idx ($bad); "
      [[ "$bad" != "on the blocklist" ]] && gpu_gate_note_bad "$uuid" "$bad"
    else
      echo "[gpu-check] gpu$idx $uuid ok (ecc_boot=${vol:-?} ecc_life=${agg:-?}" \
           "remap_pending=${rp:-?} remap_failed=${rf:-?}) $out" >&2
      GPU_GOOD_IDX+="${GPU_GOOD_IDX:+,}$idx"
    fi
  done < <(nvidia-smi --query-gpu=index,uuid,\
ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total,\
remapped_rows.pending,remapped_rows.failure --format=csv,noheader 2>/dev/null)

  if (( n_alloc == 0 )); then
    echo "[gpu-check] nvidia-smi listed NO GPU - refusing to run blind" >&2
    GPU_BAD_REPORT="no GPU visible to nvidia-smi; "
    return 1
  fi
  (( GPU_NEED > 0 )) || GPU_NEED="$n_alloc"
  local n_good=0
  [[ -n "$GPU_GOOD_IDX" ]] && n_good="$(awk -F, '{print NF}' <<< "$GPU_GOOD_IDX")"
  echo "[gpu-check] $n_good of $n_alloc allocated GPUs healthy; need $GPU_NEED" >&2
  (( n_good >= GPU_NEED ))
}

if gpu_gate_scan; then
  # keep exactly the GPUs we need, lowest indices first
  export CUDA_VISIBLE_DEVICES="$(awk -F, -v k="$GPU_NEED" \
    '{for(i=1;i<=k;i++) printf "%s%s", (i>1?",":""), $i}' <<< "$GPU_GOOD_IDX")"
  echo "[c3] using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  [[ -n "$GPU_BAD_REPORT" ]] && echo "[c3] skipped: $GPU_BAD_REPORT"
else
  echo "[c3] GPU HEALTH GATE: too few healthy GPUs - $GPU_BAD_REPORT"
  gate_tries="${SLURM_RESTART_COUNT:-0}"
  if [[ -n "${SLURM_JOB_ID:-}" ]] && (( gate_tries < 6 )); then
    [[ -n "${LOCK:-}" ]] && rm -rf "$LOCK"
    echo "[c3] requeueing, attempt $((gate_tries + 1)) of 6 - no weights loaded"
    # a bare requeue restarts within seconds and Slurm hands back the SAME
    # free cards - job 464903 burned 8 retries that way in 23 minutes. Push
    # the start out so other users' jobs rotate the free set first.
    if scontrol requeue "$SLURM_JOB_ID"; then
      scontrol update jobid="$SLURM_JOB_ID" StartTime=now+20minutes || echo "[c3] could not set a start delay; requeued immediately"
    fi
    sleep 60
    exit 0
  fi
  echo "[c3] not enough healthy GPUs and no requeue left; giving up"
  [[ -n "${LOCK:-}" ]] && rm -rf "$LOCK"
  exit 1
fi

# the pools are declared by hash; refuse to spend GPU time on a mismatch
python data-real/check_pools.py || { echo "[c3] pool check failed"; exit 1; }

model_of() {   # model_of <config> <backbone> -> HF id
  python -c "
import sys; sys.path.insert(0, '.')
from src.config import load_config
print(load_config('$1').backbones['$2'].model)"
}

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

SERVING=""

serve() {           # serve <hf-model-id>
  local model="$1"
  if [[ "$SERVING" == "$model" && -n "${VLLM_PID:-}" ]] \
     && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[c3] reusing the running server for $model"
    return 0
  fi
  stop_server
  local port
  port="$(free_port)"
  [[ -n "$port" ]] || { echo "[c3] no free port found"; return 1; }
  export VLLM_PORT="$port" TSARCH_VLLM_PORT="$port"
  echo "[c3] serving $model on 127.0.0.1:$port"
  source slurm/serve_vllm.sh "$model" || return 1
  [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null \
    || { echo "[c3] server died"; return 1; }
  curl -sf "http://127.0.0.1:${port}/v1/models" | grep -q "\"$model\"" \
    || { echo "[c3] port $port does not serve $model"; vllm_stop || true
         return 1; }
  SERVING="$model"
}

stop_server() {
  if [[ -n "$SERVING" ]]; then
    vllm_stop || true
    SERVING=""
    sleep 20
  fi
}
vllm_stop() { :; }
trap 'stop_server' EXIT

run32() {           # run32 <module> <config>  - a qwen2_5_32b run, inference + analysis
  local model
  model="$(model_of "$2" qwen2_5_32b)" && serve "$model" && \
  python -m "$1" --config "$2" --stage both --backbone qwen2_5_32b \
    --cache-dir "$CACHE"
}

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
    h1)  run32 runners.h1_history_length   "$H1_CFG"  || ok=0 ;;
    h1c) run32 runners.h1_history_length   "$H1C_CFG" || ok=0 ;;
    h2)  run32 runners.h2_series_character "$H2_CFG"  || ok=0 ;;
    h8)  run32 runners.h2_query_type       "$H8_CFG"  || ok=0 ;;
    *)   echo "[c3] unknown step '$step'"; ok=0 ;;
  esac
  if (( ok )); then
    echo "################ step $step OK in $(( SECONDS - t0 ))s"
  else
    echo "################ step $step FAILED after $(( SECONDS - t0 ))s"
    FAILED+=("$step")
    stop_server
  fi
done

stop_server
echo
echo "=== c3_run finished $(date -Is) ==="
# ---- blame the card when a step dies of ECC (added 2026-11-22) -----------
# vLLM's Worker_TP0 is the FIRST GPU of the allocation, and every ECC death
# so far named TP0. If this job died that way, record that card's UUID so
# the health gate above refuses it next time. Attribution is a heuristic,
# not proof: to undo it, delete the line (or the file).
if (( ${#FAILED[@]} )) && [[ -n "${GPU_ALLOC_UUIDS:-}" ]]; then
  _gate_log="logs/c3_${TAG:-unknown}_${SLURM_JOB_ID:-unknown}.out"
  if grep -qs 'uncorrectable ECC error' \
       "$_gate_log" vllm_"${SLURM_JOB_ID:-unknown}"_*.log 2>/dev/null; then
    _gate_tp0="${GPU_ALLOC_UUIDS%% *}"
    mkdir -p "$(dirname "$GPU_BLOCKLIST")"
    if ! grep -qsF "$_gate_tp0" "$GPU_BLOCKLIST"; then
      echo "$_gate_tp0  # uncorrectable ECC, job ${SLURM_JOB_ID:-?}, $(date -Is)" \
        >> "$GPU_BLOCKLIST"
      echo "[c3] recorded $_gate_tp0 in $GPU_BLOCKLIST"
    else
      echo "[c3] $_gate_tp0 was already on the blocklist"
    fi
  fi
fi

if (( ${#FAILED[@]} )); then
  echo "FAILED steps: ${FAILED[*]}"
  echo "Re-submit: finished shards are skipped and finished calls replay."
  [[ -n "$LOCK" ]] && rm -rf "$LOCK"
  exit 1
fi
echo "all steps OK: $STEPS"
[[ -n "$LOCK" ]] && date -Is > "$LOCK/done"
exit 0
