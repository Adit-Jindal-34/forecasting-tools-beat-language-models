#!/usr/bin/env bash
# Launch vLLM inside the current job and block until it answers.
#
#   source slurm/serve_vllm.sh <hf-model-id> [tensor-parallel-size]
#
# MUST be sourced, not executed: it exports VLLM_PID so the caller can kill
# the server, and defines vllm_stop for the EXIT trap.
#
# THE BASE URL IS ALWAYS 127.0.0.1:8000
# -------------------------------------
# `OpenAICompatibleLLM.fingerprint()` includes base_url, and the fingerprint
# feeds the call cache key. Addressing the server as `gpunode5:8000` in one
# job and `127.0.0.1:8000` in the next produces two different keys for the
# same prompt, so the second job silently re-asks every question and the
# cache buys nothing. Loopback, every time.
#
# IT SIZES ITSELF, AND REFUSES RATHER THAN OOMs
# ---------------------------------------------
# Three failure modes this guards against, all of which otherwise surface as
# a confusing CUDA OOM hours into a queue wait:
#
#   * `SLURM_GPUS_ON_NODE` is unset on older Slurm, so a naive
#     `${SLURM_GPUS_ON_NODE:-1}` silently becomes TP=1 and a 72B on four
#     allocated GPUs dies with an error that names neither cause. The count
#     is resolved from four sources here and the script ABORTS if none works.
#   * A comma-separated `--partition` list means Slurm may land the job on
#     40 GB or 80 GB or 141 GB cards. The same `--gres=gpu:1` request is
#     fine on one and impossible on another, so the fit is computed from the
#     VRAM actually present.
#   * `--dtype auto` reads bfloat16 from a Qwen config and makes vLLM raise
#     on sm_70 (V100). Compute capability is detected and fp16 substituted.
#
# Override anything: VLLM_PORT, VLLM_DTYPE, VLLM_MAX_LEN, VLLM_UTIL,
# VLLM_KV_FLOOR_GB.

VLLM_MODEL="${1:?usage: source serve_vllm.sh <model> [tp]}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-8192}"
VLLM_UTIL="${VLLM_UTIL:-0.90}"
VLLM_KV_FLOOR_GB="${VLLM_KV_FLOOR_GB:-2}"

vllm_stop() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[serve] stopping vLLM (pid $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}

_die() { echo "[serve] ABORT: $*" >&2; return 1 2>/dev/null || exit 1; }

# -- 1. how many GPUs do we actually have? --------------------------------
_gpu_count() {
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    echo "$SLURM_GPUS_ON_NODE"; return
  fi
  for v in "${SLURM_JOB_GPUS:-}" "${SLURM_STEP_GPUS:-}" "${CUDA_VISIBLE_DEVICES:-}"; do
    if [[ -n "$v" ]]; then
      echo "$v" | tr ',' '\n' | grep -c . ; return
    fi
  done
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -c '^GPU ' ; return
  fi
  echo 0
}

NGPU="$(_gpu_count)"
[[ "$NGPU" =~ ^[0-9]+$ && "$NGPU" -ge 1 ]] || _die \
  "could not determine the GPU count. SLURM_GPUS_ON_NODE, SLURM_JOB_GPUS,
   SLURM_STEP_GPUS and CUDA_VISIBLE_DEVICES are all empty and nvidia-smi
   gave nothing. Refusing to guess TP=1, which would OOM confusingly on a
   multi-GPU allocation. Run:  srun ... env | grep -i slurm_gpus"

VLLM_TP="${2:-$NGPU}"
[[ "$VLLM_TP" -le "$NGPU" ]] || _die \
  "tensor-parallel-size $VLLM_TP exceeds the $NGPU GPU(s) allocated"

# -- 2. what are they, and how much memory? -------------------------------
command -v nvidia-smi >/dev/null 2>&1 || _die "nvidia-smi not found"
PER_GPU_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
              | head -1 | tr -d ' ')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | head -1 | tr -d ' .')"
PER_GPU_GB=$(( PER_GPU_MB / 1024 ))
TOTAL_GB=$(( PER_GPU_GB * VLLM_TP ))

# -- 3. dtype: V100 (sm_70) has no bf16, but does have fp16 tensor cores --
if [[ -n "${VLLM_DTYPE:-}" ]]; then
  DTYPE="$VLLM_DTYPE"
elif [[ -n "$CC" && "$CC" -lt 80 ]]; then
  DTYPE="float16"
  echo "[serve] compute capability ${CC:0:1}.${CC:1} has no bfloat16 ->"
  echo "        --dtype float16 (NOT float32: fp16 uses tensor cores and"
  echo "        half the memory for no loss on this workload)"
else
  DTYPE="auto"
fi

# -- 4. will the weights even fit? ----------------------------------------
# bf16/fp16 weights ~= 2 bytes x params, +5% for activations and CUDA graphs.
_weights_gb() {
  case "$1" in
    *0.5B*|*0_5B*) echo 1  ;;
    *1.5B*|*1_5B*) echo 4  ;;
    *-3B*)         echo 7  ;;
    *-7B*|*-8B*)   echo 17 ;;
    *-14B*)        echo 31 ;;
    *-32B*)        echo 69 ;;
    *-70B*)        echo 148;;
    *-72B*)        echo 153;;
    *)             echo 0  ;;   # unknown: skip the check, don't block
  esac
}
NEED_GB="$(_weights_gb "$VLLM_MODEL")"
AVAIL_GB=$(awk -v t="$TOTAL_GB" -v u="$VLLM_UTIL" 'BEGIN{printf "%d", t*u}')

echo "[serve] model  : $VLLM_MODEL"
echo "[serve] gpus   : ${VLLM_TP}x ${GPU_NAME} @ ${PER_GPU_GB} GiB  (allocated: $NGPU)"
echo "[serve] budget : ${AVAIL_GB} GiB usable of ${TOTAL_GB} GiB at util ${VLLM_UTIL}"

if [[ "$NEED_GB" -gt 0 ]]; then
  KV_GB=$(( AVAIL_GB - NEED_GB ))
  echo "[serve] weights: ~${NEED_GB} GiB  ->  ~${KV_GB} GiB left for KV cache"
  if (( KV_GB < VLLM_KV_FLOOR_GB )); then
    SUGGEST=$(( (NEED_GB + 8 + PER_GPU_GB - 1) / PER_GPU_GB ))
    # tensor parallel must divide the head count; round up to 1/2/4/8
    for c in 1 2 4 8; do if (( c >= SUGGEST )); then SUGGEST=$c; break; fi; done
    _die "this model does not fit.
   weights ~${NEED_GB} GiB vs ${AVAIL_GB} GiB usable on ${VLLM_TP}x${PER_GPU_GB} GiB.
   Resubmit with --gres=gpu:${SUGGEST} (TP follows the allocation), or pick a
   partition with larger cards. Refusing now rather than OOMing after the
   weights have downloaded."
  fi
  if (( KV_GB < 8 )); then
    echo "[serve] WARNING: only ~${KV_GB} GiB of KV cache at ${VLLM_MAX_LEN} context."
    echo "        That is roughly $(( KV_GB * 1024 / 300 )) concurrent sequences; throughput"
    echo "        will be low. More GPUs would be faster, not just safer."
  fi
fi

# -- 5. go ----------------------------------------------------------------
# --max-model-len 8192: H1 at L=512 puts ~3.5k tokens of series in the bare
# arm's prompt, plus instructions, plus generation. A shorter window makes
# vLLM REJECT long-L requests, which would land in the results as bare-arm
# failures rather than as the serving misconfiguration it actually is.
LOG="vllm_${SLURM_JOB_ID:-local}_$(basename "$VLLM_MODEL").log"
# Model-specific flags (e.g. Mistral's own weight format), from the config's
# `serve.extra_args` via slurm/c2_run.sbatch. Empty for every Qwen run, so
# cycle 1 and cycle 2 start the server exactly as before.
read -r -a VLLM_EXTRA <<< "${VLLM_EXTRA_ARGS:-}"
[[ ${#VLLM_EXTRA[@]} -gt 0 ]] && echo "[serve] extra  : ${VLLM_EXTRA[*]}"
python -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL" \
  --served-model-name "$VLLM_MODEL" \
  --host 127.0.0.1 --port "$VLLM_PORT" \
  --tensor-parallel-size "$VLLM_TP" \
  --max-model-len "$VLLM_MAX_LEN" \
  --gpu-memory-utilization "$VLLM_UTIL" \
  --dtype "$DTYPE" \
  --seed 0 \
  --disable-log-requests \
  "${VLLM_EXTRA[@]}" \
  > "$LOG" 2>&1 &
export VLLM_PID=$!

echo "[serve] waiting for /health (pid $VLLM_PID, log $LOG)"
for i in $(seq 1 240); do          # up to 40 min: a 72B load is not quick
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[serve] FAILED: vLLM exited during startup. Tail of its log:"
    tail -40 "$LOG"
    return 1 2>/dev/null || exit 1
  fi
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "[serve] ready after ${i}0s"
    break
  fi
  sleep 10
done

if ! curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
  echo "[serve] FAILED: /health never came up. Tail of the log:"
  tail -40 "$LOG"
  vllm_stop
  return 1 2>/dev/null || exit 1
fi

# Prove the served name matches what the config asks for. A mismatch makes
# every request 404, which would otherwise show up as a wall of unparseable
# answers rather than as a naming error.
echo "[serve] served models:"
curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" | head -c 400; echo
