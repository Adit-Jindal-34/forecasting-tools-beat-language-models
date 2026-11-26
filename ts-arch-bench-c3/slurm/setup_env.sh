#!/usr/bin/env bash
# One-time environment setup on hpc01. Run it, don't source it.
#
#   bash slurm/setup_env.sh
#
# WHY EVERYTHING LIVES ON /scratch
# --------------------------------
# $HOME has a 40 GiB quota with ~3.8 GiB free. torch alone is ~2.5 GiB and
# the nine-rung ladder is ~410 GiB of weights. Every default cache location
# (pip, HF, torch, triton, vLLM) points inside $HOME, so each one is
# redirected here explicitly. Leave one unset and the job dies at 90% with a
# quota error.
#
# WHAT GOES WHERE, AND WHY IT MATTERS
# -----------------------------------
#   /scratch   env, model weights, HF cache   -> large, re-downloadable
#   $HOME      results, call cache, pool      -> small, IRREPLACEABLE
#
# /scratch deletes files 15 days after last modification. Model weights
# surviving that is nice; the sample pool surviving it is load-bearing,
# because a regenerated pool means `b0007-counting-003` is a different
# series and the paired design across H1/H2/H3 silently breaks. So
# artifacts are archived to $HOME by slurm/archive.sh, never left here.
set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/$USER}"
# EXPORTED, not plain shell variables: srun propagates the environment, so a
# non-exported TSARCH is invisible inside slurm/_build_venv.sh and its guard
# aborts with "TSARCH must be set". Correct of it to abort - the alternative
# was defaulting to a path nobody chose.
export TSARCH="${TSARCH:-$SCRATCH/claude-ts-arch-bench}"
export ARCHIVE="$HOME/tsarch-archive"

echo "scratch root : $TSARCH"
echo "archive root : $ARCHIVE"
mkdir -p "$TSARCH"/{env,hf,pip,torch,triton,vllm,cache,results,tmp} "$ARCHIVE"

# --------------------------------------------------------------------------
# /scratch capacity. archive.sh guards $HOME; nothing guarded this.
# --------------------------------------------------------------------------
NEED_GB=460      # ~410 GiB of ladder weights + venv + Chronos + headroom
FREE_GB="$(df -PBG "$TSARCH" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4+0}')"
echo
echo "=== /scratch capacity ==="
echo "free: ${FREE_GB} GiB   needed for the full 9-rung ladder: ~${NEED_GB} GiB"
if [[ -n "$FREE_GB" ]] && (( FREE_GB < NEED_GB )); then
  cat <<WARN
WARNING: not enough room for every rung at once.
  Per-rung weights (bf16, approx GiB):
    0.5B 1 | 1.5B 3 | 3B 6 | 7B 15 | 14B 28 | 32B 62 | 72B 145
    llama-8B 16 | llama-70B 132
  Options, in order of preference:
    1. Run the ladder in two waves and delete each wave's weights:
         rm -rf \$HF_HOME/hub/models--Qwen--Qwen2.5-72B-Instruct
       The CALL CACHE is what matters and it is tiny - shards and
       calls.jsonl survive, so deleting weights costs only re-download.
    2. Drop the two Llama robustness rungs (they are outside the
       confirmatory family anyway): ~148 GiB saved.
  Weights are re-downloadable; the pool and the call cache are not.
WARN
fi

# --------------------------------------------------------------------------
# Hugging Face token. meta-llama/Llama-3.1-* are GATED repos: without a
# token with access, rungs 7 and 8 fail at download time, deep into the
# queue, after the Qwen rungs have already succeeded.
# --------------------------------------------------------------------------
echo
echo "=== Hugging Face access ==="
HF_TOKEN_FILE="$HOME/.hf_token"
if [[ -n "${HF_TOKEN:-}" ]]; then
  printf '%s' "$HF_TOKEN" > "$HF_TOKEN_FILE"
  chmod 600 "$HF_TOKEN_FILE"
  echo "token taken from \$HF_TOKEN and saved to $HF_TOKEN_FILE (mode 600)"
elif [[ -s "$HF_TOKEN_FILE" ]]; then
  chmod 600 "$HF_TOKEN_FILE"
  echo "token found at $HF_TOKEN_FILE"
else
  cat <<'NOTOKEN'
No token found. The seven Qwen2.5 rungs are ungated and will work without
one. The two Llama-3.1 rungs are GATED and will fail at download.

To add one later (do NOT paste it into a tracked file):
  1. accept the licence at huggingface.co/meta-llama/Llama-3.1-8B-Instruct
  2. create a read token at huggingface.co/settings/tokens
  3. printf '%s' '<token>' > ~/.hf_token && chmod 600 ~/.hf_token

Or drop the Llama pair: they are the cross-family robustness check and sit
outside the confirmatory family, so H3's headline result does not need them.
NOTOKEN
fi

# --------------------------------------------------------------------------
cat > "$TSARCH/env.sh" <<'ENVEOF'
# source this at the top of every job
SCRATCH="${SCRATCH:-/scratch/$USER}"
export TSARCH="${TSARCH:-$SCRATCH/claude-ts-arch-bench}"

# every cache that would otherwise land in $HOME and blow the quota
export HF_HOME="$TSARCH/hf"
export HUGGINGFACE_HUB_CACHE="$TSARCH/hf/hub"
export TRANSFORMERS_CACHE="$TSARCH/hf/transformers"
export PIP_CACHE_DIR="$TSARCH/pip"
export TORCH_HOME="$TSARCH/torch"
export TRITON_CACHE_DIR="$TSARCH/triton"
export VLLM_CACHE_ROOT="$TSARCH/vllm"
export XDG_CACHE_HOME="$TSARCH/.cache"
export TMPDIR="$TSARCH/tmp"
mkdir -p "$TMPDIR"

# gated repos (meta-llama/*). Kept in a 600 file outside the repo, never in
# a config: a token in a tracked file is a credential leak.
if [[ -s "$HOME/.hf_token" ]]; then
  HF_TOKEN="$(cat "$HOME/.hf_token")"
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Compute nodes often have no outbound network. If so, weights must be
# pre-staged from a login node by slurm/stage_models.sh, and the job must
# not then try to reach the hub and hang.
if [[ "${TSARCH_OFFLINE:-0}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

# quieter, and deterministic where it can be
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# the shared call cache: ONE location for H1/H2/H3, so identical prompts
# are paid for once. The key is (backbone fp, PROMPT_VERSION, prompts) and
# carries neither the experiment name nor L.
export TSARCH_CACHE="$TSARCH/cache"

source "$TSARCH/env/bin/activate"
ENVEOF

echo
echo "=== creating the venv on an interactive job (not the login node) ==="
echo "login-node load average is routinely >80 and the cluster banner asks"
echo "for installs to run under srun. This will queue briefly."
# No --pty: it requires a TTY, which a non-interactive ssh session does not
# have, so the build would fail or hang when driven remotely. srun streams
# output fine without it and still works when run interactively.
#
# --cpus-per-task=8: this cluster enforces a QOS MINIMUM cpu count, and a
# smaller request is rejected outright with QOSMinCpuNotSatisfied rather
# than queued. Do not trim this to "be polite".
#
# The build itself lives in slurm/_build_venv.sh so that its quoting is not
# nested inside this string, and so a missing module ABORTS there instead of
# silently falling through to the compute nodes' system Python 3.6.
srun --partition=compute --nodes=1 --ntasks=1 --cpus-per-task=8      --mem=32G --time=01:30:00      bash "$PWD/slurm/_build_venv.sh"

echo
echo "=== done ==="
echo "source $TSARCH/env.sh   # in every job"
echo
echo "Next:"
echo "  bash slurm/preflight.sh                # settles GPU sizing + TP detection"
echo "  bash slurm/stage_models.sh             # pre-download weights (login node)"
echo "  cd \$TSARCH/ts-arch-bench && python -m pytest -q    # expect 151 passed"
echo "  slurm/submit.sh slurm/h3.sbatch '' --array=0-6   # the big spend, first"
