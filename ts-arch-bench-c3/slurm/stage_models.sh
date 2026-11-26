#!/usr/bin/env bash
# Pre-download every model from a LOGIN node, before any GPU job runs.
#
#   bash slurm/stage_models.sh            # the 7 Qwen rungs + Chronos
#   bash slurm/stage_models.sh --llama    # also the 2 gated Llama rungs
#   bash slurm/stage_models.sh --only Qwen/Qwen2.5-0.5B-Instruct
#
# WHY STAGE AT ALL
# ----------------
# Two reasons, both of which otherwise waste a queue slot:
#
#   * Compute nodes frequently have no outbound network. A GPU job that
#     tries to reach huggingface.co then hangs until its time limit, having
#     burned the allocation without producing a shard.
#   * meta-llama/Llama-3.1-* are GATED. Without a token that has accepted
#     the licence, those rungs fail at download - and they fail LAST, after
#     the Qwen rungs have already succeeded, so the failure arrives hours in.
#
# Downloading here means a GPU job only ever reads from $HF_HOME. Once this
# has run, set TSARCH_OFFLINE=1 so the jobs cannot silently try the network:
#
#   TSARCH_OFFLINE=1 slurm/submit.sh slurm/h3.sbatch '' --array=0-6
set -euo pipefail

source "/scratch/$USER/claude-ts-arch-bench/env.sh"

QWEN=(Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct
      Qwen/Qwen2.5-3B-Instruct   Qwen/Qwen2.5-7B-Instruct
      Qwen/Qwen2.5-14B-Instruct  Qwen/Qwen2.5-32B-Instruct
      Qwen/Qwen2.5-72B-Instruct)
LLAMA=(meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-70B-Instruct)
TSFM=(amazon/chronos-t5-small)

MODELS=("${QWEN[@]}" "${TSFM[@]}")
case "${1:-}" in
  --llama) MODELS=("${QWEN[@]}" "${LLAMA[@]}" "${TSFM[@]}") ;;
  --only)  MODELS=("${2:?--only needs a model id}") ;;
esac

if [[ " ${MODELS[*]} " == *meta-llama* && -z "${HF_TOKEN:-}" ]]; then
  echo "ABORT: the Llama rungs are gated and no HF_TOKEN is set."
  echo "  printf '%s' '<token>' > ~/.hf_token && chmod 600 ~/.hf_token"
  echo "  (and accept the licence at huggingface.co/meta-llama/...)"
  exit 1
fi

echo "HF_HOME = $HF_HOME"
df -PBG "$HF_HOME" | awk 'NR==2 {gsub("G","",$4); print "free: " $4 " GiB"}'
echo

for m in "${MODELS[@]}"; do
  echo "=== $m ==="
  # hf_hub's own downloader: resumable, verifies hashes, and respects
  # HF_HOME. A plain `git clone` would fetch LFS pointers or duplicate the
  # blobs outside the cache vLLM actually reads.
  python - "$m" <<'PY'
import sys
from huggingface_hub import snapshot_download
mid = sys.argv[1]
# safetensors only: skip the duplicate .bin/.pth copies many repos carry,
# which would otherwise double the download for no benefit.
p = snapshot_download(
    mid, resume_download=True,
    allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json",
                    "*.txt", "*.model", "tokenizer*"],
)
print("  ->", p)
PY
done

echo
echo "=== staged ==="
du -sh "$HF_HOME/hub" 2>/dev/null || true
echo
echo "Now submit with the network disabled, so a job cannot hang on the hub:"
echo "  TSARCH_OFFLINE=1 slurm/submit.sh slurm/h3.sbatch '' --array=0-6"
