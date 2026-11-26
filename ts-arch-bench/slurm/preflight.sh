#!/usr/bin/env bash
# Answer the questions that decide whether the runs work at all.
#
#   bash slurm/preflight.sh
#
# Run this BEFORE submitting anything. It costs ~2 minutes of queue and
# settles three things that `sinfo -o "%G"` cannot, because %G truncates the
# GRES string to ten characters and hides both the GPU count and the memory:
#
#   1. GPUs per node and VRAM per GPU -> which --gres each rung needs.
#      A 32B needs ~69 GiB of weights. That fits one 80 GiB card with little
#      room for KV cache, and does not fit a 40 GiB card at all. Since the
#      sbatch scripts list three partitions, Slurm may land the job on
#      either, so the answer must be known rather than assumed.
#   2. Whether SLURM_GPUS_ON_NODE is set on this Slurm version. If it is
#      not, a naive TP default of 1 silently misconfigures a multi-GPU job
#      instead of failing - serve_vllm.sh now aborts in that case, but you
#      want to know which branch it will take.
#   3. /scratch headroom against ~410 GiB of ladder weights.
set -uo pipefail

echo "############ 1. partitions, GPU counts and VRAM ############"
sinfo -p gpu_a100_8,gpu_h100_4,gpu_h200_8,gpu_v100_1,gpu_v100_2 \
      -O "partition:24,nodelist:12,statelong:12,gres:40,gresused:40" 2>&1 || true
echo
for n in gpunode1 gpunode2 gpunode3 gpunode4 gpunode5 gpunode6 gpunode7; do
  line="$(scontrol show node "$n" 2>/dev/null \
          | tr ' ' '\n' | grep -E '^(Gres|RealMemory|CPUTot)=' | tr '\n' ' ')"
  [[ -n "$line" ]] && printf '  %-10s %s\n' "$n" "$line"
done

echo
echo "############ 2. does Slurm export the GPU count? ############"
echo "(a 2-minute job on the least contended GPU queue)"
srun -p gpu_v100_1,gpu_v100_2 --gres=gpu:1 -t 2:00 --pty bash -lc '
  echo "  SLURM_GPUS_ON_NODE = ${SLURM_GPUS_ON_NODE:-<unset>}"
  echo "  SLURM_JOB_GPUS     = ${SLURM_JOB_GPUS:-<unset>}"
  echo "  SLURM_STEP_GPUS    = ${SLURM_STEP_GPUS:-<unset>}"
  echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "  nvidia-smi sees: $(nvidia-smi -L | grep -c "^GPU ")"
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
' 2>&1 | sed 's/^/  /' || echo "  (could not get an allocation; try again later)"

echo
echo "############ 3. filesystems ############"
echo "slurm:   $(sinfo --version 2>/dev/null || echo unknown)"
df -h "/scratch/$USER" 2>/dev/null | sed 's/^/  /' || df -h /scratch | sed 's/^/  /'
echo "  HOME = $HOME"
quota -s 2>/dev/null | sed 's/^/  /' || echo "  (quota not available)"

echo
echo "############ 4. how to size each rung ############"
cat <<'SIZING'
  Weights are ~2 bytes x params (bf16). vLLM keeps gpu-memory-utilization
  (default 0.90) and the remainder is KV cache. ~8 GiB of KV is comfortable
  at 8192 context; below ~2 GiB serve_vllm.sh refuses rather than OOMing.

  rung                  weights   on 80 GiB cards   on 141 GiB (H200)
  --------------------  -------   ---------------   -----------------
  Qwen2.5-0.5B .. 14B   1-31 GiB  --gres=gpu:1      --gres=gpu:1
  Qwen2.5-32B           ~69 GiB   gpu:1 fits but    --gres=gpu:1
                                  leaves ~3 GiB KV
                                  -> prefer gpu:2
  Qwen2.5-72B           ~153 GiB  --gres=gpu:4      --gres=gpu:2
  Llama-3.1-8B          ~17 GiB   --gres=gpu:1      --gres=gpu:1
  Llama-3.1-70B         ~148 GiB  --gres=gpu:4      --gres=gpu:2

  Tensor-parallel size follows the allocation and must divide the head
  count. Qwen2.5-72B and Llama-3.1-70B both have 64 attention heads and 8
  KV heads, so TP of 2, 4 or 8 all divide cleanly.

  If section 1 shows the A100 node is 40 GiB rather than 80, then the 32B
  (the H1/H2 backbone) needs --gres=gpu:2 there, and gpu_a100_8 should be
  dropped from the --partition list in h1.sbatch and h2.sbatch so Slurm
  cannot land you somewhere the model will not fit.
SIZING
