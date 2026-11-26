#!/usr/bin/env bash
# JOB NAME: supplied by slurm/submit.sh, which passes --job-name and thereby
# overrides the line below. Sharanga and CSIS require
#
#     name_bitsid_project[_anything]
#
# since 2026-11-10 and KILL any job named otherwise, because these are shared
# logins and the name is the only way to attribute GPU-hours to a person.
#
# `#SBATCH -J` is parsed before any shell runs, so it cannot interpolate
# $TSARCH_BITSID; and hardcoding an ID here would put one identity in seven
# files, all of them wrong for the next user of this shared login. Hence the
# placeholder below is INVALID ON PURPOSE - a direct `sbatch` on this file
# must fail visibly rather than bill its hours to nobody. Submit with:
#
#     slurm/submit.sh slurm/replay_cell.sbatch [tag] [extra sbatch args...]
#
#SBATCH --job-name=INVALID-use-slurm-submit-sh
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/replay_%j.out
#SBATCH --error=logs/replay_%j.out
#
# Run ONE cell with NO GPU, serving every model call from the call cache.
#
#   slurm/submit.sh slurm/replay_cell.sbatch h1-L128 \
#          --export=ALL,TSARCH_EXP=h1,TSARCH_L=128
#
# WHY THIS EXISTS
# ---------------
# `--stage analyse` refuses a partial grid, and a grid is partial if a cell
# has no SHARD - not if it has no model calls. A cell whose prompts are all
# already in the cache still has to RUN to write its shard. Skipping it
# because "it replays for free" leaves a hole that blocks the whole analysis,
# which is exactly what happened to H1's L=128: array indices 0,1,3,4 were
# submitted and index 2 was omitted on the reasoning that it was cached.
#
# Cached means FREE, not SKIPPABLE.
#
# WHEN A CELL IS GENUINELY CACHED
# -------------------------------
# The cache keys on (backbone fingerprint, PROMPT_VERSION, system prompt,
# user prompt) - not on experiment name or L. So two configs render
# byte-identical prompts for a cell when their `data` blocks and item counts
# match and the same backbone is used. H1 and H3 both use
# `pool_dir: data/main_n100`, n_per_query_type 100, max_history 512 and
# default to `qwen2_5_32b`, so H1's L=128 is identical to H3's 32B rung at
# L=128. H2 was the same relationship: it completed in 1m02s with 2192 cache
# hits and 0 misses.
#
# If a prompt turns out NOT to be cached, there is no vLLM server here and
# the runner will fail rather than silently produce something else. That is
# the intended behaviour: a miss means the premise was wrong, and the cell
# should be rerun on a GPU through h1.sbatch / h3.sbatch instead.
set -euo pipefail

source "/scratch/$USER/claude-ts-arch-bench/env.sh"
cd "${TSARCH}/ts-arch-bench"

# Second net for the naming policy. Cannot prevent a kill - the scheduler
# decides at submit time - but it makes the reason visible in the log and
# catches a job that somehow started under the placeholder.
source slurm/jobname.sh
tsarch_assert_jobname


EXP="${TSARCH_EXP:-h1}"
case "$EXP" in
  h1) MOD=runners.h1_history_length;    CFG=configs/h1_history_length.yaml ;;
  h2) MOD=runners.h2_query_type;        CFG=configs/h2_query_type.yaml ;;
  h3) MOD=runners.h3_backbone_strength; CFG=configs/h3_backbone_strength.yaml ;;
  *)  echo "unknown TSARCH_EXP=$EXP (want h1, h2 or h3)" >&2; exit 2 ;;
esac

BB="${TSARCH_BACKBONE:-qwen2_5_32b}"
ARGS=(--config "$CFG" --stage inference --backbone "$BB"
      --cache-dir "$TSARCH_CACHE")
if [[ -n "${TSARCH_L:-}" ]]; then
  ARGS+=(--only-L "$TSARCH_L")
fi

echo "=== replay $EXP  backbone=$BB  L=${TSARCH_L:-all}  (CPU, cache only) ==="
python -u -m "$MOD" "${ARGS[@]}"
echo "=== replay done ==="
