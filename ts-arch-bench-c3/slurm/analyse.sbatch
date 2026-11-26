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
#     slurm/submit.sh slurm/analyse.sbatch [tag] [extra sbatch args...]
#
#SBATCH --job-name=INVALID-use-slurm-submit-sh
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#
# MEASURED 2026-11-07: H2 (one backbone, 9 cells) analyses in 1m02s, but H3
# (seven rungs, clustered bootstrap at the pre-registered n_boot=5000) was
# still running at 13 minutes and job 437048 was killed by an earlier
# 30-minute limit. The fix is walltime, NOT n_boot: 5000 is declared in
# `analysis.n_boot` and cutting it to fit a deadline would be a protocol
# deviation dressed up as an ops change.
#
# A user cannot raise TimeLimit on their own running job ("Access/permission
# denied"), so the limit has to be right at submission. Two hours is still
# short enough to backfill easily on `compute`.
#SBATCH --output=logs/analyse_%j.out
#SBATCH --error=logs/analyse_%j.out
#
# CPU-ONLY. The analysis stage loads shards from disk and bootstraps; it
# never calls a model, so holding a GPU for it would be pure waste and
# would queue behind GPU demand for no reason.
#
#   slurm/submit.sh slurm/analyse.sbatch h3 --export=ALL,TSARCH_EXP=h3
#   slurm/submit.sh slurm/analyse.sbatch h1 --export=ALL,TSARCH_EXP=h1
#   slurm/submit.sh slurm/analyse.sbatch h2 --export=ALL,TSARCH_EXP=h2
#
# `--stage analyse` REFUSES a partial grid by design (`load_shards` raises
# rather than averaging over whatever happens to be present), so submitting
# this before every rung has landed fails loudly instead of quietly
# producing a family computed over a subset. That is the intended
# behaviour; use scripts/peek_shards.py for diagnostics on a partial grid.
#
# The confirmatory result should be read ONCE. Re-running this is harmless
# and deterministic, but deciding to re-run a RUNG after seeing the verdict
# is the researcher degree of freedom the pre-registration exists to remove.
set -euo pipefail

source "/scratch/$USER/claude-ts-arch-bench/env.sh"
cd "${TSARCH}/ts-arch-bench"

# Second net for the naming policy. Cannot prevent a kill - the scheduler
# decides at submit time - but it makes the reason visible in the log and
# catches a job that somehow started under the placeholder.
source slurm/jobname.sh
tsarch_assert_jobname


EXP="${TSARCH_EXP:-h3}"
case "$EXP" in
  h1) MOD=runners.h1_history_length;    CFG=configs/h1_history_length.yaml ;;
  h2) MOD=runners.h2_query_type;        CFG=configs/h2_query_type.yaml ;;
  h3) MOD=runners.h3_backbone_strength; CFG=configs/h3_backbone_strength.yaml ;;
  # cycle 2 (PREREGISTRATION_C2.md)
  c2_h1) MOD=runners.h1_history_length;    CFG=configs/c2_h1.yaml ;;
  c2_h2) MOD=runners.h2_query_type;        CFG=configs/c2_h2.yaml ;;
  c2_h3) MOD=runners.h3_backbone_strength; CFG=configs/c2_h3.yaml ;;
  # cycle 3 (PREREGISTRATION_C3.md)
  c3_h3)  MOD=runners.h3_backbone_strength; CFG=configs/c3_h3_clean.yaml ;;
  c3_h1)  MOD=runners.h1_history_length;    CFG=configs/c3_h1_clean.yaml ;;
  c3_h1c) MOD=runners.h1_history_length;    CFG=configs/c3_h1_contaminated.yaml ;;
  c3_h2)  MOD=runners.h2_series_character;  CFG=configs/c3_h2_strata.yaml ;;
  c3_h8)  MOD=runners.h2_query_type;        CFG=configs/c3_sensitivity_h8.yaml ;;
  *)  echo "unknown TSARCH_EXP=$EXP (want h1, h2, h3, c2_h1, c2_h2, c2_h3, c3_h3, c3_h1, c3_h1c, c3_h2 or c3_h8)" >&2; exit 2 ;;
esac

echo "=== analyse $EXP : $MOD on $CFG ==="
python -m "$MOD" --config "$CFG" --stage analyse

echo
echo "=== zero-score attribution (declared secondary, PREREG D8) ==="
# Descriptive only: no CI, no p-value, no verdict. Separates an unscoreable
# item from a self-contradictory one from a cleanly-wrong one, so the
# confirmatory contrast can be read as capability or instrument rather than
# an unknown mixture of both.
OUT="$(python -c "
import sys; sys.path.insert(0, '.')
from src.config import load_config
print(load_config('$CFG').out_dir)")"
python scripts/decompose_scores.py --results "$OUT" || true

echo "=== analyse $EXP done ==="
