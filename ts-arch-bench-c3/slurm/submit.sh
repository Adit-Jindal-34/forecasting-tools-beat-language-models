#!/usr/bin/env bash
# The submit entry point for Sharanga / CSIS, where a job whose name is not
#
#     name_bitsid_project[_anything]
#
# is killed immediately (policy, 2026-11-10 - shared logins mean the name is
# the only way to attribute GPU-hours to a person).
#
#   slurm/submit.sh slurm/h2.sbatch
#   slurm/submit.sh slurm/h2.sbatch seed2 --export=ALL,TSARCH_H2_CONFIG=configs/h2_query_type_seed2.yaml
#   slurm/submit.sh slurm/judge_slots.sbatch h1h3 --export=ALL,TSARCH_EXP="h1 h3"
#   slurm/submit.sh slurm/h1.sbatch '' --array=0-4
#
# Arguments: <script> [tag] [extra sbatch args...]
#   tag  optional trailing component, e.g. seed2 - the trackers ignore
#        anything after the project field, so use it freely to tell your own
#        runs apart in squeue. Pass '' to omit it.
#
# Everything after the tag is handed to sbatch untouched, so --array,
# --dependency, --export and friends all work as before.
#
# This refuses to submit LOCALLY, before anything reaches the scheduler, if
# the identity variables are unset. A clear error here is much better than a
# job that vanishes a second after squeue first shows it.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/jobname.sh"

if [ $# -lt 1 ]; then
  echo "usage: $0 <script.sbatch> [tag] [extra sbatch args...]" >&2
  exit 2
fi

SCRIPT="$1"; shift
TAG="${1-}"; [ $# -gt 0 ] && shift || true

if [ ! -f "$SCRIPT" ]; then
  echo "no such script: $SCRIPT" >&2
  exit 2
fi

# Derive a default tag from the script name so squeue stays readable when
# the caller does not pass one: slurm/h2.sbatch -> h2
if [ -z "$TAG" ]; then
  TAG="$(basename "$SCRIPT")"
  TAG="${TAG%.sbatch}"
  TAG="${TAG//_/-}"
fi

# tsarch_jobname writes its own diagnostics and returns non-zero; let that
# message stand rather than wrapping it in a second one.
JOBNAME="$(tsarch_jobname "$TAG")" || exit 1

echo "submitting $SCRIPT as --job-name=$JOBNAME"
exec sbatch --job-name="$JOBNAME" "$@" "$SCRIPT"
