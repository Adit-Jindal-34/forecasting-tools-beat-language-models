#!/usr/bin/env bash
# Shared-login job naming, required on Sharanga and CSIS from 2026-11-10.
#
# REQUIRED FORMAT
#   name_bitsid_project[_anything]
#   e.g.  kushagra_2023A7PS1191H_akshargraft_run1
#
#   name    first name, lowercase, no spaces (hyphen if needed)
#   bitsid  your BITS ID
#   project a short project tag
#   _anything  optional, ignored by the trackers
#
# WHY THIS FILE EXISTS, RATHER THAN EDITING EACH #SBATCH -J LINE
# --------------------------------------------------------------
# A job whose name does not match is KILLED IMMEDIATELY on submission. That
# makes the name a correctness requirement at SUBMIT time, and it means a
# check inside the job script is too late to prevent anything - by the time
# the script runs, the job either survived or never started.
#
# `#SBATCH -J` is also a static directive: SLURM parses it before any shell
# runs, so it cannot interpolate $TSARCH_BITSID. Hardcoding the ID into
# seven sbatch files would scatter one identity across seven places, and
# every one of them would be wrong for the next person to use this repo on a
# shared login.
#
# So the name is assembled HERE and passed with `sbatch --job-name=...`,
# which overrides the static directive. `slurm/submit.sh` is the entry point.
# The sbatch files keep a deliberately INVALID placeholder name, so a direct
# `sbatch slurm/h2.sbatch` fails loudly instead of running under a name that
# bills someone else's usage to this login.
#
# ONE-TIME SETUP, on the cluster, in /scratch/$USER/claude-ts-arch-bench/env.sh:
#
#   export TSARCH_NAME=adit            # your first name, lowercase
#   export TSARCH_BITSID=2023A7PS1191H # YOUR BITS ID - no default on purpose
#   export TSARCH_PROJECT=tsarch       # project tag
#
# TSARCH_BITSID has no default and never will. A guessed or inherited ID
# attributes GPU-hours to the wrong person, which is the exact problem this
# policy exists to fix; refusing to submit is strictly better.

# Matched loosely on purpose. The admins' checker is authoritative, and BITS
# IDs are not all shaped like 2023A7PS1191H (PhD and staff IDs differ), so a
# strict pattern here would reject valid IDs and block real work. This
# catches the failure that actually happens: a field left empty or still
# holding a placeholder.
TSARCH_JOBNAME_RE='^[a-z][a-z-]*_[A-Za-z0-9]{4,}_[A-Za-z0-9][A-Za-z0-9-]*(_.*)?$'

# tsarch_jobname <tag> -> prints name_bitsid_project_tag
tsarch_jobname() {
  local tag="${1:-}"
  local name="${TSARCH_NAME:-}"
  local bitsid="${TSARCH_BITSID:-}"
  local project="${TSARCH_PROJECT:-tsarch}"

  local missing=()
  [ -n "$name" ]   || missing+=("TSARCH_NAME")
  [ -n "$bitsid" ] || missing+=("TSARCH_BITSID")
  if [ ${#missing[@]} -gt 0 ]; then
    cat >&2 <<EOF
ERROR: cannot build a compliant SLURM job name - unset: ${missing[*]}

Sharanga and CSIS kill any job whose name is not

    name_bitsid_project[_anything]

Set these once in /scratch/\$USER/claude-ts-arch-bench/env.sh:

    export TSARCH_NAME=<your first name, lowercase>
    export TSARCH_BITSID=<your BITS ID>
    export TSARCH_PROJECT=tsarch

then re-source it. Nothing was submitted.
EOF
    return 1
  fi

  # lowercase the name and turn spaces into hyphens, per the spec
  name="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')"

  local jn="${name}_${bitsid}_${project}"
  [ -n "$tag" ] && jn="${jn}_${tag}"

  if ! [[ "$jn" =~ $TSARCH_JOBNAME_RE ]]; then
    echo "ERROR: assembled job name '$jn' does not match the required" >&2
    echo "       format name_bitsid_project[_anything]. Check TSARCH_NAME" >&2
    echo "       (lowercase letters/hyphens) and TSARCH_BITSID." >&2
    return 1
  fi
  printf '%s' "$jn"
}

# tsarch_assert_jobname - second safety net, called from inside a job.
#
# Cannot prevent the kill (see above) but turns a mysterious disappearance
# into a logged reason, and catches the case where the job DID start under a
# placeholder because the policy check was not applied to this partition.
tsarch_assert_jobname() {
  local jn="${SLURM_JOB_NAME:-}"
  [ -n "$jn" ] || return 0          # not under SLURM; nothing to check
  if ! [[ "$jn" =~ $TSARCH_JOBNAME_RE ]]; then
    cat >&2 <<EOF
ERROR: this job is running as '$jn', which does NOT match the naming policy

    name_bitsid_project[_anything]

required on Sharanga and CSIS since 2026-11-10. Jobs like this are killed
immediately and their GPU-hours are attributed to nobody. Submit through

    slurm/submit.sh <script> [tag] [extra sbatch args...]

which builds the name from TSARCH_NAME / TSARCH_BITSID / TSARCH_PROJECT.
Refusing to run rather than consuming an allocation under a bad name.
EOF
    return 1
  fi
  echo "[jobname] $jn"
}
