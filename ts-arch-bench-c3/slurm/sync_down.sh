#!/usr/bin/env bash
# Pull the irreplaceable artifacts DOWN from the cluster to this machine.
# Run this from your own computer, daily, while an experiment is live.
#
#   bash slurm/sync_down.sh                       # into ./cluster-sync/
#   bash slurm/sync_down.sh /d/backups/tsarch     # somewhere else
#
# WHY DAILY, AND WHY DOWN
# -----------------------
# $SCRATCH is purged 15 days after last modification and is NOT backed up;
# the HPC docs are explicit that data lost there cannot be restored.
# $HOME is backed up daily with 30-day retention, but has a 40 GiB quota, so
# it can hold the artifacts and nothing else.
#
#   https://sharanga.hpc.bits-hyderabad.ac.in/docs/faq/storage/
#
# So there are two independent copies to keep, and this script makes the
# second one:
#
#   slurm/archive.sh   on the cluster:  /scratch -> $HOME   (backed up)
#   slurm/sync_down.sh from your laptop: /scratch -> here   (off-site)
#
# WHAT IT COPIES, AND WHAT IT DELIBERATELY DOES NOT
# -------------------------------------------------
#   data/<pool>/     the frozen sample pool. THE critical one: a regenerated
#                    pool is only byte-identical if every data parameter and
#                    the seed match, so losing it means `b0007-counting-003`
#                    is a different series and the paired design across
#                    H1/H2/H3 breaks. That invalidates results; it does not
#                    merely cost compute.
#   cache/           the call cache. vLLM's continuous batching is not
#                    bit-deterministic even at temperature 0, so a cold
#                    re-run may not reproduce the recorded replies. This is
#                    what makes re-analysis exact.
#   results/         shards, screening.json, analysis.json, the grid.
#   logs/            setup and SLURM job logs.
#
# NOT the model weights (~260 GiB, re-downloadable) and NOT the venv
# (rebuildable in under an hour). Copying those would take hours and protect
# nothing.
#
# Incremental: rsync over ssh, so each day's run transfers only what changed.
set -euo pipefail

REMOTE="${TSARCH_REMOTE:-jagatsesh@hpc.bits-hyderabad.ac.in}"
RROOT="${TSARCH_RROOT:-/scratch/jagatsesh/claude-ts-arch-bench}"
DEST="${1:-./cluster-sync}"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$DEST"
echo "remote : $REMOTE:$RROOT"
echo "local  : $DEST"
echo

if command -v rsync >/dev/null 2>&1; then
  # --partial so an interrupted transfer resumes rather than restarting;
  # no --delete, so nothing already pulled is ever removed locally even if
  # the cluster has purged it. That is the entire point of this copy.
  for sub in ts-arch-bench/data ts-arch-bench/results ts-arch-bench/logs cache; do
    echo "=== $sub ==="
    rsync -az --partial --info=stats1 \
      -e "ssh -o BatchMode=yes" \
      "$REMOTE:$RROOT/$sub/" "$DEST/$sub/" 2>/dev/null \
      || echo "  (absent on the cluster yet)"
  done
else
  echo "rsync not found; falling back to scp (full copy, not incremental)"
  for sub in ts-arch-bench/data ts-arch-bench/results ts-arch-bench/logs cache; do
    mkdir -p "$DEST/$sub"
    scp -o BatchMode=yes -qr "$REMOTE:$RROOT/$sub/." "$DEST/$sub/" \
      2>/dev/null || echo "  (absent) $sub"
  done
fi

# a record of when each pull happened, so a gap is visible rather than assumed
{
  echo "pulled   : $(date -Is)"
  echo "remote   : $REMOTE:$RROOT"
  echo "local    : $(cd "$DEST" && pwd)"
  du -sh "$DEST" 2>/dev/null | awk '{print "size     : " $1}'
} >> "$DEST/SYNC_LOG.txt"

echo
echo "=== pulled ==="
du -sh "$DEST"/* 2>/dev/null | sed 's/^/  /' || true
echo
tail -4 "$DEST/SYNC_LOG.txt" | sed 's/^/  /'
echo
echo "Run this daily while an experiment is live. If more than 15 days pass"
echo "between the last write on /scratch and the next run, the cluster may"
echo "already have purged what you were about to copy."
