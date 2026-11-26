#!/usr/bin/env bash
# Copy the irreplaceable artifacts off /scratch.
#
#   bash slurm/archive.sh [label]
#
# WHY THIS IS NOT OPTIONAL
# ------------------------
# /scratch deletes files 15 days after last modification. Three of the things
# living there cannot be regenerated:
#
#   data/<pool>/   the frozen sample pool. Regenerating it is only identical
#                  if every data parameter AND the seed match; across a
#                  framework change that is not guaranteed. A different pool
#                  means `b0007-counting-003` is a different series, so the
#                  paired design across H1/H2/H3 silently breaks. This is the
#                  one that would invalidate results rather than cost compute.
#   cache/         the call cache. vLLM's continuous batching is not
#                  bit-deterministic even at temperature 0, so a cold re-run
#                  may not reproduce the recorded replies. The cache is what
#                  makes re-analysis exact (PREREGISTRATION.md §6,
#                  "Execution environment").
#   results/       shards and analysis artifacts.
#
# Model weights and the venv are NOT archived: they are large and simply
# re-downloadable. That is the whole split - $HOME has ~3.8 GiB free, which
# is ample for artifacts and nowhere near enough for checkpoints.
set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/$USER}"
TSARCH="${TSARCH:-$SCRATCH/claude-ts-arch-bench}"
REPO="$TSARCH/ts-arch-bench"
LABEL="${1:-$(date +%Y%m%d-%H%M%S)}"
DEST="$HOME/tsarch-archive/$LABEL"

echo "=== what would be archived ==="
for p in "$REPO/data" "$TSARCH/cache" "$REPO/results"; do
  [[ -e "$p" ]] && du -sh "$p" || echo "   (absent) $p"
done

NEED_KB=$(du -sk "$REPO/data" "$TSARCH/cache" "$REPO/results" 2>/dev/null \
          | awk '{s+=$1} END {print s+0}')
FREE_KB=$(df -Pk "$HOME" | awk 'NR==2 {print $4}')
echo "need ~$((NEED_KB/1024)) MiB, \$HOME has $((FREE_KB/1024)) MiB free"
if (( NEED_KB > FREE_KB * 90 / 100 )); then
  echo "REFUSING: this would use >90% of free \$HOME space."
  echo "Archive elsewhere, or prune results/*/shards first (they are"
  echo "regenerable from the cache at zero GPU cost; the cache is not)."
  exit 1
fi

mkdir -p "$DEST"
echo
echo "=== copying to $DEST ==="
# --exclude the venv and HF cache defensively, in case anyone ever puts the
# repo somewhere that contains them
rsync -a --info=stats1 \
  --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
  "$REPO/data"    "$DEST/" 2>/dev/null || true
rsync -a --info=stats1 "$TSARCH/cache"  "$DEST/" 2>/dev/null || true
rsync -a --info=stats1 "$REPO/results"  "$DEST/" 2>/dev/null || true

# provenance: what produced these artifacts
{
  echo "archived     : $(date -Is)"
  echo "host         : $(hostname)"
  echo "user         : $USER"
  echo "repo         : $REPO"
  echo "git describe : $(git -C "$REPO" describe --always --dirty 2>/dev/null || echo n/a)"
  echo "git HEAD     : $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo n/a)"
  echo
  echo "--- data fingerprints ---"
  for m in "$REPO"/data/*/manifest.json; do
    [[ -e "$m" ]] && echo "$m: $(python -c "
import json,sys; print(json.load(open(sys.argv[1]))['fingerprint'])" "$m" 2>/dev/null)"
  done
} > "$DEST/PROVENANCE.txt"

echo
echo "=== archived to $DEST ==="
cat "$DEST/PROVENANCE.txt"
echo
echo "Reminder: touching files on /scratch resets their 15-day clock, but"
echo "relying on that is not a backup. Re-run this after every experiment."
