#!/usr/bin/env bash
# Sweep datasets x backends, ONE MEASUREMENT PER PROCESS.
#
# One process per cell is deliberate: an earlier version looped inside a single interpreter,
# and once one backend's DataLoader workers died every later cell in that process failed too --
# which looked like dataset-specific bugs and wasn't.
#
#   LANCE_ROOT=s3://my-bucket ./run_matrix.sh pusht toto roboturk
set -u
: "${LANCE_ROOT:?set LANCE_ROOT, e.g. s3://my-bucket or /data/lance}"
OUT=${OUT:-matrix.json}
BATCH=${BATCH:-64}; WORKERS=${WORKERS:-8}; BATCHES=${BATCHES:-150}
DROP_CACHES=${DROP_CACHES:-1}

for name in "$@"; do
  for backend in s3 hub stream; do
    # Page cache makes a local re-read look like a faster backend. See PITFALLS.md #2.
    if [ "$DROP_CACHES" = 1 ]; then
      sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || \
        echo "  (could not drop caches -- local numbers will be optimistic)" >&2
      sleep 3
    fi
    root=""; [ "$backend" = s3 ] && root="--root $LANCE_ROOT/${name}-lance"
    echo "=== $name / $backend"
    python bench_loader.py --backend "$backend" --repo-id "lerobot/$name" $root \
      --batch-size "$BATCH" --num-workers "$WORKERS" --num-batches "$BATCHES" \
      --tolerance-s 0.005 --label "$name/$backend" --out "$OUT" || true
    sleep 10
  done
done
echo "wrote $OUT"
