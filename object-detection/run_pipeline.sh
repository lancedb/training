#!/usr/bin/env bash
# run_pipeline.sh — End-to-end BDD100K pipeline (training commented out).
#
# Usage:
#   bash run_pipeline.sh               # full run
#   bash run_pipeline.sh --skip-ingest # skip ingestion if table already exists
#
# Assumes:
#   - virtualenv is active (uv venv .venv && source .venv/bin/activate)
#   - BDD100K data is at data/bdd100k/ (images + annotations)
#   - GPU available for Tier 2 + Tier 3 GPU backfills

set -euo pipefail

SKIP_INGEST=false
for arg in "$@"; do
  [[ "$arg" == "--skip-ingest" ]] && SKIP_INGEST=true
done

log() { echo; echo "=== $* ==="; echo; }

# ---------------------------------------------------------------------------
# 1. Ingest
# ---------------------------------------------------------------------------
if [[ "$SKIP_INGEST" == false ]]; then
  log "Step 1 · Ingest BDD100K → LanceDB"
  python -m object_detection.ingest_bdd --splits train val --overwrite
else
  log "Step 1 · Ingest — skipped (--skip-ingest)"
fi

# ---------------------------------------------------------------------------
# 2. Backfill Tier 1 — CPU, annotation-derived
# ---------------------------------------------------------------------------
log "Step 2 · Backfill Tier 1 (CPU): has_person, has_rider, scene metadata"
python -m object_detection.backfill_geneva \
  --columns has_person has_rider white_balance scene_description \
            scene_has_crossroad scene_has_mountain --concurrency 10

# ---------------------------------------------------------------------------
# 3. Backfill Tier 2 — GPU, Faster R-CNN person detector
# ---------------------------------------------------------------------------
log "Step 3 · Backfill Tier 2 (GPU): person_bbox_area_pct"
python -m object_detection.backfill_geneva --gpu \
  --columns person_bbox_area_pct

# ---------------------------------------------------------------------------
# 4. Backfill Tier 3 — GPU dHash
# ---------------------------------------------------------------------------
log "Step 4 · Backfill Tier 3 (GPU): dhash"
python -m object_detection.backfill_geneva --gpu \
  --columns dhash

# ---------------------------------------------------------------------------
# 5. Build IVF L2 index on dhash
# ---------------------------------------------------------------------------
log "Step 5 · Build vector index on dhash"
python -m object_detection.dedup --action index

# ---------------------------------------------------------------------------
# 6. Backfill is_duplicate — CPU, vector search
# ---------------------------------------------------------------------------
log "Step 6 · Backfill is_duplicate (CPU)"
python -m object_detection.backfill_geneva \
  --columns is_duplicate --concurrency 10

# ---------------------------------------------------------------------------
# 7. Dedup stats
# ---------------------------------------------------------------------------
log "Step 7 · Dedup stats"
python -m object_detection.dedup --action stats

# ---------------------------------------------------------------------------
# 8. Validate dedup with synthetic duplicates
#    Inject 1000 exact-copy rows, verify all caught at Hamming=0, clean up.
# ---------------------------------------------------------------------------
log "Step 8 · Validate dedup with synthetic duplicates"
python -m object_detection.dedup --action inject --n 1000
python -m object_detection.dedup --action index
python -m object_detection.backfill_geneva --columns is_duplicate --overwrite
python -m object_detection.dedup --action verify
python -m object_detection.dedup --action clean
# Rebuild index clean after removing synthetic rows
python -m object_detection.dedup --action index

# ---------------------------------------------------------------------------
# 9. EDA — SQL / FTS queries
# ---------------------------------------------------------------------------
log "Step 9 · EDA queries"
python -m object_detection.spec_queries

# ---------------------------------------------------------------------------
# 10. Create materialized views (training splits)
# ---------------------------------------------------------------------------
log "Step 10 · Create materialized views"
python -m object_detection.manage_views --action curate
python -m object_detection.manage_views --action curate-person

# ---------------------------------------------------------------------------
# 11. Training  (commented out — runs separately, takes hours on full dataset)
# ---------------------------------------------------------------------------
log "Step 11 · Training — skipped (uncomment below to run)"

# python -m object_detection.train_detector \
#     --train-table bdd100k_rider_train \
#     --val-table   bdd100k_rider_val \
#     --epochs 10 --batch-size 64 --lr 0.04 --num-workers 14 \
#     --output-dir  checkpoints/rider
#
# python -m object_detection.train_detector \
#     --train-table bdd100k_nighttime_person_train \
#     --val-table   bdd100k_nighttime_person_val \
#     --epochs 10 --batch-size 64 --lr 0.04 --num-workers 14 \
#     --output-dir  checkpoints/nighttime_person
#
# python -m object_detection.train_detector \
#     --train-table bdd100k_distant_person_train \
#     --val-table   bdd100k_distant_person_val \
#     --epochs 10 --batch-size 64 --lr 0.04 --num-workers 14 \
#     --output-dir  checkpoints/distant_person

# ---------------------------------------------------------------------------
# 12. Simulate new footage arriving — incremental refresh
# ---------------------------------------------------------------------------
log "Step 12 · Simulate new footage: ingest 500 synthetic frames + refresh"
python -m object_detection.ingest_bdd --synthetic 500
python -m object_detection.backfill_geneva \
  --columns has_person has_rider white_balance scene_description \
            scene_has_crossroad scene_has_mountain
python -m object_detection.backfill_geneva --gpu --columns person_bbox_area_pct
python -m object_detection.backfill_geneva --gpu --columns dhash
python -m object_detection.dedup --action index
python -m object_detection.backfill_geneva --columns is_duplicate
python -m object_detection.manage_views --action refresh

log "Pipeline complete."
