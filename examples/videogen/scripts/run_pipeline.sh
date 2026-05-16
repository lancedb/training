#!/usr/bin/env bash
# Run the full videogen pipeline from ingest to training in one shot.
#
# Stops at the stage selected by the first positional argument:
#   ingest        manifest + clip ingest
#   tier1         + Tier-1 keyword backfill
#   tier2         + Tier-2 light-GPU backfill (CLIP / motion / MTScore)
#   tier3         + Tier-3 heavy-GPU backfill (UMT5 + Wan-VAE)  ← cache-ready
#   curate        + Tier-1 + Tier-2 materialised views + indices
#   tier4         + Tier-4 dedup (dhash + is_duplicate)
#   train         + a short Wan2.2 LoRA training run on the cached path  (default)
#
# Examples:
#   bash scripts/run_pipeline.sh                        # full pipeline → train
#   bash scripts/run_pipeline.sh tier1                  # stop after Tier 1
#   DB=/tmp/foo bash scripts/run_pipeline.sh curate     # different db path
#
# Env vars:
#   DB           Lance database path  (default: data/videos/lancedb)
#   MANIFEST     Parquet manifest path  (default: data/chronomagic_proh.parquet)
#   CLIPS        Directory of downloaded mp4s  (default: data/clips)
#   LIMIT        Manifest row cap (default: all)
#   TRAIN_STEPS  Training steps       (default: 20)
#   PYTHON       Python executable    (default: .venv/bin/python or python)
set -euo pipefail

STAGE="${1:-train}"
DB="${DB:-data/videos/lancedb}"
MANIFEST="${MANIFEST:-data/chronomagic_proh.parquet}"
CLIPS="${CLIPS:-data/clips}"
LIMIT_FLAG=""
[ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit $LIMIT"
TRAIN_STEPS="${TRAIN_STEPS:-20}"
PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python)"

cd "$(dirname "$0")/.."

# Make sure the manifest exists.
[ -f "$MANIFEST" ] || {
    echo "ERROR: manifest not found at $MANIFEST"
    echo "       run: python -m videogen.download_manifest --variant proh \\"
    echo "            --out $MANIFEST"
    exit 1
}

run() {
    echo
    echo "=== $1 ==="
    shift
    "$@"
}

# Restrict the table to rows whose clips actually exist on disk —
# downstream Tier-2/3 backfills need the mp4 bytes.
run "ingest manifest + clips" \
    "$PYTHON" -m videogen.ingest_chronomagic \
        --manifest "$MANIFEST" --video-dir "$CLIPS" \
        --require-clips --overwrite $LIMIT_FLAG --db "$DB"
[ "$STAGE" = "ingest" ] && exit 0

run "Tier 1 — caption flags" \
    "$PYTHON" -m videogen.backfill_geneva --tier 1 --db "$DB"
[ "$STAGE" = "tier1" ] && exit 0

run "Tier 2 — CLIP / motion / MTScore" \
    "$PYTHON" -m videogen.backfill_geneva --tier 2 --db "$DB"
[ "$STAGE" = "tier2" ] && exit 0

run "Tier 3 — UMT5 hidden states" \
    "$PYTHON" -m videogen.backfill_geneva --columns t5_hidden_states --db "$DB"
run "Tier 3 — Wan-VAE latents" \
    "$PYTHON" -m videogen.backfill_geneva --columns vae_latent --db "$DB"
[ "$STAGE" = "tier3" ] && exit 0

run "manage_views — curate Tier-1" \
    "$PYTHON" -m videogen.manage_views --action curate --db "$DB"
run "manage_views — curate Tier-2" \
    "$PYTHON" -m videogen.manage_views --action curate-2 --db "$DB"
[ "$STAGE" = "curate" ] && exit 0

run "Tier 4 — dHash" \
    "$PYTHON" -m videogen.backfill_geneva --columns dhash_first_last --db "$DB"
run "Tier 4 — L2 index for dedup" \
    "$PYTHON" -m videogen.manage_views --action build-indices --db "$DB"
run "Tier 4 — is_duplicate" \
    "$PYTHON" -m videogen.backfill_geneva --columns is_duplicate --db "$DB"
[ "$STAGE" = "tier4" ] && exit 0

# Pick the strongest view that exists for the train run.
TRAIN_VIEW=$(
    "$PYTHON" -c "
import lancedb
tbl = lancedb.connect('$DB')
names = set(tbl.list_tables().tables)
for v in ('phase_transitions_curated_train', 'phase_transitions_train', 'videos_raw'):
    if v in names:
        print(v); break
")

run "train Wan2.2 LoRA on '$TRAIN_VIEW' ($TRAIN_STEPS steps)" \
    "$PYTHON" -m videogen.train_wan22_lora \
        --db "$DB" --train-view "$TRAIN_VIEW" \
        --steps "$TRAIN_STEPS" --batch-size 1 --num-workers 0 \
        --rank 16 --alpha 16 --lr 1e-4 \
        --log-every "$(( TRAIN_STEPS / 4 ))" \
        --output-dir "checkpoints/pipeline_$(date +%s)"

echo
echo "=== verify ==="
"$PYTHON" -m videogen.verify_pipeline --db "$DB"
