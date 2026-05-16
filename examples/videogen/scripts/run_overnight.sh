#!/usr/bin/env bash
# Long-haul "few-thousand-clip" run, resumable from any stage.
#
# Stages:
#   download   parallel yt-dlp until $TARGET_CLIPS clips exist on disk
#   ingest     manifest + clip bytes → Lance (--require-clips)
#   tier1-4    full Geneva backfill
#   curate     Tier-1 + Tier-2 views (+ vector indices via build-indices)
#   train      $TRAIN_STEPS Wan2.2 LoRA steps
#   eval       eval_compare with --save-videos (mp4 grid + index.html)
#
# Each stage is idempotent: if it sees its output already, it skips.
# So if the box dies overnight, re-running picks up where it left off.
#
# Env vars (defaults shown):
#   DB              data/videos/lancedb
#   MANIFEST        data/chronomagic_proh.parquet
#   CLIPS           data/clips
#   TARGET_CLIPS    2500          how many on-disk clips before we move on
#   DL_PARALLEL     12            yt-dlp concurrency
#   DL_ATTEMPTS     40000         max manifest rows to attempt downloading
#   TRAIN_STEPS     4000
#   LORA_RANK       64
#   EVAL_PROMPTS    8
#   CKPT_DIR        checkpoints/overnight
#   EVAL_OUT_DIR    eval_outputs/overnight
#   PYTHON          .venv/bin/python or python

set -euo pipefail

DB="${DB:-data/videos/lancedb}"
MANIFEST="${MANIFEST:-data/chronomagic_proh.parquet}"
CLIPS="${CLIPS:-data/clips}"
TARGET_CLIPS="${TARGET_CLIPS:-2500}"
DL_PARALLEL="${DL_PARALLEL:-12}"
DL_ATTEMPTS="${DL_ATTEMPTS:-40000}"
TRAIN_STEPS="${TRAIN_STEPS:-4000}"
LORA_RANK="${LORA_RANK:-64}"
EVAL_PROMPTS="${EVAL_PROMPTS:-8}"
CKPT_DIR="${CKPT_DIR:-checkpoints/overnight}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-eval_outputs/overnight}"
PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python)"

cd "$(dirname "$0")/.."

run() { echo; echo "=== $1 ==="; shift; "$@"; }

# Confirm prereqs
[ -f "$MANIFEST" ] || {
    echo "ERROR: manifest missing at $MANIFEST" >&2
    echo "       python -m videogen.download_manifest --variant proh --out $MANIFEST" >&2
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. download — keep downloading until we hit $TARGET_CLIPS or burn through attempts
# ─────────────────────────────────────────────────────────────────────────────
current=$(find "$CLIPS" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)
if [ "$current" -lt "$TARGET_CLIPS" ]; then
    run "download → target $TARGET_CLIPS clips (have $current)" \
        "$PYTHON" -m videogen.download_clips \
            --manifest "$MANIFEST" \
            --out "$CLIPS" \
            --limit "$DL_ATTEMPTS" \
            --parallel "$DL_PARALLEL" \
            --max-duration 60 --quality 480 \
            --shuffle
    current=$(find "$CLIPS" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)
    echo "  on-disk after download: $current clips"
else
    echo "[download] already have $current ≥ $TARGET_CLIPS — skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. ingest — only rows whose mp4 exists
# ─────────────────────────────────────────────────────────────────────────────
ingested=$(
    "$PYTHON" -c "
import os, lancedb
try:
    n = len(lancedb.connect('$DB').open_table('videos_raw'))
    print(n)
except Exception:
    print(0)
")
if [ "$ingested" -lt "$current" ]; then
    run "ingest manifest + clips ($current available)" \
        "$PYTHON" -m videogen.ingest_chronomagic \
            --manifest "$MANIFEST" --video-dir "$CLIPS" \
            --require-clips --overwrite --db "$DB"
else
    echo "[ingest] already $ingested rows in '$DB' — skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3-6. backfill tiers 1-4 — backfill_geneva skips filled rows already
# ─────────────────────────────────────────────────────────────────────────────
run "Tier 1 — caption flags"                    "$PYTHON" -m videogen.backfill_geneva --tier 1 --db "$DB"
run "Tier 2 — CLIP / motion / MTScore"          "$PYTHON" -m videogen.backfill_geneva --tier 2 --db "$DB"
run "Tier 3 — UMT5 hidden states"               "$PYTHON" -m videogen.backfill_geneva --columns t5_hidden_states --db "$DB"
run "Tier 3 — Wan-VAE latents (slow)"           "$PYTHON" -m videogen.backfill_geneva --columns vae_latent --db "$DB"
run "Tier 4 — dHash"                            "$PYTHON" -m videogen.backfill_geneva --columns dhash_first_last --db "$DB"

# ─────────────────────────────────────────────────────────────────────────────
# 7. curate views + build indices (idempotent)
# ─────────────────────────────────────────────────────────────────────────────
run "manage_views — curate Tier 1" "$PYTHON" -m videogen.manage_views --action curate   --db "$DB"
run "manage_views — curate Tier 2" "$PYTHON" -m videogen.manage_views --action curate-2 --db "$DB"

# is_duplicate needs the dhash L2 index built first; curate-2 calls build_indices,
# but build_indices skips dhash if the column wasn't there yet — call it again.
run "manage_views — build all indices" "$PYTHON" -m videogen.manage_views --action build-indices --db "$DB"
run "Tier 4 — is_duplicate"            "$PYTHON" -m videogen.backfill_geneva --columns is_duplicate --db "$DB"
run "manage_views — refresh views (apply is_duplicate filter)" \
    "$PYTHON" -m videogen.manage_views --action refresh --db "$DB"

# ─────────────────────────────────────────────────────────────────────────────
# 8. train — pick the strongest available training view
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_VIEW=$(
    "$PYTHON" -c "
import lancedb
tbl = lancedb.connect('$DB')
names = set(tbl.list_tables().tables)
for v in ('phase_transitions_curated_train', 'phase_transitions_train', 'videos_raw'):
    if v in names and len(tbl.open_table(v)) > 0:
        print(v); break
")

echo
echo "training view: $TRAIN_VIEW"
"$PYTHON" -c "
import lancedb
tbl = lancedb.connect('$DB').open_table('$TRAIN_VIEW')
print(f'  rows: {len(tbl)}')
"

# Skip if a step-${TRAIN_STEPS} checkpoint already exists
target_ckpt="$CKPT_DIR/step-$(printf "%06d" $TRAIN_STEPS)"
if [ -d "$target_ckpt" ]; then
    echo "[train] $target_ckpt already exists — skipping"
else
    run "train Wan2.2 LoRA on '$TRAIN_VIEW' ($TRAIN_STEPS steps, r=$LORA_RANK)" \
        "$PYTHON" -m videogen.train_wan22_lora \
            --db "$DB" --train-view "$TRAIN_VIEW" \
            --steps "$TRAIN_STEPS" --batch-size 1 --num-workers 0 \
            --rank "$LORA_RANK" --alpha "$LORA_RANK" --lr 1e-4 \
            --log-every $(( TRAIN_STEPS / 20 )) \
            --save-every $(( TRAIN_STEPS / 4 )) \
            --output-dir "$CKPT_DIR"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 9. eval — baseline vs LoRA on the same prompts, save side-by-side mp4 + html
# ─────────────────────────────────────────────────────────────────────────────
run "eval_compare → $EVAL_OUT_DIR" \
    "$PYTHON" -m videogen.eval_compare \
        --checkpoint "$target_ckpt" \
        --n-prompts "$EVAL_PROMPTS" --num-frames 49 --steps 20 \
        --height 480 --width 704 \
        --save-videos "$EVAL_OUT_DIR" \
        --output "$EVAL_OUT_DIR/compare.json"

echo
echo "=== DONE ==="
echo "Checkpoints:  $CKPT_DIR"
echo "Eval videos:  $EVAL_OUT_DIR"
echo "Open:         file://$PWD/$EVAL_OUT_DIR/index.html"
