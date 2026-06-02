#!/usr/bin/env bash
#
# End-to-end pipeline runner.
#
# Stages (each writes a small JSON to $RUN_DIR/<stage>.json so
# bench_pipeline.py can summarise wall-clocks):
#
#   1. ingest train + validation
#   2. tier1 backfill   (CPU UDFs)
#   3. tier2 backfill   (dhash, CPU image decode)
#   4. tier3 backfill   (vision tower + SFT tokens, Geneva GPU UDFs;
#                        TIER3_BACKEND=direct for the single-process fallback)
#   5. baseline layouts (raw_fs, wds, parquet) for bench_dataloader.py
#   6. train            (Qwen2.5-VL-3B + LoRA, cached path)
#   7. eval             (base vs tuned, side-by-side)
#   8. benches          (dataloader 1:1, train-step cached-vs-raw)
#
# Each stage tee's logs to $RUN_DIR/<stage>.log.  Set TRAIN_ROWS to a
# small number for smoke runs.
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR="${RUN_DIR:-runs/e2e_$(date +%Y%m%d_%H%M)}"
DB_TRAIN="${DB_TRAIN:-data/textvqa.lance}"
DB_VAL="${DB_VAL:-data/textvqa_val.lance}"
LAYOUT_DIR="${LAYOUT_DIR:-data/baselines}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
EPOCHS="${EPOCHS:-2}"
BSZ="${BSZ:-2}"
GACC="${GACC:-4}"
TIER3_BS="${TIER3_BS:-16}"
TIER3_BACKEND="${TIER3_BACKEND:-geneva}"   # geneva (default) | direct (single-process fallback)
TRAIN_ROWS="${TRAIN_ROWS:-0}"   # 0 = full split
VAL_ROWS="${VAL_ROWS:-500}"

mkdir -p "$RUN_DIR"
{
    echo "RUN_DIR=$RUN_DIR"
    echo "stack:"
    .venv/bin/pip show pylance 2>/dev/null | grep -E "^(Name|Version)"
    .venv/bin/pip show geneva  2>/dev/null | grep -E "^(Name|Version)"
} > "$RUN_DIR/info"
cat "$RUN_DIR/info"

PY=.venv/bin/python

# Helper: time a command and emit ${RUN_DIR}/<stage>.json with wall_s
stage() {
    local name="$1"; shift
    local extra="${1:-}"
    [[ -n "$extra" ]] && shift || true
    local log="$RUN_DIR/$name.log"
    local t0 t1
    t0=$(date +%s.%N)
    echo "[$(date +%H:%M:%S)] === stage $name ===" | tee -a "$log"
    ( "$@" 2>&1 | tee -a "$log" )
    t1=$(date +%s.%N)
    local wall
    wall=$(awk "BEGIN{print $t1 - $t0}")
    .venv/bin/python -c "import json,sys; print(json.dumps({'wall_s': float(sys.argv[1]), 'rows': $extra}))" "$wall" \
        > "$RUN_DIR/$name.stage.json"
    echo "[$(date +%H:%M:%S)] === stage $name done in ${wall}s ===" | tee -a "$log"
}

# 1) ingest
if [[ ! -d "$DB_TRAIN" ]]; then
    if (( TRAIN_ROWS > 0 )); then
        stage ingest "$TRAIN_ROWS" $PY -m vlm.ingest --dst "$DB_TRAIN" --split train --limit "$TRAIN_ROWS"
    else
        stage ingest 34602 $PY -m vlm.ingest --dst "$DB_TRAIN" --split train
    fi
else
    echo "skipping ingest (DB_TRAIN exists)"
fi

if [[ ! -d "$DB_VAL" ]]; then
    stage ingest_val "$VAL_ROWS" $PY -m vlm.ingest --dst "$DB_VAL" --split validation --limit "$VAL_ROWS"
fi

# 2) tier-1 backfill
stage tier1_backfill "$(.venv/bin/python -c 'import lance;print(lance.dataset("'$DB_TRAIN'").count_rows())')" \
    $PY -m vlm.backfill_geneva --db "$DB_TRAIN" --tier 1 --concurrency 4

# 3) tier-2 backfill
stage tier2_backfill "$(.venv/bin/python -c 'import lance;print(lance.dataset("'$DB_TRAIN'").count_rows())')" \
    $PY -m vlm.backfill_geneva --db "$DB_TRAIN" --tier 2 --concurrency 4

# 4) tier-3 backfill — Geneva by default (showcases the distributed
#    GPU-UDF path); set TIER3_BACKEND=direct for the single-process fallback.
TIER3_ROWS="$(.venv/bin/python -c 'import lance;print(lance.dataset("'$DB_TRAIN'").count_rows())')"
if [[ "$TIER3_BACKEND" == "direct" ]]; then
    stage tier3_backfill "$TIER3_ROWS" \
        $PY -m vlm.backfill_direct --db "$DB_TRAIN" --batch-size "$TIER3_BS"
else
    stage tier3_backfill "$TIER3_ROWS" \
        $PY -m vlm.backfill_geneva --db "$DB_TRAIN" --tier 3 --concurrency 2
fi

# 5) baseline layouts
stage layouts_export "$(.venv/bin/python -c 'import lance;print(lance.dataset("'$DB_TRAIN'").count_rows())')" \
    $PY -c "from vlm.dataloader_baselines import prepare_baseline_layouts; \
            prepare_baseline_layouts('$DB_TRAIN', '$LAYOUT_DIR')"

# 6) train
stage train "$(.venv/bin/python -c 'import lance;print(lance.dataset("'$DB_TRAIN'").count_rows())')" \
    $PY -m vlm.train_qwen25vl_lora --db "$DB_TRAIN" \
        --out "$RUN_DIR/lora" \
        --batch-size "$BSZ" --grad-accum "$GACC" --epochs "$EPOCHS"

# 7) eval — base + tuned, side-by-side
stage eval "$EVAL_LIMIT" \
    $PY -m vlm.eval --db "$DB_VAL" \
        --adapter "$RUN_DIR/lora/lora" \
        --out "$RUN_DIR/eval" \
        --limit "$EVAL_LIMIT" --mode both --side-by-side-k 12

# 8) benches
stage bench_dataloader 0 \
    $PY -m bench.bench_dataloader --db "$DB_TRAIN" --layout-dir "$LAYOUT_DIR" \
        --bs 8 --nw 4 --batches 200 --skip-prep \
        --out "$RUN_DIR/bench_dataloader.json"

stage bench_train_step 0 \
    $PY -m bench.bench_train_step --db "$DB_TRAIN" --bs 2 --steps 30 --mode both \
        --out "$RUN_DIR/bench_train_step.json"

# 9) summarise
$PY -m bench.bench_pipeline --run-dir "$RUN_DIR" --out "$RUN_DIR/pipeline.json"

echo
echo "DONE — outputs at $RUN_DIR"
