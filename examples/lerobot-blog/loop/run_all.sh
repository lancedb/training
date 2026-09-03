#!/usr/bin/env bash
# The whole loop, end to end. Safe to re-run: finished stages are skipped.
#
#   LANCE_ROOT=/home/ubuntu/data/droid_lance RUNS=~/runs loop/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LANCE_ROOT:?}"
export RUNS=${RUNS:-$HOME/runs} RENAME_MAP=${RENAME_MAP:-config/rename_map.json}
GPUS=${GPUS:-4}; BASE_STEPS=${BASE_STEPS:-10000}; FT_STEPS=${FT_STEPS:-1500}; K=${K:-300}
SEEDS=${SEEDS:-"1 2"}; ARMS=${ARMS:-"mined hard text random"}
BASE=$RUNS/base/checkpoints/$(printf %06d "$BASE_STEPS")/pretrained_model
mkdir -p "$RUNS" out
log() { echo "[$(date +%T)] $*"; }

[ -f config/loop_subset.json ] || { log "0. subset"; python loop/select_subset.py; }

if [ ! -d "$BASE" ]; then
  log "1. base run: $BASE_STEPS steps on $GPUS GPUs"
  GPUS=$GPUS STEPS=$BASE_STEPS loop/train_base.sh
fi
[ -d "$BASE" ] || { log "base checkpoint missing at $BASE"; exit 1; }

if ! ls out/score/shard_*.parquet >/dev/null 2>&1; then
  log "2. score + embed the subset with the base model"
  for r in $(seq 0 $((GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$r python loop/score_and_embed.py --ckpt "$BASE" --rank "$r" --world "$GPUS" \
      > "out/score_rank$r.log" 2>&1 &
  done; wait
  ls out/score/shard_*.parquet >/dev/null
fi

if ! python -c "import lance,sys; sys.exit(0 if 'err_chunk_mae_base' in lance.dataset('$LANCE_ROOT/frames.lance').schema.names else 1)"; then
  log "3. merge columns into frames.lance"
  python loop/merge_columns.py | tee out/merge.log
fi

[ -f config/loop_sets.json ] || { log "4. build sets"; python loop/build_sets.py --k "$K" | tee out/build_sets.log; }

for s in $SEEDS; do
  need=0
  for a in $ARMS; do [ -d "$RUNS/ft_${a}_s$s/checkpoints/$(printf %06d "$FT_STEPS")/pretrained_model" ] || need=1; done
  if [ $need = 1 ]; then
    log "5. fine-tune arms, seed $s"
    BASE=$BASE SEED=$s STEPS=$FT_STEPS ARMS="$ARMS" loop/finetune_arms.sh
  fi
done

if ! ls out/eval/shard_*.json >/dev/null 2>&1; then
  log "6. evaluate on the holdout"
  CK="base=$BASE"
  for a in $ARMS; do for s in $SEEDS; do
    CK="$CK ${a}_s$s=$RUNS/ft_${a}_s$s/checkpoints/$(printf %06d "$FT_STEPS")/pretrained_model"
  done; done
  for r in $(seq 0 $((GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$r python loop/eval_arms.py --rank "$r" --world "$GPUS" --checkpoints $CK \
      > "out/eval_rank$r.log" 2>&1 &
  done; wait
fi
python loop/report.py | tee out/report.txt
log "done"
