#!/usr/bin/env bash
# Round 3: the data-acquisition framing. The base model never sees the 2,000-episode pool.
# Its errors on the pool decide which NEW episodes to add: 300 mined via the index, 300 with
# the highest error (no index), 300 random, or all 2,000. Moderate LR (3e-5 peak, 100 warmup).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LANCE_ROOT:?}"
export RUNS=${RUNS:-$HOME/runs} RENAME_MAP=${RENAME_MAP:-config/rename_map.json}
GPUS=${GPUS:-4}; BASE_STEPS=${BASE_STEPS:-10000}; FT_STEPS=${FT_STEPS:-1500}; K=${K:-300}
SEEDS=${SEEDS:-"1 2"}; ARMS=${ARMS:-"mined hard random pool"}; TAG=_nopool
RUN_NAME=base_nopool; SETS=config/loop_sets$TAG.json
BASE=$RUNS/$RUN_NAME/checkpoints/$(printf %06d "$BASE_STEPS")/pretrained_model
LR=${LR:-3e-5}
EXTRA_ARGS="--policy.optimizer_lr=$LR --policy.scheduler_warmup_steps=100 --policy.scheduler_decay_steps=$FT_STEPS --policy.scheduler_decay_lr=1e-6"
log() { echo "[$(date +%T)] $*"; }

if [ ! -d "$BASE" ]; then
  log "1. base run without the pool: $BASE_STEPS steps on $GPUS GPUs"
  GPUS=$GPUS STEPS=$BASE_STEPS EXCLUDE_KEYS=holdout,pool RUN_NAME=$RUN_NAME loop/train_base.sh
fi
[ -d "$BASE" ] || { log "base checkpoint missing at $BASE"; exit 1; }

if ! ls out/score$TAG/shard_*.parquet >/dev/null 2>&1; then
  log "2. score the subset with the no-pool base model (embeddings already on the table)"
  for r in $(seq 0 $((GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$r python loop/score_and_embed.py --ckpt "$BASE" --rank "$r" --world "$GPUS" \
      --no-embed --out-dir out/score$TAG > "out/score${TAG}_rank$r.log" 2>&1 &
  done; wait
  ls out/score$TAG/shard_*.parquet >/dev/null
fi

if ! python -c "import lance,sys; sys.exit(0 if 'err_chunk_mae$TAG' in lance.dataset('$LANCE_ROOT/frames.lance').schema.names else 1)"; then
  log "3. merge error columns (suffix $TAG)"
  python loop/merge_columns.py --shards "out/score$TAG/shard_*.parquet" --suffix $TAG --no-embed | tee out/merge$TAG.log
fi

[ -f $SETS ] || { log "4. build sets from the no-pool errors"; python loop/build_sets.py --suffix $TAG --k "$K" --out $SETS | tee out/build_sets$TAG.log; }
python - <<PY
import json; s=json.load(open("$SETS")); sub=json.load(open("config/loop_subset.json"))
s["arms"]["pool"]=sorted(sub["pool"]); json.dump(s, open("$SETS","w"))
print({k: len(v) for k, v in s["arms"].items()})
PY

for s in $SEEDS; do
  need=0
  for a in $ARMS; do [ -d "$RUNS/ft_${a}${TAG}_s$s/checkpoints/$(printf %06d "$FT_STEPS")/pretrained_model" ] || need=1; done
  if [ $need = 1 ]; then
    log "5. fine-tune arms on NEW episodes (round 3), seed $s"
    BASE=$BASE SEED=$s STEPS=$FT_STEPS ARMS="$ARMS" TAG=$TAG SETS=$SETS EXTRA_ARGS="$EXTRA_ARGS" loop/finetune_arms.sh
  fi
done

OUT=out/eval$TAG
if ! ls $OUT/shard_*.json >/dev/null 2>&1; then
  log "6. evaluate on the holdout (round 3)"
  CK="base=$BASE"
  for a in $ARMS; do for s in $SEEDS; do
    CK="$CK ${a}_s$s=$RUNS/ft_${a}${TAG}_s$s/checkpoints/$(printf %06d "$FT_STEPS")/pretrained_model"
  done; done
  for r in $(seq 0 $((GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$r python loop/eval_arms.py --rank "$r" --world "$GPUS" --sets $SETS --out-dir $OUT --checkpoints $CK \
      > "out/eval${TAG}_rank$r.log" 2>&1 &
  done; wait
fi
python loop/report.py --shards "$OUT/shard_*.json" --sets $SETS --out $OUT/report.json | tee out/report$TAG.txt
log "done"
