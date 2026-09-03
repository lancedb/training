#!/usr/bin/env bash
# Round 2: same loop, sane fine-tuning schedule. Peak LR 1e-5 (10x lower than the preset),
# 100 warmup steps, cosine to 1e-6 over the run. Arms: mined, random (300 episodes each),
# mixed (mined + 300 random), pool (all 2,000 pool episodes).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LANCE_ROOT:?}"
export RUNS=${RUNS:-$HOME/runs} RENAME_MAP=${RENAME_MAP:-config/rename_map.json}
GPUS=${GPUS:-4}; FT_STEPS=${FT_STEPS:-1500}; SEEDS=${SEEDS:-"1 2"}; TAG=${TAG:-_lowlr}
ARMS=${ARMS:-"mined random mixed pool"}
BASE=${BASE:-$RUNS/base/checkpoints/010000/pretrained_model}
EXTRA_ARGS="--policy.optimizer_lr=1e-5 --policy.scheduler_warmup_steps=100 --policy.scheduler_decay_steps=$FT_STEPS --policy.scheduler_decay_lr=1e-6"
log() { echo "[$(date +%T)] $*"; }
for s in $SEEDS; do
  need=0
  for a in $ARMS; do [ -d "$RUNS/ft_${a}${TAG}_s$s/checkpoints/$(printf %06d "$FT_STEPS")/pretrained_model" ] || need=1; done
  if [ $need = 1 ]; then
    log "5. fine-tune arms (round 2, $TAG), seed $s"
    BASE=$BASE SEED=$s STEPS=$FT_STEPS ARMS="$ARMS" TAG=$TAG EXTRA_ARGS="$EXTRA_ARGS" loop/finetune_arms.sh
  fi
done
OUT=out/eval$TAG
if ! ls $OUT/shard_*.json >/dev/null 2>&1; then
  log "6. evaluate on the holdout (round 2)"
  CK="base=$BASE"
  for a in $ARMS; do for s in $SEEDS; do
    CK="$CK ${a}_s$s=$RUNS/ft_${a}${TAG}_s$s/checkpoints/$(printf %06d "$FT_STEPS")/pretrained_model"
  done; done
  for r in $(seq 0 $((GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$r python loop/eval_arms.py --rank "$r" --world "$GPUS" --out-dir $OUT --checkpoints $CK \
      > "out/eval${TAG}_rank$r.log" 2>&1 &
  done; wait
fi
python loop/report.py --shards "$OUT/shard_*.json" --out $OUT/report.json | tee out/report$TAG.txt
log "done"
