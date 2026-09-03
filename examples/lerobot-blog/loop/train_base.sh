#!/usr/bin/env bash
# The base run: SmolVLA fine-tuned on DROID with the 200 random holdout episodes excluded.
# Every later arm starts from this checkpoint, so all arms share its normalization statistics.
#
#   LANCE_ROOT=/data/droid_lance RUNS=~/runs GPUS=4 STEPS=10000 loop/train_base.sh
set -euo pipefail
: "${LANCE_ROOT:?}"
RUNS=${RUNS:-$HOME/runs}; GPUS=${GPUS:-4}; STEPS=${STEPS:-10000}; BATCH=${BATCH:-32}
WORKERS=${WORKERS:-4}; SEED=${SEED:-100}; SAVE_FREQ=${SAVE_FREQ:-2500}
SUBSET=${SUBSET:-config/loop_subset.json}
HOLDOUT=$(python -c "import json;print(json.dumps(json.load(open('$SUBSET'))['holdout']))")
RMAP=$(cat "${RENAME_MAP:-config/rename_map.json}")
mkdir -p "$RUNS"
OUT=$RUNS/base
echo "base run -> $OUT  ($GPUS GPUs, $STEPS steps, batch $BATCH/GPU, excluding $(python -c "print(len($HOLDOUT))") holdout episodes)"
torchrun --nproc-per-node="$GPUS" --master_port=29501 "$(which lerobot-train)" \
  --dataset.repo_id=lerobot/droid_1.0.1 --dataset.root="$LANCE_ROOT" \
  --dataset.exclude_episodes="$HOLDOUT" \
  --policy.path=lerobot/smolvla_base --policy.push_to_hub=false --rename_map="$RMAP" \
  --dataloader_multiprocessing_context=fork --accelerator.mixed_precision=bf16 \
  --batch_size="$BATCH" --num_workers="$WORKERS" --steps="$STEPS" --log_freq=100 --save_freq="$SAVE_FREQ" \
  --eval_steps=0 --tolerance_s=0.005 --wandb.enable=false --seed="$SEED" \
  --output_dir="$OUT" 2>&1 | tee "$RUNS/base.log"
