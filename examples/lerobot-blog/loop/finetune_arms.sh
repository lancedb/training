#!/usr/bin/env bash
# Fine-tune one arm per GPU, all from the same base checkpoint, same steps, same seed.
# Only --dataset.episodes differs between the arms.
#
#   LANCE_ROOT=... BASE=~/runs/base/checkpoints/010000/pretrained_model SEED=1 STEPS=1500 \
#   ARMS="mined hard text random" loop/finetune_arms.sh
set -euo pipefail
: "${LANCE_ROOT:?}"; : "${BASE:?}"
RUNS=${RUNS:-$HOME/runs}; STEPS=${STEPS:-1500}; BATCH=${BATCH:-32}; WORKERS=${WORKERS:-8}
SEED=${SEED:-1}; ARMS=${ARMS:-"mined hard text random"}; SETS=${SETS:-config/loop_sets.json}
TAG=${TAG:-}; EXTRA_ARGS=${EXTRA_ARGS:-}   # TAG suffixes the run dir; EXTRA_ARGS are appended to lerobot-train
RMAP=$(cat "${RENAME_MAP:-config/rename_map.json}")
gpu=0; pids=()
for arm in $ARMS; do
  EPS=$(python -c "import json;print(json.dumps(json.load(open('$SETS'))['arms']['$arm']))")
  OUT=$RUNS/ft_${arm}${TAG}_s${SEED}
  echo "arm $arm -> GPU $gpu, $(python -c "print(len($EPS))") episodes, $STEPS steps -> $OUT"
  CUDA_VISIBLE_DEVICES=$gpu lerobot-train \
    --dataset.repo_id=lerobot/droid_1.0.1 --dataset.root="$LANCE_ROOT" --dataset.episodes="$EPS" \
    --policy.path="$BASE" --policy.push_to_hub=false --rename_map="$RMAP" \
    --dataloader_multiprocessing_context="${MP_CTX:-spawn}" --accelerator.mixed_precision=bf16 \
    --batch_size="$BATCH" --num_workers="$WORKERS" --steps="$STEPS" --log_freq=100 --save_freq="$STEPS" \
    --eval_steps=0 --tolerance_s=0.005 --wandb.enable=false --seed="$SEED" \
    --output_dir="$OUT" $EXTRA_ARGS > "$RUNS/ft_${arm}${TAG}_s${SEED}.log" 2>&1 &
  pids+=($!); gpu=$((gpu + 1))
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for arm in $ARMS; do
  L="$RUNS/ft_${arm}${TAG}_s${SEED}.log"
  echo "== $arm: $(grep -o 'loss:[0-9.]*' "$L" | head -1) -> $(grep -o 'loss:[0-9.]*' "$L" | tail -1)"
done
exit $fail
