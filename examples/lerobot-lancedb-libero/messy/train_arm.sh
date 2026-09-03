#!/usr/bin/env bash
# Train one arm: SmolVLA full fine-tune on a LIBERO Lance root, optionally restricted to an
# episode list. Same recipe for every arm; only --dataset.root / --dataset.episodes differ.
#
#   ROOT=/data/libero_messy_lance EPISODES_JSON=/data/libero_messy_lance/curation.json \
#   NAME=curated GPU=2 messy/train_arm.sh
set -euo pipefail
: "${ROOT:?}"; : "${NAME:?}"
RUNS=${RUNS:-$HOME/runs_libero}; GPU=${GPU:-0}; STEPS=${STEPS:-40000}; BATCH=${BATCH:-32}
WORKERS=${WORKERS:-8}; SEED=${SEED:-1000}; SAVE_FREQ=${SAVE_FREQ:-10000}
EPISODES_JSON=${EPISODES_JSON:-}   # curation.json with "curated_episodes", or empty for all
EPS_ARG=()
if [ -n "$EPISODES_JSON" ]; then
  EPS=$(python -c "import json;print(json.dumps(json.load(open('$EPISODES_JSON'))['curated_episodes']))")
  EPS_ARG=(--dataset.episodes="$EPS")
  echo "arm $NAME: $(python -c "print(len($EPS))") episodes from $EPISODES_JSON"
else
  echo "arm $NAME: all episodes of $ROOT"
fi
export MUJOCO_GL=egl
mkdir -p "$RUNS"
CUDA_VISIBLE_DEVICES=$GPU lerobot-train \
  --dataset.repo_id=HuggingFaceVLA/libero --dataset.root="$ROOT" "${EPS_ARG[@]}" \
  --policy.path=lerobot/smolvla_base --policy.input_features=null --policy.output_features=null \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false --policy.push_to_hub=false \
  --dataloader_multiprocessing_context=spawn --batch_size="$BATCH" --num_workers="$WORKERS" \
  --steps="$STEPS" --log_freq=200 --save_freq="$SAVE_FREQ" --eval_steps=0 --wandb.enable=false --seed="$SEED" \
  --output_dir="$RUNS/$NAME" > "$RUNS/$NAME.log" 2>&1
echo "arm $NAME done: $(grep -o 'loss:[0-9.]*' "$RUNS/$NAME.log" | head -1) -> $(grep -o 'loss:[0-9.]*' "$RUNS/$NAME.log" | tail -1)"
