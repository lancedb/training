#!/bin/bash
# Full 4-GPU training runs: usage: run_full_training.sh {lance|base}
set -e
source ~/work/env.sh
cd ~/work/exp
MODE=$1
STEPS=20000
BS=16          # per-rank; effective 64
NW=8
COMMON="--policy.path=lerobot/smolvla_base --policy.input_features=null --policy.output_features=null \
  --policy.device=cuda --policy.push_to_hub=false \
  --steps=$STEPS --batch_size=$BS --num_workers=$NW \
  --log_freq=100 --save_freq=5000 --save_checkpoint=true --env_eval_freq=0 \
  --seed=1000 --wandb.enable=false"

if [ "$MODE" = "lance" ]; then
  OUT=$HOME/work/runs/train_lance
  rm -rf $OUT
  python gpu_monitor.py $HOME/work/logs/gpu_lance.csv 2 &
  MON=$!
  accelerate launch --multi_gpu --num_processes=4 $HOME/work/exp/train_lance.py \
    --dataset.repo_id=local/libero_video --dataset.root=$HOME/work/data/libero_lance_video \
    --output_dir=$OUT $COMMON > ~/work/logs/train_lance.log 2>&1
  RC=$?
  kill $MON
else
  OUT=$HOME/work/runs/train_base
  rm -rf $OUT
  python gpu_monitor.py $HOME/work/logs/gpu_base.csv 2 &
  MON=$!
  accelerate launch --multi_gpu --num_processes=4 $(which lerobot-train) \
    --dataset.repo_id=HuggingFaceVLA/libero --dataset.root=$HOME/work/data/libero_src \
    --output_dir=$OUT $COMMON > ~/work/logs/train_base.log 2>&1
  RC=$?
  kill $MON
fi
echo "TRAIN_${MODE}_EXIT=$RC"
