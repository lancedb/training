#!/bin/bash
# usage: MODEL=medium STEPS=13351 BS=32 ACC=2 LR=3e-4 WARM=700 train_large.sh [extra train.py args]
# Chinchilla budgets at 512 x 1024 = 524,288 tok/step: medium 354M -> 7.0B tok = 13,351 steps; large 774M -> 15.5B tok = 29,600 steps
set -e; source /home/ubuntu/runs/env.sh; source /home/ubuntu/venv/bin/activate; cd $SRC
DB=/home/ubuntu/runs/large/db; OUT=/home/ubuntu/runs/large; MODEL=${MODEL:-medium}
torchrun --nproc-per-node 8 train.py --db $DB --model $MODEL --tokenizer hf:gpt2 \
  --pack --compile --batch-size ${BS:-32} --grad-accum ${ACC:-2} --seq-len 1024 --epochs 1 \
  --steps ${STEPS:-13351} --lr-total-steps ${STEPS:-13351} --lr ${LR:-3e-4} --warmup-steps ${WARM:-700} \
  --num-splits 128 --read-batch-size 8 --io-queue-depth 1 --transform-parallelism 2 --num-workers 2 \
  --ckpt-every 2000 --eval-every 2000 --eval-batches 16 --log-every 100 \
  --ckpt-dir $OUT/ckpt_$MODEL "$@" 2>&1 | tee $OUT/train_$MODEL.log
