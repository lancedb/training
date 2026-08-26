#!/bin/bash
# GPT-2 124M, Chinchilla 1 epoch over the 2.4M-doc table, 8xH100, packed + compiled.
# Same global batch as the 4xH100 blog run (512 x 1024 = 524k tok/step): bs32 x accum2 x 8 ranks.
set -e; source /home/ubuntu/runs/env.sh; source /home/ubuntu/venv/bin/activate; cd $SRC
DB=/home/ubuntu/runs/small/db; OUT=/home/ubuntu/runs/small
torchrun --nproc-per-node 8 train.py --db $DB --model small --tokenizer hf:gpt2 \
  --pack --compile --batch-size 32 --grad-accum 2 --seq-len 1024 --epochs 1 \
  --num-splits 128 --read-batch-size ${RB:-8} --io-queue-depth 1 --transform-parallelism 2 --ckpt-every 1000 --eval-every 1500 --eval-batches 16 \
  --num-workers ${NW:-2} --log-every 50 --ckpt-dir $OUT/ckpt_main "$@" 2>&1 | tee $OUT/train_main${NW:+_nw$NW}.log
