#!/bin/bash
# Loader-only packed throughput, one rank's shape (32 of 256 splits), read_batch_size sweep.
source /home/ubuntu/runs/env.sh; source /home/ubuntu/venv/bin/activate; cd $SRC
DB=${DB:-/home/ubuntu/runs/small/db}
for rb in 64 16 8 4; do
  python bench_loader.py --db $DB --seq-len 1024 --num-splits 32 --read-batch-size $rb --seconds 30 2>&1 | grep -E "splits=|Error|Traceback"
done
