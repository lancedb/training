#!/bin/bash
# Small corpus: 2.4M FineWeb-Edu docs (same as the 4xH100 blog run), timed per stage.
set -e; source /home/ubuntu/runs/env.sh; source /home/ubuntu/venv/bin/activate
DB=/home/ubuntu/runs/small/db; mkdir -p /home/ubuntu/runs/small; cd $SRC
t() { local s=$(date +%s); "$@"; echo "### STAGE_TIME $1 $2 $(( $(date +%s) - s ))s"; }
t python ingest.py --source fineweb-parquet --sample 10BT --files 4 --rows 2400000 --db $DB   2>&1 | tee /home/ubuntu/runs/small/ingest.log
t python curate.py --db $DB --query "photosynthesis carbon dioxide" 2>&1 | tee /home/ubuntu/runs/small/curate.log
s=$(date +%s)
/home/ubuntu/venv-geneva/bin/python geneva_backfill.py --db $DB --tokenizer hf:gpt2 --concurrency 32 --columns input_ids n_tokens 2>&1 | tee /home/ubuntu/runs/small/geneva.log
echo "### STAGE_TIME geneva tokenize $(( $(date +%s) - s ))s" | tee -a /home/ubuntu/runs/small/geneva.log
