#!/bin/bash
# Large corpus: 24 FineWeb-Edu sample-100BT shards (~17.4M docs). Stages: ingest curate tokenize (args).
set -e; source /home/ubuntu/runs/env.sh; source /home/ubuntu/venv/bin/activate
DB=/home/ubuntu/runs/large/db; mkdir -p /home/ubuntu/runs/large; cd $SRC
t() { local s=$(date +%s); "$@"; echo "### STAGE_TIME $1 $2 $(( $(date +%s) - s ))s"; }
for stage in "$@"; do case $stage in
  ingest)   t python ingest.py --source fineweb-parquet --sample 100BT --files 24 --db $DB 2>&1 | tee /home/ubuntu/runs/large/ingest.log;;
  curate)   t python curate.py --db $DB --query "photosynthesis carbon dioxide" 2>&1 | tee /home/ubuntu/runs/large/curate.log;;
  tokenize) s=$(date +%s); /home/ubuntu/venv-geneva/bin/python geneva_backfill.py --db $DB --tokenizer hf:gpt2 --concurrency ${GENEVA_WORKERS:-64} --columns input_ids n_tokens 2>&1 | tee /home/ubuntu/runs/large/geneva.log
            echo "### STAGE_TIME geneva tokenize $(( $(date +%s) - s ))s" | tee -a /home/ubuntu/runs/large/geneva.log;;
esac; done
