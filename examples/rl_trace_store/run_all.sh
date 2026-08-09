#!/usr/bin/env bash
# Reproduce everything: storage benchmark matrix + live pipeline demo.
# Needs: pip install -r requirements.txt  (and ~4 GB free disk)
set -euo pipefail
cd "$(dirname "$0")"

python bench.py --steps 16 --per-step 16 --root ./bench_data --results ./results
python demo_pipeline.py --producers 3 --batches 6 --batch-size 8 --root ./demo_data
