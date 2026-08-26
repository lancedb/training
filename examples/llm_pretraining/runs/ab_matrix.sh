#!/bin/bash
# usage: ab_matrix.sh <model> <steps> <results.md> <spec>...   spec = tag:mode:path
#   mode: corpus (lance on-the-fly pack) | lance (blocks table) | mosaic | parquet-random | parquet-seq
source /home/ubuntu/runs/env.sh
MODEL=$1; STEPS=$2; RES=$3; shift 3
COMMON="--model $MODEL --batch-size ${BS:-32} --grad-accum ${ACC:-2} --num-splits 128 --io-queue-depth 1 --transform-parallelism 2 --num-workers 2 --eval-batches 2"
for spec in "$@"; do
  IFS=: read -r tag mode path <<<"$spec"; path=${spec#*:*:}
  case $mode in
    corpus) args="--db $path --pack $COMMON";;
    *)      args="--blocks-mode $mode --blocks-path $path --db /home/ubuntu/runs/small/db $COMMON";;
  esac
  echo "=== $tag ($mode) $path"
  out=$(TMO=${TMO:-1500} /home/ubuntu/runs/ab_run.sh "$tag" "$STEPS" $args 2>&1 | grep -E "^\[|gpu util|Traceback|Error" | tail -3)
  echo "$out"
  echo "| $tag | $mode | $path | $(echo "$out" | grep -oE 'mean [0-9,]+ tok/s  mfu [0-9.]+%' ) | $(echo "$out" | grep -oE 'gpu util: mean [0-9.]+%' ) |" >> "$RES"
done
