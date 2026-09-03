#!/usr/bin/env bash
# The whole messy-data experiment: three arms trained in parallel (one GPU each), then evaluated
# closed-loop. Re-runnable; finished stages are skipped.
#
#   CLEAN=/data/libero_lance MESSY=/data/libero_messy_lance messy/run_messy.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CLEAN:?}"; : "${MESSY:?}"
export RUNS=${RUNS:-$HOME/runs_libero} STEPS=${STEPS:-40000} BATCH=${BATCH:-32}
N_EVAL=${N_EVAL:-10}
log() { echo "[$(date +%T)] $*"; }
CK() { echo "$RUNS/$1/checkpoints/$(printf %06d "$STEPS")/pretrained_model"; }

[ -f "$MESSY/curation.json" ] || { log "detect defects on the messy table"; CUDA_VISIBLE_DEVICES=0 python messy/detect.py --root "$MESSY" --z 3.0 | tee "$RUNS/detect.log"; }

pids=()
[ -d "$(CK clean)" ]   || { log "train clean";   ROOT=$CLEAN NAME=clean   GPU=0 messy/train_arm.sh & pids+=($!); }
[ -d "$(CK messy)" ]   || { log "train messy";   ROOT=$MESSY NAME=messy   GPU=1 messy/train_arm.sh & pids+=($!); }
[ -d "$(CK curated)" ] || { log "train curated"; ROOT=$MESSY NAME=curated GPU=2 EPISODES_JSON=$MESSY/curation.json messy/train_arm.sh & pids+=($!); }
fail=0; for p in "${pids[@]:-}"; do [ -n "$p" ] && { wait "$p" || fail=1; }; done
[ $fail = 0 ] || { log "a training arm failed"; exit 1; }

pids=(); g=0
for a in clean messy curated; do
  [ -f "$RUNS/eval_$a/eval_info.json" ] || { log "eval $a"; CKPT=$(CK $a) NAME=$a GPU=$g N=$N_EVAL messy/eval_arm.sh > "$RUNS/eval_${a}.summary" 2>&1 & pids+=($!); }
  g=$((g + 1))
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done
cat "$RUNS"/eval_*.summary 2>/dev/null
log "done"
