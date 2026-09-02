#!/usr/bin/env bash
# End-to-end wall clock: N steps of SmolVLA, 8 GPUs, identical config, only the data path
# differs. Reports wall clock, steady samples/s, data-wait share and mean GPU power for each
# side. Loader microbenchmarks tell you the ceiling; this tells you what training actually got.
#
#   VENV=~/venv ENV_FILE=~/aws.sh NPP_LIB=~/venv/lib/python3.12/site-packages/nvidia/npp/lib \
#   LANCE_ROOT=s3://my-bucket/droid-lance UPSTREAM_ROOT=/data/droid \
#   STEPS=10000 ./train_e2e.sh
#
# "data wait" is the share of each step the GPUs spend blocked on the loader. It is a far
# better instrument than nvidia-smi utilisation, which is nearly insensitive to loader stalls.
set -u
: "${VENV:?}"; : "${LANCE_ROOT:?}"; : "${UPSTREAM_ROOT:?}"
ENV_FILE=${ENV_FILE:-/dev/null}; NPP_LIB=${NPP_LIB:-/nonexistent}
STEPS=${STEPS:-10000}; BATCH=${BATCH:-32}; WORKERS=${WORKERS:-4}; GPUS=${GPUS:-8}
source $VENV/bin/activate; source $ENV_FILE
export AWS_DEFAULT_REGION=eu-north-1 AWS_REGION=eu-north-1
export LD_LIBRARY_PATH=$NPP_LIB:$LD_LIBRARY_PATH
S=${OUTDIR:-./e2e-out}; mkdir -p "$S"
L=$S/train_e2e.log; : > "$L"
RMAP=${RENAME_MAP:-"{}"}


run() {  # label dataset-arg...
  local lbl=$1; shift
  echo "### $lbl  start $(date -Is)" >> "$L"
  rm -rf "$S/run_$lbl"
  sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 5
  ( while true; do nvidia-smi --query-gpu=index,power.draw --format=csv,noheader,nounits; sleep 5; done ) > "$S/power_$lbl.csv" &
  local pw=$!; local t0=$(date +%s)
  timeout 18000 torchrun --nproc-per-node=$GPUS --master_port=29$((RANDOM%800+100)) \
    "$(which lerobot-train)" --dataset.repo_id=lerobot/droid_1.0.1 "$@" \
    --policy.path=lerobot/smolvla_base --policy.push_to_hub=false --rename_map="$RMAP" \
    --dataloader_multiprocessing_context=fork --accelerator.mixed_precision=bf16 \
    --batch_size=$BATCH --num_workers=$WORKERS --steps=$STEPS --log_freq=250 --save_freq=5000 \
    --eval_steps=0 --tolerance_s=0.005 --wandb.enable=false \
    --output_dir="$S/run_$lbl" > "$S/$lbl.log" 2>&1
  echo "  exit=$? WALL=$(( $(date +%s) - t0 ))s" >> "$L"; kill $pw 2>/dev/null
  python - "$S/$lbl.log" "$S/power_$lbl.csv" >> "$L" <<'PY'
import sys, re, collections
txt = open(sys.argv[1], errors="ignore").read().replace("\r", "\n")
rows = [r for r in re.findall(r"step:(\d+).*?data_s:([\d.]+).*?step_s:([\d.]+) smp/s:(\d+)", txt) if int(r[0]) >= 500]
if rows:
    d=sum(float(r[1]) for r in rows)/len(rows); s=sum(float(r[2]) for r in rows)/len(rows)
    print(f"  steady {sum(int(r[3]) for r in rows)/len(rows):,.0f} smp/s | data wait {100*d/s:.1f}%")
loss = re.findall(r"step:(\d+).*?loss:([\d.]+)", txt)
if loss: print(f"  loss  step {loss[0][0]}: {loss[0][1]}  ->  step {loss[-1][0]}: {loss[-1][1]}")
try:
    w=collections.defaultdict(list)
    for line in open(sys.argv[2]):
        i,p=line.split(","); w[int(i)].append(float(p))
    v=[x for xs in w.values() for x in xs[len(xs)//4:]]
    print(f"  GPU power mean {sum(v)/len(v):.0f} W  ({len(v)} samples)")
except Exception: pass
PY
  sleep 20
}
run lance   --dataset.root="$LANCE_ROOT"
run upstream --dataset.root="$UPSTREAM_ROOT"
echo "### FULL TRAIN COMPLETE $(date -Is)" >> "$L"
