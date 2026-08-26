#!/bin/bash
# usage: ab_run.sh <tag> <steps> <train.py args...>
# 8xH100 GPT run with GPU-util sampling; prints tok/s + MFU over the steady window.
source /home/ubuntu/runs/env.sh; source /home/ubuntu/venv/bin/activate; cd $SRC
TAG=$1; STEPS=$2; shift 2
OUT=/home/ubuntu/runs/ab; mkdir -p $OUT
/home/ubuntu/runs/gpu_sampler.sh $OUT/$TAG.gpu.csv & SAMP=$!
timeout ${TMO:-900} torchrun --nproc-per-node 8 train.py --tokenizer hf:gpt2 --seq-len 1024 --compile \
  --steps $STEPS --log-every 25 --ckpt-every 100000 --ckpt-dir $OUT/ckpt_$TAG "$@" > $OUT/$TAG.log 2>&1
kill $SAMP 2>/dev/null
python - "$OUT/$TAG.log" "$TAG" <<'PY'
import re,sys,statistics
log=open(sys.argv[1]).read()
rows=[(int(m.group(1)),float(m.group(2).replace(',','')),float(m.group(3))) for m in re.finditer(r"step (\d+)/\d+ \| loss [\d.]+ \| ([\d,]+) tok/s \| mfu ([\d.]+)%", log)]
steady=[r for r in rows if r[0]>100]
if steady:
    print(f"[{sys.argv[2]}] steps {steady[0][0]}-{steady[-1][0]}: mean {statistics.mean(r[1] for r in steady):,.0f} tok/s  mfu {statistics.mean(r[2] for r in steady):.1f}%  (min {min(r[1] for r in steady):,.0f})")
else:
    print(f"[{sys.argv[2]}] NO STEADY ROWS"); print(log[-1500:])
PY
python /home/ubuntu/runs/gpu_summary.py $OUT/$TAG.gpu.csv 45
