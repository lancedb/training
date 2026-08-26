#!/bin/bash
# kill -9 a packed 8-GPU run at ~step 260 (ckpt at 200), resume the SAME run on 4 GPUs (bs 64 keeps 512 seqs/step),
# then compare the post-resume batches against an uninterrupted 8-GPU reference via the training log losses.
source /home/ubuntu/runs/env.sh; source /home/ubuntu/venv/bin/activate; cd $SRC
OUT=/home/ubuntu/runs/small/resume_demo; rm -rf $OUT; mkdir -p $OUT
COMMON="--db /home/ubuntu/runs/small/db --model small --tokenizer hf:gpt2 --pack --compile --seq-len 1024 --num-splits 128 --io-queue-depth 1 --transform-parallelism 2 --num-workers 2 --eval-batches 2 --log-every 10 --lr-total-steps 4635 --shuffle-seed 42"
echo "### reference: uninterrupted 8 GPUs, 400 steps"
torchrun --nproc-per-node 8 train.py $COMMON --batch-size 32 --grad-accum 2 --steps 400 --ckpt-every 100000 --ckpt-dir $OUT/ref > $OUT/ref.log 2>&1
echo "### run A: 8 GPUs, ckpt every 200, kill -9 at step ~260"
torchrun --nproc-per-node 8 train.py $COMMON --batch-size 32 --grad-accum 2 --steps 400 --ckpt-every 200 --ckpt-dir $OUT/ckpt > $OUT/a.log 2>&1 &
TR=$!
until grep -q "step 260/" $OUT/a.log; do sleep 2; done
for p in $(pgrep -f "[t]rain.py --db /home/ubuntu/runs/small/db --model small.*ckpt-every 200"); do kill -9 $p; done; kill -9 $TR 2>/dev/null; sleep 3
echo "killed at: $(grep -oE 'step 2[0-9]0/' $OUT/a.log | tail -1)"; ls $OUT/ckpt
echo "### run B: resume on 4 GPUs (bs 64 x accum 2 = same 512-seq global step)"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 train.py $COMMON --batch-size 64 --grad-accum 2 --steps 400 --ckpt-every 100000 --ckpt-dir $OUT/ckpt --resume auto > $OUT/b.log 2>&1
grep -E "resumed|final" $OUT/b.log
python3 - <<'PY'
import re
def losses(p):
    return {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"step (\d+)/\d+ \| loss ([\d.]+)", open(p).read())}
ref, b = losses("/home/ubuntu/runs/small/resume_demo/ref.log"), losses("/home/ubuntu/runs/small/resume_demo/b.log")
common = sorted(set(ref) & set(b))
print("step | ref loss (8 GPUs, uninterrupted) | resumed loss (4 GPUs)")
for s in common: print(f"{s:4d} | {ref[s]:.4f} | {b[s]:.4f}")
PY
