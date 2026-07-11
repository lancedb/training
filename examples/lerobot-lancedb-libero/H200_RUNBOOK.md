# Runbook: reproduce the e2e speed benchmark on a faster GPU box (H200 / B200 / B300)

This is a self-contained prompt for an AI agent (or a careful human) on a **fresh GPU
machine** to reproduce the base-LeRobot vs lerobot-lancedb end-to-end training comparison
and measure how the delta grows on faster accelerators. It assumes nothing from the
original machine except this repository.

## Hypothesis being tested

On the original 4×H100 box (104 threads), the e2e training speedup of the Lance video
format over parquet+mp4 was a function of how data-bound the run is: ~1.06× at 26
vCPU/GPU up to ~2.6× at 4 vCPU/GPU, ceiling ≈ the dataloader per-CPU cost ratio
(~2.4–2.8×). **Faster GPUs raise sample demand while CPU stays flat, so the same recipe
at the same CPU budget should show a larger delta on H200/B200/B300 — approaching the
ceiling at higher (more flattering) CPU budgets than on H100.** The measured mechanism:
raising demand at fixed CPU (bs32→bs64 on H100) moved the delta 1.98×→2.31×.

## Environment setup (Ubuntu 22.04+, NVIDIA driver present)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/work && cd ~/work
uv venv --python 3.12 venv && source venv/bin/activate
export CMAKE_POLICY_VERSION_MINIMUM=3.5   # egl-probe under cmake 4
uv pip install "lerobot[dataset]==0.6.0" "lerobot-lancedb>=0.2.1" "torchcodec==0.10.*" accelerate
# torchcodec needs FFmpeg shared libs:
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba create -y -p ~/work/ffmpeg7 -c conda-forge "ffmpeg=7.*"
export LD_LIBRARY_PATH=$HOME/work/ffmpeg7/lib:$LD_LIBRARY_PATH
# sanity: python -c "from torchcodec.decoders import VideoDecoder; import lerobot_lancedb; print('ok')"
```

Record: `nvidia-smi` (GPU model/count), `lscpu` (threads, cores, model), disk type.

## Datasets (same two as the blog)

```bash
hf download lerobot/droid_100 --repo-type dataset --local-dir ~/work/data/droid100
lerobot-convert-to-lance-video --repo-id=lerobot/droid_100 \
  --src-root=$HOME/work/data/droid100 --output=$HOME/work/data/droid100_lance --table-name=droid100
hf download lerobot/aloha_sim_transfer_cube_human --repo-type dataset --local-dir ~/work/data/aloha_sim
lerobot-convert-to-lance-video --repo-id=lerobot/aloha_sim_transfer_cube_human \
  --src-root=$HOME/work/data/aloha_sim --output=$HOME/work/data/aloha_sim_lance --table-name=aloha_sim
```

(If the blog's final winner was RoboTwin, also mirror the `lerobot/robotwin_unified`
subset procedure from `scripts/` — trim meta to downloaded episodes before converting.)

## The workloads (identical to the blog; copy `scripts/train_lance.py` and
## `scripts/smoke_cpubudget.sh` from this example)

Grid to run — for N_GPUS ∈ {all available}, for each dataset ∈ {droid, aloha_sim}:

1. **CPU-budget training smokes** (300 steps, bs64, nw=8), budgets = {4, 8, 12, 26}
   vCPU/GPU via `taskset -c 0-$((N*NGPUS-1))`. Use `scripts/smoke_cpubudget.sh <threads>
   <preset> 64 300 8` — it runs base then lance identically and prints steady-state
   samples/s from the training log. Also run one bs128 point at 8 vCPU/GPU: on faster
   GPUs, larger batches raise demand further; record whether the delta grows (H100
   evidence says yes until the loader ceiling).
2. **Full wall-clock pair** at the two most interesting budgets (expected: 4 and 8
   vCPU/GPU): 20k steps, bs64, nw8, `save_checkpoint=true`, `time` the whole
   `accelerate launch`. Same seed (1000) both. Confirm final loss matches between
   backends (parity check).
3. **Loader-only benches** for context: `scripts/bench_throughput.py --preset
   {droid,aloha_sim} --backend {video,lance} --num-workers {4,8,16}` on the idle box.

## What to record (append to RESULTS-<gpu>.md)

- Per run: samples/s at final step, data_s, updt_s, wall seconds, final loss.
- The key comparison vs H100: at each CPU budget, delta(H200) vs delta(H100) — the
  hypothesis predicts delta grows at every budget, most visibly at 8–12 vCPU/GPU where
  H100 showed 1.2–1.9×.
- GPU power draw (nvidia-smi loop) if convenient — starved runs idle at low watts.

## Gotchas learned on the original box

- Never `pip install` into the venv while runs are live. Never use `pgrep -f` patterns
  that appear in your own command line (use `[b]racket` trick).
- `taskset` on the `accelerate launch` command caps the whole process tree; SMT siblings
  on this Xeon were adjacent CPU ids (verify with `lscpu -e=CPU,CORE`).
- Warmup: ignore the first ~100 steps (spawn workers take ~7 s each for Lance).
- lerobot logs steps as `20K` — parse accordingly. `smp/s` in its log is effective
  training samples/s (global).

## Reporting

Produce a table mirroring the blog's CPU-budget chart with an extra column per GPU type,
and one sentence per dataset: "at X vCPU/GPU, the delta moved from A× (H100) to B×
(H200)". If B ≤ A anywhere, investigate CPU model differences (per-core decode speed
shifts both supplies) before concluding.
