# Run harness for the 8x H100 results

Thin shell wrappers around the example scripts, exactly as run for the
reported numbers (paths assume `/home/ubuntu/runs/...`; `env.sh` holds
credentials and is not committed).

- `prep_small.sh`, `prep_large.sh` — timed ingest -> curate -> Geneva tokenize
- `bench_small.sh` — loader-only `read_batch_size` sweep (see `../LOADER_TUNING.md`)
- `train_small.sh`, `train_large.sh` — the 8-GPU training commands
- `ab_run.sh`, `ab_matrix.sh`, `gpu_sampler.sh`, `gpu_summary.py` — loader A/B with `nvidia-smi` sampling
- `resume_demo.sh` — kill -9 on 8 GPUs, resume on 4
- `loader_gil_repro.py` — single-file, CPU-only reproduction of why the loader knobs matter; needs nothing from this repo
- `results/` — raw outputs these produced
