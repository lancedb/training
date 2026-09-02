# LeRobot loader throughput benchmark

Measures how fast a LeRobot dataset can feed a training step, across storage backends, and
what that costs end to end in training wall clock.

Four files:

| file | what it does |
|---|---|
| `bench_loader.py` | **The measurement.** One process, one backend, one number. |
| `run_matrix.sh` | Sweeps datasets x backends, one process per cell. |
| `train_e2e.sh` | N steps of SmolVLA on 8 GPUs, two data paths, identical config. |
| `parse_train_log.py` | Parses `lerobot-train` logs. Use it; the naive regex is wrong (#9). |
| **`PITFALLS.md`** | **Read this first.** Eight ways this benchmark has produced wrong numbers. |

## Setup

```bash
pip install lerobot lancedb "lerobot-lancedb>=0.3.0"

# torchcodec's CUDA wheels link libnppicc.so.12 but don't declare the dependency
pip install nvidia-npp-cu12
export LD_LIBRARY_PATH=$VENV/lib/python3.12/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH
```

One host-level gotcha that costs a day if you hit it: systemd-logind's default
`RemoveIPC=yes` deletes POSIX shm when the user's last login session closes, which kills
multi-worker DataLoaders mid-run (worker segfaults, `could not unlink the shared memory file
/torch_*`). Fix with `sudo loginctl enable-linger $USER` and `RemoveIPC=no` in
`/etc/systemd/logind.conf.d/`. It is not a Lance fork hazard, though it looks like one.

## Use

```bash
# one backend, one number
python bench_loader.py --backend s3 --repo-id lerobot/droid_1.0.1 \
    --root s3://my-bucket/droid_1.0.1-lance \
    --batch-size 64 --num-workers 8 --num-batches 30 --tolerance-s 0.005

# compare raw IO with matched access patterns (see PITFALLS.md #1)
python bench_loader.py --backend s3     --repo-id ... --root ... --no-shuffle
python bench_loader.py --backend stream --repo-id ... --stream-buffer-size 15000

# the full table
LANCE_ROOT=s3://my-bucket ./run_matrix.sh pusht toto roboturk aloha_mobile_cabinet

# end to end
VENV=~/venv LANCE_ROOT=s3://my-bucket/droid-lance UPSTREAM_ROOT=/data/droid \
  STEPS=10000 ./train_e2e.sh
```

Backends: `local` | `s3` | `bucket` (HF Storage Bucket) | `hub` (map-style via Hub cache,
downloads) | `stream` (`StreamingLeRobotDataset`).

## What it reports

`samples_per_s` is steady state, after `--warmup` batches. Also:

- **`ttfb_s`** — time to first batch. The streaming path's weak spot: 196 s at a 15k buffer
  against 7 s for Lance on local disk.
- **`MB_per_sample`** — bytes over the wire per sample, from `/sys/class/net/*/rx_bytes`.
  Tells you whether you are bandwidth-bound and how much of a read is decode overhead rather
  than pixels.
- **`sys_mem_consumed_gb`** — `MemAvailable` delta. **The memory number to trust**; the
  per-process figure can under-report (PITFALLS.md #4).
- **`shuffle`** / **`stream_buffer_size`** — what was actually asked for. Never compare
  samples/s across rows where these differ.

## Reference results

8xH100, 112 cores, 990 GB RAM, NVMe, S3 same-region. DROID = `lerobot/droid_1.0.1`,
27.6M frames / 95,658 episodes. batch 64, 8 workers.

| backend | samples/s | RAM | shuffle coverage | first batch |
|---|---|---|---|---|
| Lance, local disk | 632.7 | 33 GB | 100% | 7 s |
| Lance, S3 (no download) | 214.5 | 36 GB | 100% | 13 s |
| streaming, 15k buffer | 613.4 | 117 GB | 0.054% | 196 s |
| streaming, 40k buffer | 600.6 | 230 GB | 0.145% | 422 s |

The streaming reader buys shuffle quality with RAM; matching Lance's coverage that way would
need ~14.3 TB. The buffer is per process, so an 8-GPU job pays it eight times.

End to end, the same 10,000-step job run to completion twice, identical except
`--dataset.root` (seed 100, batch 32/GPU x 8 GPUs, `num_workers=4`, unpinned, cold page cache,
no episode filter):

| | Lance · S3 | upstream · local NVMe |
|---|---|---|
| **wall clock** | **5,249 s** (1 h 27 m 29 s) | **7,222 s** (2 h 00 m 22 s) |
| difference | — | **+1,973 s / +33 min → 1.38x** |
| steady rate | 495 samples/s | 361 samples/s |
| step time | 0.519 s | 0.708 s |
| ...waiting on data | **1.7%** | **37.4%** |
| GPU power | 297 W | 256 W |
| final loss @10k | 0.2380 | 0.2380 |

The loss curves are identical to four decimal places at every logged step — same seed, same
samples, same order. One job served two ways, which is what makes the wall clock comparable.
Compute per step matches; the whole 33-minute gap is GPUs waiting on the loader, and the
*lower* power draw on the upstream side is that same idleness measured again.

Note the handicap direction when quoting the ratio: upstream reads local NVMe **after a 384 GB
download**, Lance streams from object storage with **nothing downloaded**, doing a global
shuffle over all 27.6M frames.

## Known limits

- **Object storage was capped by the host NIC, never by S3.** This node sustained 1.6 GB/s;
  feeding 8xH100 needs ~2.7 GB/s (~22 Gbps), under a quarter of a 100 GbE link. Every ceiling
  hit was the box's. Re-run on faster networking to find the real one.
- **Lance's own read overhead is the next thing to fix.** A shuffled sample costs 2.9 MB
  against a ~31 KB physical floor (5.2 KB frame + 10.3 KB group-of-pictures, x3 cameras). The
  decoder cache defaults to 16; raising it to 512 gave +41.8% but is not user-reachable.
- **Small datasets belong on local disk.** On a 20,000-frame ALOHA set, upstream-local beat
  Lance-from-S3 outright (538 vs 452 samples/s) because the whole dataset fits in page cache.
  Lance's advantage is random access at a scale where caching stops working.
