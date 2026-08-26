# StreamingDataset tuning notes (packed mode, lancedb 0.38.0b10)

Measured on the 2.4M-doc FineWeb-Edu table (GPT-2 tokens, `seq_len=1024`),
local disk (virtio, 80k IOPS / 5 GB/s random 64k reads — not the bottleneck),
112-core host, 8xH100. Loader-only, no GPU in the loop
(`bench_loader.py` / `runs/loader_knobs.py`). One "rank shape" = the splits one
of 8 ranks owns.

## The three knobs that matter

| knob | library default | what we use | why |
|---|---|---|---|
| `io_queue_depth` (was `prefetch_batches`) | 4 | **1** | I/O threads per rank = `owned_splits x io_queue_depth`. 32 splits x 4 = 128 threads *per rank*; storage latency on local disk is ~1-5 ms per take, so extra depth buys nothing and the threads contend for the GIL and the tokio bridge. |
| `transform_parallelism` | `os.cpu_count()` (112) | **2** | Packing's transform (`RecordBatch.to_pylist()`) and the packer itself are pure-Python, GIL-bound work. 8 ranks x 112 threads = 896 transform threads on 112 cores. 1-2 threads per rank saturate it. |
| `num_splits` (per-rank share) | `world_size` | **128 total = 16/rank** | `_commit_pack_state()` snapshots *every* owned split's token buffer after *every* emitted block, so per-block cost grows linearly with owned splits. Keep it as small as elasticity needs allow (must be divisible by every world size you'll resume on; global batch must be a multiple of it). |
| `read_batch_size` | 64 | **8** | Rows per take. With the thread fix, 8 ≈ 4 > 16 > 32; the old "super-linear per-take cost" finding was mostly thread contention, but small takes still win on local disk. On S3 larger takes amortise request latency — measure. |

## Measurements (packed, `read_batch_size=8`, one rank shape)

| config | blocks/s | tok/s |
|---|---|---|
| defaults (`io_queue_depth=4`, `transform_parallelism=112`), 32 splits | 154 | 158k |
| `transform_parallelism=8` | 257 | 263k |
| `transform_parallelism=2` | 258 | 264k |
| `io_queue_depth=1` | 457 | 468k |
| `io_queue_depth=1, transform_parallelism=4` | 685 | 701k |
| `io_queue_depth=1, transform_parallelism=2` | 776 | 795k |
| `io_queue_depth=1, transform_parallelism=1` | 769 | 787k |
| `io_queue_depth=2, transform_parallelism=4` | 447 | 458k |
| `io_queue_depth=1, transform_parallelism=4, read_batch_size=4` | 718 | 735k |
| `io_queue_depth=1, transform_parallelism=4, read_batch_size=16` | 624 | 639k |
| `io_queue_depth=1, transform_parallelism=4, read_batch_size=32` | 461 | 472k |
| `io_queue_depth=1, transform_parallelism=4`, **16 splits** | 937 | 960k |
| `io_queue_depth=1, transform_parallelism=4`, **64 splits** | 496 | 508k |
| **8 concurrent rank processes**, 128 splits (16/rank), `ioq=1, tx=2` | 8 x ~583 | **~4.8M aggregate** |

For reference the pre-merge `ayush/seq-packing` overlay measured 419k tok/s at
`rb=8`, 32 splits, on the previous 4xH100 box (its defaults were different).

## Where the remaining time goes (cProfile, main thread, tuned config, 10 s)

```
tottime  function
 4.22s   streaming.py:_commit_pack_state   <- 42%: copies all owned splits' buffers per block
 0.65s   streaming.py:arrow_tokens         <- to_pylist() of the token column
 0.45s   torch.tensor                      <- list -> LongTensor per block
 0.34s   streaming.py:_emit_block
```

Upstream suggestions, in order of payoff:

1. `_commit_pack_state` should snapshot only the split that just emitted (or
   keep buffers as immutable tuples / numpy arrays so the snapshot is a
   reference, not a copy). This alone is ~2x on the main thread.
2. Keep token buffers as numpy/int64 arrays instead of Python lists: replaces
   `to_pylist()` + `list.extend` + `torch.tensor(list)` with array slicing.
3. Defaults: `io_queue_depth=1` and `transform_parallelism=min(4, cpu_count)`
   when `pack_sequences` is set; or at least scale the transform pool by
   `1 / world_size` when `world_size > 1` so 8 ranks don't each spawn
   `cpu_count` threads.
4. `StreamingDataLoader` with `num_workers > 0`: use `forkserver` (or `spawn`) —
   forked workers deadlock inside a CUDA + DDP rank (lancedb's own fork
   warning). If workers die at start-up with
   `SemLock._rebuild -> FileNotFoundError`, check the *host*: `systemd-logind`
   `RemoveIPC=yes` deletes the user's `/dev/shm` semaphores whenever a login
   session ends (Brev VMs do this every few seconds); fix with
   `RemoveIPC=no` in `/etc/systemd/logind.conf.d/`. Full write-up and repro:
   `ISSUE_streaming_workers.md`. Upstream asks: don't pickle the whole
   permutation table (38 MB for 2.37M rows) into every worker, and document
   the start method.
5. Document that `state_dict()` in packed mode requires an aligned cycle
   (every owned split has emitted the same block count) — callers must
   checkpoint on micro-batch boundaries that are multiples of `owned_splits`.

## Memory and GC: cap the post-transform queue on long runs

The 354M run (13,351 steps, 1h37m) showed periodic slow windows — 0.8-1.1M
tok/s against a 1.34M steady state — growing more frequent as the run went
on. The loader's `q` column explains it: with no `transform_queue_depth` the
cooked queue reached ~480k rows per worker (Python lists of ints, ~2GB and
hundreds of millions of objects), and CPython's cyclic GC walks all of it on
every gen-2 collection. Mean throughput was 1.26M / 38.6% instead of the
1.34M / 41.1% median. `train.py --transform-queue-depth 16` caps the cooked
rows per split at 16 x `read_batch_size`, which is still ~100 blocks of
headroom per split. Upstream: default `transform_queue_depth` to a small
number when `pack_sequences` is set (or store cooked tokens as numpy arrays,
which the GC does not traverse).

## Memory: the I/O stage runs far ahead

With `io_queue_depth=1` the I/O pool still outruns the packer by a wide margin:
on the 354M run each rank had ~360k raw rows + ~355k cooked rows queued by step
9,000 (`q` column), i.e. ~3GB per rank held in Python lists. Harmless on a
944GB host, but on smaller nodes cap it with `transform_queue_depth` (the
post-transform backpressure knob) — the packer only needs a few thousand rows
of headroom.

## How the trainer applies this

`train.py` exposes `--io-queue-depth` (default 1), `--transform-parallelism`
(default 2), `--read-batch-size` (default 8) and `--num-splits`; the 8xH100 runs
use `--num-splits 128`.
