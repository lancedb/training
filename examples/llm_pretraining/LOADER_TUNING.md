# Packed loader settings (lancedb 0.38)

Measured loader-only, no GPU in the loop (`bench_loader.py`), on the 2.4M-doc
FineWeb-Edu table with GPT-2 tokens at `seq_len=1024`, local NVMe, 112-core
host. One "rank shape" is the splits one of 8 ranks owns.

## The knobs

| knob | library default | 8-GPU runs use | why |
|---|---|---|---|
| `io_queue_depth` | 4 | **1** | I/O threads per rank = `owned_splits x io_queue_depth`; every completed read takes the GIL from the packer |
| `transform_parallelism` | `os.cpu_count()` | **2** | Arrow->Python conversion threads; same GIL cost, and 8 ranks x 112 threads oversubscribes the host |
| `num_splits` | `world_size` | **128 = 16/rank** | the packer snapshots every owned split's buffer after every block, so per-block cost grows with owned splits; too few splits leaves too few reads in flight |
| `read_batch_size` | 64 | **8** | rows per take; small takes win on local disk, larger ones amortise latency on S3 |
| `transform_queue_depth` | unbounded | **16** | caps cooked rows per split; unbounded, long runs accumulate ~480k Python-list rows per worker and CPython's GC walks them on every full collection (periodic 20-40% dips on the 354M run) |

## Measured sweep (one rank shape, `read_batch_size=8`)

| config | tok/s |
|---|---|
| defaults (`io_queue_depth=4`, `transform_parallelism=112`), 32 splits | 158k |
| `io_queue_depth=1` | 468k |
| `io_queue_depth=1, transform_parallelism=2` | 795k |
| same, 16 splits | **960k** |
| same, 64 splits | 508k |
| 8 concurrent rank processes, 16 splits each | **~4.8M aggregate** |

cProfile of the tuned configuration, main thread: 42% in `_commit_pack_state`
(the per-block snapshot), 6% in `to_pylist`, 4% in `torch.tensor`.

## Why: the packer is starved of the interpreter lock

`runs/loader_gil_repro.py` reproduces the mechanism on any machine in about
two minutes, with no dependency on this example (it builds its own synthetic
token table). Per setting it prints tok/s, the CPU share the packer (main)
thread got, whole-process CPU and the raw-queue depth. On a 4-core box:

| io_queue_depth | threads | tok/s | packer CPU share | process CPU | raw queue (rows) |
|---:|---:|---:|---:|---:|---:|
| 1 | 20 | **1.00M** | **0.31** | 2.1 | 61k |
| 2 | 36 | 631k | 0.25 | 2.3 | 104k |
| 4 | 68 | 441k | 0.23 | 2.6 | 150k |
| 8 | 132 | 271k | 0.19 | 2.7 | 169k |

More reader threads read *more* rows ahead (storage is not the limit) and
burn *more* CPU (they are not idle), yet deliver fewer blocks while the packer
thread's CPU share falls. The packer is the one serial stage; every completed
read and every Arrow->Python transform holds the GIL to do work that cannot
speed up the output. Process CPU stays under 3 of 4 cores, so it is lock
serialization, not core oversubscription. `transform_parallelism` 1 -> 16
costs ~30% the same way (`--tx-sweep`). Raw output:
`runs/results/loader_gil_repro_4core.txt`.

## Upstream asks (filed with these numbers)

- Snapshot only the split that just emitted in `_commit_pack_state`, or keep
  buffers as immutable arrays so the snapshot is a reference, not a copy.
- Keep token buffers as numpy arrays instead of Python lists (also takes them
  out of the GC's reach).
- Default `io_queue_depth=1` and a small `transform_parallelism` and
  `transform_queue_depth` when `pack_sequences` is set.
- Don't pickle the whole permutation table (38MB for 2.37M rows) into every
  `StreamingDataLoader` worker; rebuild it from `(seed, epoch, num_splits,
  filter)` or share it. Document `forkserver`/`spawn` as the start method.
