# StreamingDataLoader workers die at start-up: `SemLock._rebuild -> FileNotFoundError` (root cause + two upstream asks)

**Repo:** lancedb/lancedb (python) — `lancedb.streaming.StreamingDataLoader` / `StreamingDataset`
**Version:** 0.38.0b10 (built from tag `v0.38.0-beta.10`), torch 2.13, Python 3.12.14, Ubuntu 24.04 (Brev VM, 8xH100)

## Symptom
`StreamingDataLoader(ds, num_workers=2, multiprocessing_context="spawn"|"forkserver")` on a real table (2.4M-row
permutation) fails on the first `next()`:

```
RuntimeError: DataLoader worker (pid(s) ...) exited unexpectedly
# worker stderr:
  File ".../multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
  File ".../multiprocessing/synchronize.py", line 115, in __setstate__
    self._semlock = _multiprocessing.SemLock._rebuild(*state)
FileNotFoundError: [Errno 2] No such file or directory
```
At parent exit its own `SemLock._cleanup -> sem_unlink` also raises FileNotFoundError, and the resource tracker
reports "leaked semaphore objects" it cannot find. With `fork` (default context) the workers instead deadlock inside a
CUDA/DDP rank (lancedb already warns fork support is experimental).

Size/time dependent: a 2k-row synthetic table works; 20k rows works with spawn but not forkserver; 300k+ rows fails
with both. Passes if the resource tracker happened to be started >~5 s before the loader.

## Root cause (environment, not lancedb)
Traced with `bpftrace` on `unlinkat`: **`systemd-logind`** deletes every file the user owns in `/dev/shm` a few
seconds after creation — its default `RemoveIPC=yes` wipes a user's POSIX IPC whenever one of that user's login
sessions ends, and on this VM a host agent opens/closes an `ubuntu` session every few seconds. Named semaphores
(`/dev/shm/sem.mp-*`) backing torch's `index_queue`/`done_event` vanish before a slow-starting worker can
`sem_open` them. Repro without lancedb: `touch /dev/shm/x` — gone in 4-7 s.

Fix on the host:
```
printf '[Login]\nRemoveIPC=no\n' | sudo tee /etc/systemd/logind.conf.d/no-remove-ipc.conf
sudo systemctl restart systemd-logind
```
(The same gotcha bites PostgreSQL; their docs recommend `RemoveIPC=no`.)

## Why the loader is exposed, and two asks
Worker start-up is slow enough to cross that window because each worker re-imports torch + lancedb and receives the
whole permutation table by pickle: `StreamingDataset.__getstate__` serialises `_perm_table.to_arrow()` — 38 MB for
2.37M rows (0.7 s to produce, 0.85 s to rebuild as a `memory://` table in the worker), ×`num_workers`×`world_size`.

1. **Don't ship the permutation table to every worker.** Options: rebuild it in the worker from
   `(seed, epoch, num_splits, filter)` (it is deterministic), or spill it to a Lance file under the db dir and pass
   the path, or share it via `pa.Table` in shared memory. This also cuts start-up time on 8-rank hosts
   (8 ranks × 2 workers × 38 MB).
2. **Document the start method.** `fork` deadlocks under CUDA + the async runtime; `forkserver` is the fast, safe
   choice (the `RawArray` comment in `streaming.py` already assumes it). A one-line note in the
   `StreamingDataLoader` docstring plus a startup check (`if get_start_method() == "fork" and torch.cuda.is_initialized(): warn`)
   would have saved a day of debugging here.

## Minimal repro
```python
# repro.py  — run:  python repro.py forkserver   (fails on a host with RemoveIPC=yes; passes after the fix)
import sys, multiprocessing as mp, warnings, lancedb
from lancedb.streaming import StreamingDataset, StreamingDataLoader
if __name__ == "__main__":
    mp.set_start_method(sys.argv[1], force=True); warnings.simplefilter("ignore")
    tbl = lancedb.connect("<db>").open_table("corpus")          # any table with a few hundred k rows
    ds = StreamingDataset(tbl, columns=["input_ids"], num_splits=16, shuffle_seed=42,
                          pack_sequences=1024, eos_id=50256, pad_id=50257, blocks_per_epoch=16 * 64)
    dl = StreamingDataLoader(ds, batch_size=16, num_workers=2, multiprocessing_context=sys.argv[1])
    print(next(iter(dl))["input_ids"].shape)
```
