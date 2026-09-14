"""Why more loader threads made LanceDB's packed StreamingDataset slower.

Minimal, self-contained reproducer.  Builds a synthetic token table, then runs
the packed loader with io_queue_depth = 1, 2, 4, 8 and prints, per setting:

  tok/s        packed-block throughput of one iterator
  packer CPU   CPU-seconds the main (packer) thread got per wall-second.
               The packer is the serial stage: every block passes through it.
  process CPU  CPU-seconds the whole process used per wall-second.
  raw queue    rows already read from storage but not yet packed.

The pattern to look for: more I/O threads -> MORE rows read ahead (storage is
not the limit) and MORE process CPU (threads are busy), yet FEWER tok/s and a
SMALLER packer CPU share.  Every completed read must build Python objects
while holding the GIL; that time comes straight out of the one thread that
sets the pace.  Same story for transform_parallelism (pass --tx-sweep).

    pip install "lancedb>=0.38" torch pyarrow
    python loader_gil_repro.py                 # ~2 min on a laptop
"""

import argparse
import os
import resource
import threading
import time

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.streaming import StreamingDataset

SEQ, EOS, PAD = 1024, 1, 0


def build_table(db_path: str, rows: int, seed: int = 0):
    """Documents of 80-600 random tokens, like a text corpus after tokenization."""
    db = lancedb.connect(db_path)
    if "corpus" in db.list_tables():
        return db.open_table("corpus")
    rng = np.random.default_rng(seed)
    lengths = rng.integers(80, 600, size=rows)
    flat = rng.integers(2, 50_000, size=int(lengths.sum()), dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    ids = pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat))
    tbl = pa.table({"id": pa.array(np.arange(rows, dtype=np.int64)), "input_ids": ids,
                    "n_tokens": pa.array(lengths.astype(np.int32))})
    return db.create_table("corpus", tbl)


def measure(tbl, *, splits, ioq, tx, seconds):
    n = tbl.count_rows()
    total_tokens = sum(r["n_tokens"] for r in tbl.search().select(["n_tokens"]).limit(n).to_list())
    budget = (total_tokens + n) // SEQ
    ds = StreamingDataset(tbl, columns=["input_ids"], num_splits=splits, shuffle_seed=0,
                          read_batch_size=8, io_queue_depth=ioq, transform_parallelism=tx,
                          pack_sequences=SEQ, eos_id=EOS, pad_id=PAD,
                          blocks_per_epoch=budget - budget % splits)
    it = iter(ds)
    for _ in range(splits):          # warm-up: get every pipeline stage busy
        next(it)
    ru0, cpu0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.thread_time(), time.perf_counter()
    blocks = 0
    while time.perf_counter() - t0 < seconds:
        next(it)
        blocks += 1
    wall = time.perf_counter() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    print(f"  splits={splits:>3} io_queue_depth={ioq} transform_parallelism={tx:>3} | threads {threading.active_count():>4} | "
          f"{blocks * SEQ / wall:>9,.0f} tok/s | packer CPU {(time.thread_time() - cpu0) / wall:.2f} | "
          f"process CPU {((ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)) / wall:.2f} | "
          f"raw queue {ds.raw_queue_depth:>7} rows | unread {ds.unscanned_rows:>7}")
    del it, ds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="./gil_repro_db")
    p.add_argument("--rows", type=int, default=400_000, help="documents; must outlast the window at the fastest read rate")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--splits", type=int, default=16)
    p.add_argument("--tx-sweep", action="store_true", help="also sweep transform_parallelism at io_queue_depth=1")
    a = p.parse_args()

    tbl = build_table(a.db, a.rows)
    print(f"{tbl.count_rows():,} docs | cpu_count={os.cpu_count()} | lancedb {lancedb.__version__}\n")
    print(f"io_queue_depth sweep (I/O threads = splits x io_queue_depth), transform_parallelism=2:")
    for ioq in (1, 2, 4, 8):
        measure(tbl, splits=a.splits, ioq=ioq, tx=2, seconds=a.seconds)
    if a.tx_sweep:
        print(f"\ntransform_parallelism sweep (Arrow->Python threads), io_queue_depth=1:")
        for tx in (1, 2, 4, 16, os.cpu_count() or 1):
            measure(tbl, splits=a.splits, ioq=1, tx=tx, seconds=a.seconds)
    print("\nIf 'raw queue' and 'process CPU' rise while 'tok/s' and 'packer CPU' fall, the threads are not waiting on"
          "\nstorage and are not idle: they are taking the interpreter lock from the one thread that produces blocks.")


if __name__ == "__main__":
    main()
