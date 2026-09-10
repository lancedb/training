"""Why the packed loader's thread knobs matter: a self-contained reproduction.

The 8xH100 runs found that the released packed loader at library defaults
(`io_queue_depth=4`, `transform_parallelism=os.cpu_count()`, 32 splits per
rank) delivered ~158k tok/s per rank, and that `io_queue_depth=1`,
`transform_parallelism=2` and 16 splits per rank delivered ~960k.  This
script reproduces the *mechanism* on any machine, CPU only, in a few minutes,
by measuring three things per configuration instead of guessing:

  tok/s          packed-block throughput of one iterator (one rank's shape)
  main CPU       CPU seconds the main (packer) thread got per wall second.
                 The packer is the serial stage; its throughput is bounded by
                 how much of the GIL it can hold.  <1.0 with full queues means
                 it is waiting for the GIL, not for data.
  process CPU    CPU seconds burned by the whole process per wall second
                 (all threads).  Rising process CPU with falling tok/s is
                 contention overhead, not useful work.
  queues         raw / cooked rows waiting at the end of the window.  Full
                 queues rule out "starved for data" as the explanation.
  commit share   (--profile) fraction of main-thread time spent in
                 `_commit_pack_state`, the per-block snapshot of every owned
                 split's token buffer.  Grows with the number of splits.

Usage
-----
python loader_knobs_repro.py --db ./lance_pretrain_db                # sweep all three knobs
python loader_knobs_repro.py --db ./lance_pretrain_db --profile      # add the commit-share column
python loader_knobs_repro.py --db ./lance_pretrain_db --seconds 20 --splits 16 32 64

The table must have `input_ids` and `n_tokens` columns (ingest.py -> curate.py
-> tokenize_data.py, or the Geneva backfill).  Any tokenizer; `--eos-id` and
`--pad-id` default to the byte tokenizer used by the offline pipeline.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import resource
import sys
import threading
import time

import lancedb
import pyarrow.compute as pc
from lancedb.streaming import StreamingDataset

TRAIN_FILTER = "NOT is_dup AND score >= 1.0 AND (id % 100 != 0)"


def blocks_budget(tbl, seq_len: int, num_splits: int) -> int:
    n = tbl.count_rows(TRAIN_FILTER)
    toks = tbl.search().select(["n_tokens"]).where(TRAIN_FILTER).limit(n).to_arrow()
    b = (pc.sum(toks.column("n_tokens")).as_py() + n) // seq_len
    return b - b % num_splits


class GilProbe(threading.Thread):
    """Measures how long a thread waits to get the GIL back.

    Sleeps 2 ms in a loop and records how late each wake-up is.  After a
    sleep the thread is runnable but must re-acquire the GIL before it can
    execute the next bytecode, so the oversleep is (scheduler jitter +) GIL
    wait.  With one thread it is ~0.1 ms; with N GIL-hungry threads it
    approaches N x the switch interval (5 ms).  Reported as the median.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.samples: list[float] = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            t0 = time.perf_counter()
            time.sleep(0.002)
            self.samples.append(time.perf_counter() - t0 - 0.002)

    def median_ms(self) -> float:
        s = sorted(self.samples)
        return 1000 * s[len(s) // 2] if s else float("nan")


def run_one(tbl, *, seq_len, num_splits, ioq, tx, read_batch, seconds, eos_id, pad_id, profile):
    ds = StreamingDataset(
        tbl,
        columns=["input_ids"],
        filter=TRAIN_FILTER,
        num_splits=num_splits,
        shuffle_seed=0,
        read_batch_size=read_batch,
        io_queue_depth=ioq,
        transform_parallelism=tx,
        pack_sequences=seq_len,
        eos_id=eos_id,
        pad_id=pad_id,
        blocks_per_epoch=blocks_budget(tbl, seq_len, num_splits),
    )
    it = iter(ds)
    # Warm up until the pipeline has data in every stage; the first blocks
    # measure I/O ramp, not the steady state we care about.
    for _ in range(max(8, num_splits)):
        next(it)

    prof = cProfile.Profile() if profile else None
    probe = GilProbe()
    probe.start()
    cpu0, wall0 = time.thread_time(), time.perf_counter()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    blocks = 0
    if prof:
        prof.enable()
    while time.perf_counter() - wall0 < seconds:
        next(it)
        blocks += 1
    if prof:
        prof.disable()
    wall = time.perf_counter() - wall0
    probe.stop.set()
    probe.join()
    main_cpu = (time.thread_time() - cpu0) / wall
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    proc_cpu = ((ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)) / wall

    commit_share = None
    if prof:
        st = pstats.Stats(prof)
        total = st.total_tt
        commit = sum(v[2] for k, v in st.stats.items() if k[2] == "_commit_pack_state")
        commit_share = commit / total if total else 0.0

    row = dict(
        splits=num_splits,
        ioq=ioq,
        tx=tx,
        threads=threading.active_count(),
        tok_s=blocks * seq_len / wall,
        main_cpu=main_cpu,
        proc_cpu=proc_cpu,
        gil_wait_ms=probe.median_ms(),
        unscanned=ds.unscanned_rows,
        raw_q=ds.raw_queue_depth,
        cooked_q=ds.prefetch_queue_depth,
        fetch_s=ds.fetch_time,
        tx_s=ds.transform_time,
        commit_share=commit_share,
    )
    # Let the pools exit before the next config so thread counts don't leak.
    del it, ds
    return row


def fmt(row) -> str:
    s = (
        f"splits={row['splits']:>3} ioq={row['ioq']} tx={row['tx']:>3} | threads {row['threads']:>4} | "
        f"{row['tok_s']:>9,.0f} tok/s | main CPU {row['main_cpu']:.2f} | proc CPU {row['proc_cpu']:.2f} | "
        f"GIL wait {row['gil_wait_ms']:>5.1f} ms | unread {row['unscanned']:>7} raw {row['raw_q']:>6} cooked {row['cooked_q']:>6} | "
        f"fetch {row['fetch_s']:.0f}s tx {row['tx_s']:.0f}s"
    )
    if row["commit_share"] is not None:
        s += f" | commit {row['commit_share']:.0%} of main"
    return s


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", required=True)
    p.add_argument("--table", default="corpus")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--read-batch-size", type=int, default=8)
    p.add_argument("--seconds", type=float, default=15.0, help="measurement window per config")
    p.add_argument("--eos-id", type=int, default=257, help="byte tokenizer EOS (offline pipeline)")
    p.add_argument("--pad-id", type=int, default=258, help="byte tokenizer PAD (offline pipeline)")
    p.add_argument("--splits", type=int, nargs="*", default=[4, 16, 64], help="split sweep (at ioq=1, tx=2)")
    p.add_argument("--tx", type=int, nargs="*", default=[1, 2, 4, 16, 64, os.cpu_count() or 1],
                   help="transform_parallelism sweep (at ioq=1, 16 splits)")
    p.add_argument("--ioq", type=int, nargs="*", default=[1, 2, 4, 8], help="io_queue_depth sweep (at tx=2, 16 splits)")
    p.add_argument("--base-splits", type=int, default=16)
    p.add_argument("--profile", action="store_true", help="also report the _commit_pack_state share of main-thread time")
    args = p.parse_args(argv)

    tbl = lancedb.connect(args.db).open_table(args.table)
    print(f"table {args.table}: {tbl.count_rows():,} rows, {tbl.count_rows(TRAIN_FILTER):,} after filter, "
          f"cpu_count={os.cpu_count()}, python {sys.version.split()[0]}, lancedb {lancedb.__version__}")
    common = dict(seq_len=args.seq_len, read_batch=args.read_batch_size, seconds=args.seconds,
                  eos_id=args.eos_id, pad_id=args.pad_id, profile=args.profile)

    print(f"\n== A. transform_parallelism (threads doing Arrow->Python), io_queue_depth=1, {args.base_splits} splits")
    print("   Expect: tok/s falls and main-thread CPU share falls as tx grows, while queues stay full.")
    for tx in args.tx:
        print("  " + fmt(run_one(tbl, num_splits=args.base_splits, ioq=1, tx=tx, **common)))

    print(f"\n== B. io_queue_depth (I/O threads = splits x ioq), transform_parallelism=2, {args.base_splits} splits")
    print("   Expect: no gain past 1 on fast storage; extra threads only add GIL hand-offs.")
    for ioq in args.ioq:
        print("  " + fmt(run_one(tbl, num_splits=args.base_splits, ioq=ioq, tx=2, **common)))

    print("\n== C. splits owned by this iterator, io_queue_depth=1, transform_parallelism=2")
    print("   Expect: a sweet spot. Too few splits = too few reads in flight (raw queue empty); too many = more I/O")
    print("   threads plus a bigger per-block snapshot (with --profile, _commit_pack_state's share of main-thread time grows).")
    for sp in args.splits:
        print("  " + fmt(run_one(tbl, num_splits=sp, ioq=1, tx=2, **common)))

    print("\nHow to read it: the packer is a single Python thread. Every extra I/O or transform thread that also needs the GIL"
          "\ntakes turns with it. When the raw/cooked queues are full, the packer is not waiting for data; a main-thread CPU"
          "\nshare below 1.0 is time spent waiting for the interpreter lock. Fewer threads = more of the lock for the packer."
          "\nMore owned splits = a bigger per-block state snapshot on that same thread.")


if __name__ == "__main__":
    main()
