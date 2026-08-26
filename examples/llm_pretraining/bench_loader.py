"""Loader-only throughput: packed blocks/s and tokens/s, no GPU in the loop.

Usage
-----
python bench_loader.py --db ~/blogrun/db --seq-len 1024 --num-splits 128 --seconds 30
"""

from __future__ import annotations

import argparse
import time

from lancedb.streaming import StreamingDataset

from common import DEFAULT_TABLE, TRAIN_FILTER, connect_table, load_tokenizer


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True)
    p.add_argument("--table", default=DEFAULT_TABLE)
    p.add_argument("--tokenizer", default="hf:gpt2")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--num-splits", type=int, default=128)
    p.add_argument("--read-batch-size", type=int, default=8)
    p.add_argument("--io-queue-depth", type=int, default=1)
    p.add_argument("--transform-parallelism", type=int, default=2)
    p.add_argument("--clump-size", type=int, default=0)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument(
        "--blocks-per-epoch",
        default="",
        help="default: exact budget from the n_tokens column (as train.py); "
        "'auto' = the loader's sampled estimate",
    )
    p.add_argument("--min-score", type=float, default=1.0)
    args = p.parse_args(argv)

    tok = load_tokenizer(args.tokenizer)
    tbl = connect_table(args.db, args.table)
    filt = f"NOT is_dup AND score >= {args.min_score} AND ({TRAIN_FILTER})"
    if args.blocks_per_epoch == "auto":
        blocks_per_epoch = "auto"
    elif args.blocks_per_epoch:
        blocks_per_epoch = int(args.blocks_per_epoch)
    else:  # exact: same computation as train.py
        import pyarrow.compute as pc

        n = tbl.count_rows(filt)
        toks = tbl.search().select(["n_tokens"]).where(filt).limit(n).to_arrow()
        b = (pc.sum(toks.column("n_tokens")).as_py() + n) // args.seq_len
        blocks_per_epoch = b - b % args.num_splits
    ds = StreamingDataset(
        tbl,
        columns=["input_ids"],
        filter=filt,
        num_splits=args.num_splits,
        shuffle_seed=0,
        read_batch_size=args.read_batch_size,
        io_queue_depth=args.io_queue_depth,
        transform_parallelism=args.transform_parallelism,
        shuffle_clump_size=args.clump_size or None,
        pack_sequences=args.seq_len,
        eos_id=tok.eos_token_id,
        pad_id=tok.pad_token_id,
        blocks_per_epoch=blocks_per_epoch,
    )
    it = iter(ds)
    next(it)  # warm the pipeline
    t0 = time.perf_counter()
    blocks = 0
    while time.perf_counter() - t0 < args.seconds:
        next(it)
        blocks += 1
    dt = time.perf_counter() - t0
    print(
        f"splits={args.num_splits} rb={args.read_batch_size} ioq={args.io_queue_depth} "
        f"tx={args.transform_parallelism} clump={args.clump_size}: "
        f"{blocks / dt:,.0f} blocks/s = {blocks * args.seq_len / dt:,.0f} tok/s "
        f"(fetch {ds.fetch_time:.1f}s, transform {ds.transform_time:.1f}s over {dt:.0f}s wall)"
    )


if __name__ == "__main__":
    main()
