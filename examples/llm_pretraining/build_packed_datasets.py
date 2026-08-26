"""Materialize identical pre-packed 1024-token blocks in every "standard
workflow" format, from one deterministic pass over the curated Lance table.

This pass is the materialization tax the incumbent workflows require: their
loaders stream pre-packed, pre-shuffled samples, they do not pack or shuffle.
Every output holds the byte-identical block set, so training A/Bs
(`train.py --blocks-mode ...`) differ only in the loader.

Outputs (under --out):
  blocks_parquet/part-XXX.parquet   packed stream order, 1024-row groups
  blocks_parquet_shuffled/          globally shuffled copy, one shard per rank
  mds_blocks/                       MosaicML Streaming MDS shards
  blocks_db/blocks                  Lance table of blocks

Usage
-----
python build_packed_datasets.py --db ~/runs/small/db --out ~/runs/small/blocks --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import TRAIN_FILTER, connect_table, load_tokenizer

SEQ_LEN = 1024
ROW_GROUP = 1024  # rows per parquet row group (4MB of int32 blocks)
NUM_SPLITS = 128
SEED = 1234
SCHEMA = pa.schema([pa.field("input_ids", pa.list_(pa.int32(), SEQ_LEN))])


def blocks_budget(tbl, filt: str) -> int:
    import pyarrow.compute as pc

    n = tbl.count_rows(filt)
    tot = pc.sum(
        tbl.search().select(["n_tokens"]).where(filt).limit(n).to_arrow().column("n_tokens")
    ).as_py() + n
    b = tot // SEQ_LEN
    return b - b % NUM_SPLITS


def worker(args) -> None:
    """One simulated rank: pack its splits, write one parquet shard."""
    from lancedb.streaming import StreamingDataset

    tok = load_tokenizer("hf:gpt2")
    tbl = connect_table(args.db, "corpus")
    filt = f"NOT is_dup AND score >= {args.min_score} AND ({TRAIN_FILTER})"
    ds = StreamingDataset(
        tbl,
        columns=["input_ids"],
        filter=filt,
        num_splits=NUM_SPLITS,
        shuffle_seed=SEED,
        rank=args.rank,
        world_size=args.workers,
        read_batch_size=8,
        io_queue_depth=1,
        transform_parallelism=2,
        pack_sequences=SEQ_LEN,
        eos_id=tok.eos_token_id,
        pad_id=tok.pad_token_id,
        blocks_per_epoch=args.blocks,
    )
    path = f"{args.out}/blocks_parquet/part-{args.rank:03d}.parquet"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = pq.ParquetWriter(path, SCHEMA, compression=None)
    buf, n = [], 0
    for block in ds:
        buf.append(block["input_ids"].numpy().astype(np.int32))
        if len(buf) == ROW_GROUP:
            w.write_table(pa.table({"input_ids": pa.FixedSizeListArray.from_arrays(pa.array(np.concatenate(buf)), SEQ_LEN)}))
            n += len(buf)
            buf = []
    if buf:
        w.write_table(pa.table({"input_ids": pa.FixedSizeListArray.from_arrays(pa.array(np.concatenate(buf)), SEQ_LEN)}))
        n += len(buf)
    w.close()
    print(f"[worker {args.rank}] {n:,} blocks -> {path}", flush=True)


def build_derived(args) -> None:
    import lancedb
    from streaming import MDSWriter

    parts = sorted(
        f"{args.out}/blocks_parquet/{f}" for f in os.listdir(f"{args.out}/blocks_parquet")
    )
    t0 = time.perf_counter()
    tables = [pq.read_table(p) for p in parts]
    total = sum(t.num_rows for t in tables)
    print(f"read {total:,} blocks from {len(parts)} parquet shards in {time.perf_counter() - t0:.0f}s")

    # Lance table of blocks
    t0 = time.perf_counter()
    ldb = lancedb.connect(f"{args.out}/blocks_db")
    if "blocks" in ldb.list_tables():
        ldb.drop_table("blocks")
    lt = ldb.create_table("blocks", tables[0], schema=SCHEMA)
    for t in tables[1:]:
        lt.add(t)
    print(f"lance blocks table: {lt.count_rows():,} rows in {time.perf_counter() - t0:.0f}s")

    # MDS shards (Mosaic's loader streams these; it does not pack or shuffle on write)
    t0 = time.perf_counter()
    mds_dir = f"{args.out}/mds_blocks"
    shutil.rmtree(mds_dir, ignore_errors=True)
    with MDSWriter(
        out=mds_dir,
        columns={"input_ids": f"ndarray:int32:{SEQ_LEN}"},
        compression=None,
        size_limit=1 << 28,
    ) as mds:
        for t in tables:
            mat = t.column("input_ids").combine_chunks().values.to_numpy().reshape(-1, SEQ_LEN)
            for row in mat:
                mds.write({"input_ids": np.ascontiguousarray(row)})
    print(f"mds shards written in {time.perf_counter() - t0:.0f}s")

    # Pre-shuffled parquet: the standard "materialize the shuffle" step, one shard per rank
    t0 = time.perf_counter()
    all_mat = np.concatenate(
        [t.column("input_ids").combine_chunks().values.to_numpy().reshape(-1, SEQ_LEN) for t in tables]
    )
    perm = np.random.default_rng(SEED).permutation(len(all_mat))
    out = f"{args.out}/blocks_parquet_shuffled"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for r, idx in enumerate(np.array_split(perm, args.shards)):
        mat = all_mat[idx]
        pq.write_table(
            pa.table({"input_ids": pa.FixedSizeListArray.from_arrays(pa.array(mat.reshape(-1)), SEQ_LEN)}),
            f"{out}/part-{r:03d}.parquet",
            row_group_size=ROW_GROUP,
            compression=None,
        )
    print(f"pre-shuffled parquet ({args.shards} shards) written in {time.perf_counter() - t0:.0f}s")

    def du(p):
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)

    sizes = {k: du(f"{args.out}/{k}") for k in ["blocks_parquet", "blocks_parquet_shuffled", "mds_blocks", "blocks_db"]}
    json.dump({"blocks": total, "bytes": sizes}, open(f"{args.out}/manifest.json", "w"), indent=1)
    for k, v in sizes.items():
        print(f"  {k:26s} {v / 1e9:6.2f} GB")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shards", type=int, default=8, help="pre-shuffled parquet shards")
    p.add_argument("--min-score", type=float, default=1.0)
    p.add_argument("--blocks", type=int, default=0)
    p.add_argument("--rank", type=int, default=-1, help="(internal) worker rank")
    p.add_argument("--derived-only", action="store_true")
    args = p.parse_args(argv)
    if args.rank >= 0:
        return worker(args)

    os.makedirs(args.out, exist_ok=True)
    if not args.derived_only:
        tbl = connect_table(args.db, "corpus")
        filt = f"NOT is_dup AND score >= {args.min_score} AND ({TRAIN_FILTER})"
        args.blocks = args.blocks or blocks_budget(tbl, filt)
        print(f"blocks_per_epoch={args.blocks:,}; packing with {args.workers} workers")
        t0 = time.perf_counter()
        shutil.rmtree(f"{args.out}/blocks_parquet", ignore_errors=True)
        procs = [
            subprocess.Popen(
                [sys.executable, __file__, "--db", args.db, "--out", args.out, "--workers",
                 str(args.workers), "--blocks", str(args.blocks), "--rank", str(r),
                 "--min-score", str(args.min_score)]
            )
            for r in range(args.workers)
        ]
        for pr in procs:
            assert pr.wait() == 0, "pack worker failed"
        print(f"PACK STAGE: {args.blocks:,} blocks -> parquet in {time.perf_counter() - t0:.0f}s")
    build_derived(args)


if __name__ == "__main__":
    main()
