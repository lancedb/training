"""Materialize identical pre-packed 1024-token samples as MDS shards and a
Lance table — the controlled A/B substrate for the Mosaic Streaming
comparison.

One deterministic pass over the curated corpus: documents (train filter,
fixed shuffle seed) are EOS-joined and sliced into fixed-length blocks; every
block is written to BOTH outputs, so the two loaders serve byte-identical
sample sets. This pass is itself the "materialization tax" the Mosaic flow
requires (its loader streams pre-packed samples; it does not pack).

Usage
-----
python build_packed_datasets.py --db ~/blogrun/db \
    --mds-out ~/blogrun/mds_blocks --lance-out ~/blogrun/blocks_db
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow as pa

import lancedb
from lancedb.streaming import StreamingDataset
from streaming import MDSWriter

from common import TRAIN_FILTER, connect_table, load_tokenizer

SEQ_LEN = 1024
WRITE_BATCH = 4096


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True)
    p.add_argument("--mds-out", required=True)
    p.add_argument("--lance-out", required=True)
    p.add_argument("--min-score", type=float, default=1.0)
    args = p.parse_args(argv)

    tok = load_tokenizer("hf:gpt2")
    tbl = connect_table(args.db, "corpus")
    # The packed stream itself defines the samples: single split, fixed seed.
    ds = StreamingDataset(
        tbl,
        columns=["input_ids"],
        filter=f"NOT is_dup AND score >= {args.min_score} AND ({TRAIN_FILTER})",
        num_splits=1,
        shuffle_seed=1234,
        read_batch_size=64,
        pack_sequences=SEQ_LEN,
        eos_id=tok.eos_token_id,
        pad_id=tok.pad_token_id,
    )

    schema = pa.schema([pa.field("input_ids", pa.list_(pa.int32(), SEQ_LEN))])
    ldb = lancedb.connect(args.lance_out)
    if "blocks" in ldb.list_tables():
        ldb.drop_table("blocks")
    lance_tbl = None

    t0 = time.perf_counter()
    n = 0
    buf: list[list[int]] = []
    with MDSWriter(
        out=args.mds_out,
        columns={"input_ids": f"ndarray:int32:{SEQ_LEN}"},
        compression=None,
        size_limit=1 << 28,  # 256MB shards
    ) as mds:
        for block in ds:
            ids = block["input_ids"].numpy().astype(np.int32)
            mds.write({"input_ids": ids})
            buf.append(ids.tolist())
            n += 1
            if len(buf) >= WRITE_BATCH:
                batch = pa.table({"input_ids": pa.array(buf, schema.field(0).type)})
                if lance_tbl is None:
                    lance_tbl = ldb.create_table("blocks", batch, schema=schema)
                else:
                    lance_tbl.add(batch)
                buf = []
    if buf:
        batch = pa.table({"input_ids": pa.array(buf, schema.field(0).type)})
        if lance_tbl is None:
            lance_tbl = ldb.create_table("blocks", batch, schema=schema)
        else:
            lance_tbl.add(batch)

    dt = time.perf_counter() - t0
    print(
        f"packed {n:,} blocks of {SEQ_LEN} tokens ({n * SEQ_LEN / 1e9:.2f}B tokens) "
        f"into MDS + Lance in {dt:,.1f}s"
    )
    print(f"lance rows: {lance_tbl.count_rows():,}")


if __name__ == "__main__":
    main()
