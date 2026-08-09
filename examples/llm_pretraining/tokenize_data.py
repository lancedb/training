"""Tokenize the corpus ONCE, storing token ids as a new column — zero-copy.

This replaces the classic "materialize a tokenized+shuffled copy of the
dataset" preprocessing step.  The `input_ids` column is appended to the same
table via Lance's zero-copy schema evolution: raw text and token ids live
side by side, no data is rewritten, and training reads only the columns it
asks for.

Usage
-----
python tokenize_data.py --db ./lance_pretrain_db --tokenizer byte
python tokenize_data.py --tokenizer hf:Qwen/Qwen2.5-0.5B     # needs network
"""

from __future__ import annotations

import argparse
import time

import pyarrow as pa

from common import (
    DEFAULT_DB,
    DEFAULT_TABLE,
    banner,
    connect_table,
    data_file_stats,
    load_tokenizer,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--tokenizer", default="byte", help="'byte' or 'hf:<model>'")
    args = parser.parse_args(argv)

    tok = load_tokenizer(args.tokenizer)
    tbl = connect_table(args.db, args.table)

    banner(f"TOKENIZE ({args.tokenizer}) -> zero-copy `input_ids` column")
    files_before, bytes_before = data_file_stats(args.db, args.table)

    def tokenize_udf(batch: pa.RecordBatch) -> pa.RecordBatch:
        encoded = [tok.encode(t) for t in batch.column("text").to_pylist()]
        return pa.RecordBatch.from_arrays(
            [
                pa.array(encoded, pa.list_(pa.int32())),
                pa.array([len(e) for e in encoded], pa.int32()),
            ],
            names=["input_ids", "n_tokens"],
        )

    t0 = time.perf_counter()
    tbl.to_lance().add_columns(tokenize_udf, read_columns=["text"])
    dt = time.perf_counter() - t0

    tbl = connect_table(args.db, args.table)  # reopen: see the new columns
    files_after, bytes_after = data_file_stats(args.db, args.table)
    n_tokens = tbl.search().select(["n_tokens"]).to_arrow().column("n_tokens")
    total_tokens = pa.compute.sum(n_tokens).as_py()

    print(f"tokenized {tbl.count_rows()} docs -> {total_tokens:,} tokens in {dt:.1f}s")
    print(
        f"data files: {files_before} -> {files_after}, "
        f"bytes: {bytes_before:,} -> {bytes_after:,} "
        f"(+{bytes_after - bytes_before:,} for token columns; nothing rewritten)"
    )
    print(f"columns now: {tbl.schema.names}")
    print(f"table version: {tbl.version} (previous versions still readable)")


if __name__ == "__main__":
    main()
