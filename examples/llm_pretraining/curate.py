"""Curate the corpus in place: EDA, full-text search, and dedup — one table.

Everything here happens on the table written by ingest.py.  The dedup flag is
added as a new column via zero-copy schema evolution: existing data files are
never rewritten, only new column files are appended.

Usage
-----
python curate.py --db ./lance_pretrain_db
"""

from __future__ import annotations

import argparse
import hashlib

import pyarrow as pa
from lancedb.index import FTS

from common import DEFAULT_DB, DEFAULT_TABLE, banner, connect_table, data_file_stats


def eda(tbl) -> None:
    banner("EDA (SQL on the training table)")
    total = tbl.count_rows()
    print(f"rows: {total}")
    for lo in range(0, 5):
        n = tbl.count_rows(f"score >= {lo} AND score < {lo + 1}")
        print(
            f"  score in [{lo},{lo + 1}): {n:>8}  {'#' * int(50 * n / max(total, 1))}"
        )
    short = tbl.count_rows("n_chars < 200")
    print(f"docs shorter than 200 chars: {short}")


def full_text_search(tbl, query: str) -> None:
    banner(f"FULL-TEXT SEARCH: {query!r}")
    tbl.create_index("text", config=FTS(), replace=True)
    hits = (
        tbl.search(query, query_type="fts").select(["id", "score", "_score"]).limit(3).to_list()
    )
    for h in hits:
        print(f"  id={h['id']:<8} quality={h['score']:.2f} bm25={h['_score']:.3f}")


def find_duplicate_ids(tbl) -> set[int]:
    """Pass 1 of exact dedup: scan only (id, text), keep the first occurrence
    of each whitespace-normalized content hash, return the ids of the rest.

    Shared by the offline path below and by ``geneva_backfill.py``, which
    turns the same set into an ``is_dup`` column with a Geneva UDF.
    """
    seen: set[str] = set()
    dup_ids: set[int] = set()
    for batch in tbl.search().select(["id", "text"]).to_batches(1024):
        for rid, text in zip(
            batch.column("id").to_pylist(), batch.column("text").to_pylist()
        ):
            h = hashlib.md5(" ".join(text.split()).encode()).hexdigest()
            if h in seen:
                dup_ids.add(rid)
            else:
                seen.add(h)
    return dup_ids


def flag_duplicates(tbl, db_uri: str, table_name: str) -> None:
    """Exact-dedup on normalized text; write the flag as a new column.

    Pass 2 backfills an ``is_dup`` column with a Python UDF through the
    underlying Lance dataset — LanceDB's own add_columns takes SQL
    expressions, so computed columns drop down to ``tbl.to_lance()``.
    This keeps the offline path dependency-free; ``geneva_backfill.py
    --columns is_dup`` writes the same column as a distributed Geneva
    backfill.
    """
    banner("DEDUP -> zero-copy `is_dup` column")
    files_before, bytes_before = data_file_stats(db_uri, table_name)

    dup_ids = find_duplicate_ids(tbl)

    def is_dup_udf(batch: pa.RecordBatch) -> pa.RecordBatch:
        flags = [rid in dup_ids for rid in batch.column("id").to_pylist()]
        return pa.RecordBatch.from_arrays(
            [pa.array(flags, pa.bool_())], names=["is_dup"]
        )

    tbl.to_lance().add_columns(is_dup_udf, read_columns=["id"])

    files_after, bytes_after = data_file_stats(db_uri, table_name)
    print(f"flagged {len(dup_ids)} duplicate rows")
    print(
        f"data files: {files_before} -> {files_after}, "
        f"bytes: {bytes_before:,} -> {bytes_after:,} "
        f"(+{bytes_after - bytes_before:,} for the new column; nothing rewritten)"
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--query", default="gradient scale law")
    args = parser.parse_args(argv)

    tbl = connect_table(args.db, args.table)
    eda(tbl)
    full_text_search(tbl, args.query)
    flag_duplicates(tbl, args.db, args.table)

    tbl = connect_table(args.db, args.table)  # reopen: see the new column
    print(f"\ncolumns now: {tbl.schema.names}")
    print(f"clean training rows: {tbl.count_rows('NOT is_dup')}")
    print(f"table version: {tbl.version}")


if __name__ == "__main__":
    main()
