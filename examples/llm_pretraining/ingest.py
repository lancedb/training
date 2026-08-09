"""Ingest a text corpus into a single LanceDB table.

This is the only time raw data is written.  Every later stage — EDA, dedup,
tokenization, training, retrieval — works against this same table.

Usage
-----
# Offline synthetic corpus (used by verify_e2e.py):
python ingest.py --source synthetic --rows 5000

# Real corpus (FineWeb-Edu, streamed from HuggingFace — needs network):
python ingest.py --source fineweb --rows 2000000 --db ./lance_pretrain_db
"""

from __future__ import annotations

import argparse
import hashlib
import random

import lancedb
import pyarrow as pa

from common import DEFAULT_DB, DEFAULT_TABLE, banner

SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("source", pa.string()),
        # Quality score. FineWeb-Edu ships an educational-quality score (0-5);
        # the synthetic source draws one at random so filters have something
        # to bite on.
        pa.field("score", pa.float32()),
        pa.field("n_chars", pa.int32()),
    ]
)

WRITE_BATCH_ROWS = 2048

_WORDS = (
    "model data loss token batch gradient layer attention memory cache "
    "vector search index shuffle split stream train eval curve scale law "
    "corpus sample epoch step rank node cluster shard column table row"
).split()


def synthetic_docs(rows: int, seed: int = 0, dup_fraction: float = 0.05):
    """Generate pseudo-documents, injecting exact duplicates.

    A small fraction of rows duplicate an earlier document so the dedup stage
    in curate.py has real work to do.
    """
    rng = random.Random(seed)
    originals: list[str] = []
    for i in range(rows):
        if originals and rng.random() < dup_fraction:
            text = rng.choice(originals)
        else:
            n = rng.randint(80, 400)
            text = " ".join(rng.choice(_WORDS) for _ in range(n))
            originals.append(text)
        yield {
            "id": i,
            "text": text,
            "source": "synthetic",
            "score": round(rng.uniform(0.0, 5.0), 2),
            "n_chars": len(text),
        }


def fineweb_docs(rows: int):
    """Stream FineWeb-Edu from HuggingFace (requires network access)."""
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    for i, rec in enumerate(ds):
        if i >= rows:
            break
        yield {
            "id": i,
            "text": rec["text"],
            "source": "fineweb-edu",
            "score": float(rec.get("score", 0.0)),
            "n_chars": len(rec["text"]),
        }


def record_batches(doc_iter):
    rows = []
    for doc in doc_iter:
        rows.append(doc)
        if len(rows) >= WRITE_BATCH_ROWS:
            yield pa.RecordBatch.from_pylist(rows, schema=SCHEMA)
            rows = []
    if rows:
        yield pa.RecordBatch.from_pylist(rows, schema=SCHEMA)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument(
        "--source", choices=["synthetic", "fineweb"], default="synthetic"
    )
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    banner(f"INGEST  {args.source} -> {args.db}/{args.table}")
    docs = (
        synthetic_docs(args.rows, seed=args.seed)
        if args.source == "synthetic"
        else fineweb_docs(args.rows)
    )
    reader = pa.RecordBatchReader.from_batches(SCHEMA, record_batches(docs))

    db = lancedb.connect(args.db)
    if args.table in db.table_names():
        db.drop_table(args.table)
    tbl = db.create_table(
        args.table,
        data=reader,
        schema=SCHEMA,
        storage_options={"new_table_enable_stable_row_ids": "true"},
    )
    n = tbl.count_rows()
    digest = hashlib.md5(str(n).encode()).hexdigest()[:8]
    print(f"ingested {n} rows (checksum {digest})")
    print(f"table version: {tbl.version}")


if __name__ == "__main__":
    main()
