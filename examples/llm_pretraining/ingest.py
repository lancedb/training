"""Ingest a text corpus into a single LanceDB table.

This is the only time raw data is written.  Every later stage — EDA, dedup,
tokenization, training, retrieval — works against this same table.

Usage
-----
# Offline synthetic corpus (used by verify_e2e.py):
python ingest.py --source synthetic --rows 5000

# Real corpus (FineWeb-Edu, streamed from HuggingFace — needs network):
python ingest.py --source fineweb --rows 2000000 --db ./lance_pretrain_db

# Larger corpus: FineWeb-Edu sample-100BT parquet files, downloaded in
# parallel and ingested file-by-file (~10x faster than the streaming iterator):
python ingest.py --source fineweb-parquet --files 24 --db ./lance_pretrain_db
# (`--sample 10BT --rows 2400000` reproduces the streaming ingest's first 2.4M docs)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import time

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


def fineweb_parquet_batches(
    n_files: int, rows: int = 0, workers: int = 8, sample: str = "100BT"
):
    """Yield FineWeb-Edu RecordBatches straight from its parquet shards.

    Files are downloaded concurrently (hf_hub_download caches them) and read
    row-group by row-group with pyarrow, so ingest runs at disk speed rather
    than at the speed of a Python row iterator.  ``id`` is assigned
    sequentially across files so the train/val split filters stay stable.
    """
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    tree = HfApi().list_repo_tree(
        "HuggingFaceFW/fineweb-edu", f"sample/{sample}", repo_type="dataset"
    )
    names = sorted(f.path for f in tree if f.path.endswith(".parquet"))[:n_files]

    def fetch(name):
        return hf_hub_download(
            "HuggingFaceFW/fineweb-edu", name, repo_type="dataset",
            local_dir=os.environ.get("FINEWEB_LOCAL_DIR", "./fineweb_parquet"),
        )

    next_id = 0
    with ThreadPoolExecutor(workers) as ex:
        for path in ex.map(fetch, names):
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                if rows and next_id >= rows:
                    return
                t = pf.read_row_group(rg, columns=["text", "score"])
                if rows:
                    t = t.slice(0, rows - next_id)
                n = t.num_rows
                text = t.column("text")
                yield pa.RecordBatch.from_arrays(
                    [
                        pa.array(range(next_id, next_id + n), pa.int64()),
                        text.combine_chunks(),
                        pa.array(["fineweb-edu"] * n, pa.string()),
                        pc.cast(t.column("score"), pa.float32()).combine_chunks(),
                        pc.cast(pc.utf8_length(text), pa.int32()).combine_chunks(),
                    ],
                    schema=SCHEMA,
                )
                next_id += n


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
        "--source",
        choices=["synthetic", "fineweb", "fineweb-parquet"],
        default="synthetic",
    )
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument(
        "--files", type=int, default=4, help="fineweb-parquet: shards to ingest"
    )
    parser.add_argument(
        "--sample", default="100BT", help="fineweb-parquet: 10BT or 100BT sample"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    banner(f"INGEST  {args.source} -> {args.db}/{args.table}")
    t0 = time.perf_counter()
    if args.source == "fineweb-parquet":
        batches = fineweb_parquet_batches(args.files, args.rows, sample=args.sample)
    else:
        docs = (
            synthetic_docs(args.rows, seed=args.seed)
            if args.source == "synthetic"
            else fineweb_docs(args.rows)
        )
        batches = record_batches(docs)
    reader = pa.RecordBatchReader.from_batches(SCHEMA, batches)

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
    print(f"ingested {n} rows (checksum {digest}) in {time.perf_counter() - t0:.1f}s")
    print(f"table version: {tbl.version}")


if __name__ == "__main__":
    main()
