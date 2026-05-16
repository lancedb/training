"""
B1 — Ingest throughput.

Times two ingest paths and reports rows/sec:

  * synthetic:  ``videogen.ingest_chronomagic.synthetic_rows`` → Lance
  * manifest :  parquet manifest → Lance  (videos optional)

Both go through the same ``ingest()`` function so the only thing that
changes is the source generator.

Usage
-----
python -m bench.bench_ingest --synthetic 2000
python -m bench.bench_ingest --manifest data/chronomagic_pro.parquet --limit 5000
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from videogen.ingest_chronomagic import (
    DEFAULT_BATCH,
    ingest,
    manifest_rows,
    synthetic_rows,
)


def _run(label: str, rows_iter, *, db: str, table: str, batch_size: int) -> None:
    if Path(db).exists():
        shutil.rmtree(db)
    t0 = time.perf_counter()
    ingest(db_path=db, table_name=table, rows=rows_iter,
           overwrite=True, batch_size=batch_size)
    dt = time.perf_counter() - t0

    import lancedb
    tbl = lancedb.connect(db).open_table(table)
    n = len(tbl)
    print(f"\n[B1 {label}]  {n:,} rows  {dt:.2f}s  → {n / dt:,.0f} rows/sec")


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="/tmp/videogen_bench/ingest")
    p.add_argument("--table", default="videos_raw")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", type=int)
    src.add_argument("--manifest",  type=Path)
    p.add_argument("--video-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    if args.synthetic is not None:
        rows = synthetic_rows(args.synthetic)
        label = f"synthetic n={args.synthetic}"
    else:
        rows = manifest_rows(args.manifest, args.video_dir, args.limit)
        label = f"manifest {args.manifest.name} limit={args.limit}"
    _run(label, rows, db=args.db, table=args.table, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
