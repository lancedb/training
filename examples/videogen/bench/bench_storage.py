"""
B8 — Storage footprint.

Walk the Lance dataset directories and report on-disk size, file count, and
average bytes per row.  Compares the source table to each materialised view.

For the "Lance vs Parquet vs raw mp4 dir" comparison referenced in
PROPOSAL.md you would also run this against:

  * a directory of mp4 files
  * a single Parquet file containing the same rows
  * (optional) a Hugging Face datasets cache

That part is environment-specific and is left to the user; this script
handles the Lance side cleanly.

Usage
-----
python -m bench.bench_storage --db data/videos/lancedb
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import lancedb


def _du(path: Path) -> tuple[int, int]:
    """Total bytes and file count under ``path``."""
    total = files = 0
    for root, _, names in os.walk(path):
        for n in names:
            total += os.path.getsize(os.path.join(root, n))
            files += 1
    return total, files


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:7.1f} {unit}"
        n /= 1024
    return f"{n:7.1f} PB"


def run(db_path: str) -> None:
    db = lancedb.connect(db_path)
    tables = [t for t in db.list_tables().tables if not t.startswith("__")]
    if not tables:
        raise SystemExit(f"No tables in '{db_path}'.")

    print(f"\n[B8] database = {db_path}\n")
    print(f"  {'table':<40} {'rows':>10}  {'size':>12}  {'files':>6}  {'bytes/row':>10}")
    print("  " + "-" * 84)
    for name in sorted(tables):
        tbl = db.open_table(name)
        rows = tbl.count_rows()
        size, files = _du(Path(db_path) / f"{name}.lance")
        per_row = size / rows if rows else 0
        print(f"  {name:<40} {rows:>10,}  {_human(size):>12}  {files:>6}  {_human(int(per_row)):>10}")
    print()


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/videos/lancedb")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    run(args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
