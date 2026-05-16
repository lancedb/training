"""
B3 / B4 — Geneva backfill throughput.

  B3  raw cost     : time the first backfill of a given column over N rows.
  B4  incremental  : append M more rows, time the second backfill.  Only
                     the new M rows should re-run; the original N is
                     skipped via Geneva's default ``<col> IS NULL`` filter.

The script ingests a fresh synthetic Lance table inside ``--db`` to keep
the numbers reproducible, then runs the requested columns through
backfill once, appends 25% more rows, and runs them again.

Usage
-----
python -m bench.bench_backfill --db /tmp/videogen_bench/backfill \\
    --n 200 --extra 50 --columns keyword_melting motion_strength
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from videogen.backfill_geneva import backfill
from videogen.ingest_chronomagic import ingest, synthetic_rows


def _ingest(db: str, table: str, n: int, *, overwrite: bool, seed: int) -> float:
    t0 = time.perf_counter()
    ingest(db_path=db, table_name=table,
           rows=synthetic_rows(n, seed=seed),
           overwrite=overwrite, batch_size=64)
    return time.perf_counter() - t0


def _backfill(db: str, table: str, columns: list[str], *, dedup_threshold: int = 12) -> float:
    t0 = time.perf_counter()
    backfill(db_path=db, table_name=table, columns=columns,
             concurrency=1, overwrite=False, force_stub=False,
             dedup_threshold=dedup_threshold)
    return time.perf_counter() - t0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="/tmp/videogen_bench/backfill")
    p.add_argument("--table", default="videos_raw")
    p.add_argument("--n",       type=int, default=100,
                   help="Initial row count.")
    p.add_argument("--extra",   type=int, default=25,
                   help="Rows appended for the incremental backfill.")
    p.add_argument("--columns", nargs="+", required=True,
                   help="One or more Geneva column names to time.")
    p.add_argument("--seed",    type=int, default=0)
    args = p.parse_args(argv)

    db_path = Path(args.db)
    if db_path.exists():
        shutil.rmtree(db_path)

    print(f"\n[B3]  initial ingest: {args.n:,} rows")
    t_ing = _ingest(args.db, args.table, args.n, overwrite=True, seed=args.seed)
    print(f"       ingest done in {t_ing:.2f}s  ({args.n / t_ing:.1f} rows/sec)\n")

    print(f"[B3]  backfill {args.columns}  on {args.n:,} rows")
    t_b3 = _backfill(args.db, args.table, args.columns)
    rps_b3 = args.n / t_b3
    print(f"       backfill done in {t_b3:.2f}s  ({rps_b3:.2f} rows/sec)\n")

    print(f"[B4]  append +{args.extra:,} rows")
    t_app = _ingest(args.db, args.table, args.extra, overwrite=False, seed=args.seed + 1)
    print(f"       append done in {t_app:.2f}s\n")

    print(f"[B4]  incremental backfill {args.columns}  (expects only +{args.extra:,} rows re-run)")
    t_b4 = _backfill(args.db, args.table, args.columns)
    rps_b4 = args.extra / t_b4
    print(f"       backfill done in {t_b4:.2f}s  ({rps_b4:.2f} new-rows/sec)\n")

    # Geneva has a fixed Ray actor-spinup cost per backfill (~5-10s on this box).
    # For cheap per-row UDFs that dominates the wall-clock and B3/B4 ratios
    # look ugly.  Subtract out a rough fixed cost to show the actual per-row
    # advantage; for heavy UDFs (Tier 2/3) the fixed cost is in the noise.
    fixed_floor = 5.0  # rough Ray startup floor in seconds
    var_b3 = max(t_b3 - fixed_floor, 1e-6)
    var_b4 = max(t_b4 - fixed_floor, 1e-6)
    print("─" * 64)
    print(f"  B3  initial         : {t_b3:6.2f}s   ({rps_b3:5.2f} rows/s)")
    print(f"  B4  incremental     : {t_b4:6.2f}s   ({rps_b4:5.2f} new rows/s)")
    print(f"      B4 / B3 wall    : {t_b4 / t_b3:.3f}  "
          f"(would be 1.0 if re-deriving everything)")
    print(f"      variable-only   : {var_b4 / var_b3:.3f}  "
          f"(after subtracting ~{fixed_floor:.0f}s Ray startup floor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
