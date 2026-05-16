"""
B3 / B4 — Geneva backfill throughput.

  B3  raw cost     : time the first backfill of a given column over N rows.
  B4  incremental  : append M more rows via a second manifest slice; time the
                     second backfill.  Only the new M rows should re-run; the
                     original N is skipped via Geneva's default
                     ``<col> IS NULL`` filter.

Usage
-----
# B3 only — assumes the table already has the rows ingested.
python -m bench.bench_backfill --db data/videos/lancedb \\
    --columns keyword_melting any_phase_keyword

# B3 + B4 — appends 25 more manifest rows for the second pass.
python -m bench.bench_backfill --db data/videos/lancedb \\
    --columns keyword_melting any_phase_keyword \\
    --append-from data/chronomagic_proh.parquet --append-n 25
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from videogen.backfill_geneva import backfill
from videogen.ingest_chronomagic import ingest, manifest_rows


def _backfill(db: str, table: str, columns: list[str],
              *, dedup_threshold: int = 12) -> float:
    t0 = time.perf_counter()
    backfill(db_path=db, table_name=table, columns=columns,
             concurrency=1, overwrite=False, force_stub=False,
             dedup_threshold=dedup_threshold)
    return time.perf_counter() - t0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db",        default="data/videos/lancedb")
    p.add_argument("--table",     default="videos_raw")
    p.add_argument("--columns",   nargs="+", required=True,
                   help="One or more Geneva column names to time.")
    p.add_argument("--append-from", type=Path, default=None,
                   help="Optional parquet manifest to append from (for B4).")
    p.add_argument("--append-n",  type=int, default=25,
                   help="How many manifest rows to append for B4.")
    p.add_argument("--video-dir", type=Path, default=None,
                   help="If set, pull clip bytes from this dir during append.")
    args = p.parse_args(argv)

    import lancedb
    tbl = lancedb.connect(args.db).open_table(args.table)
    n_before = tbl.count_rows()
    print(f"\n[B3]  backfill {args.columns}  on {n_before:,} rows")
    t_b3 = _backfill(args.db, args.table, args.columns)
    rps_b3 = n_before / max(t_b3, 1e-6)
    print(f"       backfill done in {t_b3:.2f}s  ({rps_b3:.2f} rows/s)\n")

    if args.append_from is None:
        print("(no --append-from given — skipping B4)")
        return 0

    print(f"[B4]  append +{args.append_n:,} rows from {args.append_from}")
    # Skip ids already in the table to avoid duplicates.
    have_ids = set(
        tbl.search().select(["clip_id"]).to_arrow().column("clip_id").to_pylist()
    )
    fresh = []
    for r in manifest_rows(args.append_from, args.video_dir, limit=None):
        if r["clip_id"] in have_ids:
            continue
        fresh.append(r)
        if len(fresh) >= args.append_n:
            break

    t0 = time.perf_counter()
    ingest(db_path=args.db, table_name=args.table,
           rows=iter(fresh), overwrite=False, batch_size=64)
    t_app = time.perf_counter() - t0
    print(f"       append done in {t_app:.2f}s\n")

    print(f"[B4]  incremental backfill {args.columns} "
          f"(expects only +{len(fresh):,} rows re-run)")
    t_b4 = _backfill(args.db, args.table, args.columns)
    rps_b4 = len(fresh) / max(t_b4, 1e-6)
    print(f"       backfill done in {t_b4:.2f}s  ({rps_b4:.2f} new-rows/s)\n")

    # Geneva has a fixed Ray actor-spinup cost per backfill (~5-10s on this box).
    fixed_floor = 5.0
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
