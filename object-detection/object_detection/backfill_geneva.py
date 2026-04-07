"""
Run Geneva UDF backfills against the BDD100K Lance table.

This script registers UDF columns on the table (if they don't exist yet) and
then runs Geneva's checkpointed backfill to populate them.  It is safe to
re-run: already-computed rows are skipped automatically (Geneva's default
``where`` filter is ``<col> IS NULL``).

Lightweight vs heavy UDFs
--------------------------
By default only the lightweight (CPU-friendly, SSDLite-based) UDFs are run.
Pass ``--columns vehicle_label vehicle_confidence ...`` to opt into the heavy
Faster R-CNN UDFs (GPU recommended).

Usage
-----
# Local smoke test — backfill lightweight UDFs on the default table:
PYTHONPATH=object-detection/src python -m object_detection.backfill_geneva

# Backfill specific columns only:
PYTHONPATH=object-detection/src python -m object_detection.backfill_geneva \\
    --columns vehicle_light_label vehicle_light_confidence

# Heavy backfill on a GPU cluster (set higher concurrency):
PYTHONPATH=object-detection/src \\
GENEVA_PIPELINE_STALL_TIMEOUT_S=7200 \\
python -m object_detection.backfill_geneva \\
    --columns vehicle_label vehicle_confidence vehicle_bbox_area_pct \\
    --concurrency 8
"""

from __future__ import annotations

import argparse
import sys

import pyarrow as pa
import lancedb

import geneva
from object_detection.schema import GENEVA_UDF_COLUMNS
from object_detection.geneva_udfs import ALL_UDFS, LIGHT_UDFS

# Default LanceDB path and table name (override via CLI)
DEFAULT_DB = "data/bdd100k/lancedb"
DEFAULT_TABLE = "bdd100k"

# Map column name → PyArrow field (from schema)
_FIELD_BY_NAME: dict[str, pa.Field] = {f.name: f for f in GENEVA_UDF_COLUMNS}


# Map PyArrow type → the SQL null expression LanceDB expects when adding a column
_NULL_EXPR: dict[pa.DataType, str] = {
    pa.string():  "cast(null as string)",
    pa.float32(): "cast(null as float)",
    pa.bool_():   "cast(null as boolean)",
}


def _ensure_columns(tbl, columns: list[str]) -> None:
    """Add null-initialised columns to the table for any that are missing."""
    existing = set(tbl.schema.names)
    to_add = {}
    for col in columns:
        if col in existing:
            continue
        field = _FIELD_BY_NAME.get(col)
        if field is None:
            raise ValueError(f"Unknown UDF column: '{col}'.  Valid options: {list(_FIELD_BY_NAME)}")
        sql_expr = _NULL_EXPR.get(field.type)
        if sql_expr is None:
            raise NotImplementedError(f"No SQL null expression for type {field.type}")
        to_add[col] = sql_expr

    if to_add:
        print(f"  Adding {len(to_add)} new column(s): {list(to_add)}")
        tbl.add_columns(to_add)


def backfill(
    db_path: str,
    table_name: str,
    columns: list[str],
    concurrency: int,
    min_checkpoint_size: int,
    max_checkpoint_size: int,
) -> None:
    conn = geneva.connect(db_path)
    tbl = conn.open_table(table_name)

    print(f"Opened table '{table_name}' — {tbl.count_rows()} rows")
    print(f"Columns to backfill: {columns}\n")

    _ensure_columns(tbl, columns)

    with conn.local_ray_context():
        for col in columns:
            udf_fn = ALL_UDFS.get(col)
            if udf_fn is None:
                print(f"  [SKIP] No UDF registered for column '{col}'")
                continue

            print(f"  [backfill] {col} …")
            job_id = tbl.backfill(
                col,
                udf=udf_fn,
                concurrency=concurrency,
                min_checkpoint_size=min_checkpoint_size,
                max_checkpoint_size=max_checkpoint_size,
            )
            print(f"  [done]     {col}  (job_id={job_id})\n")

    print("All backfills complete.")


def _parse_args(argv=None):
    default_light = list(LIGHT_UDFS.keys())

    p = argparse.ArgumentParser(
        description="Run Geneva UDF backfills on the BDD100K Lance table."
    )
    p.add_argument("--db", default=DEFAULT_DB, help="LanceDB database path")
    p.add_argument("--table", default=DEFAULT_TABLE, help="Lance table name")
    p.add_argument(
        "--columns", nargs="+", default=default_light,
        help=(
            "UDF columns to backfill.  Defaults to all lightweight columns: "
            + ", ".join(default_light)
        ),
    )
    p.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of parallel Ray actor processes (default: 1 for local CPU)",
    )
    p.add_argument(
        "--min-checkpoint-size", type=int, default=10,
        help="Minimum rows per checkpoint (default: 10)",
    )
    p.add_argument(
        "--max-checkpoint-size", type=int, default=50,
        help="Maximum rows per checkpoint (default: 50)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    backfill(
        db_path=args.db,
        table_name=args.table,
        columns=args.columns,
        concurrency=args.concurrency,
        min_checkpoint_size=args.min_checkpoint_size,
        max_checkpoint_size=args.max_checkpoint_size,
    )


if __name__ == "__main__":
    main()
