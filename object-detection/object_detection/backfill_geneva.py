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
# Backfill all columns (CPU, SSDLite):
python -m object_detection.backfill_geneva

# Backfill vehicle columns using Faster R-CNN on GPU:
python -m object_detection.backfill_geneva --gpu --columns vehicle_label vehicle_confidence vehicle_bbox_area_pct

# Restart a stuck job:
python -m object_detection.backfill_geneva --gpu --columns vehicle_label --overwrite
"""

from __future__ import annotations

import argparse

import pyarrow as pa
import lancedb

import geneva
from object_detection.schema import GENEVA_UDF_COLUMNS
from object_detection.geneva_udfs import ALL_UDFS, CPU_VEHICLE_UDFS, GPU_VEHICLE_UDFS, METADATA_UDFS

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


def _drop_column(tbl, col: str) -> None:
    """Drop a column so it gets re-added from scratch on the next backfill."""
    if col in tbl.schema.names:
        print(f"  [drop]     {col}")
        tbl.drop_columns([col])


def backfill(
    db_path: str,
    table_name: str,
    columns: list[str],
    concurrency: int,
    overwrite: bool = False,
    gpu: bool = False,
) -> None:
    # Merge the right vehicle-detector variant into the lookup table.
    udf_registry = {**ALL_UDFS, **(GPU_VEHICLE_UDFS if gpu else CPU_VEHICLE_UDFS)}

    conn = geneva.connect(db_path)
    tbl = conn.open_table(table_name)

    print(f"Opened table '{table_name}' — {tbl.count_rows()} rows")
    print(f"Columns to backfill: {columns}")
    if overwrite:
        print("Mode: overwrite — resetting columns to NULL before backfill\n")

    if overwrite:
        for col in columns:
            _drop_column(tbl, col)
        # Re-open to get a fresh schema after drops
        tbl = conn.open_table(table_name)

    _ensure_columns(tbl, columns)

    with conn.local_ray_context():
        for col in columns:
            udf_fn = udf_registry.get(col)
            if udf_fn is None:
                print(f"  [SKIP] No UDF registered for column '{col}'")
                continue

            print(f"  [backfill] {col} …")
            job_id = tbl.backfill(
                col,
                udf=udf_fn,
                concurrency=concurrency,
                batch_size=32,
            )
            print(f"  [done]     {col}  (job_id={job_id})\n")

    print("All backfills complete.")


def _parse_args(argv=None):
    default_cols = list({**CPU_VEHICLE_UDFS, **METADATA_UDFS}.keys())

    p = argparse.ArgumentParser(
        description="Run Geneva UDF backfills on the BDD100K Lance table."
    )
    p.add_argument("--db", default=DEFAULT_DB, help="LanceDB database path")
    p.add_argument("--table", default=DEFAULT_TABLE, help="Lance table name")
    p.add_argument(
        "--columns", nargs="+", default=default_cols,
        help=(
            "UDF columns to backfill.  Defaults to all columns: "
            + ", ".join(default_cols)
        ),
    )
    p.add_argument(
        "--gpu", action="store_true",
        help=(
            "Use Faster R-CNN (GPU) for vehicle_* columns instead of SSDLite (CPU). "
            "Recommended on a GPU cluster for the final backfill."
        ),
    )
    p.add_argument(
        "--concurrency", type=int, default=2,
        help="Parallel Ray actor processes (default: 2)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Drop and re-add columns before backfilling — use to restart a stuck job",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    backfill(
        db_path=args.db,
        table_name=args.table,
        columns=args.columns,
        concurrency=args.concurrency,
        overwrite=args.overwrite,
        gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
