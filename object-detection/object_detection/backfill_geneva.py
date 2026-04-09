"""
Run Geneva UDF backfills against the BDD100K Lance table.

This script registers UDF columns on the table (if they don't exist yet) and
then runs Geneva's checkpointed backfill to populate them.  It is safe to
re-run: already-computed rows are skipped automatically (Geneva's default
``where`` filter is ``<col> IS NULL``).

Two tiers of UDFs:

  Tier 1 — Annotation-derived (CPU, fast, no image decoding):
    has_person, has_rider, scene_description, scene_has_crossroad,
    scene_has_mountain, white_balance.

  Tier 2 — GPU inference:
    person_bbox_area_pct — area of the largest detected person as % of frame.
    CPU fallback (SSDLite) runs without --gpu; Faster R-CNN runs with --gpu.

Usage
-----
# Tier 1 — fast annotation-based columns:
python -m object_detection.backfill_geneva --columns has_person has_rider

# Tier 2 — person area (CPU SSDLite, local dev):
python -m object_detection.backfill_geneva --columns person_bbox_area_pct

# Tier 2 — person area (GPU Faster R-CNN, recommended for full dataset):
python -m object_detection.backfill_geneva --gpu --columns person_bbox_area_pct

# Restart a stuck job:
python -m object_detection.backfill_geneva --gpu --columns person_bbox_area_pct --overwrite
"""

from __future__ import annotations

import argparse

import geneva
from object_detection.geneva_udfs import (
    ALL_UDFS, CPU_PERSON_UDFS, GPU_PERSON_UDFS, GPU_EMBED_UDFS, METADATA_UDFS,
    _IsDuplicateCPU,
)

# Default LanceDB path and table name (override via CLI)
DEFAULT_DB = "data/bdd100k/lancedb"
DEFAULT_TABLE = "bdd100k"


def _ensure_columns(tbl, udf_registry: dict, columns: list[str], silent: bool = False) -> None:
    existing = set(tbl.schema.names)
    to_add = {col: udf_registry[col] for col in columns if col not in existing and col in udf_registry}

    if to_add:
        if not silent:
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
    dedup_threshold: float = 0.97,
) -> None:
    # is_duplicate UDF is instantiated here so it picks up the correct db_path + threshold.
    dedup_udfs = {"is_duplicate": _IsDuplicateCPU(db_path=db_path, threshold=dedup_threshold)}

    udf_registry = {
        **ALL_UDFS,
        **(GPU_PERSON_UDFS if gpu else CPU_PERSON_UDFS),
        **(GPU_EMBED_UDFS if gpu else {}),   # embedding only available with --gpu
        **dedup_udfs,
    }

    conn = geneva.connect(db_path)
    tbl = conn.open_table(table_name)

    print(f"Opened table '{table_name}' — {tbl.count_rows()} rows")
    print(f"Columns to backfill: {columns}")
    if overwrite:
        print("Mode: overwrite — dropping columns and recreating from scratch\n")
        for col in columns:
            _drop_column(tbl, col)
        # Re-open to get a fresh schema after drops
        tbl = conn.open_table(table_name)
        _ensure_columns(tbl, udf_registry, columns, silent=True)
    else:
        _ensure_columns(tbl, udf_registry, columns)

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
                checkpoint_size=32,
                task_size=2048,
            )
            print(f"  [done]     {col}  (job_id={job_id})\n")

    print("All backfills complete.")


def _parse_args(argv=None):
    default_cols = list(METADATA_UDFS.keys())

    p = argparse.ArgumentParser(
        description="Run Geneva UDF backfills on the BDD100K Lance table."
    )
    p.add_argument("--db", default=DEFAULT_DB, help="LanceDB database path")
    p.add_argument("--table", default=DEFAULT_TABLE, help="Lance table name")
    p.add_argument(
        "--columns", nargs="+", default=default_cols,
        help=(
            "UDF columns to backfill.  Defaults to Tier 1 annotation columns: "
            + ", ".join(default_cols)
        ),
    )
    p.add_argument(
        "--gpu", action="store_true",
        help=(
            "Use Faster R-CNN (GPU) for person_bbox_area_pct instead of SSDLite (CPU). "
            "Recommended on a GPU cluster for the final backfill."
        ),
    )
    p.add_argument(
        "--concurrency", type=int, default=None,
        help=(
            "Parallel Ray actor processes. "
            "Defaults to 1 for --gpu (one worker per GPU), 2 for CPU."
        ),
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Drop and re-add columns before backfilling — use to restart a stuck job",
    )
    p.add_argument(
        "--dedup-threshold", type=float, default=0.97,
        help=(
            "Cosine similarity threshold for is_duplicate backfill (default: 0.97). "
            "Only used when 'is_duplicate' is in --columns. "
            "0.97 = near-pixel-identical frames; lower = more aggressive dedup."
        ),
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    concurrency = args.concurrency if args.concurrency is not None else (1 if args.gpu else 2)
    backfill(
        db_path=args.db,
        table_name=args.table,
        columns=args.columns,
        concurrency=concurrency,
        overwrite=args.overwrite,
        gpu=args.gpu,
        dedup_threshold=args.dedup_threshold,
    )


if __name__ == "__main__":
    main()
