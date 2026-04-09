"""
Deduplication helpers for BDD100K.

The full dedup pipeline follows the same Geneva backfill pattern as every
other feature column:

  1. backfill_geneva --gpu --columns embedding
        ResNet18 (512-d, L2-normalised) written to bdd100k.embedding

  2. dedup --action index
        IVF-PQ cosine index on bdd100k.embedding (required before step 3)

  3. backfill_geneva --columns is_duplicate
        Per-row: nearest non-self neighbour found via vector search.
        is_duplicate = True when cosine similarity >= 0.85.

  4. manage_views --action curate / curate-person
        Views already filter is_duplicate = false on train splits.

This module provides two supporting actions:

  index   Build the IVF-PQ cosine index on bdd100k.embedding.
          Must run between the embedding and is_duplicate backfills.

  stats   Report duplicate rate using the already-backfilled is_duplicate
          column. Requires is_duplicate to be present — errors otherwise.

CLI usage
---------
  python -m object_detection.dedup --action index
  python -m object_detection.dedup --action stats
"""

from __future__ import annotations

import argparse

import lancedb

SOURCE_TABLE = "bdd100k"
DEFAULT_DB = "data/bdd100k/lancedb"


def index(db_path: str) -> None:
    """Build IVF-PQ cosine index on bdd100k.embedding."""
    db = lancedb.connect(db_path)
    src_tbl = db.open_table(SOURCE_TABLE)
    if "embedding" not in src_tbl.schema.names:
        raise RuntimeError(
            "Column 'embedding' not found. "
            "Run: python -m object_detection.backfill_geneva --gpu --columns embedding"
        )
    n = src_tbl.count_rows()
    print(f"[index] Building IVF-PQ cosine index on {n} rows …")
    src_tbl.create_index(metric="cosine", vector_column_name="embedding")
    print("[index] Index built.")


_DEDUP_CLAUSE = "(is_duplicate IS NULL OR is_duplicate = false)"

# Train-split filters — mirrors manage_views.py BUILTIN_VIEWS + PERSON_VIEWS
_TRAIN_SPLITS: dict[str, str] = {
    "rider_train":            "has_rider = true AND split = 'train'",
    "nighttime_person_train": "timeofday = 'night' AND has_person = true AND split = 'train'",
    "distant_person_train":   "has_person = true AND person_bbox_area_pct < 30.0 AND split = 'train'",
}


def stats(db_path: str) -> None:
    """Report duplicate rate from the backfilled is_duplicate column, globally and per training split."""
    db = lancedb.connect(db_path)
    src_tbl = db.open_table(SOURCE_TABLE)
    if "is_duplicate" not in src_tbl.schema.names:
        raise RuntimeError(
            "Column 'is_duplicate' not found. "
            "Run: python -m object_detection.backfill_geneva --columns is_duplicate"
        )

    total      = src_tbl.count_rows()
    backfilled = src_tbl.count_rows(filter="is_duplicate IS NOT NULL")
    dup_count  = src_tbl.count_rows(filter="is_duplicate = true")

    print(f"\n{'Global':}")
    print(f"  total frames:       {total:>8}")
    print(f"  backfilled:         {backfilled:>8}  ({backfilled / total * 100:.1f}%)")
    print(f"  duplicates:         {dup_count:>8}  ({dup_count / total * 100:.1f}%)")
    print(f"  training-eligible:  {total - dup_count:>8}  ({(total - dup_count) / total * 100:.1f}%)")

    print(f"\n{'Training split impact':}")
    print(f"  {'split':<30}  {'before':>8}  {'after':>8}  {'removed':>8}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name, filt in _TRAIN_SPLITS.items():
        try:
            before = src_tbl.count_rows(filter=filt)
            after  = src_tbl.count_rows(filter=f"{filt} AND {_DEDUP_CLAUSE}")
            removed = before - after
            print(f"  {name:<30}  {before:>8}  {after:>8}  {removed:>8}  ({removed / before * 100:.1f}%)")
        except Exception:
            print(f"  {name:<30}  (column not yet backfilled — skip)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="BDD100K dedup helpers (index + stats).")
    p.add_argument("--action", choices=["index", "stats"], required=True)
    p.add_argument("--db", default=DEFAULT_DB)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.action == "index":
        index(args.db)
    elif args.action == "stats":
        stats(args.db)


if __name__ == "__main__":
    main()
