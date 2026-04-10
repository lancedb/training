"""
Deduplication helpers for BDD100K.

The full dedup pipeline follows the same Geneva backfill pattern as every
other feature column:

  1. backfill_geneva --gpu --columns dhash
        dHash (64-bit perceptual hash, binary float32 vector) written to bdd100k.dhash.
        Computed on GPU: JPEG decode via nvjpeg, grayscale + 9×8 resize as tensor ops.

  2. dedup --action index
        IVF L2 index on bdd100k.dhash (required before step 3).
        For binary vectors, L2² = Hamming distance — no conversion needed.

  3. backfill_geneva --columns is_duplicate
        Per-row: nearest non-self neighbour found via vector search.
        is_duplicate = True when Hamming distance ≤ 10 (L2² ≤ 10).

  4. manage_views --action curate / curate-person
        Views already filter is_duplicate = false on train splits.

This module provides two supporting actions:

  index   Build the IVF L2 index on bdd100k.dhash.
          Must run between the dhash and is_duplicate backfills.

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
import pyarrow as pa

SOURCE_TABLE = "bdd100k"
DEFAULT_DB = "data/bdd100k/lancedb"


def index(db_path: str) -> None:
    """Build IVF L2 index on bdd100k.dhash."""
    db = lancedb.connect(db_path)
    src_tbl = db.open_table(SOURCE_TABLE)
    if "dhash" not in src_tbl.schema.names:
        raise RuntimeError(
            "Column 'dhash' not found. "
            "Run: python -m object_detection.backfill_geneva --gpu --columns dhash"
        )
    n = src_tbl.count_rows()
    print(f"[index] Building IVF L2 index on dhash ({n} rows) …")
    src_tbl.create_index(metric="l2", vector_column_name="dhash")
    print("[index] Index built.")


def inject(db_path: str, n: int) -> None:
    """
    Append N exact-copy rows (new image_id prefixed 'syndup_') to bdd100k.

    Injected rows have dhash=NULL and is_duplicate=NULL so the backfill pipeline
    picks them up. After backfill, every injected row should have Hamming=0 with
    its source — the strictest possible test of the dedup pipeline.

    Remove injected rows afterwards with: dedup --action clean
    """
    db = lancedb.connect(db_path)
    tbl = db.open_table(SOURCE_TABLE)

    schema_names = set(tbl.schema.names)

    # Select all base columns + dhash (copy hash directly — no backfill needed)
    select_cols = ["image_id", "image_bytes", "width", "height",
                   "weather", "scene", "timeofday", "timestamp",
                   "ann_categories", "ann_bboxes", "ann_occluded",
                   "ann_truncated", "ann_traffic_light_colors", "num_annotations",
                   "split"]
    if "dhash" in schema_names:
        select_cols.append("dhash")

    rows = (
        tbl.search()
        .where("split = 'train' AND dhash IS NOT NULL")
        .select(select_cols)
        .limit(n)
        .to_arrow()
    )

    cols = {name: rows[name] for name in rows.schema.names}

    # New image_ids so vector search `image_id != iid` filter works correctly
    cols["image_id"] = pa.array([f"syndup_{iid}" for iid in rows["image_id"].to_pylist()])

    # Copy metadata columns that exist; leave is_duplicate NULL for backfill
    for col in ["has_person", "has_rider", "white_balance", "scene_description",
                "scene_has_crossroad", "scene_has_mountain", "person_bbox_area_pct"]:
        if col in schema_names:
            cols[col] = rows[col] if col in rows.schema.names else pa.nulls(len(rows), type=tbl.schema.field(col).type)
    if "is_duplicate" in schema_names:
        cols["is_duplicate"] = pa.nulls(len(rows), type=tbl.schema.field("is_duplicate").type)

    tbl.add(pa.table(cols))
    print(f"[inject] Appended {len(rows)} synthetic duplicate rows (image_id prefix: 'syndup_')")
    print(f"         dhash copied directly — no dhash backfill needed.")
    print(f"         Now run: dedup --action index")
    print(f"                  backfill_geneva --columns is_duplicate --overwrite")
    print(f"                  dedup --action verify")


def verify(db_path: str) -> None:
    """Check that injected synthetic duplicates were correctly flagged."""
    db = lancedb.connect(db_path)
    tbl = db.open_table(SOURCE_TABLE)

    total_injected  = tbl.count_rows(filter="starts_with(image_id, 'syndup_')")
    flagged         = tbl.count_rows(filter="starts_with(image_id, 'syndup_') AND is_duplicate = true")
    not_flagged     = total_injected - flagged

    print(f"\nSynthetic duplicate verification")
    print(f"  injected rows:   {total_injected}")
    print(f"  flagged:         {flagged}  ({flagged / max(total_injected, 1) * 100:.1f}%)")
    print(f"  missed:          {not_flagged}")
    if not_flagged == 0:
        print(f"  PASS — all synthetic duplicates caught")
    else:
        print(f"  FAIL — {not_flagged} synthetic duplicates missed (threshold too strict?)")


def clean(db_path: str) -> None:
    """Remove injected synthetic duplicate rows from bdd100k."""
    db = lancedb.connect(db_path)
    tbl = db.open_table(SOURCE_TABLE)
    before = tbl.count_rows()
    tbl.delete("starts_with(image_id, 'syndup_')")
    after = tbl.count_rows()
    print(f"[clean] Removed {before - after} synthetic rows ({before} → {after})")


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
    p = argparse.ArgumentParser(description="BDD100K dedup helpers.")
    p.add_argument("--action", choices=["index", "stats", "inject", "verify", "clean"], required=True)
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--n", type=int, default=1000,
                   help="Number of rows to inject (default: 1000). Only used with --action inject.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.action == "index":
        index(args.db)
    elif args.action == "stats":
        stats(args.db)
    elif args.action == "inject":
        inject(args.db, args.n)
    elif args.action == "verify":
        verify(args.db)
    elif args.action == "clean":
        clean(args.db)


if __name__ == "__main__":
    main()
