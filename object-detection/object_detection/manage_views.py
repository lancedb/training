"""
Manage Geneva materialized views for BDD100K curated training splits.

Each view is a permanent, refreshable slice of the parent table defined
by a SQL WHERE clause.  The curation definition lives here — not in the
training command.  Training scripts just open the view by name.

Lifecycle
---------
  ingest → backfill → curate (once) → train
                          ↑               ↓
              new data arrives         checkpoint
                  ↓                       ↓
              refresh views ←── mv.version logged in training run

Two tiers of views, matching the two tiers of backfill:

  Tier 1  Annotation-based (has_person, has_rider) — fast, no GPU needed.
          Covers nighttime pedestrians, riders, nighttime riders.

  Tier 2  Model-inference-based (person_bbox_area_pct) — requires GPU UDF
          backfill first.  Covers prominent close-range pedestrians: frames
          where a person occupies >5% of the frame area, ensuring the
          training signal focuses on legible, nearby pedestrians rather than
          distant background figures.

Actions
-------
  curate          Create Tier 1 views (run once after Tier 1 backfill).
  curate-person   Create Tier 2 close-range pedestrian views (after GPU backfill).
  add             Create any custom view from an arbitrary SQL filter.
  refresh         Refresh all views after new data is ingested + backfilled.
  status          Print row counts and versions for parent + all views.

Usage
-----
# Tier 1 — annotation-based views:
python -m object_detection.manage_views --action curate

# Tier 2 — close-range pedestrian views (requires person_bbox_area_pct backfill):
python -m object_detection.manage_views --action curate-person

# Custom view (any SQL filter over any backfilled column):
python -m object_detection.manage_views --action add \\
    --name bdd100k_foggy \\
    --filter "weather = 'foggy' AND split = 'train'"

# After new data arrives:
python -m object_detection.manage_views --action refresh

# Check what exists:
python -m object_detection.manage_views --action status
"""

from __future__ import annotations

import argparse

import lancedb
import geneva

PARENT_TABLE = "bdd100k"

# Tier 1 — annotation-based filters, no GPU backfill required.
# Run: python -m object_detection.manage_views --action curate
BUILTIN_VIEWS: dict[str, str] = {
    "bdd100k_nighttime_person_train": "timeofday = 'night' AND has_person = true AND split = 'train'",
    "bdd100k_nighttime_person_val":   "timeofday = 'night' AND has_person = true AND split = 'val'",
    "bdd100k_rider_train":            "has_rider = true AND split = 'train'",
    "bdd100k_rider_val":              "has_rider = true AND split = 'val'",
    "bdd100k_nighttime_rider_train":  "timeofday = 'night' AND has_rider = true AND split = 'train'",
    "bdd100k_nighttime_rider_val":    "timeofday = 'night' AND has_rider = true AND split = 'val'",
}

# Tier 2 — requires person_bbox_area_pct GPU backfill first.
# Frames where a detected person covers >5% of the frame — close-range pedestrians.
# Run: python -m object_detection.manage_views --action curate-person
_CP_FILTER = "has_person = true AND person_bbox_area_pct > 15.0"
PERSON_VIEWS: dict[str, str] = {
    "bdd100k_close_range_person_train": f"{_CP_FILTER} AND split = 'train'",
    "bdd100k_close_range_person_val":   f"{_CP_FILTER} AND split = 'val'",
}


def _connect(db_path: str):
    """Return (geneva_conn, lancedb_conn) — both are needed for different ops."""
    return geneva.connect(db_path), lancedb.connect(db_path)


def _all_views(lconn) -> list[str]:
    """All tables in the DB that aren't the parent or a system table."""
    return [
        t for t in lconn.list_tables().tables
        if t != PARENT_TABLE and not t.startswith("__")
    ]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def curate(db_path: str) -> None:
    """Create Tier 1 materialized views (annotation-based, no GPU needed)."""
    gconn, lconn = _connect(db_path)
    existing = set(lconn.list_tables().tables)
    gtbl = gconn.open_table(PARENT_TABLE)

    with gconn.local_ray_context():
        for name, sql_filter in BUILTIN_VIEWS.items():
            _create_or_refresh(gconn, gtbl, name, sql_filter, existing)

    print("All Tier 1 views ready.")


def curate_person(db_path: str) -> None:
    """Create Tier 2 close-range pedestrian views (requires person_bbox_area_pct backfill)."""
    gconn, lconn = _connect(db_path)
    existing = set(lconn.list_tables().tables)
    gtbl = gconn.open_table(PARENT_TABLE)

    with gconn.local_ray_context():
        for name, sql_filter in PERSON_VIEWS.items():
            _create_or_refresh(gconn, gtbl, name, sql_filter, existing)

    print("Close-range pedestrian views ready.")


def add(db_path: str, name: str, sql_filter: str) -> None:
    """Create a single custom materialized view."""
    gconn, lconn = _connect(db_path)
    existing = set(lconn.list_tables().tables)
    gtbl = gconn.open_table(PARENT_TABLE)

    with gconn.local_ray_context():
        _create_or_refresh(gconn, gtbl, name, sql_filter, existing)


def refresh(db_path: str) -> None:
    """Refresh all views after new data has been ingested + backfilled."""
    gconn, lconn = _connect(db_path)
    views = _all_views(lconn)

    parent = lconn.open_table(PARENT_TABLE)
    print(f"\nParent '{PARENT_TABLE}': {parent.count_rows()} rows (version {parent.version})")

    if not views:
        print("No views found. Run --action curate or --action add first.")
        return

    with gconn.local_ray_context():
        for view_name in views:
            mv = gconn.open_table(view_name)
            before = mv.count_rows()
            mv.refresh()
            after = mv.count_rows()
            delta = after - before
            sign = f"+{delta}" if delta >= 0 else str(delta)
            print(f"[{view_name}]  {before} → {after} rows  ({sign})  version {mv.version}")

    print("\nRefresh complete.")


def status(db_path: str) -> None:
    """Print row counts and versions for the parent table and all views."""
    gconn, lconn = _connect(db_path)
    views = _all_views(lconn)

    parent = lconn.open_table(PARENT_TABLE)
    print(f"\n{'table':<40}  {'rows':>8}  {'version':>8}")
    print("-" * 65)
    print(f"  {PARENT_TABLE:<38}  {parent.count_rows():>8}  {parent.version:>8}  (source)")

    for view_name in views:
        tbl = lconn.open_table(view_name)
        print(f"  {view_name:<38}  {tbl.count_rows():>8}  {tbl.version:>8}")
    print()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _create_or_refresh(gconn, gtbl, name: str, sql_filter: str, existing: set) -> None:
    if name in existing:
        print(f"[{name}] already exists — refreshing …")
        mv = gconn.open_table(name)
        mv.refresh()
    else:
        print(f"[{name}] creating … (filter: {sql_filter})")
        query = gtbl.search().where(sql_filter)
        mv = gconn.create_materialized_view(name, query)
        mv.refresh()

    print(f"[{name}] ✓  {mv.count_rows()} rows  (version {mv.version})\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Manage Geneva materialized views for BDD100K.")
    p.add_argument("--action",
                   choices=["status", "curate", "curate-person", "refresh", "add"],
                   default="status")
    p.add_argument("--db",     default="data/bdd100k/lancedb")
    p.add_argument("--name",   default=None,
                   help="View name (required for --action add)")
    p.add_argument("--filter", default=None,
                   help="SQL WHERE clause (required for --action add)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.action == "add":
        if not args.name or not args.filter:
            print("--action add requires both --name and --filter")
            raise SystemExit(1)
        add(args.db, args.name, args.filter)
    elif args.action == "curate":
        curate(args.db)
    elif args.action == "curate-person":
        curate_person(args.db)
    elif args.action == "refresh":
        refresh(args.db)
    elif args.action == "status":
        status(args.db)


if __name__ == "__main__":
    main()
