"""
Manage Geneva materialized views for BDD100K curated training splits.

This is the maintenance layer of the LanceDB + Geneva lifecycle:

  ingest → backfill → curate (once) → train
                          ↑               ↓
              new data arrives         checkpoint
                  ↓                       ↓
              refresh views ←── mv.version logged in training run

Views are Geneva-maintained child tables of the `bdd100k` parent.
When new footage is ingested and backfilled, a single `make refresh`
call propagates updates to all curated splits — no training code changes.

Usage
-----
# Create all materialized views (run once after initial backfill):
python -m object_detection.manage_views --action curate --db data/bdd100k/lancedb

# Show current parent + view sizes:
python -m object_detection.manage_views --action status --db data/bdd100k/lancedb

# Refresh all views after new ingest + backfill:
python -m object_detection.manage_views --action refresh --db data/bdd100k/lancedb
"""

from __future__ import annotations

import argparse

import lancedb
import geneva

# ---------------------------------------------------------------------------
# View definitions
# Each view is a named Geneva materialized view over the parent bdd100k table.
# The WHERE clause lives here, not in training scripts — that's the point.
# ---------------------------------------------------------------------------

VIEWS: dict[str, str] = {
    "bdd100k_nighttime_person": (
        "timeofday = 'night' AND has_person = true"
    ),
    "bdd100k_rider": (
        "has_rider = true"
    ),
    "bdd100k_nighttime_rider": (
        "timeofday = 'night' AND has_rider = true"
    ),
}

PARENT_TABLE = "bdd100k"


def _connect(db_path: str):
    """Open both a Geneva connection (for view ops) and a plain LanceDB connection (for row counts)."""
    return geneva.connect(db_path), lancedb.connect(db_path)


def status(db_path: str) -> None:
    """Print current row counts for parent table and all views."""
    gconn, lconn = _connect(db_path)
    existing = set(lconn.list_tables().tables)

    parent = lconn.open_table(PARENT_TABLE)
    print(f"\n{'table':<35}  {'rows':>8}  {'version':>8}  {'filter'}")
    print("-" * 90)
    print(f"  {PARENT_TABLE:<33}  {parent.count_rows():>8}  {parent.version:>8}  (source)")

    for view_name, sql_filter in VIEWS.items():
        if view_name in existing:
            tbl = lconn.open_table(view_name)
            print(f"  {view_name:<33}  {tbl.count_rows():>8}  {tbl.version:>8}  WHERE {sql_filter}")
        else:
            print(f"  {view_name:<33}  {'—':>8}  {'—':>8}  (not yet created)")
    print()


def curate(db_path: str) -> None:
    """Create all materialized views (idempotent — refreshes if already exists)."""
    gconn, lconn = _connect(db_path)
    existing = set(lconn.list_tables().tables)
    gtbl = gconn.open_table(PARENT_TABLE)

    with gconn.local_ray_context():
        for view_name, sql_filter in VIEWS.items():
            query = gtbl.search().where(sql_filter)

            if view_name in existing:
                print(f"[{view_name}] already exists — refreshing …")
                mv = gconn.open_table(view_name)
                mv.refresh()
            else:
                print(f"[{view_name}] creating … (filter: {sql_filter})")
                mv = gconn.create_materialized_view(view_name, query)
                mv.refresh()

            rows = mv.count_rows()
            print(f"[{view_name}] ✓  {rows} rows  (version {mv.version})\n")

    print("All views ready.")


def refresh(db_path: str) -> None:
    """
    Refresh all views after new data has been ingested + backfilled.

    Call this whenever:
      - New BDD frames are appended to the parent table
      - Geneva backfill has run on the new rows
    The views will pick up any new rows that match their filter.
    """
    gconn, lconn = _connect(db_path)
    existing = set(lconn.list_tables().tables)

    parent = lconn.open_table(PARENT_TABLE)
    print(f"\nParent '{PARENT_TABLE}': {parent.count_rows()} rows (version {parent.version})")

    with gconn.local_ray_context():
        for view_name in VIEWS:
            if view_name not in existing:
                print(f"[{view_name}] not found — run `--action curate` first")
                continue

            mv = gconn.open_table(view_name)
            before = mv.count_rows()
            mv.refresh()
            after = mv.count_rows()
            delta = after - before
            sign = f"+{delta}" if delta >= 0 else str(delta)
            print(f"[{view_name}]  {before} → {after} rows  ({sign})  version {mv.version}")

    print("\nRefresh complete.")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Manage Geneva materialized views for BDD100K.")
    p.add_argument("--action", choices=["status", "curate", "refresh"],
                   default="status")
    p.add_argument("--db", default="data/bdd100k/lancedb")
    p.add_argument("--table", default=PARENT_TABLE)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.action == "status":
        status(args.db)
    elif args.action == "curate":
        curate(args.db)
    elif args.action == "refresh":
        refresh(args.db)


if __name__ == "__main__":
    main()
