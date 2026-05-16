"""
Manage Geneva materialised views for the videogen pipeline.

Each named view is a SQL WHERE clause over ``videos_raw`` (or a parent view).
The training script opens the view by name and never sees the filter; the
filter definition lives here.

Two tiers of views, mirroring the two tiers of Geneva backfill:

  Tier 1 — keyword-only.  Runnable right after ``backfill_geneva --tier 1``.
           One view per phase transition × {train, val}.

  Tier 2 — adds quality gates that need Tier-2 backfills:
              motion_strength BETWEEN 2.0 AND 12.0
              metamorphic_score > 0.6
           Plus the union view ``phase_transitions_curated_{train,val}``.

Once Tier 4 dedup runs we patch ``is_duplicate = false`` into every train view.

Actions
-------
  status   Print row counts and versions for parent + all views.
  curate   Create / refresh Tier 1 views (one per transition).
  curate-2 Create / refresh Tier 2 views (requires Tier-2 backfill done).
  refresh  Refresh every existing view.
  drop     Drop every view (keeps the parent table intact).

Usage
-----
python -m videogen.manage_views --action curate
python -m videogen.manage_views --action curate-2
python -m videogen.manage_views --action status
python -m videogen.manage_views --action refresh
"""

from __future__ import annotations

import argparse
import sys

import geneva
import lancedb

from videogen.schema import PHASE_TRANSITIONS


PARENT_TABLE = "videos_raw"
DEFAULT_DB   = "data/videos/lancedb"

UNION_BASE = " OR ".join(f"keyword_{t} = true" for t in PHASE_TRANSITIONS)

# -- Tier 1: keyword views ---------------------------------------------------

def _tier1_views() -> dict[str, str]:
    views: dict[str, str] = {}
    for t in PHASE_TRANSITIONS:
        views[f"phase_{t}_train"] = f"keyword_{t} = true AND split = 'train'"
        views[f"phase_{t}_val"]   = f"keyword_{t} = true AND split = 'val'"
    views["phase_transitions_train"] = f"({UNION_BASE}) AND split = 'train'"
    views["phase_transitions_val"]   = f"({UNION_BASE}) AND split = 'val'"
    return views


# -- Tier 2: quality-gated views (needs motion_strength + metamorphic_score) -

_QUALITY_CLAUSE = (
    "motion_strength BETWEEN 2.0 AND 12.0 "
    "AND metamorphic_score > 0.6 "
    "AND duration_s BETWEEN 4.0 AND 8.0"
)

def _tier2_views() -> dict[str, str]:
    return {
        "phase_transitions_curated_train":
            f"({UNION_BASE}) AND {_QUALITY_CLAUSE} AND split = 'train'",
        "phase_transitions_curated_val":
            f"({UNION_BASE}) AND {_QUALITY_CLAUSE} AND split = 'val'",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect(db_path: str):
    return geneva.connect(db_path), lancedb.connect(db_path)


def _all_views(lconn) -> list[str]:
    return [t for t in lconn.list_tables().tables
            if t != PARENT_TABLE and not t.startswith("__")]


def _dedup_clause(lconn) -> str:
    """Tier-4 dedup gate — applied only on _train views, and only if the
    ``is_duplicate`` column exists yet."""
    try:
        tbl = lconn.open_table(PARENT_TABLE)
        if "is_duplicate" in tbl.schema.names:
            return " AND (is_duplicate IS NULL OR is_duplicate = false)"
    except Exception:
        pass
    return ""


def _create_or_refresh(gconn, gtbl, name: str, sql_filter: str, existing) -> None:
    if name in existing:
        print(f"  [{name}] dropping (filter may have changed) …")
        gconn.drop_table(name)
    print(f"  [{name}] creating  filter: {sql_filter}")
    query = gtbl.search().where(sql_filter)
    mv = gconn.create_materialized_view(name, query)
    mv.refresh()
    print(f"  [{name}] ✓  {mv.count_rows()} rows  (version {mv.version})\n")


def _curate(db_path: str, views: dict[str, str]) -> None:
    gconn, lconn = _connect(db_path)
    if PARENT_TABLE not in set(lconn.list_tables().tables):
        raise SystemExit(f"Parent table '{PARENT_TABLE}' missing — run ingest first.")
    existing = set(lconn.list_tables().tables)
    gtbl = gconn.open_table(PARENT_TABLE)
    dedup = _dedup_clause(lconn)

    with gconn.local_ray_context():
        for name, sql_filter in views.items():
            d = dedup if name.endswith("_train") else ""
            _create_or_refresh(gconn, gtbl, name, sql_filter + d, existing)

    print("Done.")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def status(db_path: str) -> None:
    _, lconn = _connect(db_path)
    if PARENT_TABLE not in set(lconn.list_tables().tables):
        print(f"Parent table '{PARENT_TABLE}' missing — run ingest first.")
        return
    parent = lconn.open_table(PARENT_TABLE)
    print(f"\n{'table':<40} {'rows':>10} {'version':>8}")
    print("-" * 65)
    print(f"  {PARENT_TABLE:<38} {parent.count_rows():>10,} {parent.version:>8}  (source)")
    for name in sorted(_all_views(lconn)):
        tbl = lconn.open_table(name)
        print(f"  {name:<38} {tbl.count_rows():>10,} {tbl.version:>8}")
    print()


def curate(db_path: str) -> None:
    _curate(db_path, _tier1_views())


def curate_2(db_path: str) -> None:
    _, lconn = _connect(db_path)
    if PARENT_TABLE not in set(lconn.list_tables().tables):
        raise SystemExit(f"Parent table '{PARENT_TABLE}' missing — run ingest first.")
    have = set(lconn.open_table(PARENT_TABLE).schema.names)
    missing = [c for c in ("motion_strength", "metamorphic_score") if c not in have]
    if missing:
        raise SystemExit(
            f"Tier-2 view curation needs columns {missing} — "
            f"run `backfill_geneva --tier 2` first."
        )
    _curate(db_path, _tier2_views())


def refresh(db_path: str) -> None:
    gconn, lconn = _connect(db_path)
    views = _all_views(lconn)
    if not views:
        print("No views found.")
        return
    parent = lconn.open_table(PARENT_TABLE)
    print(f"\nParent '{PARENT_TABLE}': {parent.count_rows():,} rows  "
          f"(version {parent.version})\n")
    with gconn.local_ray_context():
        for name in views:
            mv = gconn.open_table(name)
            before = mv.count_rows()
            mv.refresh()
            after = mv.count_rows()
            delta = after - before
            sign = f"+{delta}" if delta >= 0 else str(delta)
            print(f"  [{name}]  {before:,} → {after:,}  ({sign})  v{mv.version}")
    print("\nRefresh complete.")


def drop(db_path: str) -> None:
    gconn, lconn = _connect(db_path)
    views = _all_views(lconn)
    if not views:
        print("No views found.")
        return
    for name in views:
        try:
            gconn.drop_table(name)
            print(f"  [{name}] dropped")
        except Exception as e:
            print(f"  [{name}] drop failed: {e}")
    print(f"\nDropped {len(views)} view(s).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Manage Geneva materialised views for the videogen pipeline."
    )
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--action",
                   choices=["status", "curate", "curate-2", "refresh", "drop"],
                   default="status")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    {
        "status":   status,
        "curate":   curate,
        "curate-2": curate_2,
        "refresh":  refresh,
        "drop":     drop,
    }[args.action](args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
