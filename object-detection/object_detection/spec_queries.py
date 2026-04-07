"""
Curation helpers — query, split, and materialize Geneva-enriched BDD100K slices.

Covers three high-level operations:

1. Counting / previewing rows matching a spec (EDA)
   Uses table.count_rows(filter=...) and table.search().where(...) — the
   correct LanceDB API for plain SQL filtering without vector search.

2. Full-text search on Geneva scene descriptions
   Requires an FTS index on scene_description; lets you iterate on filters
   like "ambulance near crossroad" purely in natural language.

3. Reproducible train/val splits via permutation_builder
   Splits live inside LanceDB — no external CSV manifests needed.

4. Materialising a spec as a Geneva materialized view
   Uses geneva.db.Connection.create_materialized_view() so the view stays
   in sync with source data and can be refreshed incrementally.

Usage
-----
import lancedb, geneva
from object_detection.spec_queries import (
    count_spec, preview_spec, make_fts_index, fts_search,
    make_split, materialize_spec,
)

db  = lancedb.connect("data/bdd100k/lancedb")
tbl = db.open_table("bdd100k")

# How many red-ambulance frames cover >= 20% of the frame?
print(count_spec(tbl, "red_ambulance", bbox_pct=5.0))

# Preview 10 rows
preview_spec(tbl, "red_ambulance").head(10)

# FTS: find frames whose scene description mentions an intersection
make_fts_index(tbl)
fts_search(tbl, "city street crossroad daytime").head(5)

# Reproducible 80/20 split stored inside Lance
perm_tbl = make_split(tbl, seed=42)

# Materialise as a Geneva view (requires a Geneva connection)
gconn = geneva.connect("data/bdd100k/lancedb")
gtbl  = gconn.open_table("bdd100k")
mv    = materialize_spec(gconn, gtbl, spec="red_ambulance", bbox_pct=5.0)
"""

from __future__ import annotations

import lancedb
from lancedb.permutation import Permutation, permutation_builder

# Columns to hide from human-readable previews and search results
_SKIP_COLS = {"image_bytes", "ann_bboxes", "ann_occluded", "ann_truncated"}

# ---------------------------------------------------------------------------
# SQL filter expressions for each spec
# Keep as plain strings so they compose easily with additional WHERE clauses.
# ---------------------------------------------------------------------------

SPEC_FILTERS: dict[str, str] = {
    "red_ambulance": (
        "vehicle_label = 'red_ambulance' "
        "AND vehicle_bbox_area_pct >= {bbox_pct}"
    ),
    "yellow_ambulance": (
        "vehicle_label = 'yellow_ambulance' "
        "AND vehicle_bbox_area_pct >= {bbox_pct}"
    ),
    "traffic_light": (
        "vehicle_label = 'traffic_light' "
        "AND vehicle_confidence >= {min_confidence}"
    ),
    "daytime_clear": (
        "timeofday = 'daytime' AND weather = 'clear'"
    ),
    # Nighttime / rare-class specs (require has_person / has_rider UDF columns)
    "nighttime_person": (
        "timeofday = 'night' AND has_person = true"
    ),
    "rider": (
        "has_rider = true"
    ),
    "nighttime_rider": (
        "timeofday = 'night' AND has_rider = true"
    ),
}


def _build_filter(spec: str, bbox_pct: float = 5.0, min_confidence: float = 0.5, **kwargs) -> str:
    template = SPEC_FILTERS.get(spec)
    if template is None:
        raise ValueError(f"Unknown spec '{spec}'. Choose from: {list(SPEC_FILTERS)}")
    return template.format(bbox_pct=bbox_pct, min_confidence=min_confidence, **kwargs)


# ---------------------------------------------------------------------------
# 1. EDA helpers
# ---------------------------------------------------------------------------

def count_spec(tbl, spec: str, **kwargs) -> int:
    """
    Return the number of rows matching a spec — fast metadata-only count.

    Parameters
    ----------
    tbl     : open LanceDB table (must have Geneva UDF columns backfilled)
    spec    : one of 'red_ambulance', 'yellow_ambulance', 'traffic_light',
              'daytime_clear', 'nighttime_person', 'rider', 'nighttime_rider'
    **kwargs: spec threshold overrides, e.g. bbox_pct=5.0, min_confidence=0.5

    Examples
    --------
    count_spec(tbl, "rider")
    count_spec(tbl, "red_ambulance", bbox_pct=5.0)
    """
    return tbl.count_rows(filter=_build_filter(spec, **kwargs))


def preview_spec(tbl, spec: str, limit: int = 20, **kwargs):
    """
    Return a pandas DataFrame of up to `limit` rows matching a spec.

    Excludes image_bytes to keep the preview readable.
    """
    display_cols = [f for f in tbl.schema.names if f not in _SKIP_COLS]
    return (
        tbl.search()
        .where(_build_filter(spec, **kwargs))
        .select(display_cols)
        .limit(limit)
        .to_pandas()
    )


def spec_counts_summary(tbl) -> dict[str, int]:
    """
    Print a quick count for every spec at default thresholds.
    Useful for an EDA sanity check right after Geneva backfill.

    Example output
    --------------
    red_ambulance      :      423 rows  (bbox_pct >= 20%)
    yellow_ambulance   :      187 rows  (bbox_pct >= 20%)
    traffic_light      :     1832 rows  (conf    >= 0.50)
    daytime_clear      :     3241 rows
    """
    defaults = {
        "red_ambulance":    {"bbox_pct": 5.0},
        "yellow_ambulance": {"bbox_pct": 5.0},
        "traffic_light":    {"min_confidence": 0.5},
        "daytime_clear":    {},
    }
    counts = {}
    for spec, kwargs in defaults.items():
        counts[spec] = count_spec(tbl, spec, **kwargs)
    return counts


# ---------------------------------------------------------------------------
# 2. Full-text search on Geneva scene descriptions
# ---------------------------------------------------------------------------

def make_fts_index(tbl, replace: bool = False) -> None:
    """
    Create an FTS index on the Geneva-generated `scene_description` column.

    Only needed once.  Safe to call again with replace=True to rebuild.
    """
    tbl.create_fts_index("scene_description", replace=replace)
    print("FTS index created on 'scene_description'.")


def fts_search(tbl, query: str, limit: int = 20):
    """
    Full-text search over Geneva scene descriptions.

    Lets researchers iterate on filters without writing SQL — e.g.:
        fts_search(tbl, "city street daytime ambulance")
        fts_search(tbl, "intersection rainy night")

    Returns a pandas DataFrame (image_bytes excluded).
    """
    display_cols = [f for f in tbl.schema.names if f not in _SKIP_COLS]
    return (
        tbl.search(query, query_type="fts", fts_columns="scene_description")
        .select(display_cols)
        .limit(limit)
        .to_pandas()
    )


# ---------------------------------------------------------------------------
# 3. Reproducible train / val split via PermutationBuilder
# ---------------------------------------------------------------------------

def make_split(
    tbl,
    train_ratio: float = 0.8,
    seed: int = 42,
    spec: str | None = None,
    **spec_kwargs,
) -> lancedb.table.LanceTable:
    """
    Build and persist a reproducible train/val split inside LanceDB.

    Returns the permutation table (a Lance table of indices).  Pass it to
    Permutation.from_tables() in your DataLoader to get each split's rows.

    Parameters
    ----------
    tbl         : source LanceDB table
    train_ratio : fraction of rows for training (remainder → val)
    seed        : random seed for reproducibility
    spec        : optional SQL filter to scope the split to a spec subset
    **spec_kwargs : passed through to _build_filter if spec is given

    Example
    -------
    perm_tbl = make_split(tbl, seed=42)
    train = Permutation.from_tables(tbl, perm_tbl, split="train")
    val   = Permutation.from_tables(tbl, perm_tbl, split="val")
    # Use in DataLoader:
    train[0]                          # first training sample
    train.__getitems__([0, 1, 2])     # batch fetch
    """
    val_ratio = round(1.0 - train_ratio, 4)
    builder = permutation_builder(tbl).shuffle(seed=seed)

    if spec:
        builder = builder.filter(_build_filter(spec, **spec_kwargs))

    perm_tbl = (
        builder
        .split_random(
            ratios=[train_ratio, val_ratio],
            split_names=["train", "val"],
            seed=seed,
        )
        .execute()
    )

    # split_id=0 → train, split_id=1 → val (names stored in schema metadata)
    n_train = perm_tbl.count_rows(filter="split_id = 0")
    n_val   = perm_tbl.count_rows(filter="split_id = 1")
    print(f"Split created — train: {n_train}  val: {n_val}  (seed={seed})")
    return perm_tbl


# ---------------------------------------------------------------------------
# 4. Geneva materialized view
# ---------------------------------------------------------------------------

def materialize_spec(
    gconn,
    gtbl,
    spec: str,
    view_name: str | None = None,
    **spec_kwargs,
):
    """
    Create (or refresh) a Geneva materialized view scoped to a spec filter.

    The view is queryable like any Lance table and stays in sync with the
    source table when you call mv.refresh().

    Parameters
    ----------
    gconn     : open Geneva connection (geneva.connect(...))
    gtbl      : source Geneva table
    spec      : one of 'red_ambulance', 'yellow_ambulance', etc.
    view_name : destination table name (default: bdd100k_<spec>)
    **spec_kwargs : spec threshold overrides, e.g. bbox_pct=15.0

    Example
    -------
    import geneva
    gconn = geneva.connect("data/bdd100k/lancedb")
    gtbl  = gconn.open_table("bdd100k")
    mv    = materialize_spec(gconn, gtbl, "red_ambulance", bbox_pct=5.0)

    # Later — refresh to pick up newly ingested rows:
    mv.refresh()

    # Query it like a normal table:
    mv.count_rows()
    mv.search().where("timeofday = 'daytime'").to_pandas()
    """
    spec_kwargs.setdefault("bbox_pct", 20.0)
    spec_kwargs.setdefault("min_confidence", 0.5)
    view_name = view_name or f"bdd100k_{spec}"

    sql_filter = _build_filter(spec, **spec_kwargs)
    query = gtbl.search().where(sql_filter)

    if view_name in gconn.list_tables().tables:
        print(f"View '{view_name}' already exists — refreshing …")
        mv = gconn.open_table(view_name)
        mv.refresh()
    else:
        print(f"Creating materialized view '{view_name}' …")
        mv = gconn.create_materialized_view(view_name, query)
        with gconn.local_ray_context():
            mv.refresh()

    print(f"Done — '{view_name}' has {mv.count_rows()} rows")
    return mv


# ---------------------------------------------------------------------------
# CLI — quick EDA summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Print spec row counts for EDA.")
    p.add_argument("--db", default="data/bdd100k/lancedb")
    p.add_argument("--table", default="bdd100k")
    p.add_argument("--bbox-pct", type=float, default=20.0)
    args = p.parse_args()

    db  = lancedb.connect(args.db)
    tbl = db.open_table(args.table)

    print(f"\nSpec counts — table '{args.table}'  ({tbl.count_rows()} rows total)\n")
    for spec, n in spec_counts_summary(tbl).items():
        print(f"  {spec:25s}  {n:>8,} rows")
