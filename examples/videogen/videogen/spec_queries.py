"""
Curation helpers for the videogen pipeline.

Three concerns:

  1. **SQL specs** — named WHERE clauses we reuse across EDA + view creation.
     Tier 1 specs are keyword-only.  Tier 2 specs add quality gates
     (``motion_strength``, ``metamorphic_score``, etc.) and only resolve
     after a Tier-2 backfill.

  2. **Count / preview / sample** helpers — what the EDA notebook calls into.

  3. **Full-text search + (eventually) CLIP retrieval** helpers — the same
     pattern as ``object-detection/spec_queries.py``.
"""

from __future__ import annotations

from typing import Iterable

import lancedb
import pyarrow as pa

from videogen.schema import PHASE_TRANSITIONS


# Columns to hide when previewing rows (video_bytes is a blob descriptor,
# the fixed-size-list latents are humongous, etc.).
_HIDE_COLS = {
    "video_bytes",
    "t5_input_ids", "t5_hidden_states", "vae_latent",
    "clip_emb_text", "clip_emb_video",
    "dhash_first_last",
}


# ---------------------------------------------------------------------------
# SQL specs
# ---------------------------------------------------------------------------

def _split(split: str) -> str:
    return f"split = '{split}'"


def keyword_spec(transition: str, split: str = "train") -> str:
    """Tier 1: rows whose caption matches one phase-transition keyword."""
    if transition not in PHASE_TRANSITIONS:
        raise ValueError(f"Unknown transition '{transition}'. "
                         f"Available: {PHASE_TRANSITIONS}")
    return f"keyword_{transition} = true AND {_split(split)}"


def union_keyword_spec(split: str = "train") -> str:
    """Tier 1: union over all five phase transitions."""
    union = " OR ".join(f"keyword_{t} = true" for t in PHASE_TRANSITIONS)
    return f"({union}) AND {_split(split)}"


def curated_spec(split: str = "train",
                 motion: tuple[float, float] = (2.0, 12.0),
                 mtscore_min: float = 0.6,
                 duration: tuple[float, float] = (4.0, 8.0)) -> str:
    """Tier 2: keyword union ∩ quality gates."""
    lo, hi   = motion
    d_lo, dhi = duration
    return (
        f"({' OR '.join(f'keyword_{t} = true' for t in PHASE_TRANSITIONS)})"
        f" AND motion_strength BETWEEN {lo} AND {hi}"
        f" AND metamorphic_score > {mtscore_min}"
        f" AND duration_s BETWEEN {d_lo} AND {dhi}"
        f" AND {_split(split)}"
    )


# ---------------------------------------------------------------------------
# Lance helpers
# ---------------------------------------------------------------------------

def count(tbl, where: str) -> int:
    """Row count under a SQL WHERE."""
    return tbl.count_rows(filter=where)


def preview(tbl, where: str, n: int = 5, columns: Iterable[str] | None = None):
    """Return a pandas DataFrame of up to ``n`` matching rows for EDA.

    Drops large/blob columns by default so the result is readable in a
    notebook.  Pass ``columns=[...]`` to override.
    """
    schema_cols = set(tbl.schema.names)
    if columns is None:
        columns = [c for c in tbl.schema.names if c not in _HIDE_COLS]
    columns = [c for c in columns if c in schema_cols]
    return (
        tbl.search()
        .where(where)
        .select(columns)
        .limit(n)
        .to_pandas()
    )


def sample_ids(tbl, where: str, n: int = 20) -> list[str]:
    """Cheap qualitative sampler — return ``n`` clip_ids matching ``where``."""
    arr = (
        tbl.search()
        .where(where)
        .select(["clip_id"])
        .limit(n)
        .to_arrow()
    )
    return arr.column("clip_id").to_pylist()


def distribution(tbl, column: str, where: str | None = None) -> "pa.Table":
    """Group-by histogram on a scalar column (small-cardinality only)."""
    q = tbl.search()
    if where:
        q = q.where(where)
    tbl_arrow = q.select([column]).to_arrow()
    return tbl_arrow.group_by([column]).aggregate([([], "count_all")])


def ensure_caption_fts(tbl, column: str = "caption") -> None:
    """Create an FTS index on ``caption`` if missing (idempotent)."""
    try:
        tbl.create_fts_index(column, replace=False)
    except Exception:
        # Already exists or backend doesn't need a replace=False — fine.
        pass


def fts(tbl, query: str, n: int = 10, where: str | None = None):
    """Full-text search the caption column.  Falls back to a simple
    case-insensitive substring scan if FTS isn't available."""
    try:
        q = tbl.search(query, query_type="fts").limit(n)
        if where:
            q = q.where(where, prefilter=True)
        return q.to_pandas()
    except Exception:
        like = query.lower().split()[0]
        sql_where = f"lower(caption) LIKE '%{like}%'"
        if where:
            sql_where = f"({sql_where}) AND ({where})"
        return preview(tbl, sql_where, n)


# ---------------------------------------------------------------------------
# Pre-canned curation summary — used by EDA notebook + verify_pipeline
# ---------------------------------------------------------------------------

def summarise(tbl) -> list[tuple[str, int]]:
    """One row per transition + grand totals.  Returns ``(label, count)``
    pairs for tabular display.

    Skips tiers that haven't been backfilled yet.
    """
    rows: list[tuple[str, int]] = [
        ("total rows", len(tbl)),
        ("train", count(tbl, "split = 'train'")),
        ("val",   count(tbl, "split = 'val'")),
    ]
    schema_cols = set(tbl.schema.names)
    if "any_phase_keyword" in schema_cols:
        rows.append(("any phase keyword", count(tbl, "any_phase_keyword = true")))
    for t in PHASE_TRANSITIONS:
        col = f"keyword_{t}"
        if col in schema_cols:
            rows.append((f"  {t}", count(tbl, f"{col} = true")))
    if {"motion_strength", "metamorphic_score"}.issubset(schema_cols):
        rows.append(("curated (Tier 2)", count(tbl, curated_spec("train"))))
    if "is_duplicate" in schema_cols:
        rows.append(("duplicates flagged",
                     count(tbl, "is_duplicate = true")))
    return rows
