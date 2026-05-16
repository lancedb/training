"""
Pipeline verification — prints a single status table covering every stage.

Walks (in order):

  1. Source table health (row count, schema completeness).
  2. Tier-1 backfill coverage (caption_length, keyword_*, any_phase_keyword).
  3. Tier-2 backfill coverage (clip_emb_*, motion_strength, metamorphic_score).
  4. Tier-3 backfill coverage (t5_input_ids, t5_hidden_states, vae_latent).
  5. Tier-4 dedup coverage (dhash_first_last, is_duplicate).
  6. Materialised view row counts (one per phase transition + the curated union).
  7. Per-transition row counts after dedup (if Tier 4 done).

A column that hasn't been backfilled yet is reported as ``PEND`` (pending,
not a failure) so the script is safe to run at every stage of the pipeline.

Usage
-----
python -m videogen.verify_pipeline
python -m videogen.verify_pipeline --db data/videos/lancedb
"""

from __future__ import annotations

import argparse

import lancedb

from videogen.schema import PHASE_TRANSITIONS, TIER_FIELD_NAMES


DEFAULT_DB    = "data/videos/lancedb"
PARENT_TABLE  = "videos_raw"

# Tier-1 views — one per transition, train+val, plus the union train+val.
_TIER1_VIEWS = (
    [f"phase_{t}_{s}" for t in PHASE_TRANSITIONS for s in ("train", "val")]
    + ["phase_transitions_train", "phase_transitions_val"]
)
_TIER2_VIEWS = ["phase_transitions_curated_train", "phase_transitions_curated_val"]


# Status labels.
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
PEND = "PEND"  # column not yet backfilled — not a failure


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "—"


def _check_source(db) -> list[tuple]:
    rows = []
    if PARENT_TABLE not in db.list_tables().tables:
        rows.append((FAIL, "source table", f"'{PARENT_TABLE}' not found — run ingest"))
        return rows
    tbl = db.open_table(PARENT_TABLE)
    rows.append((PASS, "source table",
                 f"{tbl.count_rows():,} rows  version={tbl.version}"))
    return rows


def _check_columns(db) -> list[tuple]:
    rows = []
    if PARENT_TABLE not in db.list_tables().tables:
        return rows
    tbl = db.open_table(PARENT_TABLE)
    total = tbl.count_rows()
    schema_cols = set(tbl.schema.names)

    for tier, cols in TIER_FIELD_NAMES.items():
        for col in cols:
            if col not in schema_cols:
                rows.append((PEND, f"T{tier} column: {col}",
                             "not yet added — run backfill_geneva"))
                continue
            try:
                nulls = tbl.count_rows(filter=f"{col} IS NULL")
            except Exception as e:
                rows.append((WARN, f"T{tier} column: {col}",
                             f"could not count NULLs: {e}"))
                continue
            filled = total - nulls
            if filled == 0:
                rows.append((PEND, f"T{tier} column: {col}",
                             f"present but unfilled — run backfill_geneva"))
            elif nulls == 0:
                rows.append((PASS, f"T{tier} column: {col}",
                             f"{filled:,} / {total:,} filled (100%)"))
            else:
                rows.append((WARN, f"T{tier} column: {col}",
                             f"{filled:,} / {total:,} filled "
                             f"({_pct(filled, total)}) — {nulls:,} NULL"))
    return rows


def _check_views(db) -> list[tuple]:
    rows = []
    tables = set(db.list_tables().tables)
    for view in _TIER1_VIEWS + _TIER2_VIEWS:
        if view in tables:
            mv = db.open_table(view)
            rows.append((PASS, f"view: {view}",
                         f"{mv.count_rows():,} rows  version={mv.version}"))
        else:
            tier = "Tier 1" if view in _TIER1_VIEWS else "Tier 2"
            rows.append((PEND, f"view: {view}",
                         f"missing — run `manage_views --action curate"
                         f"{'' if tier == 'Tier 1' else '-2'}`"))
    return rows


def _check_dedup(db) -> list[tuple]:
    rows = []
    if PARENT_TABLE not in db.list_tables().tables:
        return rows
    tbl = db.open_table(PARENT_TABLE)
    if "is_duplicate" not in tbl.schema.names:
        rows.append((PEND, "dedup",
                     "is_duplicate column missing — Tier 4 backfill pending"))
        return rows
    flagged  = tbl.count_rows(filter="is_duplicate = true")
    eligible = tbl.count_rows(filter="is_duplicate IS NULL OR is_duplicate = false")
    rows.append((PASS, "dedup",
                 f"{flagged:,} flagged  ({_pct(flagged, flagged + eligible)})  "
                 f"— {eligible:,} training-eligible"))
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_PAINT = {
    PASS: "\033[32m",  # green
    WARN: "\033[33m",  # yellow
    PEND: "\033[36m",  # cyan
    FAIL: "\033[31m",  # red
}
_RESET = "\033[0m"


def _render(rows: list[tuple], use_color: bool = True) -> None:
    print()
    print("  STATUS  CHECK                                       DETAIL")
    print("  " + "-" * 72)
    counts = {PASS: 0, WARN: 0, FAIL: 0, PEND: 0}
    for status, check, detail in rows:
        counts[status] = counts.get(status, 0) + 1
        s = f"{_PAINT[status]}✓{_RESET} {status}" if use_color and status == PASS else \
            f"{_PAINT[status]}!{_RESET} {status}" if use_color and status == WARN else \
            f"{_PAINT[status]}…{_RESET} {status}" if use_color and status == PEND else \
            f"{_PAINT[status]}✗{_RESET} {status}" if use_color and status == FAIL else \
            status
        print(f"  {s:<7} {check:<42}  {detail}")
    print()
    print(f"  {counts[PASS]} passed  {counts[WARN]} warnings  "
          f"{counts[PEND]} pending  {counts[FAIL]} failed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def verify(db_path: str, no_color: bool = False) -> int:
    db = lancedb.connect(db_path)
    rows: list[tuple] = []
    rows += _check_source(db)
    rows += _check_columns(db)
    rows += _check_dedup(db)
    rows += _check_views(db)
    _render(rows, use_color=not no_color)
    return 0 if not any(s == FAIL for s, *_ in rows) else 1


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Verify the videogen pipeline state.")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--no-color", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    return verify(args.db, no_color=args.no_color)


if __name__ == "__main__":
    raise SystemExit(main())
