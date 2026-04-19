"""
Pipeline verification — prints a full end-to-end status summary.

Covers:
  1. Source table health (row count, column presence, backfill coverage)
  2. Dedup stats (dhash + is_duplicate)
  3. Materialized view row counts
  4. Model eval — baseline COCO vs fine-tuned checkpoints (if present)

Usage
-----
python -m object_detection.verify_pipeline
python -m object_detection.verify_pipeline --db data/bdd100k/lancedb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lancedb

DEFAULT_DB = "data/bdd100k/lancedb"
SOURCE_TABLE = "bdd100k"

# Columns expected after a full backfill, in display order
_BACKFILL_COLUMNS = [
    "has_person", "has_rider",
    "white_balance", "scene_description", "scene_has_crossroad", "scene_has_mountain",
    "person_bbox_area_pct",
    "dhash", "is_duplicate",
]

# Materialized views expected after manage_views curate + curate-person
_EXPECTED_VIEWS = [
    "bdd100k_rider_train",            "bdd100k_rider_val",
    "bdd100k_nighttime_person_train", "bdd100k_nighttime_person_val",
    "bdd100k_distant_person_train",   "bdd100k_distant_person_val",
]

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "—"


def _check_source_table(db: lancedb.DBConnection) -> list[tuple]:
    """Returns rows for the summary table."""
    rows = []
    try:
        tbl = db.open_table(SOURCE_TABLE)
    except Exception:
        rows.append((FAIL, "source table", f"'{SOURCE_TABLE}' not found — run ingest_bdd"))
        return rows

    total = tbl.count_rows()
    rows.append((PASS, "source table", f"{total:,} rows"))

    schema_names = set(tbl.schema.names)
    for col in _BACKFILL_COLUMNS:
        if col not in schema_names:
            rows.append((FAIL, f"column: {col}", "missing — run backfill_geneva"))
            continue
        nulls = tbl.count_rows(filter=f"{col} IS NULL")
        filled = total - nulls
        if nulls == 0:
            rows.append((PASS, f"column: {col}", f"{filled:,} / {total:,} filled (100%)"))
        else:
            status = FAIL if filled == 0 else "WARN"
            rows.append((status, f"column: {col}",
                         f"{filled:,} / {total:,} filled ({_pct(filled, total)}) — {nulls:,} NULL"))
    return rows


def _check_dedup(db: lancedb.DBConnection) -> list[tuple]:
    rows = []
    try:
        tbl = db.open_table(SOURCE_TABLE)
    except Exception:
        return rows

    if "is_duplicate" not in tbl.schema.names:
        rows.append((SKIP, "dedup", "is_duplicate column missing"))
        return rows

    total = tbl.count_rows()
    dups   = tbl.count_rows(filter="is_duplicate = true")
    clean  = total - dups
    rows.append((PASS, "dedup: duplicate rate",
                 f"{dups:,} flagged ({_pct(dups, total)}) — {clean:,} training-eligible"))
    return rows


_CHECKPOINT_DIRS: dict[str, str] = {
    "Rider":            "checkpoints/rider",
    "Nighttime person": "checkpoints/nighttime_person",
    "Distant person":   "checkpoints/distant_person",
}


def _check_models() -> list[tuple]:
    rows = []
    for mode, ckpt_dir in _CHECKPOINT_DIRS.items():
        metrics_path = Path(ckpt_dir) / "metrics.json"
        ckpt_path    = Path(ckpt_dir) / "fasterrcnn_bdd_finetuned.pt"
        if not ckpt_path.exists():
            rows.append((SKIP, f"train: {mode}", f"checkpoint not found — run train_detector.py"))
            continue
        if not metrics_path.exists():
            rows.append(("WARN", f"train: {mode}", "checkpoint found but metrics.json missing"))
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        b  = m["baseline"]
        ft = m["finetuned"]
        delta_map    = ft["map_50"]   - b["map_50"]
        delta_recall = ft["recall"]   - b["recall"]
        status = PASS if delta_map > 0 else FAIL
        rows.append((status, f"train: {mode}",
                     f"mAP  {b['map_50']:.4f} → {ft['map_50']:.4f}  ({delta_map:+.4f})  |  "
                     f"recall  {b['recall']:.4f} → {ft['recall']:.4f}  ({delta_recall:+.4f})  "
                     f"[{m['epochs']} epochs, val={m['val_table']}]"))
    return rows


def _check_views(db: lancedb.DBConnection) -> list[tuple]:
    rows = []
    for view in _EXPECTED_VIEWS:
        try:
            n = db.open_table(view).count_rows()
            rows.append((PASS, f"view: {view}", f"{n:,} rows"))
        except Exception:
            rows.append((FAIL, f"view: {view}", "missing — run manage_views"))
    return rows


def _print_summary(all_rows: list[tuple]) -> None:
    w_status = max(len(r[0]) for r in all_rows)
    w_key    = max(len(r[1]) for r in all_rows)

    print()
    print("=" * 80)
    print("  PIPELINE VERIFICATION SUMMARY")
    print("=" * 80)
    header = f"  {'STATUS':<{w_status}}  {'CHECK':<{w_key}}  DETAIL"
    print(header)
    print("  " + "-" * (len(header) - 2))

    section = ""
    for status, key, detail in all_rows:
        cur_section = key.split(":")[0]
        if cur_section != section:
            section = cur_section
            print()
        icon = {"PASS": "✓", "FAIL": "✗", "WARN": "!", "SKIP": "–"}.get(status, " ")
        print(f"  {icon} {status:<{w_status}}  {key:<{w_key}}  {detail}")

    print()
    n_fail = sum(1 for r in all_rows if r[0] == FAIL)
    n_warn = sum(1 for r in all_rows if r[0] == "WARN")
    n_pass = sum(1 for r in all_rows if r[0] == PASS)
    print(f"  {n_pass} passed  {n_warn} warnings  {n_fail} failed")
    print("=" * 80)
    print()


def verify(db_path: str) -> None:
    db = lancedb.connect(db_path)
    rows: list[tuple] = []
    rows += _check_source_table(db)
    rows += _check_dedup(db)
    rows += _check_views(db)
    rows += _check_models()
    _print_summary(rows)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Verify the full BDD100K pipeline.")
    p.add_argument("--db", default=DEFAULT_DB)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    verify(args.db)


if __name__ == "__main__":
    main()
