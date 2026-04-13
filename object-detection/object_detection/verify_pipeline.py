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

# Checkpoints produced by train_detector.py, keyed by failure mode
_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "Rider":              ("checkpoints/rider/fasterrcnn_bdd_finetuned.pt",            "bdd100k_rider_val"),
    "Nighttime person":   ("checkpoints/nighttime_person/fasterrcnn_bdd_finetuned.pt", "bdd100k_nighttime_person_val"),
    "Distant person":     ("checkpoints/distant_person/fasterrcnn_bdd_finetuned.pt",   "bdd100k_distant_person_val"),
}

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


def _check_views(db: lancedb.DBConnection) -> list[tuple]:
    rows = []
    for view in _EXPECTED_VIEWS:
        try:
            n = db.open_table(view).count_rows()
            rows.append((PASS, f"view: {view}", f"{n:,} rows"))
        except Exception:
            rows.append((FAIL, f"view: {view}", "missing — run manage_views"))
    return rows


def _run_eval_subprocess(db_path: str, checkpoint: str, table: str) -> dict | None:
    """
    Run eval.py in a fresh subprocess — avoids Lance handle conflicts with the
    open connections already held by verify_pipeline in the parent process.
    """
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "object_detection.eval",
         "--checkpoint", checkpoint,
         "--db", db_path,
         "--table", table,
         "--batch-size", "8",
         "--num-workers", "4",
         "--output-json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    [eval stderr] {result.stderr.strip()[-500:]}")
        return None
    # JSON is the last line of stdout
    for line in reversed(result.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except Exception:
            continue
    return None


def _check_models(db: lancedb.DBConnection, db_path: str) -> list[tuple]:
    rows = []
    for mode, (ckpt_path, val_view) in _CHECKPOINTS.items():
        try:
            db.open_table(val_view)
        except Exception:
            rows.append((SKIP, f"eval: {mode}", f"val view '{val_view}' missing"))
            continue

        if not Path(ckpt_path).exists():
            rows.append((SKIP, f"eval: {mode}", f"checkpoint not found: {ckpt_path}"))
            continue

        print(f"  evaluating {mode} baseline …")
        base = _run_eval_subprocess(db_path, "pretrained", val_view)
        print(f"  evaluating {mode} fine-tuned …")
        ft   = _run_eval_subprocess(db_path, ckpt_path, val_view)

        if base is None or ft is None:
            rows.append(("WARN", f"eval: {mode}", "eval subprocess failed — run eval.py manually"))
            continue

        delta = ft["map_50"] - base["map_50"]
        status = PASS if delta > 0 else FAIL
        rows.append((status, f"eval: {mode}",
                     f"mAP  baseline={base['map_50']:.4f}  fine-tuned={ft['map_50']:.4f}  "
                     f"Δ={delta:+.4f}  |  "
                     f"recall  {base['recall']:.4f} → {ft['recall']:.4f}"))
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
    rows += _check_models(db, db_path)
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
