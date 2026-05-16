"""
Run Geneva UDF backfills against the ``videos_raw`` Lance table.

The pipeline has four tiers (see ``geneva_udfs.py``):

  Tier 1 — CPU caption/annotation UDFs (always safe to run).
  Tier 2 — light GPU UDFs (CLIP, RAFT, MTScore).
  Tier 3 — heavy GPU UDFs (T5 hidden states, Wan-VAE latents — the
           headline trick: their outputs become the columns the train loop reads).
  Tier 4 — dedup (GPU dHash + CPU NN lookup).

Each tier has its own ``--tier N`` selector.  By default we only run Tier 1
because Tier 2-4 currently have placeholder implementations (they raise
``NotImplementedError``) — the GPU bodies will be wired up once the H100
finishes the current job.

Behaviour mirrors ``object-detection/backfill_geneva.py``:
  * Re-runs are safe: Geneva's default ``where`` filter is ``<col> IS NULL``.
  * ``--overwrite`` drops the columns and re-derives from scratch.
  * Columns are registered via ``table.add_columns({name: udf})`` before
    backfill, so Lance knows the schema even before any row is computed.

Usage
-----
# Tier 1 (CPU, safe alongside a running GPU job):
python -m videogen.backfill_geneva --tier 1

# Specific column(s):
python -m videogen.backfill_geneva --columns keyword_melting any_phase_keyword

# Tier 2 — refuses to run today (placeholder UDFs):
python -m videogen.backfill_geneva --tier 2

# Force-run a stub for diagnostics only:
python -m videogen.backfill_geneva --columns clip_emb_text --force-stub
"""

from __future__ import annotations

import argparse
import sys

import geneva

from videogen.geneva_udfs import (
    IMPLEMENTED_COLUMNS,
    TIER_UDFS,
    _IsDuplicate,
)


DEFAULT_DB    = "data/videos/lancedb"
DEFAULT_TABLE = "videos_raw"


def _resolve_columns(columns: list[str] | None, tier: int | None) -> list[str]:
    if columns:
        return columns
    if tier is not None:
        if tier not in TIER_UDFS:
            raise SystemExit(f"--tier must be one of {sorted(TIER_UDFS)}; got {tier}")
        return list(TIER_UDFS[tier])
    # No selection — default to Tier 1.
    return list(TIER_UDFS[1])


def _udf_for(column: str, *,
             db_path: str = DEFAULT_DB,
             table_name: str = DEFAULT_TABLE,
             dedup_threshold: int = 12) -> object | None:
    """Find the UDF instance registered for ``column``.

    ``is_duplicate`` is instantiated lazily here so it can pick up the
    correct db path + threshold from the CLI.
    """
    if column == "is_duplicate":
        return _IsDuplicate(db_path=db_path, table_name=table_name,
                            hamming_threshold=dedup_threshold)
    for regs in TIER_UDFS.values():
        if column in regs:
            return regs[column]
    return None


def backfill(
    *,
    db_path: str,
    table_name: str,
    columns: list[str],
    concurrency: int,
    overwrite: bool,
    force_stub: bool,
    checkpoint_size: int = 64,
    task_size: int = 1024,
    dedup_threshold: int = 12,
) -> None:
    skipped = [c for c in columns if c not in IMPLEMENTED_COLUMNS]
    if skipped and not force_stub:
        print(f"\nThe following columns are not yet implemented "
              f"(placeholder GPU UDFs):\n  {skipped}\n"
              f"Pass --force-stub to register their schemas without running, "
              f"or wait for the GPU UDFs to land.\n")
        columns = [c for c in columns if c in IMPLEMENTED_COLUMNS]

    if not columns:
        print("Nothing to backfill — exiting.")
        return

    conn = geneva.connect(db_path)
    tbl = conn.open_table(table_name)
    n_rows = tbl.count_rows()
    print(f"Opened table '{table_name}'  rows={n_rows:,}  version={tbl.version}")
    print(f"Columns to backfill: {columns}")

    if overwrite:
        existing = set(tbl.schema.names)
        to_drop = [c for c in columns if c in existing]
        if to_drop:
            print(f"  [overwrite] dropping {to_drop}")
            tbl.drop_columns(to_drop)
            tbl = conn.open_table(table_name)

    # Register any missing columns by attaching the UDF — Geneva picks the
    # data_type from the decorator.
    existing = set(tbl.schema.names)
    new_cols = {}
    for col in columns:
        if col not in existing:
            udf_obj = _udf_for(col, db_path=db_path, table_name=table_name,
                               dedup_threshold=dedup_threshold)
            if udf_obj is None:
                print(f"  [SKIP] no UDF registered for column '{col}'")
                continue
            new_cols[col] = udf_obj
    if new_cols:
        print(f"  Adding {len(new_cols)} new column(s): {list(new_cols)}")
        tbl.add_columns(new_cols)

    with conn.local_ray_context():
        for col in columns:
            udf_obj = _udf_for(col, db_path=db_path, table_name=table_name,
                               dedup_threshold=dedup_threshold)
            if udf_obj is None:
                continue
            print(f"  [backfill] {col} …")
            job_id = tbl.backfill(
                col,
                udf=udf_obj,
                concurrency=concurrency,
                checkpoint_size=checkpoint_size,
                task_size=task_size,
            )
            print(f"  [done]     {col}  (job_id={job_id})\n")

    print("All requested backfills complete.")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run Geneva UDF backfills on the videos_raw Lance table."
    )
    p.add_argument("--db",       default=DEFAULT_DB)
    p.add_argument("--table",    default=DEFAULT_TABLE)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--tier",    type=int, default=None, choices=sorted(TIER_UDFS),
                     help="Run all UDFs in a given tier.  Default: tier 1.")
    sel.add_argument("--columns", nargs="+", default=None,
                     help="Explicit list of UDF columns to backfill.")
    p.add_argument("--concurrency",     type=int, default=None,
                   help="Parallel Ray actor processes. Defaults: 4 for CPU, 1 for GPU tiers.")
    p.add_argument("--checkpoint-size", type=int, default=64)
    p.add_argument("--task-size",       type=int, default=128,
                   help="Geneva task size; lower means more frequent commits "
                        "but more orchestration overhead.  Default 128 keeps "
                        "heavy UDFs (T5, VAE) inside Geneva's 10-min stall "
                        "watchdog even at modest per-row latency.")
    p.add_argument("--overwrite",       action="store_true",
                   help="Drop the columns first, then recompute.")
    p.add_argument("--force-stub",      action="store_true",
                   help="Run UDFs that are still placeholder stubs (will crash on __call__).")
    p.add_argument("--dedup-threshold", type=int, default=12,
                   help="Hamming distance threshold for is_duplicate "
                        "(only used when 'is_duplicate' is in --columns).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    columns = _resolve_columns(args.columns, args.tier)

    if args.concurrency is None:
        # Tier 1 is pure-CPU regex; the others go through CUDA workers.
        cpu_only = all(c in TIER_UDFS[1] for c in columns)
        args.concurrency = 4 if cpu_only else 1

    backfill(
        db_path=args.db,
        table_name=args.table,
        columns=columns,
        concurrency=args.concurrency,
        overwrite=args.overwrite,
        force_stub=args.force_stub,
        checkpoint_size=args.checkpoint_size,
        task_size=args.task_size,
        dedup_threshold=args.dedup_threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
