"""Run Geneva backfills over the local TextVQA Lance table.

Every feature column — including the heavy Tier-3 ``vision_tower_hiddens``
— is a Geneva UDF defined in ``vlm/geneva_udfs.py`` and backfilled here.
The UDFs are stateful classes that lazy-load their model in the worker
process, so the driver never touches a GPU and Geneva distributes the
forward passes across its actor pool.

``vlm/backfill_direct.py`` is a single-process fallback for the same
Tier-3 columns (Lance ``add_columns`` with no Ray) — handy on a single
box / for the Colab bake, or if you hit an actor-pool issue.

Usage:

    python -m vlm.backfill_geneva --tier 1     # CPU text columns
    python -m vlm.backfill_geneva --tier 2     # dhash (image decode)
    python -m vlm.backfill_geneva --tier 3     # vision tower + SFT tokens (GPU)
    python -m vlm.backfill_geneva --tier all   # 1 + 2 + 3
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import geneva

from .geneva_udfs import TIER1_UDFS, TIER2_UDFS, TIER3_UDFS

LOG = logging.getLogger("vlm.backfill_geneva")


def _udfs_for_tier(tier: str) -> dict:
    if tier == "1":
        return TIER1_UDFS
    if tier == "2":
        return TIER2_UDFS
    if tier == "3":
        return TIER3_UDFS
    if tier == "all":
        return {**TIER1_UDFS, **TIER2_UDFS, **TIER3_UDFS}
    raise SystemExit(f"unknown tier {tier!r}")


def _add_missing(table, udfs: dict) -> None:
    """Geneva's add_columns errors if a column already exists; filter first."""
    have = {f.name for f in table.schema}
    to_add = {n: u for n, u in udfs.items() if n not in have}
    if to_add:
        LOG.info("adding columns: %s", list(to_add))
        table.add_columns(to_add)
    else:
        LOG.info("no new columns to add (all present)")


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--db",          default="data/textvqa.lance",
                   help="Lance dataset path")
    p.add_argument("--tier",        default="1", choices=["1", "2", "3", "all"])
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--task-size",   type=int, default=256)
    p.add_argument("--checkpoint",  type=int, default=128)
    args = p.parse_args()

    db_path = Path(args.db).resolve()
    parent = str(db_path.parent)
    table_name = db_path.name[:-len(".lance")] if db_path.name.endswith(".lance") else db_path.name

    udfs = _udfs_for_tier(args.tier)
    LOG.info("backfilling %d columns into %s/%s: %s",
             len(udfs), parent, table_name, list(udfs))

    g = geneva.connect(parent)
    table = g.open_table(table_name)

    _add_missing(table, udfs)

    with g.local_ray_context():
        for name, fn in udfs.items():
            LOG.info("backfill column %s ...", name)
            t0 = time.time()
            table.backfill(
                name, udf=fn,
                concurrency=args.concurrency,
                task_size=args.task_size,
                checkpoint_size=args.checkpoint,
            )
            LOG.info("  %s done in %.1fs", name, time.time() - t0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
