"""
B9 — Recipe-change cost.

When you change one piece of the curation recipe — say, swap to a
different VAE, or re-tokenise with a different max length — what
proportion of the cache do you have to re-derive?

With a flat filesystem cache: **everything**.  The cache key is usually
"hash of all preprocessing params" so any change invalidates the whole
directory.

With Lance + Geneva: only the **changed column**.  Every other column
(raw video bytes, T5 hidden states, motion strength, dedup hash, …)
stays put.  This bench measures the wall-clock for "re-derive one
column" vs "re-derive everything".

It's a synthetic comparison — we don't actually have a second VAE to
swap to — but the cost ratio is what matters for the story.

Usage
-----
python -m bench.bench_recipe_change \\
    --db /tmp/videogen_b9 --n 16 \\
    --recipe-column motion_strength
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from videogen.backfill_geneva import backfill
from videogen.ingest_chronomagic import ingest, synthetic_rows


# Columns that any "Tier-1 + Tier-2" pipeline would derive.  We exclude
# Tier-3 (T5 / VAE) and Tier-4 (dHash / dedup) for runtime — feel free to
# extend if you have the GPU time.
PIPELINE_COLUMNS = [
    "caption_length",
    "keyword_melting", "keyword_freezing", "keyword_dissolving",
    "keyword_boiling", "keyword_evaporating",
    "any_phase_keyword",
    "motion_strength",          # the one we re-derive in stage B
    "metamorphic_score",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db",      default="/tmp/videogen_b9")
    p.add_argument("--table",   default="videos_raw")
    p.add_argument("--n",       type=int, default=16)
    p.add_argument("--recipe-column", default="motion_strength",
                   help="Column whose recipe we simulate changing.")
    args = p.parse_args(argv)

    if args.recipe_column not in PIPELINE_COLUMNS:
        raise SystemExit(f"--recipe-column must be in {PIPELINE_COLUMNS}")

    if Path(args.db).exists():
        shutil.rmtree(args.db)

    # --- Stage 1: bring up a full Tier 1+2 pipeline -------------------------
    print(f"[stage 0] ingest {args.n} synthetic rows")
    ingest(db_path=args.db, table_name=args.table,
           rows=synthetic_rows(args.n, seed=0),
           overwrite=True, batch_size=64)

    print(f"[stage A] backfill ALL pipeline columns ({len(PIPELINE_COLUMNS)})")
    t0 = time.perf_counter()
    backfill(db_path=args.db, table_name=args.table,
             columns=PIPELINE_COLUMNS,
             concurrency=1, overwrite=False, force_stub=False)
    full_cost = time.perf_counter() - t0
    print(f"  full pipeline wall-clock: {full_cost:.1f}s\n")

    # --- Stage 2: simulate a recipe change ---------------------------------
    print(f"[stage B] recipe change → re-derive only '{args.recipe_column}'")
    t0 = time.perf_counter()
    backfill(db_path=args.db, table_name=args.table,
             columns=[args.recipe_column],
             concurrency=1, overwrite=True, force_stub=False)
    one_col_cost = time.perf_counter() - t0
    print(f"  single-column re-derive: {one_col_cost:.1f}s\n")

    print("─" * 60)
    print(f"  full pipeline    : {full_cost:>7.2f}s")
    print(f"  one column re-do : {one_col_cost:>7.2f}s")
    print(f"  ratio            : {full_cost / max(one_col_cost, 1e-6):>7.2f}× "
          f"(higher = bigger win for the Lance/Geneva approach)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
