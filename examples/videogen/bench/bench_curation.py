"""
B2 — Curation query throughput.

How fast can we go from "raw table" to "training-ready subset"?  Times the
three idioms a practitioner actually uses:

  1. ``count_rows(filter=...)``    pure scalar SQL
  2. ``search(text, query_type="fts")``  FTS over caption
  3. Composite: FTS  ∩  scalar filter   (e.g. "phase transition" AND duration_s>4)

For a real comparison, also runs a *grep-style* baseline: iterate all
captions in Python and apply the same predicate.  This is the "naïve
manifest pipeline" most teams start with.

Usage
-----
python -m bench.bench_curation --db data/videos/lancedb
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import lancedb

from videogen.spec_queries import (
    curated_spec,
    ensure_caption_fts,
    union_keyword_spec,
)


def _time(label: str, fn, *, repeats: int = 3) -> tuple[float, object]:
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    print(f"  {label:<48} {best*1000:>10.2f} ms")
    return best, result


def run(db_path: str, table: str = "videos_raw", repeats: int = 3) -> None:
    db = lancedb.connect(db_path)
    if table not in db.list_tables().tables:
        raise SystemExit(f"Table '{table}' missing in {db_path} — run ingest first.")
    tbl = db.open_table(table)
    print(f"\n[B2] table='{table}'  rows={len(tbl):,}\n")

    # ── 1) Pure scalar SQL ────────────────────────────────────────────────
    _time("count_rows(any phase keyword)",
          lambda: tbl.count_rows(filter=union_keyword_spec("train")),
          repeats=repeats)

    # Tier-2 spec needs Tier-2 columns; only run if present
    if {"motion_strength", "metamorphic_score"}.issubset(tbl.schema.names):
        _time("count_rows(curated_spec — Tier 2)",
              lambda: tbl.count_rows(filter=curated_spec("train")),
              repeats=repeats)
    else:
        print("  (skipping curated_spec — Tier 2 columns absent)")

    # ── 2) Full-text search ───────────────────────────────────────────────
    ensure_caption_fts(tbl)
    _time("FTS 'melting' limit=100",
          lambda: tbl.search("melting", query_type="fts").limit(100).to_arrow(),
          repeats=repeats)

    _time("FTS 'ice cream melting'  limit=100",
          lambda: tbl.search("ice cream melting", query_type="fts").limit(100).to_arrow(),
          repeats=repeats)

    # ── 3) Naïve Python baseline — iterate captions, regex match ──────────
    pat = re.compile(r"\b(melt|melts|melting)\b", re.IGNORECASE)
    def naive():
        caps = tbl.search().select(["caption"]).to_arrow().column("caption").to_pylist()
        return sum(1 for c in caps if c and pat.search(c))
    _time("naive python regex over all captions",
          naive, repeats=repeats)

    print()


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--db",     default="data/videos/lancedb")
    p.add_argument("--table",  default="videos_raw")
    p.add_argument("--repeats", type=int, default=3,
                   help="Times each query is timed; best wall-clock is reported.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    run(args.db, args.table, args.repeats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
