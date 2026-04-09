"""
Deduplication helpers for BDD100K.

The full dedup pipeline follows the same Geneva backfill pattern as every
other feature column — no separate embed/flag steps needed:

  1. backfill_geneva --gpu --columns embedding
        ResNet18 (512-d, L2-normalised) written to bdd100k.embedding

  2. dedup --action index
        IVF-PQ cosine index on bdd100k.embedding

  3. backfill_geneva --columns is_duplicate
        Per-row: nearest non-self neighbour found via vector search.
        is_duplicate = True when cosine similarity >= 0.85.

  4. manage_views --action curate / curate-person
        Views already filter is_duplicate — no extra step needed.

This module provides two supporting actions:

  index   Build the IVF-PQ vector index (must run between embedding and
          is_duplicate backfills).

  stats   Preview the estimated duplicate rate at a given threshold
          by sampling rows from the index, without writing anything.

CLI usage
---------
  python -m object_detection.dedup --action index
  python -m object_detection.dedup --action stats --threshold 0.85
"""

from __future__ import annotations

import argparse

import lancedb
import pyarrow as pa

SOURCE_TABLE = "bdd100k"
DEFAULT_THRESHOLD = 0.85
DEFAULT_DB = "data/bdd100k/lancedb"


def index(db_path: str) -> None:
    """Build IVF-PQ cosine index on bdd100k.embedding."""
    db = lancedb.connect(db_path)
    src_tbl = db.open_table(SOURCE_TABLE)
    n = src_tbl.count_rows()
    print(f"[index] Building IVF-PQ cosine index on {n} rows …")
    src_tbl.create_index(metric="cosine", vector_column_name="embedding")
    print("[index] Index built.")


def stats(db_path: str, threshold: float, sample_size: int = 1000) -> None:
    """Estimate duplicate rate by sampling rows and querying the index."""
    db = lancedb.connect(db_path)
    src_tbl = db.open_table(SOURCE_TABLE)
    distance_threshold = 1.0 - threshold
    dup_count = 0
    sampled = 0

    print(f"[stats] Sampling {sample_size} images at threshold={threshold} …")
    # Stream — never load the full embedding column into memory
    for batch in src_tbl.to_batches(columns=["image_id", "embedding"]):
        for image_id, embedding in zip(
            batch.column("image_id").to_pylist(),
            batch.column("embedding").to_pylist(),
        ):
            if sampled >= sample_size:
                break
            if embedding is None:
                sampled += 1
                continue
            result = (
                src_tbl.search(embedding, vector_column_name="embedding")
                .metric("cosine")
                .where(f"image_id != '{image_id}'")
                .limit(1)
                .to_arrow()
            )
            if len(result) > 0 and result["_distance"][0].as_py() <= distance_threshold:
                dup_count += 1
            sampled += 1
        if sampled >= sample_size:
            break

    rate = dup_count / sampled * 100 if sampled > 0 else 0.0
    print(f"[stats] Estimated duplicate rate: {dup_count}/{sampled} = {rate:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="BDD100K dedup helpers (index + stats).")
    p.add_argument("--action", choices=["index", "stats"], required=True)
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Cosine similarity threshold for stats preview (default: 0.85)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.action == "index":
        index(args.db)
    elif args.action == "stats":
        stats(args.db, args.threshold)


if __name__ == "__main__":
    main()
