#!/usr/bin/env python
"""Merge the scored shards into the frames table as new columns.

Lance ``merge`` joins on the ``index`` column and appends columns without touching any existing
data: the 386 GB of video is not read, and the tabular fragments are not rewritten. Readers
pinned to the previous version keep working. Rows outside the scored subset are null.
"""
import argparse
import glob
import time

import lance
import pyarrow as pa
import pyarrow.parquet as pq

from common import env_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="out/score/shard_*.parquet")
    ap.add_argument("--suffix", default="_base", help="appended to the error column names")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--root", default=None, help="dataset root to write to (default $LANCE_ROOT); may be s3://")
    a = ap.parse_args()

    files = sorted(glob.glob(a.shards))
    if not files:
        raise SystemExit(f"no shards match {a.shards}")
    t = pa.concat_tables([pq.read_table(f) for f in files])
    keep = ["index", "err_chunk_mae", "err_next_mae", "err_gripper_next"] + \
           ([] if a.no_embed or "emb_siglip2" not in t.column_names else ["emb_siglip2"])
    t = t.select(keep)
    names = {c: (c + a.suffix if c.startswith("err_") else c) for c in keep}
    t = t.rename_columns([names[c] for c in keep])

    uri = f"{a.root or env_root()}/frames.lance"
    ds = lance.dataset(uri)
    v0 = ds.version
    existing = [c for c in t.column_names if c != "index" and c in ds.schema.names]
    if existing:
        print(f"dropping existing columns {existing} first")
        ds.drop_columns(existing)
        ds = lance.dataset(uri)
    t0 = time.perf_counter()
    ds.merge(t, left_on="index", right_on="index")
    el = time.perf_counter() - t0
    ds = lance.dataset(uri)
    print(f"merged {t.num_rows:,} rows x {len(t.column_names) - 1} columns in {el:.1f}s; "
          f"version {v0} -> {ds.version}; rows {ds.count_rows():,}")
    for c in t.column_names:
        if c != "index":
            print(f"  {c}: {ds.count_rows(f'{c} IS NOT NULL'):,} non-null")


if __name__ == "__main__":
    main()
