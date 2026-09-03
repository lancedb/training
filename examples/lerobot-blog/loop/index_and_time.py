#!/usr/bin/env python
"""Build the vector index on the embedding column and re-time the mining queries.

build_sets.py ran its 60 queries as flat scans because the table had no vector index yet
(7.7 s each). This adds an IVF-PQ index over the embedded rows and a scalar index on
episode_index, then times the exact same seed queries again.
"""
import argparse
import json
import time

import lance
import lancedb
import numpy as np

from common import env_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="config/loop_sets.json")
    ap.add_argument("--subset", default="config/loop_subset.json")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="out/index_timing.json")
    a = ap.parse_args()
    root = a.root or env_root()
    db = lancedb.connect(root)
    tbl = db.open_table("frames")
    res = {"rows": tbl.count_rows(), "embedded": tbl.count_rows("emb_siglip2 IS NOT NULL")}

    t = time.perf_counter()
    tbl.create_index(metric="cosine", vector_column_name="emb_siglip2", index_type="IVF_PQ",
                     num_partitions=64, num_sub_vectors=48, replace=True)
    res["vector_index_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    tbl.create_scalar_index("episode_index", index_type="BTREE", replace=True)
    res["scalar_index_s"] = round(time.perf_counter() - t, 1)
    print(f"vector index {res['vector_index_s']}s, scalar index {res['scalar_index_s']}s", flush=True)

    sets = json.load(open(a.sets))
    sub = json.load(open(a.subset))
    pool_sql = ",".join(map(str, sorted(sub["pool"])))
    ds = lance.dataset(f"{root}/frames.lance")
    seeds = sets["seeds"]
    ms_flat, ms_idx, agree = [], [], []
    for s in seeds:
        row = ds.to_table(columns=["emb_siglip2"], filter=f"episode_index = {s['ep']} AND frame_index = {s['frame']}")
        v = np.asarray(row.column("emb_siglip2")[0].as_py(), dtype=np.float32)
        q = (tbl.search(v, vector_column_name="emb_siglip2").metric("cosine")
             .where(f"episode_index IN ({pool_sql}) AND emb_siglip2 IS NOT NULL", prefilter=True)
             .select(["episode_index", "frame_index"]).limit(600))
        t = time.perf_counter(); hits = q.to_list(); ms_idx.append(1000 * (time.perf_counter() - t))
        t = time.perf_counter(); flat = q.bypass_vector_index().to_list(); ms_flat.append(1000 * (time.perf_counter() - t))
        top = {(r["episode_index"]) for r in hits[:100]}
        agree.append(len(top & {(r["episode_index"]) for r in flat[:100]}) / max(len(top), 1))
    res.update(queries=len(seeds), indexed_ms_mean=round(float(np.mean(ms_idx)), 1), indexed_ms_max=round(float(np.max(ms_idx)), 1),
               flat_ms_mean=round(float(np.mean(ms_flat)), 1), top100_episode_overlap=round(float(np.mean(agree)), 3),
               table_version=ds.version)
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
