#!/usr/bin/env python
"""Attach the embedding column to the frames table, index it, and search it by text.

Everything happens on the table the trainer reads. Adding the column is schema evolution
(a commit that adds a column) -- the video blobs are never rewritten.
"""
import os
import argparse, json, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("LANCE_FE_ROOT", os.environ.get("LANCE_ROOT", "./data/droid_lance")))
    ap.add_argument("--embeddings", default="out/embeddings.parquet")
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--col", default="emb_siglip2")
    ap.add_argument("--out", default="out/feature_eng.json")
    a = ap.parse_args()

    import lance, lancedb, pyarrow.parquet as pq, torch
    from transformers import AutoModel, AutoTokenizer

    res = {}
    frames_uri = f"{a.root}/frames.lance"

    # ---- 1. attach the column (schema evolution; video bytes untouched)
    before = lance.dataset(frames_uri)
    res["rows"] = before.count_rows()
    res["version_before"] = before.version
    size_before = sum(f.stat().st_size for f in Path(a.root).rglob("*") if f.is_file())

    tbl = pq.read_table(a.embeddings)
    print(f"merging {tbl.num_rows:,} embeddings into {frames_uri}", flush=True)
    t = time.perf_counter()
    ds = lance.dataset(frames_uri)
    ds.merge(tbl, left_on="index")
    res["merge_s"] = round(time.perf_counter() - t, 1)
    after = lance.dataset(frames_uri)
    res["version_after"] = after.version
    size_after = sum(f.stat().st_size for f in Path(a.root).rglob("*") if f.is_file())
    res["added_MB"] = round((size_after - size_before) / 1e6, 1)
    res["dataset_MB"] = round(size_after / 1e6, 1)
    print(f"  merged in {res['merge_s']}s; version {res['version_before']} -> "
          f"{res['version_after']}; +{res['added_MB']} MB", flush=True)

    # ---- 2. vector index
    db = lancedb.connect(a.root)
    T = db.open_table("frames")
    t = time.perf_counter()
    T.create_index(metric="cosine", vector_column_name=a.col)
    res["index_s"] = round(time.perf_counter() - t, 1)
    print(f"  IVF-PQ index built in {res['index_s']}s", flush=True)

    # ---- 3. text -> frame search with the SigLIP2 text tower
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, dtype=torch.float32).eval()
    queries = ["a robot gripper holding a white mug",
               "a person's hand entering the frame",
               "an open dishwasher",
               "a cluttered kitchen countertop"]
    res["queries"] = []
    for q in queries:
        with torch.inference_mode():
            ids = tok([q], padding="max_length", max_length=64, return_tensors="pt")
            v = model.get_text_features(**ids)
            if not torch.is_tensor(v):
                v = v.pooler_output
            v = torch.nn.functional.normalize(v, dim=-1)[0].numpy()
        t = time.perf_counter()
        hits = (T.search(v, vector_column_name=a.col).metric("cosine")
                 .select(["index", "episode_index", "frame_index", "building"])
                 .limit(5).to_arrow())
        ms = round((time.perf_counter() - t) * 1000, 1)
        top = [{k: hits.column(k)[i].as_py() for k in hits.schema.names if k != a.col}
               for i in range(hits.num_rows)]
        res["queries"].append({"query": q, "latency_ms": ms, "top": top[:3]})
        print(f'  "{q}" -> {ms} ms; top episode {top[0].get("episode_index") if top else "-"}',
              flush=True)

    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
