#!/usr/bin/env python3
"""Hybrid queries: a learned embedding column and a derived score column, composed.

Both columns live on the table the trainer reads -- emb_siglip2 (SigLIP2, backfilled) and
jerk_score (a Geneva UDF over action_joint_velocity). A curation question like "frames that
look like a grasp AND are unusually jerky AND come from a failed episode" is therefore one
query, not a join across an embedding store, a metrics table and the dataset.
"""
import os
import base64, io, json, time
import numpy as np, lance, lancedb, torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image

ROOT = os.environ.get("LANCE_FE_ROOT", os.environ.get("LANCE_ROOT", "./data/droid_lance"))
MODEL = "google/siglip2-base-patch16-224"
CAM = "observation.images.exterior_1_left"
P95, P99 = 0.5144, 1.2875

db = lancedb.connect(ROOT); T = db.open_table("frames")
fr = lance.dataset(f"{ROOT}/frames.lance")
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModel.from_pretrained(MODEL, dtype=torch.float32).eval()

from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("lerobot/droid_1.0.1", root=ROOT, tolerance_s=5e-3)

def vec(q):
    with torch.inference_mode():
        ids = tok([q], padding="max_length", max_length=64, return_tensors="pt")
        v = mdl.get_text_features(**ids)
        if not torch.is_tensor(v): v = v.pooler_output
        return torch.nn.functional.normalize(v, dim=-1)[0].numpy()

def thumb(gi, w=190):
    r = fr.take([gi], columns=["episode_index", "frame_index"]).to_pydict()
    ep, fi = r["episode_index"][0], r["frame_index"][0]
    item = ds[int(ds.meta.episodes[ep]["dataset_from_index"]) + fi]
    a = item[CAM].numpy()
    if a.ndim == 3 and a.shape[0] in (1, 3): a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.max() <= 1.01 else a.astype(np.uint8)
    im = Image.fromarray(a).convert("RGB"); im = im.resize((w, int(w*im.height/im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=72, optimize=True)
    return dict(ep=ep, fi=fi, b64=base64.b64encode(b.getvalue()).decode())

CASES = [
  ("a gripper closing on an object", f"jerk_score > {P99}",
   "looks like a grasp, and the motion is in the roughest 1%"),
  ("a cluttered kitchen counter", "is_episode_successful = false",
   "looks like clutter, and the episode failed"),
  ("a robot arm near a drawer", f"jerk_score < {P95} AND is_episode_successful = true",
   "looks like a drawer task, smooth, and it succeeded"),
]

out = {"p95": P95, "p99": P99, "cases": []}
for q, where, desc in CASES:
    v = vec(q)
    t0 = time.perf_counter()
    hits = (T.search(v, vector_column_name="emb_siglip2").metric("cosine")
             .where(where)
             .select(["index", "episode_index", "frame_index", "jerk_score", "building"])
             .limit(4).to_arrow())
    ms = round((time.perf_counter() - t0) * 1000, 1)
    idxs = [hits.column("index")[i].as_py() for i in range(hits.num_rows)]
    js = [round(hits.column("jerk_score")[i].as_py(), 3) for i in range(hits.num_rows)]
    tiles = []
    for k, gi in enumerate(idxs):
        try:
            t = thumb(gi); t["jerk"] = js[k]; tiles.append(t)
        except Exception as e:
            print("  thumb fail", gi, str(e)[:50], flush=True)
    out["cases"].append({"query": q, "where": where, "desc": desc,
                         "latency_ms": ms, "hits": hits.num_rows, "tiles": tiles})
    print(f'"{q}" WHERE {where} -> {ms} ms, {hits.num_rows} hits, jerk {js}', flush=True)

json.dump(out, open("out/hybrid_search.json", "w"))
print("RESULT written", flush=True)
