#!/usr/bin/env python
"""Curation demo on the Lance-backed LIBERO dataset.

Prereq: embed_frames.py has merged an `emb_image` column into the frames table.

What this shows (none of it possible on parquet+mp4 without external systems):
  1. schema evolution: join task text into the frames table as a real column
  2. secondary indexes on the SAME table the trainer reads: IVF-PQ vector,
     full-text (BM25), and btree scalar indexes
  3. text -> frame semantic search over 273k frames (SigLIP2 dual encoder)
  4. SQL-style filtered scans accelerated by the btree index
  5. building a curated episode list -> feed straight back into training via
     `LeRobotLanceVideoDataset(episodes=...)` + EpisodeAwareSampler

    python curate.py --lance-root ~/work/data/libero_lance_video --out-dir assets/
"""

import argparse
import time
from pathlib import Path

import lance
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import torch

MODEL_ID = "google/siglip2-base-patch16-256"


def log_step(title):
    print(f"\n=== {title} ===", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lance-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    root = Path(args.lance_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(root)
    tbl = db.open_table("libero")
    frames_ds = lance.dataset(str(root / "libero.lance"))
    print("frames table rows:", tbl.count_rows(), "| columns:", tbl.schema.names)

    # ── 1. schema evolution: task text as a column (merge, no rewrite) ──
    if "task" not in tbl.schema.names:
        log_step("merging task text column")
        tasks = pd.read_parquet(root / "meta" / "tasks.parquet").reset_index()
        tasks.columns = ["task", "task_index"]
        idx_task = frames_ds.to_table(columns=["index", "task_index"]).to_pandas()
        joined = idx_task.merge(tasks, on="task_index", how="left")
        frames_ds.merge(
            pa.table({"index": joined["index"].values, "task": joined["task"].astype(str).values}),
            left_on="index", right_on="index",
        )
        print("task column merged, version:", frames_ds.version)

    # ── 2. indexes ──
    log_step("building indexes (vector IVF_PQ + FTS + btree)")
    t0 = time.time()
    tbl.create_index(metric="cosine", vector_column_name="emb_image", index_type="IVF_PQ", num_partitions=256, num_sub_vectors=48, replace=True)
    print(f"vector index: {time.time()-t0:.1f}s")
    t0 = time.time()
    tbl.create_fts_index("task", replace=True)
    print(f"fts index: {time.time()-t0:.1f}s")
    t0 = time.time()
    tbl.create_scalar_index("episode_index", index_type="BTREE", replace=True)
    print(f"btree index: {time.time()-t0:.1f}s")

    # ── 3. text -> frame semantic search ──
    log_step("semantic search")
    from transformers import AutoModel, AutoProcessor

    device = f"cuda:{args.gpu}"
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    def text_vec(q: str) -> np.ndarray:
        with torch.no_grad():
            ins = processor(text=[q], return_tensors="pt", padding="max_length", max_length=64).to(device)
            v = model.get_text_features(**ins)
            if not torch.is_tensor(v):  # transformers>=5 returns BaseModelOutputWithPooling
                v = v.pooler_output
            return torch.nn.functional.normalize(v, dim=-1)[0].float().cpu().numpy()

    queries = [
        "the robot gripper is right above the stove",
        "a wine bottle on the table",
        "the microwave door is open",
        "robot arm reaching into a drawer",
    ]
    hits = {}
    for q in queries:
        t0 = time.time()
        res = tbl.search(text_vec(q), vector_column_name="emb_image").limit(6).to_pandas()
        ms = (time.time() - t0) * 1000
        hits[q] = res
        eps = res["episode_index"].tolist()
        print(f"[{ms:6.1f} ms] {q!r} -> episodes {eps}")

    # ── 4. FTS + btree-accelerated filters ──
    log_step("FTS + scalar filters")
    t0 = time.time()
    fts = tbl.search("microwave", query_type="fts").limit(5).to_pandas()
    print(f"FTS 'microwave': {len(fts)} hits in {(time.time()-t0)*1000:.1f} ms; e.g. {fts['task'].iloc[0]!r}")
    t0 = time.time()
    sub = tbl.search().where("episode_index BETWEEN 900 AND 950 AND timestamp < 1.0").limit(1000).to_pandas()
    print(f"btree window scan: {len(sub)} rows in {(time.time()-t0)*1000:.1f} ms")

    # ── 5. curated subset -> training-ready episode list ──
    log_step("curated subset")
    # example: all episodes whose task mentions a stove OR that are visually
    # near the 'gripper above stove' query -> a focused finetuning split
    task_eps = tbl.search().where("task LIKE '%stove%'").select(["episode_index"]).limit(300000).to_pandas()["episode_index"].unique()
    vec_eps = tbl.search(text_vec(queries[0]), vector_column_name="emb_image").limit(200).to_pandas()["episode_index"].unique()
    curated = sorted(set(task_eps) | set(vec_eps))
    np.save(out / "curated_episodes.npy", np.array(curated))
    print(f"curated {len(curated)} episodes -> curated_episodes.npy")
    print("train with: LeRobotLanceVideoDataset(root=..., episodes=curated) + EpisodeAwareSampler")

    extras(root, out, tbl, frames_ds, text_vec)

    # ── frame grids for the blog ──
    log_step("saving search-result frame grids")
    from lerobot_lancedb import LeRobotLanceVideoDataset
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = LeRobotLanceVideoDataset(root=str(root), return_uint8=True)
    for qi, (q, res) in enumerate(hits.items()):
        fig, axes = plt.subplots(1, 6, figsize=(15, 2.9), dpi=120)
        for ax, (_, r) in zip(axes, res.iterrows()):
            item = ds[int(r["index"])]
            ax.imshow(item["observation.images.image"].permute(1, 2, 0).numpy())
            ax.set_title(f"ep {int(r['episode_index'])} t={float(r['timestamp']):.1f}s", fontsize=8)
            ax.axis("off")
        fig.suptitle(f'text query: "{q}"', fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"search_{qi}.png")
    print("done ->", out)




def extras(root: Path, out: Path, tbl, frames_ds, text_vec):
    """Extended curation examples: dedup, outliers, hybrid query, time travel."""
    import numpy as np
    log_step("near-duplicate episodes (redundancy pruning)")
    t0 = time.time()
    df = frames_ds.to_table(columns=["episode_index", "frame_index", "emb_image"]).to_pandas()
    mid = df[df.frame_index == 30].dropna(subset=["emb_image"])
    E = np.stack(mid["emb_image"].to_numpy())
    eps = mid["episode_index"].to_numpy()
    sims = E @ E.T
    np.fill_diagonal(sims, 0)
    ii, jj = np.where(np.triu(sims, 1) > 0.995)
    print(f"episode pairs with >0.995 mid-episode similarity: {len(ii)} "
          f"(e.g. {[(int(eps[a]), int(eps[b])) for a, b in list(zip(ii, jj))[:5]]}) "
          f"in {time.time()-t0:.1f}s — candidates for dedup before training")

    log_step("visual outliers within a task (teleop glitch mining)")
    task_rows = tbl.search().where("task LIKE '%microwave%'").select(["index", "episode_index", "emb_image"]).limit(50000).to_pandas()
    T = np.stack(task_rows["emb_image"].to_numpy())
    centroid = T.mean(0); centroid /= np.linalg.norm(centroid)
    d = 1 - T @ centroid
    worst = task_rows.iloc[np.argsort(d)[-5:]]
    print("frames farthest from the task's visual centroid (potential anomalies):")
    print(worst[["index", "episode_index"]].to_string(index=False))

    log_step("hybrid query: task text + visual state")
    t0 = time.time()
    hits = (tbl.search(text_vec("the gripper is holding the object in the air"), vector_column_name="emb_image")
            .where("task LIKE '%basket%'", prefilter=True).limit(6).to_pandas())
    print(f"'holding object' ∩ task~basket -> episodes {hits['episode_index'].tolist()} "
          f"in {(time.time()-t0)*1000:.0f} ms")

    log_step("time travel: versioned, reproducible splits")
    versions = frames_ds.versions()
    print(f"table has {len(versions)} versions; e.g. v{versions[1]['version']} = pre-embedding table.")
    print("checkout any version for a bit-reproducible training split:")
    print("  lance.dataset(uri, version=N)  # the split you trained on, forever")


if __name__ == "__main__":
    main()
