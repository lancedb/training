#!/usr/bin/env python
"""One pass over the subset per GPU: the base policy's error on every frame, and a SigLIP2
embedding of the same frame. Both become columns on the frames table (see merge_columns.py).

The frames are read through lerobot's LeRobotDataset on the Lance root, so what gets scored
is exactly what the trainer trains on. Run one process per GPU:

  for r in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$r python loop/score_and_embed.py \
      --ckpt runs/base/checkpoints/010000/pretrained_model --rank $r --world 4 & done; wait
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from common import env_root, load_meta, load_policy, load_rename_map, open_dataset, relative_indices

EMB_MODEL = "google/siglip2-base-patch16-224"
EMB_DIM = 768
CAM = "observation.images.exterior_1_left"


class Embedder:
    def __init__(self, device="cuda"):
        from transformers import AutoModel, AutoProcessor

        self.model = AutoModel.from_pretrained(EMB_MODEL, dtype=torch.float16).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(EMB_MODEL)
        self.device = device

    @torch.no_grad()
    def __call__(self, frames_uint8: torch.Tensor) -> np.ndarray:  # (B, 3, H, W) uint8
        imgs = [f.permute(1, 2, 0).numpy() for f in frames_uint8]
        px = self.proc(images=imgs, return_tensors="pt")
        px = {k: v.to(self.device, dtype=torch.float16 if v.is_floating_point() else None) for k, v in px.items()}
        e = self.model.get_image_features(**px)
        if not torch.is_tensor(e):
            e = e.pooler_output
        return torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="<run>/checkpoints/<step>/pretrained_model")
    ap.add_argument("--subset", default="config/loop_subset.json")
    ap.add_argument("--keys", default="holdout,pool", help="which lists of the subset file to cover")
    ap.add_argument("--stride", type=int, default=3, help="score every k-th frame of each episode")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--out-dir", default="out/score")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    from common import BatchScorer

    sub = json.load(open(a.subset))
    episodes = sorted({e for k in a.keys.split(",") for e in sub[k]})
    shard = episodes[a.rank::a.world]
    root = env_root()
    rmap = load_rename_map()
    meta = load_meta(root)
    policy, pre, delta_ts, cfg = load_policy(a.ckpt, meta, rmap)
    ds = open_dataset(root, shard, delta_ts)
    rel = relative_indices(ds, shard, stride=a.stride)
    print(f"[rank {a.rank}] {len(shard)} episodes, {len(rel):,} frames at stride {a.stride}", flush=True)
    embed = None if a.no_embed else Embedder()
    scorer = BatchScorer(policy, cfg)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, sampler=rel, num_workers=a.num_workers,
        multiprocessing_context="fork", prefetch_factor=4, pin_memory=True)
    cols = {k: [] for k in ("index", "episode_index", "frame_index",
                            "err_chunk_mae", "err_next_mae", "err_gripper_next")}
    embs = []
    t0 = time.perf_counter()
    done = 0
    for bi, raw in enumerate(loader):
        idx = raw["index"].numpy().astype(np.int64)
        cols["index"].append(idx)
        cols["episode_index"].append(raw["episode_index"].numpy().astype(np.int64))
        cols["frame_index"].append(raw["frame_index"].numpy().astype(np.int64))
        if embed is not None:
            embs.append(embed(raw[CAM]))
        batch = _preprocess_dataset_batch(raw, ds.meta.camera_keys, rmap, pre)
        batch = {k: (v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        for k, v in scorer(batch, batch_seed=int(idx[0])).items():
            cols[k].append(v)
        done += len(idx)
        if (bi + 1) % 25 == 0:
            print(f"[rank {a.rank}] {done:,}/{len(rel):,}  {done / (time.perf_counter() - t0):.0f} frames/s",
                  flush=True)

    arrays = {k: pa.array(np.concatenate(v)) for k, v in cols.items()}
    if embed is not None:
        E = np.concatenate(embs).astype(np.float32)
        arrays["emb_siglip2"] = pa.FixedSizeListArray.from_arrays(pa.array(E.reshape(-1), pa.float32()), EMB_DIM)
    table = pa.table(arrays)
    out = os.path.join(a.out_dir, f"shard_{a.rank:02d}.parquet")
    pq.write_table(table, out)
    el = time.perf_counter() - t0
    print(f"[rank {a.rank}] wrote {table.num_rows:,} rows to {out} in {el:.0f}s "
          f"({table.num_rows / el:.0f} frames/s)", flush=True)


if __name__ == "__main__":
    main()
