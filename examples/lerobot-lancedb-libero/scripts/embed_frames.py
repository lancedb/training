#!/usr/bin/env python
"""Backfill SigLIP2 image embeddings into the Lance frames table (zero-copy schema evolution).

Decodes frames straight from the Lance video table (blob v2 -> torchcodec),
embeds the agentview camera with SigLIP2, and merges the vectors back into the
frames table as a new column — no rewrite of existing data files.

    python embed_frames.py --lance-root PATH [--batch-size 256] [--gpu 0] [--stride 1]
"""

import argparse
import time

import lance
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader

from lerobot_lancedb import LeRobotLanceVideoDataset

MODEL_ID = "google/siglip2-base-patch16-256"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lance-root", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--camera", default="observation.images.image")
    p.add_argument("--column", default="emb_image")
    args = p.parse_args()

    from transformers import AutoModel, AutoProcessor

    device = f"cuda:{args.gpu}"
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    ds = LeRobotLanceVideoDataset(root=args.lance_root, return_uint8=True)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    all_idx = []
    all_emb = []
    t0 = time.time()
    n = 0
    with torch.no_grad():
        for batch in dl:
            imgs = batch[args.camera]  # uint8 CHW
            pixel = processor(images=[im.permute(1, 2, 0).numpy() for im in imgs], return_tensors="pt")
            pixel = {k: v.to(device, dtype=torch.float16 if v.is_floating_point() else None) for k, v in pixel.items()}
            emb = model.get_image_features(**pixel)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_emb.append(emb.cpu().float().numpy())
            all_idx.append(batch["index"].numpy())
            n += len(imgs)
            if n % (args.batch_size * 20) == 0:
                print(f"{n}/{len(ds)} embedded | {n / (time.time() - t0):.0f} fps", flush=True)

    idx = np.concatenate(all_idx)
    emb = np.concatenate(all_emb)
    dim = emb.shape[1]
    tbl = pa.table(
        {
            "index": pa.array(idx, pa.int64()),
            args.column: pa.FixedSizeListArray.from_arrays(pa.array(emb.reshape(-1), pa.float32()), dim),
        }
    )
    frames_uri = next(p for p in __import__("pathlib").Path(args.lance_root).glob("*.lance") if not p.name.endswith("_videos.lance"))
    dset = lance.dataset(str(frames_uri))
    dset.merge(tbl, left_on="index", right_on="index")
    print(f"merged column {args.column}[{dim}] into {frames_uri} | version {dset.version}")


if __name__ == "__main__":
    main()
