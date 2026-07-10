#!/usr/bin/env python
"""DataLoader throughput benchmark: upstream image-parquet vs upstream video vs Lance video.

Measures steady-state samples/s through a torch DataLoader with a SmolVLA-style
read pattern (1 observation frame per camera + 50-step action chunk), the same
delta_timestamps lerobot-train resolves for `lerobot/smolvla_base` on a 10 fps
dataset.

    python bench_throughput.py --backend {image,video,lance} --root PATH \
        --batch-size 64 --num-workers 8 --num-batches 60 --warmup 10 \
        [--uri s3://...  --meta-root PATH] [--out results.json]
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

FPS = 10
CHUNK = 50  # smolvla chunk_size


def smolvla_delta_timestamps(image_keys):
    dt = {k: [0.0] for k in image_keys}
    dt["observation.state"] = [0.0]
    dt["action"] = [i / FPS for i in range(CHUNK)]
    return dt


def build_dataset(args):
    if args.backend == "lance":
        from lerobot_lancedb import LeRobotLanceVideoDataset

        kwargs = dict(return_uint8=True)
        if args.uri:
            kwargs.update(uri=args.uri, meta_root=args.meta_root)
        else:
            kwargs.update(root=args.root)
        image_keys = ["observation.images.image", "observation.images.image2"]
        return LeRobotLanceVideoDataset(delta_timestamps=smolvla_delta_timestamps(image_keys), **kwargs)
    else:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        image_keys = ["observation.images.image", "observation.images.image2"]
        return LeRobotDataset(
            repo_id="HuggingFaceVLA/libero",
            root=args.root,
            delta_timestamps=smolvla_delta_timestamps(image_keys),
            return_uint8=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["image", "video", "lance"], required=True)
    p.add_argument("--root")
    p.add_argument("--uri")
    p.add_argument("--meta-root")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-batches", type=int, default=60)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out")
    args = p.parse_args()

    ds = build_dataset(args)
    n = len(ds)
    print(f"backend={args.backend} len={n}")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True,
        pin_memory=True,
    )
    t_start = time.perf_counter()
    it = iter(dl)
    next(it)
    ttfb = time.perf_counter() - t_start
    for _ in range(args.warmup - 1):
        next(it)
    t0 = time.perf_counter()
    for _ in range(args.num_batches):
        batch = next(it)
    el = time.perf_counter() - t0
    sps = args.num_batches * args.batch_size / el
    res = {
        "backend": args.backend,
        "uri": args.uri,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_batches": args.num_batches,
        "time_to_first_batch_s": round(ttfb, 2),
        "seconds": round(el, 2),
        "samples_per_s": round(sps, 1),
        "batches_per_s": round(args.num_batches / el, 2),
    }
    print(json.dumps(res))
    if args.out:
        out = Path(args.out)
        rows = json.loads(out.read_text()) if out.exists() else []
        rows.append(res)
        out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
