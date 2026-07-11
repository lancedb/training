#!/usr/bin/env python
"""Build a standard v3 *video-format* variant of an image-format LeRobot dataset.

HuggingFaceVLA/libero ships camera frames as PNG bytes inside parquet (26 GB).
Real-world LeRobot datasets are recorded with video features (mp4). This script
re-materializes the same dataset with `dtype: video` using lerobot's own writer
(`LeRobotDataset.create` + `add_frame`/`save_episode`), so the output is a
by-the-book v3 dataset with lerobot's default encoder (libsvtav1, yuv420p).

Worker mode (one shard of episodes):
    python make_video_variant.py worker --src-root SRC --out-dir OUT --shard I --num-shards N
Aggregate mode (merge all shards into the final dataset):
    python make_video_variant.py aggregate --out-dir OUT --final-root FINAL --num-shards N
"""

import argparse
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


def load_episodes_meta(src_root: Path) -> pd.DataFrame:
    files = sorted((src_root / "meta" / "episodes").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def video_features(src_root: Path) -> dict:
    info = json.loads((src_root / "meta" / "info.json").read_text())
    feats = {}
    for key, spec in info["features"].items():
        if key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            continue  # bookkeeping columns are added by the writer
        spec = dict(spec)
        if spec["dtype"] == "image":
            spec["dtype"] = "video"
        feats[key] = {"dtype": spec["dtype"], "shape": tuple(spec["shape"]), "names": spec.get("names")}
    return feats, info["fps"]


def run_worker(args) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src_root = Path(args.src_root)
    out_dir = Path(args.out_dir)
    eps = load_episodes_meta(src_root)
    # contiguous blocks keep global episode order in the aggregated output
    bounds = np.linspace(0, len(eps), args.num_shards + 1).astype(int)
    shard_eps = eps.iloc[bounds[args.shard] : bounds[args.shard + 1]]
    feats, fps = video_features(src_root)
    img_keys = [k for k, v in feats.items() if v["dtype"] == "video"]
    vec_keys = [k for k in feats if k not in img_keys]

    shard_root = out_dir / f"shard_{args.shard:04d}"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    ds = LeRobotDataset.create(
        repo_id=f"local/libero_video_shard_{args.shard:04d}",
        fps=int(fps),
        features=feats,
        root=shard_root,
        streaming_encoding=True,
    )

    # The episodes metadata in HuggingFaceVLA/libero carries stale
    # data/chunk_index+file_index values and files are not globally ordered,
    # so build an explicit episode -> parquet-file map from the data itself.
    data_files = sorted((src_root / "data").rglob("*.parquet"))
    ep_files: dict[int, list[Path]] = {}
    for f in data_files:
        col = pq.read_table(f, columns=["episode_index"])["episode_index"].to_numpy()
        for e in np.unique(col):
            ep_files.setdefault(int(e), []).append(f)

    cache: dict = {}

    def read_episode(ep_idx: int) -> pd.DataFrame:
        parts = []
        for f in ep_files[ep_idx]:
            if f not in cache:
                if len(cache) > 1:
                    cache.clear()
                cache[f] = pq.read_table(f).to_pandas()
            df = cache[f]
            parts.append(df[df["episode_index"] == ep_idx])
        return pd.concat(parts, ignore_index=True)

    t0 = time.time()
    done_frames = 0
    for _, ep in shard_eps.iterrows():
        rows = read_episode(int(ep["episode_index"])).sort_values("frame_index")
        assert len(rows) == int(ep["length"]), (
            f"episode {int(ep['episode_index'])}: got {len(rows)} rows, expected {int(ep['length'])}"
        )
        task = ep["tasks"][0] if len(ep["tasks"]) else ""
        for _, row in rows.iterrows():
            frame = {"task": task}
            for k in img_keys:
                frame[k] = np.asarray(Image.open(io.BytesIO(row[k]["bytes"])).convert("RGB"))
            for k in vec_keys:
                frame[k] = np.asarray(row[k], dtype=np.float32)
            ds.add_frame(frame)
        ds.save_episode()
        done_frames += len(rows)
        el = time.time() - t0
        print(
            f"[shard {args.shard}] ep {int(ep['episode_index'])} done | {done_frames} frames "
            f"| {done_frames / el:.1f} fps",
            flush=True,
        )
    ds.finalize()
    print(f"[shard {args.shard}] FINISHED {done_frames} frames in {time.time() - t0:.0f}s", flush=True)


def run_aggregate(args) -> None:
    from lerobot.datasets.aggregate import aggregate_datasets

    out_dir = Path(args.out_dir)
    roots = [out_dir / f"shard_{i:04d}" for i in range(args.num_shards)]
    for r in roots:
        assert r.exists(), f"missing shard root {r}"
    aggregate_datasets(
        repo_ids=[f"local/libero_video_shard_{i:04d}" for i in range(args.num_shards)],
        aggr_repo_id="local/libero_video",
        roots=roots,
        aggr_root=Path(args.final_root),
    )
    print("AGGREGATED ->", args.final_root, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--src-root", required=True)
    w.add_argument("--out-dir", required=True)
    w.add_argument("--shard", type=int, required=True)
    w.add_argument("--num-shards", type=int, required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--final-root", required=True)
    a.add_argument("--num-shards", type=int, required=True)
    args = p.parse_args()
    if args.mode == "worker":
        run_worker(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    sys.exit(main())
