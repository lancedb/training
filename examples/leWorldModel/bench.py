"""
leWorldModel dataloader throughput benchmark: LanceDB vs HDF5.

Measures how fast each backend can feed batches to the GPU, independently
of training compute.  Three backends:

  LanceDB S3/local   — our implementation, parallel workers, no download step
  HDF5 local         — reads from a local file (best-case for HDF5)
  HDF5 s3fs          — reads directly from S3 via s3fs (realistic, no download)

Usage:
  # LanceDB S3 vs HDF5 local
  python bench.py \\
    --lance-uri s3://my-bucket/lewm \\
    --table-name lewm_pusht \\
    --hdf5-local /dev/shm/pusht.hdf5

  # All three backends
  python bench.py \\
    --lance-uri s3://my-bucket/lewm \\
    --table-name lewm_pusht \\
    --hdf5-local /dev/shm/pusht.hdf5 \\
    --hdf5-s3-key hdf5/pusht.hdf5 \\
    --s3-bucket my-bucket

  # Credentials via environment variables (AWS_ACCESS_KEY_ID etc.)
  python bench.py --lance-uri s3://my-bucket/lewm --table-name lewm_pusht
"""

import argparse
import multiprocessing
import os
import time

import h5py
import hdf5plugin  # noqa: F401 — registers HDF5 decompression filters
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.insert(0, os.path.dirname(__file__))
from lewm_loader import make_lewm_lance_loader


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BATCH_SIZE      = 128
NUM_STEPS       = 4
IMAGE_SIZE      = 224
NUM_WORKERS     = 8
PREFETCH_FACTOR = 3
WARMUP_BATCHES  = 5
BENCH_BATCHES   = 50
COLUMNS         = ["pixels", "action", "proprio", "state"]


# ---------------------------------------------------------------------------
# HDF5 dataset  (mirrors the original stable_worldmodel.data.HDF5Dataset)
# ---------------------------------------------------------------------------

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class HDF5LeWMDataset(torch.utils.data.Dataset):
    """
    HDF5-backed temporal-window dataset matching stable-worldmodel's HDF5Dataset.

    The HDF5 schema uses per-episode metadata arrays:
      ep_len    — shape (n_episodes,) episode lengths
      ep_offset — shape (n_episodes,) global start row per episode

    Valid clip_indices are (episode_idx, local_start) pairs where a full window
    of span = num_steps * frameskip rows fits within the episode.  At read time,
    the global slice [offset + local_start : offset + local_start + span] is
    fetched and every frameskip-th frame is selected.

    Pixels are stored as (N, H, W, C) uint8 — no transpose needed before PIL.
    Non-pixel columns (action, proprio, etc.) are cached in RAM at init time;
    only pixels are read from the file at __getitem__ time.

    hdf5_src can be a local file path (str) or an s3fs file object.
    h5py is opened lazily per worker because handles are not fork-safe.
    """

    def __init__(self, hdf5_src, columns, num_steps=NUM_STEPS, frameskip=1):
        self._src      = hdf5_src
        self.columns   = columns
        self.num_steps = num_steps
        self.frameskip = frameskip
        self._span     = num_steps * frameskip
        self._file     = None

        with h5py.File(self._src, "r", rdcc_nbytes=256 * 1024 * 1024) as f:
            ep_len    = np.array(f["ep_len"],    dtype=np.int32)
            ep_offset = np.array(f["ep_offset"], dtype=np.int32)
            # Cache all non-pixel columns in RAM — avoids repeated random HDF5 seeks
            self._cached: dict[str, np.ndarray] = {}
            for col in columns:
                if col != "pixels":
                    self._cached[col] = np.array(f[col], dtype=np.float32)

        # Build (ep_idx, local_start) pairs for all valid windows
        self._clip_indices: list[tuple[int, int]] = []
        for ep_idx, (off, length) in enumerate(zip(ep_offset.tolist(), ep_len.tolist())):
            if length < self._span:
                continue
            for local_start in range(length - self._span + 1):
                self._clip_indices.append((ep_idx, local_start))

        self._ep_offset = ep_offset

    def __len__(self):
        return len(self._clip_indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None       # h5py handle can't be pickled
        return state

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self._src, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)

    def __getitem__(self, clip_idx):
        self._ensure_open()
        ep_idx, local_start = self._clip_indices[clip_idx]
        g_start = int(self._ep_offset[ep_idx]) + local_start
        g_end   = g_start + self._span

        pixels_raw = self._file["pixels"][g_start:g_end:self.frameskip]   # (T, H, W, C)
        frames = [
            _TRANSFORM(Image.fromarray(pixels_raw[t].astype(np.uint8)))
            for t in range(self.num_steps)
        ]

        sample = {"pixels": torch.stack(frames)}
        for col in self.columns:
            if col == "pixels":
                continue
            data = self._cached[col][g_start:g_end:self.frameskip]
            sample[col] = torch.from_numpy(np.nan_to_num(data, nan=0.0))
        return sample


def _collate(samples):
    return {k: torch.stack([s[k] for s in samples], dim=0) for k in samples[0]}


def make_hdf5_loader(hdf5_src, columns, batch_size, num_workers, prefetch_factor):
    return DataLoader(
        HDF5LeWMDataset(hdf5_src, columns),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def measure_throughput(loader, label, warmup, steps):
    """
    Iterate the loader for `warmup` batches (discarded), then time `steps` batches.
    Returns samples/sec and average batch latency in ms.
    """
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  batch_size={loader.batch_size}  workers={loader.num_workers}")
    print(f"{'─' * 60}")

    it = iter(loader)

    print(f"  warming up ({warmup} batches)...")
    for _ in range(warmup):
        batch = next(it, None)
        if batch is None:
            it = iter(loader)
            batch = next(it)
        # Touch the pixels tensor to ensure decoding actually happened
        _ = batch["pixels"].shape

    print(f"  benchmarking ({steps} batches)...")
    batch_times = []
    t_total = time.perf_counter()

    for _ in range(steps):
        t0 = time.perf_counter()
        batch = next(it, None)
        if batch is None:
            it = iter(loader)
            batch = next(it)
        _ = batch["pixels"].shape
        batch_times.append(time.perf_counter() - t0)

    elapsed = time.perf_counter() - t_total
    samples_sec = (steps * loader.batch_size) / elapsed
    avg_ms = np.mean(batch_times) * 1000
    p99_ms = np.percentile(batch_times, 99) * 1000

    print(f"\n  samples/sec : {samples_sec:,.0f}")
    print(f"  batch avg   : {avg_ms:.1f} ms")
    print(f"  batch p99   : {p99_ms:.1f} ms")

    return {"label": label, "samples_sec": samples_sec, "avg_ms": avg_ms, "p99_ms": p99_ms}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description="Benchmark LanceDB vs HDF5 dataloader throughput for leWorldModel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lance-uri",    required=True)
    p.add_argument("--table-name",   required=True)
    p.add_argument("--hdf5-local",   default=None,  help="Path to local HDF5 file")
    p.add_argument("--hdf5-s3-key",  default=None,  help="S3 object key for HDF5 file")
    p.add_argument("--s3-bucket",    default=None,  help="S3 bucket (for --hdf5-s3-key)")
    p.add_argument("--columns",      nargs="+",     default=COLUMNS)
    p.add_argument("--batch-size",   type=int,      default=BATCH_SIZE)
    p.add_argument("--num-workers",  type=int,      default=NUM_WORKERS)
    p.add_argument("--warmup",       type=int,      default=WARMUP_BATCHES)
    p.add_argument("--steps",        type=int,      default=BENCH_BATCHES)

    s3 = p.add_argument_group("S3 credentials (fall back to AWS_* env vars)")
    s3.add_argument("--aws-access-key-id",     default=os.environ.get("AWS_ACCESS_KEY_ID"))
    s3.add_argument("--aws-secret-access-key", default=os.environ.get("AWS_SECRET_ACCESS_KEY"))
    s3.add_argument("--aws-session-token",     default=os.environ.get("AWS_SESSION_TOKEN"))
    s3.add_argument("--aws-region",            default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    s3.add_argument("--s3-endpoint",           default=os.environ.get("AWS_ENDPOINT_URL"))
    return p


def main():
    args = _build_parser().parse_args()

    print(f"\nleWorldModel dataloader benchmark")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  num_workers : {args.num_workers}")
    print(f"  T (frames)  : {NUM_STEPS}")
    print(f"  warmup      : {args.warmup} batches  bench: {args.steps} batches")

    # Build S3 storage_options for LanceDB
    storage_options = {}
    if args.aws_access_key_id:
        storage_options["aws_access_key_id"] = args.aws_access_key_id
    if args.aws_secret_access_key:
        storage_options["aws_secret_access_key"] = args.aws_secret_access_key
    if args.aws_session_token:
        storage_options["aws_session_token"] = args.aws_session_token
    if args.aws_region:
        storage_options["region"] = args.aws_region
    if args.s3_endpoint:
        storage_options["endpoint_url"] = args.s3_endpoint
        storage_options["aws_virtual_hosted_style_request"] = "false"

    connect_kwargs = {"storage_options": storage_options} if storage_options else {}

    results = []

    # 1. LanceDB
    lance_loader = make_lewm_lance_loader(
        uri=args.lance_uri,
        table_name=args.table_name,
        columns=args.columns,
        batch_size=args.batch_size,
        num_steps=NUM_STEPS,
        img_size=IMAGE_SIZE,
        num_workers=args.num_workers,
        prefetch_factor=PREFETCH_FACTOR,
        **connect_kwargs,
    )
    backend = "S3" if args.lance_uri.startswith("s3://") else "local"
    results.append(measure_throughput(
        lance_loader,
        f"LanceDB {backend}  ({args.table_name})",
        args.warmup, args.steps,
    ))

    # 2. HDF5 local
    if args.hdf5_local:
        hdf5_local_loader = make_hdf5_loader(
            args.hdf5_local, args.columns, args.batch_size, args.num_workers, PREFETCH_FACTOR,
        )
        results.append(measure_throughput(
            hdf5_local_loader,
            f"HDF5 local  ({os.path.basename(args.hdf5_local)})",
            args.warmup, args.steps,
        ))

    # 3. HDF5 via s3fs (reads directly from S3, no local copy)
    if args.hdf5_s3_key and args.s3_bucket:
        import s3fs
        s3_kwargs = {}
        if args.aws_access_key_id:
            s3_kwargs["key"] = args.aws_access_key_id
        if args.aws_secret_access_key:
            s3_kwargs["secret"] = args.aws_secret_access_key
        if args.aws_session_token:
            s3_kwargs["token"] = args.aws_session_token
        client_kwargs = {}
        if args.aws_region:
            client_kwargs["region_name"] = args.aws_region
        if args.s3_endpoint:
            client_kwargs["endpoint_url"] = args.s3_endpoint
        if client_kwargs:
            s3_kwargs["client_kwargs"] = client_kwargs

        fs = s3fs.S3FileSystem(**s3_kwargs)
        s3_file = fs.open(f"{args.s3_bucket}/{args.hdf5_s3_key}", "rb")

        hdf5_s3_loader = make_hdf5_loader(
            s3_file, args.columns, args.batch_size, args.num_workers, PREFETCH_FACTOR,
        )
        results.append(measure_throughput(
            hdf5_s3_loader,
            f"HDF5 s3fs  (s3://{args.s3_bucket}/{args.hdf5_s3_key})",
            args.warmup, args.steps,
        ))

    # Summary table
    if len(results) > 1:
        baseline = results[-1]["samples_sec"]
        print(f"\n{'=' * 60}")
        print(f"  {'Backend':<46} {'samples/sec':>12}  {'avg ms':>8}  {'speedup':>8}")
        print(f"{'─' * 60}")
        for r in sorted(results, key=lambda x: -x["samples_sec"]):
            speedup = r["samples_sec"] / baseline
            print(f"  {r['label']:<46} {r['samples_sec']:>12,.0f}  {r['avg_ms']:>7.1f}  {speedup:>7.1f}×")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
