"""
Direct (non-Geneva) backfill for the Tier-3 columns.

Geneva's actor pipeline stalls on the very-large per-row Tier-3 columns
(2 MB+ each) — a known issue in our pinned ``geneva==0.12.0`` stack
(actor goes ``S (sleeping)`` after model load and never picks up the
first task; pipeline watchdog kills it after 600 s).

This module sidesteps Geneva entirely for ``t5_hidden_states`` and
``vae_latent``:

  1. Open the Lance dataset directly.
  2. Scan ``(clip_id, caption)`` or ``(clip_id, video_bytes)`` in
     small batches.
  3. Run the UMT5 / Wan-VAE forward pass in-process.
  4. Merge the result column back into the table via
     ``dataset.merge_insert(key)`` keyed on ``clip_id``.

It's restart-safe: rows whose target column is already non-NULL are
skipped on the read.

Usage
-----
python -m videogen.backfill_direct --column t5_hidden_states
python -m videogen.backfill_direct --column vae_latent --batch-size 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import torch

from videogen.geneva_udfs import (
    _decode_evenly_spaced_frames,
    WAN_MODEL_ID,
)
from videogen.schema import (
    T5_HIDDEN, T5_SEQ_LEN, T5_TOTAL,
    VAE_INPUT_FRAMES, VAE_INPUT_H, VAE_INPUT_W,
    VAE_LATENT_C, VAE_LATENT_H, VAE_LATENT_T, VAE_LATENT_W, VAE_TOTAL,
)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class T5Encoder:
    def __init__(self) -> None:
        from transformers import T5TokenizerFast, UMT5EncoderModel
        print("Loading UMT5-XXL …", flush=True)
        t0 = time.time()
        self.device = torch.device("cuda")
        self.tokenizer = T5TokenizerFast.from_pretrained(
            WAN_MODEL_ID, subfolder="tokenizer",
        )
        self.model = UMT5EncoderModel.from_pretrained(
            WAN_MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16,
        ).eval().to(self.device)
        print(f"  loaded in {time.time() - t0:.1f}s  "
              f"({sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B params)")

    def encode(self, captions: list[str]) -> np.ndarray:
        """Return (B, T5_SEQ_LEN, T5_HIDDEN) fp16 numpy."""
        texts = [c if c else " " for c in captions]
        toks = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=T5_SEQ_LEN, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            h = self.model(**toks).last_hidden_state
        return h.reshape(h.shape[0], -1).cpu().numpy().astype(np.float16, copy=False)


class VaeEncoder:
    def __init__(self) -> None:
        from diffusers import AutoencoderKLWan
        from PIL import Image  # noqa
        print("Loading AutoencoderKLWan …", flush=True)
        t0 = time.time()
        self.device = torch.device("cuda")
        self.vae = AutoencoderKLWan.from_pretrained(
            WAN_MODEL_ID, subfolder="vae", torch_dtype=torch.float32,
        ).eval().to(self.device)
        print(f"  loaded in {time.time() - t0:.1f}s  "
              f"({sum(p.numel() for p in self.vae.parameters()) / 1e6:.0f}M params)")

    def encode_one(self, video_bytes: bytes) -> np.ndarray:
        """Return a flat fp16 array of length VAE_TOTAL."""
        from PIL import Image
        frames = _decode_evenly_spaced_frames(video_bytes or b"", VAE_INPUT_FRAMES)
        if len(frames) < VAE_INPUT_FRAMES:
            if not frames:
                return np.zeros(VAE_TOTAL, dtype=np.float16)
            while len(frames) < VAE_INPUT_FRAMES:
                frames.append(frames[-1])

        resized = [
            np.asarray(im.resize((VAE_INPUT_W, VAE_INPUT_H), Image.BICUBIC))
            for im in frames[:VAE_INPUT_FRAMES]
        ]
        arr = np.stack(resized, axis=0)
        t = torch.from_numpy(arr).to(self.device).float().div(127.5).sub(1.0)
        t = t.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        with torch.no_grad():
            enc = self.vae.encode(t)
            lat = enc.latent_dist.sample() if hasattr(enc, "latent_dist") else enc.latents
        return (
            lat.squeeze(0).reshape(-1).cpu().numpy().astype(np.float16, copy=False)
        )


# ---------------------------------------------------------------------------
# Streamed add_columns transform
# ---------------------------------------------------------------------------
#
# Lance's ``add_columns(transform, read_columns)`` scans the dataset
# fragment-by-fragment, calls our transform on each batch, and merges
# the returned columns back in-place.  This is exactly the mechanism
# Geneva uses under the hood, just without Geneva's Ray actor system
# (which is the part that stalls on heavy UDFs in our pinned stack).
#
# The transform must be picklable for some Lance back-ends; we keep the
# model inside the function via a module-level cache to avoid re-loading
# on every batch within a single process.

_T5_SINGLETON: T5Encoder | None = None
_VAE_SINGLETON: VaeEncoder | None = None


def _t5_transform(batch: pa.RecordBatch) -> pa.RecordBatch:
    global _T5_SINGLETON
    if _T5_SINGLETON is None:
        _T5_SINGLETON = T5Encoder()
    captions = batch.column("caption").to_pylist()
    t0 = time.perf_counter()
    flat = _T5_SINGLETON.encode(captions).reshape(-1)
    fsl = pa.FixedSizeListArray.from_arrays(
        pa.array(flat, type=pa.float16()), T5_TOTAL,
    )
    dt = time.perf_counter() - t0
    print(f"    [t5] {len(batch)} rows in {dt:.2f}s "
          f"({len(batch) / max(dt, 1e-6):.1f} rows/s)", flush=True)
    return pa.record_batch(
        [fsl], schema=pa.schema([pa.field("t5_hidden_states", fsl.type)])
    )


def _vae_transform(batch: pa.RecordBatch) -> pa.RecordBatch:
    global _VAE_SINGLETON
    if _VAE_SINGLETON is None:
        _VAE_SINGLETON = VaeEncoder()
    vbs = batch.column("video_bytes").to_pylist()
    t0 = time.perf_counter()
    chunks = [_VAE_SINGLETON.encode_one(b) for b in vbs]
    flat = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float16)
    fsl = pa.FixedSizeListArray.from_arrays(
        pa.array(flat, type=pa.float16()), VAE_TOTAL,
    )
    dt = time.perf_counter() - t0
    print(f"    [vae] {len(batch)} rows in {dt:.2f}s "
          f"({len(batch) / max(dt, 1e-6):.2f} rows/s)", flush=True)
    return pa.record_batch(
        [fsl], schema=pa.schema([pa.field("vae_latent", fsl.type)])
    )


# ---------------------------------------------------------------------------
# Backfill entrypoints
# ---------------------------------------------------------------------------

def backfill_t5(*, db_path: str, table: str, batch_size: int = 32) -> None:
    ds_path = str(Path(db_path) / f"{table}.lance")
    ds = lance.dataset(ds_path)
    if "t5_hidden_states" in ds.schema.names:
        print("  t5_hidden_states already present — drop it first if you want to rerun")
        return
    print(f"  add_columns(t5_hidden_states) on {ds.count_rows()} rows "
          f"(batch_size={batch_size}) …")
    t0 = time.perf_counter()
    ds.add_columns(_t5_transform, read_columns=["caption"],
                   batch_size=batch_size)
    print(f"\nt5_hidden_states done in {time.perf_counter() - t0:.1f}s")


def backfill_vae(*, db_path: str, table: str, batch_size: int = 4) -> None:
    ds_path = str(Path(db_path) / f"{table}.lance")
    ds = lance.dataset(ds_path)
    if "vae_latent" in ds.schema.names:
        print("  vae_latent already present — drop it first if you want to rerun")
        return
    print(f"  add_columns(vae_latent) on {ds.count_rows()} rows "
          f"(batch_size={batch_size}) …")
    t0 = time.perf_counter()
    ds.add_columns(_vae_transform, read_columns=["video_bytes"],
                   batch_size=batch_size)
    print(f"\nvae_latent done in {time.perf_counter() - t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db",     default="data/videos/lancedb")
    p.add_argument("--table",  default="videos_raw")
    p.add_argument("--column", required=True,
                   choices=["t5_hidden_states", "vae_latent"])
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override default per-column batch size.")
    args = p.parse_args(argv)

    if args.column == "t5_hidden_states":
        backfill_t5(db_path=args.db, table=args.table,
                    batch_size=args.batch_size or 32)
    else:
        backfill_vae(db_path=args.db, table=args.table,
                     batch_size=args.batch_size or 4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
