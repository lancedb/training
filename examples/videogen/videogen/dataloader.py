"""
PyTorch DataLoader factories for the videogen training pipeline.

Two paths:

  ``make_cached_loader(uri, view, ...)``   ← **the headline path**
      Projects only ``t5_hidden_states`` + ``vae_latent``.  The VAE and T5
      models never load in the training process.  All VRAM goes to the DiT.

  ``make_raw_loader(uri, view, ...)``      ← baseline path
      Projects ``video_bytes`` + ``caption``.  Used to measure the cost of
      the classic "decode-and-encode-every-step" pipeline.

Both follow the proven pattern from object-detection / ViT MFU bench:

  * Dataset holds only connection params; each worker reopens its own
    ``Permutation`` handle lazily in a child process.
  * ``__getstate__`` zeros the Permutation handle before pickling.
  * ``__getitems__`` returns a ``pa.RecordBatch`` so the collate function
    decodes the whole batch at once instead of per row.
  * ``multiprocessing_context="spawn"`` — Lance + multiprocessing is unsafe
    with ``fork`` (LanceDB docs warn about it explicitly).
"""

from __future__ import annotations

from typing import Iterable

import lancedb
import numpy as np
import pyarrow as pa
import torch
from lancedb.permutation import Permutation
from torch.utils.data import DataLoader, Dataset, RandomSampler

from videogen.schema import (
    CACHED_TRAIN_COLUMNS,
    RAW_TRAIN_COLUMNS,
    T5_HIDDEN,
    T5_SEQ_LEN,
    VAE_LATENT_C,
    VAE_LATENT_H,
    VAE_LATENT_T,
    VAE_LATENT_W,
)


# ---------------------------------------------------------------------------
# Shared dataset — only the projection differs between paths
# ---------------------------------------------------------------------------

class LanceVideoDataset(Dataset):
    """Random-access dataset over a Lance table or materialised view.

    Stores only ``(uri, table_name, columns)`` so it pickles cleanly into
    worker processes.  Each worker reopens its own connection + Permutation
    handle in ``_ensure_open``.
    """

    def __init__(self, uri: str, table_name: str, columns: Iterable[str]):
        self.uri        = uri
        self.table_name = table_name
        self.columns    = list(columns)
        self._perm      = None
        # Length needs a connection; this is on the driver so it's fine.
        db = lancedb.connect(uri)
        self.length = len(db.open_table(table_name))

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_perm"] = None
        return state

    def _ensure_open(self) -> None:
        if self._perm is not None:
            return
        db = lancedb.connect(self.uri)
        self._perm = (
            Permutation.identity(db.open_table(self.table_name))
            .select_columns(self.columns)
            .with_format("arrow")
        )

    def __getitem__(self, idx: int):
        self._ensure_open()
        return self._perm[idx]

    def __getitems__(self, indices: list[int]):
        self._ensure_open()
        return self._perm.__getitems__(indices)


# ---------------------------------------------------------------------------
# Cached path: T5 hidden states + Wan-VAE latent → straight to the DiT
# ---------------------------------------------------------------------------

def _flat_fp16_to_tensor(arr: pa.Array, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    """Zero-copy reshape of a Lance ``list<f16|bf16>[N]`` column into a tensor.

    Lance stores these as a flat fixed-size list; we convert to numpy with
    zero copy, then to torch.  ``dtype`` is the numpy/torch type of the
    underlying storage (Lance does not natively support bfloat16 yet, so the
    VAE latent column is stored as fp16 and cast at train time if needed).
    """
    np_dtype_map = {torch.float16: np.float16,
                    torch.float32: np.float32,
                    torch.int32:   np.int32}
    np_dtype = np_dtype_map[dtype]
    # zero_copy_only=False because the FSL → flat numpy conversion does one
    # contiguous copy.  Still 10-100× cheaper than per-row decode + VAE.
    flat = arr.flatten().to_numpy(zero_copy_only=False).astype(np_dtype, copy=False)
    flat = flat.reshape((len(arr),) + shape)
    return torch.from_numpy(flat)


def _cached_collate(batch: pa.RecordBatch):
    t5  = _flat_fp16_to_tensor(batch.column("t5_hidden_states"), torch.float16,
                               (T5_SEQ_LEN, T5_HIDDEN))
    lat = _flat_fp16_to_tensor(batch.column("vae_latent"), torch.float16,
                               (VAE_LATENT_C, VAE_LATENT_T,
                                VAE_LATENT_H, VAE_LATENT_W))
    return {"prompt_embeds": t5, "vae_latent": lat}


def make_cached_loader(
    uri: str,
    table_name: str,
    *,
    batch_size: int       = 1,
    num_workers: int      = 8,
    prefetch_factor: int  = 4,
    shuffle: bool         = True,
    seed: int             = 42,
) -> DataLoader:
    """DataLoader over (``t5_hidden_states``, ``vae_latent``) only — no VAE/T5
    process touch them."""
    dataset = LanceVideoDataset(uri, table_name, CACHED_TRAIN_COLUMNS)
    return _make_loader(
        dataset,
        collate_fn=_cached_collate,
        batch_size=batch_size, num_workers=num_workers,
        prefetch_factor=prefetch_factor, shuffle=shuffle, seed=seed,
    )


# ---------------------------------------------------------------------------
# Raw path: video bytes + caption  → decode + VAE + T5 inside the train loop
# (this is what we benchmark against, NOT the recommended path)
# ---------------------------------------------------------------------------

def _raw_collate(batch: pa.RecordBatch):
    # NOTE: video_bytes on a blob v2 column comes back as a `struct<position,size>`
    # descriptor here, not the actual bytes — for the raw baseline you must
    # fetch real bytes with ``ds.take_blobs(...)``.  This collate is therefore
    # only useful as a benchmark scaffold; we will replace it with a real
    # blob-aware variant in bench_dataloader.py.
    captions = batch.column("caption").to_pylist()
    return {"video_bytes_descriptor": batch.column("video_bytes"),
            "captions":               captions}


def make_raw_loader(
    uri: str,
    table_name: str,
    *,
    batch_size: int       = 1,
    num_workers: int      = 8,
    prefetch_factor: int  = 4,
    shuffle: bool         = True,
    seed: int             = 42,
) -> DataLoader:
    """DataLoader over (``video_bytes``, ``caption``) — baseline path used to
    measure how much the cached path saves."""
    dataset = LanceVideoDataset(uri, table_name, RAW_TRAIN_COLUMNS)
    return _make_loader(
        dataset,
        collate_fn=_raw_collate,
        batch_size=batch_size, num_workers=num_workers,
        prefetch_factor=prefetch_factor, shuffle=shuffle, seed=seed,
    )


# ---------------------------------------------------------------------------
# Shared loader builder
# ---------------------------------------------------------------------------

def _make_loader(
    dataset: Dataset,
    *,
    collate_fn,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    sampler = None
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=g)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        multiprocessing_context="spawn" if num_workers > 0 else None,
        drop_last=True,
    )
