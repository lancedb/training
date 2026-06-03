"""LanceDB-backed dataloaders for the VLM SFT task, via the Permutation API.

Follows the same pattern as ``object-detection/object_detection/dataloader.py``:

  * a ``torch.utils.data.Dataset`` that stores only connection params; each
    DataLoader worker reopens its own ``Permutation`` lazily,
  * ``__getitems__`` returns a ``pa.RecordBatch`` (``with_format("arrow")``),
  * a ``collate_fn`` turns that RecordBatch into the model batch.

Two loaders:

  * **cached** (``make_cached_loader``) — reads the pre-computed
    ``vision_tower_hiddens`` + token columns; the train loop pays zero cost
    for image decode, vision-tower forward, or tokenisation.
  * **raw** (``make_raw_loader``) — reads ``image`` + ``question`` +
    ``answer`` and decodes inline; the "what you'd do without caching"
    baseline, still served by LanceDB.

Both open the table with ``lancedb.connect(uri).open_table(name)``.  The
local table written by ``vlm/ingest.py`` lives at ``<uri>/<name>.lance``,
so a ``data/textvqa.lance`` path splits into ``uri="data"``,
``name="textvqa"`` (see ``_split_db``).
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import torch
from lancedb.permutation import Permutation
from PIL import Image

from .schema import IMAGE_PX, LLM_TOKENS_PER_IMAGE, MAX_TEXT_TOKENS, VISION_HIDDEN

LOG = logging.getLogger("vlm.dataloader")

_TOKEN_FIELDS = ("input_ids", "attention_mask", "labels")
_CACHED_FLAT_COLS = ["vision_tower_hiddens", *_TOKEN_FIELDS]
_CACHED_STRUCT_COLS = ["vision_tower_hiddens", "sft_tokens"]
_RAW_COLS = ["image", "question", "answer"]


def _split_db(db: str) -> tuple[str, str]:
    """``data/textvqa.lance`` -> ``("data", "textvqa")`` for lancedb.connect."""
    p = Path(db)
    name = p.name[:-len(".lance")] if p.name.endswith(".lance") else p.name
    return str(p.parent), name


def _as_array(col):
    """RecordBatch columns are Arrays; Table columns are ChunkedArrays."""
    return col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col


# ---------------------------------------------------------------------------
# batch containers
# ---------------------------------------------------------------------------

@dataclass
class CachedBatch:
    vision_hiddens:  torch.Tensor   # fp16 [B, LLM_TOKENS_PER_IMAGE, VISION_HIDDEN]
    input_ids:       torch.Tensor   # int64 [B, MAX_TEXT_TOKENS]
    attention_mask:  torch.Tensor   # int64 [B, MAX_TEXT_TOKENS]
    labels:          torch.Tensor   # int64 [B, MAX_TEXT_TOKENS]

    def to(self, device: torch.device, non_blocking: bool = True) -> "CachedBatch":
        return CachedBatch(
            vision_hiddens = self.vision_hiddens.to(device, non_blocking=non_blocking),
            input_ids      = self.input_ids.to(device,      non_blocking=non_blocking),
            attention_mask = self.attention_mask.to(device, non_blocking=non_blocking),
            labels         = self.labels.to(device,         non_blocking=non_blocking),
        )


@dataclass
class RawBatch:
    images:    list[Image.Image]
    questions: list[str]
    answers:   list[str]


# ---------------------------------------------------------------------------
# collate fns (receive a pa.RecordBatch from Permutation.__getitems__)
# ---------------------------------------------------------------------------

def _cached_collate(batch: pa.RecordBatch) -> CachedBatch:
    bsz = batch.num_rows

    flat_v = _as_array(batch.column("vision_tower_hiddens")).values.to_numpy(zero_copy_only=False)
    vision = torch.from_numpy(flat_v.reshape(bsz, LLM_TOKENS_PER_IMAGE, VISION_HIDDEN))  # fp16

    # Tokens live either as flat columns (direct backfill) or inside an
    # ``sft_tokens`` struct (Geneva sft_tokens UDF). Handle both.
    if "sft_tokens" in batch.schema.names:
        struct = _as_array(batch.column("sft_tokens"))
        get = struct.field
    else:
        get = lambda f: _as_array(batch.column(f))

    def _to_long(arr) -> torch.Tensor:
        flat = arr.values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        return torch.from_numpy(flat.reshape(bsz, MAX_TEXT_TOKENS)).to(torch.long)

    return CachedBatch(
        vision_hiddens = vision,
        input_ids      = _to_long(get("input_ids")),
        attention_mask = _to_long(get("attention_mask")),
        labels         = _to_long(get("labels")),
    )


def _raw_collate(batch: pa.RecordBatch) -> RawBatch:
    questions = batch.column("question").to_pylist()
    answers   = batch.column("answer").to_pylist()
    images = [
        Image.open(io.BytesIO(b)).convert("RGB").resize((IMAGE_PX, IMAGE_PX), Image.LANCZOS)
        for b in batch.column("image").to_pylist()
    ]
    return RawBatch(images=images, questions=questions, answers=answers)


# ---------------------------------------------------------------------------
# Dataset (Permutation per worker)
# ---------------------------------------------------------------------------

class _LancePermutationDataset(torch.utils.data.Dataset):
    """Stores connection params; each worker reopens its own Permutation."""

    def __init__(self, uri: str, table_name: str, columns: list[str]):
        self.uri        = uri
        self.table_name = table_name
        self.columns    = columns
        self._perm      = None

        db = lancedb.connect(uri)
        self.length = len(db.open_table(table_name))

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict:
        # Permutation holds Rust async state — zero it so each worker reopens its own.
        state = self.__dict__.copy()
        state["_perm"] = None
        return state

    def _ensure_open(self) -> None:
        if self._perm is None:
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
# Factories
# ---------------------------------------------------------------------------

def _make_loader(dataset, collate_fn, batch_size, num_workers, shuffle, seed):
    sampler = None
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(dataset, generator=g)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


def make_cached_loader(
    db: str,
    batch_size: int = 2,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """DataLoader over the cached tier-3 columns (zero vision-tower work)."""
    uri, table_name = _split_db(db)
    tbl_names = set(lancedb.connect(uri).open_table(table_name).schema.names)
    struct = "sft_tokens" in tbl_names and not set(_TOKEN_FIELDS) <= tbl_names
    columns = _CACHED_STRUCT_COLS if struct else _CACHED_FLAT_COLS
    LOG.info("cached token layout: %s", "sft_tokens struct" if struct else "flat columns")
    dataset = _LancePermutationDataset(uri, table_name, columns)
    return _make_loader(dataset, _cached_collate, batch_size, num_workers, shuffle, seed)


def make_raw_loader(
    db: str,
    batch_size: int = 2,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """DataLoader over raw (image, question, answer); decode inline.

    Same data served by LanceDB, minus the cache — the "without the trick"
    comparison for the cached path.
    """
    uri, table_name = _split_db(db)
    dataset = _LancePermutationDataset(uri, table_name, _RAW_COLS)
    return _make_loader(dataset, _raw_collate, batch_size, num_workers, shuffle, seed)
