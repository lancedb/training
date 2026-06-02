"""Lance-backed dataloader for the VLM SFT task.

Two paths:

  * **cached**: read pre-computed ``vision_tower_hiddens`` + ``input_ids``
    + ``attention_mask`` + ``labels`` directly.  The training loop pays
    zero cost for image decode, vision-tower forward, or tokenisation.

  * **raw**: read ``image`` (jpeg bytes) + ``question`` + ``answer`` and
    do all of that work inline.  This is the "what you'd write without
    Lance" baseline path; the cached path beats it because the
    vision-tower + tokeniser have been amortised at backfill time.

Both paths use ``lance.sampler.maybe_sample`` for shuffled batched
reads with the Lance ``Permutation`` API.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Iterator

import lance
import numpy as np
import pyarrow as pa
import torch
from PIL import Image

from .schema import (
    IMAGE_PX,
    LLM_TOKENS_PER_IMAGE,
    MAX_TEXT_TOKENS,
    VISION_HIDDEN,
)

LOG = logging.getLogger("vlm.dataloader")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _shuffled_indices(n_rows: int, batch_size: int, seed: int) -> Iterator[list[int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    for i in range(0, n_rows, batch_size):
        yield order[i:i + batch_size].tolist()


@dataclass
class CachedBatch:
    """Output of the cached path."""
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
    """Output of the raw path (vision tower + tokeniser run inline)."""
    images:    list[Image.Image]
    questions: list[str]
    answers:   list[str]


# ---------------------------------------------------------------------------
# Lance cached path
# ---------------------------------------------------------------------------

class LanceCachedLoader:
    """Reads pre-computed tier-3 columns; zero vision-tower / tokeniser work.

    Tolerates both Tier-3 layouts:

      * **flat** — ``input_ids`` / ``attention_mask`` / ``labels`` as
        their own top-level columns (``backfill_direct.py``), and
      * **struct** — a single ``sft_tokens`` struct column with those
        three fields (the ``sft_tokens`` Geneva UDF in
        ``backfill_geneva.py --tier 3``).

    ``vision_tower_hiddens`` is a flat fixed-size-list column in both.
    """

    _TOKEN_FIELDS = ("input_ids", "attention_mask", "labels")

    def __init__(
        self,
        db_path: str,
        batch_size: int = 4,
        seed: int = 0,
        infinite: bool = False,
    ) -> None:
        self.ds = lance.dataset(db_path)
        self.batch_size = batch_size
        self.seed = seed
        self.infinite = infinite
        self._n_rows = self.ds.count_rows()

        # Detect token layout once from the schema.
        names = set(self.ds.schema.names)
        self._struct_tokens = "sft_tokens" in names and not (
            set(self._TOKEN_FIELDS) <= names
        )
        if self._struct_tokens:
            self.CACHED_COLUMNS = ["vision_tower_hiddens", "sft_tokens"]
        else:
            self.CACHED_COLUMNS = ["vision_tower_hiddens", *self._TOKEN_FIELDS]
        LOG.info("cached token layout: %s",
                 "sft_tokens struct" if self._struct_tokens else "flat columns")

    def __len__(self) -> int:
        return (self._n_rows + self.batch_size - 1) // self.batch_size

    def _decode(self, table: pa.Table) -> CachedBatch:
        bsz = table.num_rows

        flat_v = table.column("vision_tower_hiddens").combine_chunks().values.to_numpy(zero_copy_only=False)
        vision = torch.from_numpy(
            flat_v.reshape(bsz, LLM_TOKENS_PER_IMAGE, VISION_HIDDEN)
        )  # fp16

        # Resolve the three token arrays from whichever layout is present.
        if self._struct_tokens:
            struct = table.column("sft_tokens").combine_chunks()
            token_arrays = {f: struct.field(f) for f in self._TOKEN_FIELDS}
        else:
            token_arrays = {f: table.column(f).combine_chunks() for f in self._TOKEN_FIELDS}

        def _to_long(arr) -> torch.Tensor:
            flat = arr.values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            return torch.from_numpy(flat.reshape(bsz, MAX_TEXT_TOKENS)).to(torch.long)

        return CachedBatch(
            vision_hiddens = vision,
            input_ids      = _to_long(token_arrays["input_ids"]),
            attention_mask = _to_long(token_arrays["attention_mask"]),
            labels         = _to_long(token_arrays["labels"]),
        )

    def __iter__(self) -> Iterator[CachedBatch]:
        epoch = 0
        while True:
            for indices in _shuffled_indices(self._n_rows, self.batch_size, self.seed + epoch):
                tbl = self.ds.take(indices, columns=self.CACHED_COLUMNS)
                yield self._decode(tbl)
            if not self.infinite:
                return
            epoch += 1


# ---------------------------------------------------------------------------
# Lance raw path (apples-to-apples vs HF/raw-FS/WDS)
# ---------------------------------------------------------------------------

class LanceRawLoader:
    """Reads raw (image_bytes, question, answer); decode + tokenise inline.

    Decode happens in the worker process via PIL.  The processor /
    vision-tower forward is the train script's responsibility — this
    loader's job is just to deliver the same input columns four ways
    can deliver them (HF datasets, raw FS, WebDataset).
    """

    RAW_COLUMNS = ["image", "question", "answer"]

    def __init__(
        self,
        db_path: str,
        batch_size: int = 4,
        seed: int = 0,
        infinite: bool = False,
        decode_images: bool = True,
    ) -> None:
        self.ds = lance.dataset(db_path)
        self.batch_size = batch_size
        self.seed = seed
        self.infinite = infinite
        self.decode_images = decode_images
        self._n_rows = self.ds.count_rows()

    def __len__(self) -> int:
        return (self._n_rows + self.batch_size - 1) // self.batch_size

    def _decode(self, table: pa.Table) -> RawBatch:
        questions = table.column("question").to_pylist()
        answers   = table.column("answer").to_pylist()
        images    = table.column("image").to_pylist()
        if self.decode_images:
            images = [Image.open(io.BytesIO(b)).convert("RGB").resize(
                (IMAGE_PX, IMAGE_PX), Image.LANCZOS) for b in images]
        return RawBatch(images=images, questions=questions, answers=answers)

    def __iter__(self) -> Iterator[RawBatch]:
        epoch = 0
        while True:
            for indices in _shuffled_indices(self._n_rows, self.batch_size, self.seed + epoch):
                tbl = self.ds.take(indices, columns=self.RAW_COLUMNS)
                yield self._decode(tbl)
            if not self.infinite:
                return
            epoch += 1
