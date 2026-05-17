"""Baseline dataloaders for the 1:1 throughput bench.

Each one delivers the same ``RawBatch`` (image, question, answer) as
``LanceRawLoader`` so the bench script can swap one for another and
measure pure read-path cost without changing the workload.

Three baselines:

  * **HuggingFace datasets** — parquet under the hood, ``.shuffle().iter``.
  * **Raw filesystem** — directory of JPEGs + JSON manifest, PyTorch
    ``DataLoader`` over an ``Dataset.__getitem__``.
  * **WebDataset** — tar-shard streaming with ``wds.WebDataset(urls)``.

The three layouts are produced on-the-fly from the same Lance table by
``prepare_baseline_layouts`` so the bench can be reproduced in one shot.
"""
from __future__ import annotations

import io
import json
import logging
import os
import random
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import lance
import numpy as np
import pyarrow as pa
import torch
from PIL import Image

from .dataloader import RawBatch
from .schema import IMAGE_PX

LOG = logging.getLogger("vlm.baselines")


# ---------------------------------------------------------------------------
# layout prep (one-shot, run once before benching)
# ---------------------------------------------------------------------------

def prepare_baseline_layouts(
    db_path: str,
    out_dir: str,
    n_rows: int | None = None,
    n_wds_shards: int = 8,
) -> dict[str, str]:
    """Export the Lance table to two on-disk layouts the baselines need.

    Returns a dict ``{"raw_fs": <dir>, "wds": <dir>, "manifest": <file>}``.

    We deliberately do NOT export to parquet for the HF datasets baseline
    because ``datasets`` can load from disk via ``load_from_disk`` after
    a one-time conversion — but a more honest comparison is to use the
    same parquet format Lance encodes JPEGs into (Lance stores jpeg
    bytes natively).  We rebuild parquet shards for the HF baseline.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    ds = lance.dataset(db_path)
    if n_rows is None:
        n_rows = ds.count_rows()
    LOG.info("exporting %d rows from %s to %s", n_rows, db_path, out)

    cols = ["image", "question", "answer", "image_id", "question_id"]

    raw_dir = out / "raw_fs"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / "images").mkdir(exist_ok=True)

    wds_dir = out / "wds"
    wds_dir.mkdir(exist_ok=True)

    parquet_dir = out / "parquet"
    parquet_dir.mkdir(exist_ok=True)

    # 1) Stream and write
    manifest_lines: list[dict] = []
    wds_writers: list[tarfile.TarFile] = []
    for s in range(n_wds_shards):
        wds_writers.append(tarfile.open(wds_dir / f"shard-{s:05d}.tar", "w"))

    parquet_batches: list[pa.RecordBatch] = []

    n_done = 0
    for batch in ds.scanner(columns=cols, batch_size=128).to_batches():
        n_batch = batch.num_rows
        if n_done + n_batch > n_rows:
            batch = batch.slice(0, n_rows - n_done)
            n_batch = batch.num_rows
        images       = batch.column("image").to_pylist()
        questions    = batch.column("question").to_pylist()
        answers      = batch.column("answer").to_pylist()
        image_ids    = batch.column("image_id").to_pylist()
        question_ids = batch.column("question_id").to_pylist()

        for i in range(n_batch):
            global_i = n_done + i
            key = f"{question_ids[i]:>0}"
            img_path = raw_dir / "images" / f"{key}.jpg"
            img_path.write_bytes(images[i])
            manifest_lines.append({
                "id": global_i, "question_id": question_ids[i], "image_id": image_ids[i],
                "image": str(img_path.relative_to(raw_dir)),
                "question": questions[i], "answer": answers[i],
            })

            sh = wds_writers[global_i % n_wds_shards]
            for ext, payload in (
                (".jpg",  images[i]),
                (".txt",  questions[i].encode("utf-8")),
                (".ans",  answers[i].encode("utf-8")),
            ):
                info = tarfile.TarInfo(name=key + ext)
                info.size = len(payload)
                sh.addfile(info, io.BytesIO(payload))

        parquet_batches.append(batch)
        n_done += n_batch
        if n_done >= n_rows:
            break

    for w in wds_writers:
        w.close()

    # parquet: one file is fine at this scale (~5 GB)
    parquet_table = pa.Table.from_batches(parquet_batches)
    import pyarrow.parquet as pq
    pq.write_table(parquet_table, parquet_dir / "data.parquet")

    manifest_path = raw_dir / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for line in manifest_lines:
            f.write(json.dumps(line) + "\n")

    LOG.info("layouts ready: raw_fs=%s wds=%s parquet=%s",
             raw_dir, wds_dir, parquet_dir)

    return {
        "raw_fs":   str(raw_dir),
        "wds":      str(wds_dir),
        "parquet":  str(parquet_dir),
        "manifest": str(manifest_path),
    }


# ---------------------------------------------------------------------------
# HF datasets baseline (parquet)
# ---------------------------------------------------------------------------

class HFDatasetsLoader:
    """``datasets.load_dataset("parquet", ...).shuffle().iter(batch_size)``"""

    def __init__(
        self,
        parquet_dir: str,
        batch_size: int = 4,
        seed: int = 0,
        infinite: bool = False,
        decode_images: bool = True,
    ) -> None:
        from datasets import load_dataset
        self.ds = load_dataset(
            "parquet",
            data_files={"train": os.path.join(parquet_dir, "*.parquet")},
            split="train",
        )
        self.batch_size = batch_size
        self.seed = seed
        self.infinite = infinite
        self.decode_images = decode_images

    def __iter__(self) -> Iterator[RawBatch]:
        epoch = 0
        while True:
            shuffled = self.ds.shuffle(seed=self.seed + epoch)
            it = shuffled.iter(batch_size=self.batch_size)
            for batch in it:
                imgs = batch["image"]
                if self.decode_images:
                    imgs = [Image.open(io.BytesIO(b)).convert("RGB").resize(
                        (IMAGE_PX, IMAGE_PX), Image.LANCZOS) for b in imgs]
                yield RawBatch(images=imgs, questions=list(batch["question"]),
                               answers=list(batch["answer"]))
            if not self.infinite:
                return
            epoch += 1


# ---------------------------------------------------------------------------
# Raw filesystem baseline (PyTorch Dataset / DataLoader)
# ---------------------------------------------------------------------------

class _RawFsDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dir: str, decode_images: bool = True) -> None:
        self.raw_dir = Path(raw_dir)
        self.decode_images = decode_images
        with (self.raw_dir / "manifest.jsonl").open() as f:
            self.entries = [json.loads(l) for l in f]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        e = self.entries[i]
        path = self.raw_dir / e["image"]
        if self.decode_images:
            img = Image.open(path).convert("RGB").resize(
                (IMAGE_PX, IMAGE_PX), Image.LANCZOS)
        else:
            img = path.read_bytes()
        return img, e["question"], e["answer"]


def _rawfs_collate(items) -> RawBatch:
    imgs, qs, ans = zip(*items)
    return RawBatch(images=list(imgs), questions=list(qs), answers=list(ans))


def make_raw_fs_loader(
    raw_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 0,
    decode_images: bool = True,
):
    ds = _RawFsDataset(raw_dir, decode_images=decode_images)
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, generator=g, collate_fn=_rawfs_collate,
        pin_memory=False, persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# WebDataset baseline
# ---------------------------------------------------------------------------

class WebDatasetLoader:
    """``wds.WebDataset(urls).shuffle().decode().batched(...)``"""

    def __init__(
        self,
        wds_dir: str,
        batch_size: int = 4,
        seed: int = 0,
        shuffle_buffer: int = 1024,
        decode_images: bool = True,
    ) -> None:
        import webdataset as wds
        shards = sorted(str(p) for p in Path(wds_dir).glob("shard-*.tar"))
        if not shards:
            raise FileNotFoundError(f"no shards in {wds_dir}")

        def _decode(sample: dict) -> tuple:
            jpg = sample["jpg"]
            if decode_images:
                jpg = Image.open(io.BytesIO(jpg)).convert("RGB").resize(
                    (IMAGE_PX, IMAGE_PX), Image.LANCZOS)
            return jpg, sample["txt"].decode("utf-8"), sample["ans"].decode("utf-8")

        def _batch_collate(items) -> RawBatch:
            imgs, qs, ans = zip(*items)
            return RawBatch(images=list(imgs), questions=list(qs), answers=list(ans))

        self.pipe = (
            wds.WebDataset(shards, shardshuffle=True, seed=seed, empty_check=False)
               .shuffle(shuffle_buffer)
               .map(_decode)
               .batched(batch_size, collation_fn=_batch_collate)
        )

    def __iter__(self) -> Iterator[RawBatch]:
        return iter(self.pipe)
