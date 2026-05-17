"""Ingest the TextVQA-Lance corpus into a local working Lance table.

The source corpus ``hf://datasets/lance-format/textvqa-lance`` is stored
in Lance format, but it was written with a newer Lance version than the
pylance pinned in the videogen env (3.0.0).  Rather than upgrade the
whole stack (which would break Geneva 0.12.0's bindings), we stream the
dataset through HuggingFace ``datasets`` and re-encode it as a Lance
dataset we own.  Each fragment then has clean v2.2 storage we can add
Geneva columns to.

Usage:

    python -m vlm.ingest --dst data/textvqa.lance
"""
from __future__ import annotations

import argparse
import io
import logging
import shutil
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from datasets import load_dataset

from .schema import BASE_SCHEMA

LOG = logging.getLogger("vlm.ingest")

HF_REPO = "lance-format/textvqa-lance"
BATCH   = 256


def _pil_to_jpeg(img) -> bytes:
    """Re-encode a PIL.Image (RGB or palette) into JPEG bytes."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _row_to_arrow(row: dict) -> dict:
    return {
        "id":            int(row["id"]),
        "image":         _pil_to_jpeg(row["image"]),
        "image_id":      str(row["image_id"]),
        "question_id":   str(row["question_id"]),
        "question":      str(row["question"]),
        "answers":       [str(a) for a in (row["answers"] or [])],
        "answer":        str(row["answer"] or ""),
        "image_emb":     np.asarray(row["image_emb"],    dtype=np.float32).tolist(),
        "question_emb":  np.asarray(row["question_emb"], dtype=np.float32).tolist(),
        "ocr_tokens":    [str(t) for t in (row["ocr_tokens"] or [])],
        "image_classes": [str(c) for c in (row["image_classes"] or [])],
        "set_name":      str(row.get("set_name", "")),
    }


def _batch_iter(split: str, batch_size: int, limit: int | None):
    ds = load_dataset(HF_REPO, split=split, streaming=True)
    buf: list[dict] = []
    n_total = 0
    t0 = time.time()
    for i, row in enumerate(ds):
        if limit is not None and n_total >= limit:
            break
        buf.append(_row_to_arrow(row))
        n_total += 1
        if len(buf) >= batch_size:
            arrays = [pa.array([r[f.name] for r in buf], type=f.type)
                      for f in BASE_SCHEMA]
            yield pa.RecordBatch.from_arrays(arrays, schema=BASE_SCHEMA)
            buf = []
            if n_total % (batch_size * 4) == 0:
                rate = n_total / max(time.time() - t0, 1e-6)
                LOG.info("  ... %d rows ingested  (%.1f rows/s)", n_total, rate)
    if buf:
        arrays = [pa.array([r[f.name] for r in buf], type=f.type)
                  for f in BASE_SCHEMA]
        yield pa.RecordBatch.from_arrays(arrays, schema=BASE_SCHEMA)
    LOG.info("split=%s done: %d rows in %.1fs", split, n_total, time.time() - t0)


def _ingest_split(split: str, dst: Path, limit: int | None) -> int:
    if dst.exists():
        LOG.info("removing existing %s", dst)
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    reader = pa.RecordBatchReader.from_batches(
        BASE_SCHEMA, _batch_iter(split, BATCH, limit)
    )
    lance.write_dataset(
        reader,
        str(dst),
        schema=BASE_SCHEMA,
        mode="create",
        data_storage_version="2.2",
        enable_v2_manifest_paths=True,
        storage_options={"new_table_enable_stable_row_ids": "true"},
    )
    n = lance.dataset(str(dst)).count_rows()
    LOG.info("FINAL %s -> %d rows", dst, n)
    return n


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--dst",   default="data/textvqa.lance",
                   help="output Lance dataset directory")
    p.add_argument("--split", default="train", choices=["train", "validation"])
    p.add_argument("--limit", type=int, default=None,
                   help="cap rows (useful for smoke runs)")
    args = p.parse_args()
    _ingest_split(args.split, Path(args.dst).resolve(), args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
