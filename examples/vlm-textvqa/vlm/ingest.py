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
import time
from pathlib import Path

import lancedb
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


def _batch_iter(split: str, batch_size: int, limit: int | None,
                predicate=None, scan_cap: int | None = None):
    """Stream the split into Arrow RecordBatches.

    ``predicate(row)`` (raw HF row) selects a curation slice: when set,
    ``limit`` counts *kept* rows and ``scan_cap`` bounds how many source
    rows we read looking for them.
    """
    ds = load_dataset(HF_REPO, split=split, streaming=True)
    buf: list[dict] = []
    n_kept = 0
    n_seen = 0
    t0 = time.time()
    for row in ds:
        if limit is not None and n_kept >= limit:
            break
        if scan_cap is not None and n_seen >= scan_cap:
            break
        n_seen += 1
        if predicate is not None and not predicate(row):
            continue
        buf.append(_row_to_arrow(row))
        n_kept += 1
        if len(buf) >= batch_size:
            arrays = [pa.array([r[f.name] for r in buf], type=f.type)
                      for f in BASE_SCHEMA]
            yield pa.RecordBatch.from_arrays(arrays, schema=BASE_SCHEMA)
            buf = []
            if n_kept % (batch_size * 4) == 0:
                rate = n_kept / max(time.time() - t0, 1e-6)
                LOG.info("  ... %d kept / %d scanned  (%.1f rows/s)",
                         n_kept, n_seen, rate)
    if buf:
        arrays = [pa.array([r[f.name] for r in buf], type=f.type)
                  for f in BASE_SCHEMA]
        yield pa.RecordBatch.from_arrays(arrays, schema=BASE_SCHEMA)
    LOG.info("split=%s done: %d kept / %d scanned in %.1fs",
             split, n_kept, n_seen, time.time() - t0)


def _ingest_split(split: str, dst: Path, limit: int | None,
                  predicate=None, scan_cap: int | None = None,
                  single_fragment: bool = False) -> int:
    """Write the split as a LanceDB table at ``<dst.parent>/<dst.stem>``.

    ``predicate`` + ``scan_cap`` apply a curation-slice filter while
    streaming (see ``_batch_iter``); leave them ``None`` for the full split.

    ``single_fragment=True`` buffers every kept row and writes the table in
    one ``create_table`` (one Lance fragment).  The small Colab bake uses
    this so the cached table is a single compact file — fastest for the
    shuffled ``Permutation`` reads the dataloader does (one fewer thing to
    seek across).  The full-corpus path leaves this ``False`` and streams
    batch-by-batch to keep memory bounded.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    uri = str(dst.parent)
    table_name = dst.name[:-len(".lance")] if dst.name.endswith(".lance") else dst.name

    db = lancedb.connect(uri)
    if table_name in db.list_tables().tables:
        LOG.info("dropping existing table %s", table_name)
        db.drop_table(table_name)

    batches = list(_batch_iter(split, BATCH, limit, predicate, scan_cap))
    if not batches:
        raise RuntimeError(
            f"no rows ingested for split={split} (predicate too strict / "
            f"scan_cap too small?)")

    if single_fragment:
        # One create_table from the whole (small) subset -> one fragment.
        full = pa.Table.from_batches(batches, schema=BASE_SCHEMA)
        tbl = db.create_table(
            table_name, data=full, schema=BASE_SCHEMA,
            storage_options={"new_table_enable_stable_row_ids": "true"},
        )
    else:
        # Stream batch-by-batch: ``create_table`` from the first batch (so
        # the schema + stable-row-id storage option are set), then ``add``
        # the rest.  (Handing a streaming RecordBatchReader straight to
        # create_table no longer commits on recent lancedb.)
        tbl = None
        for batch in batches:
            chunk = pa.Table.from_batches([batch], schema=BASE_SCHEMA)
            if tbl is None:
                tbl = db.create_table(
                    table_name, data=chunk, schema=BASE_SCHEMA,
                    storage_options={"new_table_enable_stable_row_ids": "true"},
                )
            else:
                tbl.add(chunk)
    n = tbl.count_rows()
    LOG.info("FINAL %s/%s -> %d rows", uri, table_name, n)
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
