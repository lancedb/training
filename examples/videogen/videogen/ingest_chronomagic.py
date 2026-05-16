"""
Ingest ChronoMagic-Pro / ChronoMagic-ProH into ``videos_raw.lance``.

Two modes:

  1. ``--manifest PATH``
       Captions only — read a parquet manifest of (videoid, caption) pairs
       and write empty ``video_bytes``.  Useful before clips are downloaded
       (Tier-1 keyword + FTS curation runs over captions alone).

  2. ``--manifest PATH --video-dir PATH``
       Captions + clip bytes — for every row whose mp4 exists at
       ``<video-dir>/<videoid>.mp4`` the raw bytes are read in.  Captions
       whose clip is missing get an empty ``video_bytes`` cell and remain
       Tier-1 / curation-eligible.

Schema-wise: the table is created with ``data_storage_version="2.2"`` and
``new_table_enable_stable_row_ids=true`` so blob v2 takes effect and Geneva
materialised views refresh incrementally.

Usage
-----
# Captions-only ingest, useful for keyword EDA before any clip download:
python -m videogen.ingest_chronomagic \\
    --manifest data/chronomagic_proh.parquet --limit 5000 --overwrite

# Once you've downloaded clips with `videogen.download_clips`:
python -m videogen.ingest_chronomagic \\
    --manifest data/chronomagic_proh.parquet \\
    --video-dir data/clips \\
    --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Iterator

import lancedb
import pyarrow as pa
import pyarrow.parquet as pq

from videogen.schema import BASE_SCHEMA


DEFAULT_DB    = "data/videos/lancedb"
DEFAULT_TABLE = "videos_raw"
DEFAULT_BATCH = 64


# ---------------------------------------------------------------------------
# Real-corpus reader: parquet manifest (+ optional video dir).
# ---------------------------------------------------------------------------

def _probe_mp4(path: Path) -> tuple[int, int, float, int, float]:
    """Return (width, height, fps, n_frames, duration_s).  Falls back gracefully."""
    try:
        import av
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            w = stream.codec_context.width or 0
            h = stream.codec_context.height or 0
            fps = float(stream.average_rate or 0.0)
            n_frames = stream.frames or 0
            duration = float(container.duration or 0) / 1_000_000.0
            if not n_frames and fps > 0 and duration > 0:
                n_frames = int(round(fps * duration))
            return w, h, fps, n_frames, duration
    except Exception:
        return 0, 0, 0.0, 0, 0.0


def manifest_rows(
    manifest_path: Path,
    video_dir: Path | None,
    limit: int | None,
    val_every: int = 50,
    require_clips: bool = False,
) -> Iterator[dict]:
    """Stream rows from a ChronoMagic-Pro-style parquet manifest.

    The manifest only needs columns ``videoid`` and one of ``name`` /
    ``caption``.  If ``video_dir`` is set we look for
    ``video_dir/<videoid>.mp4`` and read its bytes; otherwise the
    ``video_bytes`` column is empty.

    When ``require_clips=True`` we drop rows that don't have a matching mp4
    — useful for restricting a training table to rows whose clips exist.
    """
    table = pq.read_table(manifest_path)
    cols = {c.lower(): c for c in table.column_names}
    id_col   = cols.get("videoid") or cols.get("id") or cols.get("clip_id")
    cap_col  = cols.get("caption") or cols.get("name") or cols.get("text")
    if id_col is None or cap_col is None:
        raise ValueError(
            f"Manifest must contain (videoid|id|clip_id) and (caption|name|text). "
            f"Got columns: {table.column_names}"
        )

    ids      = table.column(id_col).to_pylist()
    captions = table.column(cap_col).to_pylist()

    yielded = 0
    for i, (clip_id, caption) in enumerate(zip(ids, captions)):
        if limit is not None and yielded >= limit:
            return
        video_bytes = b""
        w = h = n_frames = 0
        fps = duration = 0.0
        if video_dir is not None:
            candidate = video_dir / f"{clip_id}.mp4"
            if candidate.exists():
                video_bytes = candidate.read_bytes()
                w, h, fps, n_frames, duration = _probe_mp4(candidate)
        if require_clips and not video_bytes:
            continue
        yield {
            "clip_id":    str(clip_id),
            "source":     "chronomagic-pro",
            "split":      "val" if yielded % val_every == 0 else "train",
            "video_bytes": video_bytes,
            "width":      w,
            "height":     h,
            "fps":        fps,
            "n_frames":   n_frames,
            "duration_s": duration,
            "caption":    str(caption) if caption is not None else "",
            "source_url": f"https://www.youtube.com/watch?v={clip_id}",
        }
        yielded += 1


# ---------------------------------------------------------------------------
# Common: stream rows → RecordBatches → Lance
# ---------------------------------------------------------------------------

def _rows_to_batch(rows: list[dict]) -> pa.RecordBatch:
    arrays = []
    for field in BASE_SCHEMA:
        col = [r[field.name] for r in rows]
        if field.type == pa.large_binary():
            arrays.append(pa.array(col, type=pa.large_binary()))
        else:
            arrays.append(pa.array(col, type=field.type))
    return pa.RecordBatch.from_arrays(arrays, schema=BASE_SCHEMA)


def _batches(rows: Iterable[dict], batch_size: int) -> Iterator[pa.RecordBatch]:
    buf: list[dict] = []
    for r in rows:
        buf.append(r)
        if len(buf) == batch_size:
            yield _rows_to_batch(buf)
            buf = []
    if buf:
        yield _rows_to_batch(buf)


def ingest(
    *,
    db_path: str,
    table_name: str,
    rows: Iterable[dict],
    overwrite: bool,
    batch_size: int = DEFAULT_BATCH,
) -> None:
    db = lancedb.connect(db_path)
    existing = set(db.list_tables().tables)
    if table_name in existing and overwrite:
        db.drop_table(table_name)
        existing = set(db.list_tables().tables)

    reader = pa.RecordBatchReader.from_batches(BASE_SCHEMA, _batches(rows, batch_size))

    if table_name in existing:
        tbl = db.open_table(table_name)
        before = len(tbl)
        tbl.add(reader)
        added = len(tbl) - before
        print(f"Appended {added:,} rows  (table now {len(tbl):,} rows)")
    else:
        tbl = db.create_table(
            table_name,
            data=reader,
            schema=BASE_SCHEMA,
            storage_options={
                "new_table_enable_stable_row_ids": "true",
                "data_storage_version": "2.2",
            },
        )
        print(f"Created table '{table_name}'  ({len(tbl):,} rows)")
    print(f"  database: {db_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Ingest ChronoMagic-Pro / -ProH manifest (+ optional clips) into Lance."
    )
    p.add_argument("--db",         default=DEFAULT_DB,    help="LanceDB database path")
    p.add_argument("--table",      default=DEFAULT_TABLE, help="Lance table name")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                   help="RecordBatch size for the streaming write")
    p.add_argument("--overwrite",  action="store_true",
                   help="Drop the table first if it exists")
    p.add_argument("--manifest",   type=Path, required=True,
                   help="Path to the parquet manifest "
                        "(videoid + caption columns).")
    p.add_argument("--video-dir",  type=Path, default=None,
                   help="Directory of downloaded mp4s (one per videoid).")
    p.add_argument("--limit",      type=int,  default=None,
                   help="Cap the number of rows ingested from the manifest.")
    p.add_argument("--require-clips", action="store_true",
                   help="Skip rows whose mp4 isn't present in --video-dir. "
                        "Useful for building a training-only table.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    rows = manifest_rows(
        args.manifest, args.video_dir, args.limit,
        require_clips=args.require_clips,
    )
    ingest(
        db_path=args.db, table_name=args.table,
        rows=rows, overwrite=args.overwrite, batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
