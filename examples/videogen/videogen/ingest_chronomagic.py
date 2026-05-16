"""
Ingest ChronoMagic-Pro (or a synthetic stand-in) into ``videos_raw.lance``.

Three modes:

  1. ``--synthetic N``
       Generate N short fake clips (single-colour MP4s, fake captions).
       No network, no GPU.  Used to smoke-test the pipeline end-to-end.

  2. ``--manifest PATH``
       Read a parquet manifest of (videoid, caption) pairs.  For ChronoMagic-Pro
       this is just the parquet that HuggingFace auto-generates from the dataset
       CSV.  Each row is consumed lazily; ``--limit N`` caps how many.

  3. ``--video-dir PATH --manifest PATH``
       Same as (2), but the actual mp4 bytes are read from ``PATH/<videoid>.mp4``
       on disk.  Use this once you've actually downloaded a subset.

Schema-wise: the table is created with ``data_storage_version="2.2"`` and
``new_table_enable_stable_row_ids=true`` so blob v2 takes effect and Geneva
materialised views refresh incrementally.

Usage
-----
# Synthetic smoke test (no network):
python -m videogen.ingest_chronomagic --synthetic 200 --overwrite

# Real subset (manifest only — captions but no clip bytes yet, for keyword EDA):
python -m videogen.ingest_chronomagic \\
    --manifest data/chronomagic_pro.parquet --limit 5000 --overwrite

# Real subset with locally-downloaded mp4s:
python -m videogen.ingest_chronomagic \\
    --manifest data/chronomagic_pro.parquet \\
    --video-dir data/videos/raw \\
    --limit 1000 --overwrite
"""

from __future__ import annotations

import argparse
import io
import random
import struct
import sys
from pathlib import Path
from typing import Iterable, Iterator

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from videogen.schema import BASE_SCHEMA


DEFAULT_DB    = "data/videos/lancedb"
DEFAULT_TABLE = "videos_raw"
DEFAULT_BATCH = 64


# ---------------------------------------------------------------------------
# Synthetic generator — no network, no GPU.  Produces actual playable MP4
# bytes (tiny, single-colour) so blob-column read paths exercise real codec
# decoding when Tier 2 GPU UDFs run.
# ---------------------------------------------------------------------------

_SYNTHETIC_PHRASES = [
    "An ice cube slowly {kw} into water on a bright kitchen counter.",
    "Time-lapse of butter {kw} in a hot pan, steam rising.",
    "Honey slowly {kw} from a wooden spoon, forming a viscous pool.",
    "Sugar crystals {kw} into hot tea, a swirl forming in the cup.",
    "A wax candle {kw} down its sides under a steady flame.",
    "A chocolate truffle {kw} under heat, glossy surface flowing.",
    "Rain droplets {kw} from the surface of a leaf at dawn.",
    "Snow on a black roof {kw} as the morning sun rises.",
    "A pile of dry ice {kw}, releasing a thick white fog.",
    "Boiling water on the stove, bubbles {kw} into vapour.",
]

_KW_FOR_TRANSITION = {
    "melting":     ["melts", "is melting", "melting away"],
    "freezing":    ["freezes", "is freezing", "starts to freeze"],
    "dissolving":  ["dissolves", "is dissolving", "is dissolving slowly"],
    "boiling":     ["boils", "is boiling", "comes to a boil"],
    "evaporating": ["evaporates", "is evaporating", "starts to evaporate"],
}


def _synthetic_caption(rng: random.Random) -> str:
    transition = rng.choice(list(_KW_FOR_TRANSITION))
    kw = rng.choice(_KW_FOR_TRANSITION[transition])
    return rng.choice(_SYNTHETIC_PHRASES).format(kw=kw)


def _synthetic_mp4(width: int, height: int, n_frames: int, fps: int,
                   rng: random.Random) -> bytes:
    """Encode a tiny solid-colour MP4 in memory using imageio[ffmpeg]."""
    import imageio.v3 as iio

    rgb = (rng.randint(20, 240), rng.randint(20, 240), rng.randint(20, 240))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = rgb
    # Add a small moving square so motion-strength UDFs don't see exactly 0.
    frames = np.stack([frame.copy() for _ in range(n_frames)])
    sq = 16
    for i in range(n_frames):
        x = (i * 4) % (width - sq)
        y = (i * 3) % (height - sq)
        frames[i, y:y + sq, x:x + sq] = 255 - np.array(rgb, dtype=np.uint8)

    buf = io.BytesIO()
    iio.imwrite(buf, frames, extension=".mp4", fps=fps, codec="libx264",
                output_params=["-pix_fmt", "yuv420p", "-crf", "30",
                               "-preset", "ultrafast", "-loglevel", "error"])
    return buf.getvalue()


def synthetic_rows(n: int, *, seed: int = 0) -> Iterator[dict]:
    """Yield ``n`` synthetic rows matching ``BASE_SCHEMA``."""
    rng = random.Random(seed)
    for i in range(n):
        w, h = 320, 192
        fps = 8
        n_frames = rng.randint(33, 49)
        try:
            mp4 = _synthetic_mp4(w, h, n_frames, fps, rng)
        except Exception:
            # If ffmpeg isn't available we still want the pipeline to ingest;
            # fall back to a 4-byte sentinel so Tier 1 (CPU) UDFs still work.
            mp4 = struct.pack(">I", i)
        yield {
            "clip_id":    f"synthetic_{i:07d}",
            "source":     "synthetic",
            "split":      "val" if i % 10 == 0 else "train",
            "video_bytes": mp4,
            "width":      w,
            "height":     h,
            "fps":        float(fps),
            "n_frames":   n_frames,
            "duration_s": n_frames / float(fps),
            "caption":    _synthetic_caption(rng),
            "source_url": "",
        }


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
) -> Iterator[dict]:
    """Stream rows from a ChronoMagic-Pro-style parquet manifest.

    The manifest only needs columns ``videoid`` and one of ``name`` / ``caption``.
    If ``video_dir`` is set we look for ``video_dir/<videoid>.mp4`` and read its
    bytes; otherwise the ``video_bytes`` column is empty (manifest-only ingest,
    useful for Tier-1 caption-driven curation before any clips are downloaded).
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

    for i, (clip_id, caption) in enumerate(zip(ids, captions)):
        if limit is not None and i >= limit:
            return
        video_bytes = b""
        w = h = n_frames = 0
        fps = duration = 0.0
        if video_dir is not None:
            candidate = video_dir / f"{clip_id}.mp4"
            if candidate.exists():
                video_bytes = candidate.read_bytes()
                w, h, fps, n_frames, duration = _probe_mp4(candidate)
        yield {
            "clip_id":    str(clip_id),
            "source":     "chronomagic-pro",
            "split":      "val" if i % val_every == 0 else "train",
            "video_bytes": video_bytes,
            "width":      w,
            "height":     h,
            "fps":        fps,
            "n_frames":   n_frames,
            "duration_s": duration,
            "caption":    str(caption) if caption is not None else "",
            "source_url": f"https://www.youtube.com/watch?v={clip_id}",
        }


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
        description="Ingest video-text pairs into a Lance table for the videogen pipeline."
    )
    p.add_argument("--db",         default=DEFAULT_DB,    help="LanceDB database path")
    p.add_argument("--table",      default=DEFAULT_TABLE, help="Lance table name")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                   help="RecordBatch size for the streaming write")
    p.add_argument("--overwrite",  action="store_true",
                   help="Drop the table first if it exists")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", type=int, metavar="N",
                     help="Generate N synthetic clips (no network, no GPU).")
    src.add_argument("--manifest",  type=Path,
                     help="Path to a parquet manifest with (videoid, name|caption) columns.")

    p.add_argument("--video-dir",  type=Path, default=None,
                   help="Directory of downloaded mp4s (used only with --manifest).")
    p.add_argument("--limit",      type=int,  default=None,
                   help="Cap the number of rows ingested from --manifest.")
    p.add_argument("--seed",       type=int,  default=0,
                   help="Seed for the synthetic generator.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.synthetic is not None:
        rows: Iterable[dict] = synthetic_rows(args.synthetic, seed=args.seed)
    else:
        rows = manifest_rows(args.manifest, args.video_dir, args.limit)

    ingest(
        db_path=args.db,
        table_name=args.table,
        rows=rows,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
