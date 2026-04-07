"""
Ingest BDD100K detection data into a LanceDB table.

Supports two modes:
  1. Local BDD100K files  (downloaded from https://bdd-data.berkeley.edu)
     --image-root  path/to/bdd100k/images/100k
     --annotation-root  path/to/bdd100k/labels/det_20

  2. Synthetic data  (no download needed — great for local smoke tests)
     --synthetic 1000   generates N fake frames with random annotations

Ingestion uses a RecordBatch generator so the full dataset is never loaded
into memory at once.  Each batch is written with table.add(reader=...) which
Lance handles as a streaming write.

Usage examples
--------------
# Synthetic smoke test (1k frames, CPU laptop friendly):
python -m object_detection.ingest_bdd --synthetic 1000 --output data/bdd100k/lancedb

# Real BDD100K val split (10k frames):
python -m object_detection.ingest_bdd \\
    --splits val \\
    --image-root data/bdd100k/images/100k \\
    --annotation-root data/bdd100k/labels/det_20 \\
    --output data/bdd100k/lancedb

# Sample 5k frames from train split for quick iteration:
python -m object_detection.ingest_bdd \\
    --splits train \\
    --image-root data/bdd100k/images/100k \\
    --annotation-root data/bdd100k/labels/det_20 \\
    --limit 5000 \\
    --output data/bdd100k/lancedb
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Generator

import lancedb
import pyarrow as pa
from PIL import Image, ImageDraw

from object_detection.schema import BDD_SCHEMA

# BDD100K ten-class detection taxonomy
BDD_CATEGORIES = [
    "car", "truck", "bus", "person", "rider",
    "bicycle", "motorcycle", "traffic light", "traffic sign", "train",
]

WEATHER_VALUES = ["clear", "overcast", "rainy", "snowy", "foggy", "undefined"]
SCENE_VALUES = [
    "city street", "highway", "residential", "parking lot",
    "gas stations", "tunnel", "undefined",
]
TIMEOFDAY_VALUES = ["daytime", "night", "dawn/dusk", "undefined"]
TRAFFIC_LIGHT_COLORS = ["none", "red", "green", "yellow"]

BATCH_SIZE = 512   # rows per RecordBatch written to Lance


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def _random_jpeg(width: int = 640, height: int = 360) -> bytes:
    """Return a small JPEG with random coloured rectangles (fast to generate)."""
    img = Image.new("RGB", (width, height), color=(
        random.randint(50, 200),
        random.randint(50, 200),
        random.randint(50, 200),
    ))
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(1, 4)):
        x1 = random.randint(0, width - 50)
        y1 = random.randint(0, height - 50)
        x2 = random.randint(x1 + 20, min(x1 + 200, width))
        y2 = random.randint(y1 + 20, min(y1 + 150, height))
        draw.rectangle([x1, y1, x2, y2], fill=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


def _synthetic_annotations(width: int, height: int):
    """Generate random BDD100K-style annotation lists for one frame."""
    n = random.randint(0, 6)
    categories, bboxes, occluded, truncated, tl_colors = [], [], [], [], []
    for _ in range(n):
        cat = random.choice(BDD_CATEGORIES)
        x1 = random.uniform(0, width - 20)
        y1 = random.uniform(0, height - 20)
        x2 = random.uniform(x1 + 10, min(x1 + 200, width))
        y2 = random.uniform(y1 + 10, min(y1 + 150, height))
        categories.append(cat)
        bboxes.append([x1, y1, x2, y2])
        occluded.append(random.choice([True, False]))
        truncated.append(random.choice([True, False]))
        tl_colors.append(
            random.choice(TRAFFIC_LIGHT_COLORS) if cat == "traffic light" else "none"
        )
    return categories, bboxes, occluded, truncated, tl_colors


def synthetic_record_batches(
    n: int, split: str = "val"
) -> Generator[pa.RecordBatch, None, None]:
    """Yield RecordBatches of synthetic BDD100K-style rows."""
    rows: list[dict] = []

    for i in range(n):
        w, h = 640, 360
        cats, bboxes, occ, trunc, tlc = _synthetic_annotations(w, h)
        rows.append(
            {
                "image_id": f"synthetic_{split}_{i:06d}",
                "split": split,
                "image_bytes": _random_jpeg(w, h),
                "width": w,
                "height": h,
                "weather": random.choice(WEATHER_VALUES),
                "scene": random.choice(SCENE_VALUES),
                "timeofday": random.choice(TIMEOFDAY_VALUES),
                "timestamp": 10000 + i * 100,
                "ann_categories": cats,
                "ann_bboxes": bboxes,
                "ann_occluded": occ,
                "ann_truncated": trunc,
                "ann_traffic_light_colors": tlc,
                "num_annotations": len(cats),
            }
        )

        if len(rows) == BATCH_SIZE:
            yield _rows_to_batch(rows)
            rows = []

    if rows:
        yield _rows_to_batch(rows)


# ---------------------------------------------------------------------------
# Real BDD100K reader
# ---------------------------------------------------------------------------

def _load_label(annotation_root: Path, split: str, image_stem: str) -> dict:
    """
    Load the per-frame label JSON for one image.

    Two naming conventions exist in BDD100K:
      100k:  "b1c66a42-6f7d68ca.jpg"         → label "b1c66a42-6f7d68ca.json"
      det20: "0000f77c-6257be58-0000100.jpg"  → label "0000f77c-6257be58.json"
             (det20 appends a "-NNNNNNN" frame-number suffix)

    Returns an empty dict if no label file is found.
    """
    # Try the stem as-is first (100k format), then strip the last segment (det20)
    for label_stem in (image_stem, image_stem.rsplit("-", 1)[0]):
        label_path = annotation_root / split / f"{label_stem}.json"
        if label_path.exists():
            with open(label_path) as f:
                return json.load(f)
    return {}


def _parse_labels(objects: list[dict], width: int, height: int):
    """Extract parallel annotation lists from BDD100K per-frame objects."""
    categories, bboxes, occluded, truncated, tl_colors = [], [], [], [], []
    for lbl in objects or []:
        box = lbl.get("box2d") or lbl.get("bbox")
        if box is None:
            continue
        x1 = float(box.get("x1", box.get("xmin", 0)))
        y1 = float(box.get("y1", box.get("ymin", 0)))
        x2 = float(box.get("x2", box.get("xmax", width)))
        y2 = float(box.get("y2", box.get("ymax", height)))
        attrs = lbl.get("attributes", {})
        categories.append(lbl.get("category", "unknown"))
        bboxes.append([x1, y1, x2, y2])
        occluded.append(bool(attrs.get("occluded", False)))
        truncated.append(bool(attrs.get("truncated", False)))
        tl_colors.append(str(attrs.get("trafficLightColor", "none")))
    return categories, bboxes, occluded, truncated, tl_colors


def bdd_record_batches(
    image_root: Path,
    annotation_root: Path,
    split: str,
    limit: int | None = None,
) -> Generator[pa.RecordBatch, None, None]:
    """
    Yield RecordBatches by streaming image files + per-frame label JSONs.

    image_root       : directory containing {split}/ subdirs with JPEGs
    annotation_root  : directory containing {split}/ subdirs with per-frame JSONs
                       (one JSON per image, named {video_id}-{frame_id}.json)
    """
    split_image_dir = image_root / split
    if not split_image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {split_image_dir}")

    image_paths = sorted(split_image_dir.glob("*.jpg"))
    if limit:
        image_paths = image_paths[:limit]

    rows: list[dict] = []
    for img_path in image_paths:
        image_id = img_path.stem   # e.g. "b1c66a42-6f7d68ca-0000100"
        label_rec = _load_label(annotation_root, split, image_id)

        with open(img_path, "rb") as f:
            raw_bytes = f.read()
        with Image.open(io.BytesIO(raw_bytes)) as img:
            w, h = img.size

        # Per-frame labels live under "frames[0].objects" (one timestamp per file)
        frame = (label_rec.get("frames") or [{}])[0]
        cats, bboxes, occ, trunc, tlc = _parse_labels(frame.get("objects", []), w, h)

        # Top-level attributes contain weather/scene/timeofday for the whole clip
        top_attrs = label_rec.get("attributes", {})

        rows.append({
            "image_id": image_id,
            "split": split,
            "image_bytes": raw_bytes,
            "width": w,
            "height": h,
            "weather": top_attrs.get("weather", "undefined"),
            "scene": top_attrs.get("scene", "undefined"),
            "timeofday": top_attrs.get("timeofday", "undefined"),
            "timestamp": int(frame.get("timestamp", 0)),
            "ann_categories": cats,
            "ann_bboxes": bboxes,
            "ann_occluded": occ,
            "ann_truncated": trunc,
            "ann_traffic_light_colors": tlc,
            "num_annotations": len(cats),
        })

        if len(rows) == BATCH_SIZE:
            yield _rows_to_batch(rows)
            rows = []

    if rows:
        yield _rows_to_batch(rows)


# ---------------------------------------------------------------------------
# Row-list → RecordBatch helper
# ---------------------------------------------------------------------------

def _rows_to_batch(rows: list[dict]) -> pa.RecordBatch:
    """Convert a list of row dicts to a RecordBatch matching BDD_SCHEMA."""
    arrays = []
    for field in BDD_SCHEMA:
        col = [r[field.name] for r in rows]

        if field.type == pa.large_binary():
            arrays.append(pa.array(col, type=pa.large_binary()))

        elif field.type == pa.list_(pa.string()):
            arrays.append(pa.array(col, type=pa.list_(pa.string())))

        elif field.type == pa.list_(pa.bool_()):
            arrays.append(pa.array(col, type=pa.list_(pa.bool_())))

        elif field.type == pa.list_(pa.list_(pa.float32())):
            # ann_bboxes: list of [[x1,y1,x2,y2], ...]
            arrays.append(pa.array(col, type=pa.list_(pa.list_(pa.float32()))))

        else:
            arrays.append(pa.array(col, type=field.type))

    return pa.RecordBatch.from_arrays(arrays, schema=BDD_SCHEMA)


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

def ingest(
    output: str,
    splits: list[str],
    image_root: Path | None,
    annotation_root: Path | None,
    synthetic: int | None,
    limit: int | None,
    table_name: str,
    overwrite: bool,
) -> None:
    db = lancedb.connect(output)

    # Decide whether to overwrite or append
    existing = db.list_tables().tables  # .tables extracts the list from ListTablesResponse
    if table_name in existing:
        if overwrite:
            db.drop_table(table_name)
            tbl = None
        else:
            tbl = db.open_table(table_name)
            print(f"Appending to existing table '{table_name}' ({len(tbl)} rows)")
    else:
        tbl = None

    total = 0
    for split in splits:
        print(f"[{split}] starting ingestion …")

        if synthetic:
            batches = synthetic_record_batches(synthetic, split=split)
        else:
            batches = bdd_record_batches(image_root, annotation_root, split, limit)

        reader = pa.RecordBatchReader.from_batches(BDD_SCHEMA, batches)

        if tbl is None:
            tbl = db.create_table(
                table_name,
                data=reader,
                schema=BDD_SCHEMA,
                # Stable row IDs are required for Geneva materialized view refresh
                # across table versions (i.e. after new data is appended).
                storage_options={"new_table_enable_stable_row_ids": "true"},
            )
        else:
            tbl.add(reader)

        count = len(tbl)
        added = count - total
        total = count
        print(f"[{split}] done — {added} rows added (total: {total})")

    print(f"\nTable '{table_name}' written to {output}  ({total} rows total)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Ingest BDD100K detection data (or synthetic data) into LanceDB."
    )
    p.add_argument(
        "--output", default="data/bdd100k/lancedb",
        help="LanceDB database directory (default: data/bdd100k/lancedb)",
    )
    p.add_argument(
        "--table-name", default="bdd100k",
        help="Lance table name (default: bdd100k)",
    )
    p.add_argument(
        "--splits", nargs="+", default=["val"],
        choices=["train", "val", "test"],
        help="Dataset splits to ingest (default: val)",
    )
    p.add_argument(
        "--image-root", type=Path, default=None,
        help="Root directory containing <split>/<image>.jpg files",
    )
    p.add_argument(
        "--annotation-root", type=Path, default=None,
        help="Directory containing det_<split>.json annotation files",
    )
    p.add_argument(
        "--synthetic", type=int, default=None, metavar="N",
        help="Generate N synthetic frames instead of reading real BDD100K files",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of images read per split (real data only)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Drop and recreate the table if it already exists",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.synthetic is None and (args.image_root is None or args.annotation_root is None):
        print(
            "Error: either --synthetic N or both --image-root and --annotation-root are required.",
            file=sys.stderr,
        )
        sys.exit(1)

    ingest(
        output=args.output,
        splits=args.splits,
        image_root=args.image_root,
        annotation_root=args.annotation_root,
        synthetic=args.synthetic,
        limit=args.limit,
        table_name=args.table_name,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
