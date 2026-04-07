"""
Ingest BDD100K detection data into a LanceDB table.

Supports two modes:
  1. Real BDD100K — downloaded automatically on first run if data/bdd100k/ is empty.
     --splits train val        ingest both splits
     --limit 5000              optional: cap frames per split for local dev

  2. Synthetic data — useful for verifying the pipeline without downloading BDD100K.
     --synthetic 500           generate N fake frames with random annotations

Usage
-----
# Full dataset (GPU training — downloads automatically on first run):
python -m object_detection.ingest_bdd --splits train val --overwrite

# Subset for local dev:
python -m object_detection.ingest_bdd --splits train val --limit 5000 --overwrite

# Synthetic smoke test:
python -m object_detection.ingest_bdd --synthetic 500 --overwrite
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Generator

import lancedb
import pyarrow as pa
from PIL import Image, ImageDraw

from object_detection.download_bdd import ensure_dataset
from object_detection.schema import BDD_SCHEMA

BDD_CATEGORIES = [
    "car", "truck", "bus", "person", "rider",
    "bicycle", "motorcycle", "traffic light", "traffic sign", "train",
]
WEATHER_VALUES  = ["clear", "overcast", "rainy", "snowy", "foggy", "undefined"]
SCENE_VALUES    = ["city street", "highway", "residential", "parking lot",
                   "gas stations", "tunnel", "undefined"]
TIMEOFDAY_VALUES        = ["daytime", "night", "dawn/dusk", "undefined"]
TRAFFIC_LIGHT_COLORS    = ["none", "red", "green", "yellow"]

BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def _random_jpeg(width: int = 640, height: int = 360) -> bytes:
    img = Image.new("RGB", (width, height), color=(
        random.randint(50, 200), random.randint(50, 200), random.randint(50, 200),
    ))
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(1, 4)):
        x1 = random.randint(0, width - 50)
        y1 = random.randint(0, height - 50)
        draw.rectangle(
            [x1, y1, random.randint(x1 + 20, min(x1 + 200, width)),
                      random.randint(y1 + 20, min(y1 + 150, height))],
            fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


def _synthetic_annotations(width: int, height: int):
    categories, bboxes, occluded, truncated, tl_colors = [], [], [], [], []
    for _ in range(random.randint(0, 6)):
        cat = random.choice(BDD_CATEGORIES)
        x1, y1 = random.uniform(0, width - 20), random.uniform(0, height - 20)
        categories.append(cat)
        bboxes.append([x1, y1,
                        random.uniform(x1 + 10, min(x1 + 200, width)),
                        random.uniform(y1 + 10, min(y1 + 150, height))])
        occluded.append(random.choice([True, False]))
        truncated.append(random.choice([True, False]))
        tl_colors.append(random.choice(TRAFFIC_LIGHT_COLORS) if cat == "traffic light" else "none")
    return categories, bboxes, occluded, truncated, tl_colors


def synthetic_record_batches(n: int, split: str = "val") -> Generator[pa.RecordBatch, None, None]:
    rows: list[dict] = []
    for i in range(n):
        w, h = 640, 360
        cats, bboxes, occ, trunc, tlc = _synthetic_annotations(w, h)
        rows.append({
            "image_id": f"synthetic_{split}_{i:06d}", "split": split,
            "image_bytes": _random_jpeg(w, h), "width": w, "height": h,
            "weather": random.choice(WEATHER_VALUES), "scene": random.choice(SCENE_VALUES),
            "timeofday": random.choice(TIMEOFDAY_VALUES), "timestamp": 10000 + i * 100,
            "ann_categories": cats, "ann_bboxes": bboxes, "ann_occluded": occ,
            "ann_truncated": trunc, "ann_traffic_light_colors": tlc, "num_annotations": len(cats),
        })
        if len(rows) == BATCH_SIZE:
            yield _rows_to_batch(rows)
            rows = []
    if rows:
        yield _rows_to_batch(rows)


# ---------------------------------------------------------------------------
# Real BDD100K reader
# ---------------------------------------------------------------------------

def _load_label(annotation_root: Path, split: str, image_stem: str) -> dict:
    for label_stem in (image_stem, image_stem.rsplit("-", 1)[0]):
        label_path = annotation_root / split / f"{label_stem}.json"
        if label_path.exists():
            with open(label_path) as f:
                return json.load(f)
    return {}


def _parse_labels(objects: list[dict], width: int, height: int):
    categories, bboxes, occluded, truncated, tl_colors = [], [], [], [], []
    for lbl in objects or []:
        box = lbl.get("box2d") or lbl.get("bbox")
        if box is None:
            continue
        categories.append(lbl.get("category", "unknown"))
        bboxes.append([
            float(box.get("x1", box.get("xmin", 0))),
            float(box.get("y1", box.get("ymin", 0))),
            float(box.get("x2", box.get("xmax", width))),
            float(box.get("y2", box.get("ymax", height))),
        ])
        attrs = lbl.get("attributes", {})
        occluded.append(bool(attrs.get("occluded", False)))
        truncated.append(bool(attrs.get("truncated", False)))
        tl_colors.append(str(attrs.get("trafficLightColor", "none")))
    return categories, bboxes, occluded, truncated, tl_colors


def bdd_record_batches(
    image_root: Path, annotation_root: Path, split: str, limit: int | None = None,
) -> Generator[pa.RecordBatch, None, None]:
    split_image_dir = image_root / split
    if not split_image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {split_image_dir}")

    image_paths = sorted(split_image_dir.glob("*.jpg"))
    if limit:
        image_paths = image_paths[:limit]

    rows: list[dict] = []
    for img_path in image_paths:
        image_id  = img_path.stem
        label_rec = _load_label(annotation_root, split, image_id)

        with open(img_path, "rb") as f:
            raw_bytes = f.read()
        with Image.open(io.BytesIO(raw_bytes)) as img:
            w, h = img.size

        frame     = (label_rec.get("frames") or [{}])[0]
        top_attrs = label_rec.get("attributes", {})
        cats, bboxes, occ, trunc, tlc = _parse_labels(frame.get("objects", []), w, h)

        rows.append({
            "image_id": image_id, "split": split,
            "image_bytes": raw_bytes, "width": w, "height": h,
            "weather":   top_attrs.get("weather",   "undefined"),
            "scene":     top_attrs.get("scene",     "undefined"),
            "timeofday": top_attrs.get("timeofday", "undefined"),
            "timestamp": int(frame.get("timestamp", 0)),
            "ann_categories": cats, "ann_bboxes": bboxes, "ann_occluded": occ,
            "ann_truncated": trunc, "ann_traffic_light_colors": tlc, "num_annotations": len(cats),
        })
        if len(rows) == BATCH_SIZE:
            yield _rows_to_batch(rows)
            rows = []
    if rows:
        yield _rows_to_batch(rows)


def _rows_to_batch(rows: list[dict]) -> pa.RecordBatch:
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
            arrays.append(pa.array(col, type=pa.list_(pa.list_(pa.float32()))))
        else:
            arrays.append(pa.array(col, type=field.type))
    return pa.RecordBatch.from_arrays(arrays, schema=BDD_SCHEMA)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest(
    output: str, splits: list[str], image_root: Path | None, annotation_root: Path | None,
    synthetic: int | None, limit: int | None, table_name: str, overwrite: bool,
) -> None:
    db = lancedb.connect(output)

    existing = db.list_tables().tables
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
                table_name, data=reader, schema=BDD_SCHEMA,
                storage_options={"new_table_enable_stable_row_ids": "true"},
            )
        else:
            tbl.add(reader)

        added  = len(tbl) - total
        total  = len(tbl)
        print(f"[{split}] done — {added} rows added (total: {total})")

    print(f"\nTable '{table_name}' written to {output}  ({total} rows total)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Ingest BDD100K detection data (or synthetic data) into LanceDB."
    )
    p.add_argument("--output",     default="data/bdd100k/lancedb")
    p.add_argument("--table-name", default="bdd100k")
    p.add_argument("--splits", nargs="+", default=["val"], choices=["train", "val", "test"])
    p.add_argument("--data-root",  type=Path, default=Path("data/bdd100k"),
                   help="Where to download BDD100K (default: data/bdd100k)")
    p.add_argument("--synthetic",  type=int, default=None, metavar="N",
                   help="Generate N synthetic frames instead of downloading BDD100K")
    p.add_argument("--limit",      type=int, default=None,
                   help="Cap images per split (real data only)")
    p.add_argument("--overwrite",  action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.synthetic is not None:
        image_root = annotation_root = None
    else:
        image_root, annotation_root = ensure_dataset(args.data_root)

    ingest(
        output=args.output, splits=args.splits,
        image_root=image_root, annotation_root=annotation_root,
        synthetic=args.synthetic, limit=args.limit,
        table_name=args.table_name, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
