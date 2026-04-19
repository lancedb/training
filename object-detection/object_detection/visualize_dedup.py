"""
Visualize near-duplicate frame pairs from the BDD100K table.

For each frame flagged as is_duplicate=true, finds its nearest neighbour
via vector search and saves the pair side by side so you can see what
"near-duplicate" actually looks like at the chosen threshold.

Usage
-----
python -m object_detection.visualize_dedup --output-dir viz/dedup/ --n 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lancedb
import torch
import torchvision.io as tio
from PIL import Image, ImageDraw

DEFAULT_DB    = "data/bdd100k/lancedb"
SOURCE_TABLE  = "bdd100k"
_GAP_COLOR    = (40, 40, 40)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_pil(image_bytes: bytes) -> Image.Image:
    buf = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
    t = tio.decode_image(buf, tio.ImageReadMode.RGB)
    return Image.fromarray(t.permute(1, 2, 0).numpy())


def _label(img: Image.Image, text: str, color: tuple) -> Image.Image:
    bar = Image.new("RGB", (img.width, 28), color)
    ImageDraw.Draw(bar).text((6, 5), text, fill=(255, 255, 255))
    out = Image.new("RGB", (img.width, img.height + 28))
    out.paste(bar, (0, 0))
    out.paste(img, (0, 28))
    return out


def _hstack(left: Image.Image, right: Image.Image, gap: int = 8) -> Image.Image:
    h = max(left.height, right.height)
    w = left.width + right.width + gap
    out = Image.new("RGB", (w, h), _GAP_COLOR)
    out.paste(left,  (0, 0))
    out.paste(right, (left.width + gap, 0))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize_dedup(db_path: str, output_dir: Path, n: int, pool: int, threshold: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    db  = lancedb.connect(db_path)
    tbl = db.open_table(SOURCE_TABLE)

    if "is_duplicate" not in tbl.schema.names:
        raise RuntimeError("is_duplicate column not found — run backfill_geneva --columns is_duplicate first")
    if "dhash" not in tbl.schema.names:
        raise RuntimeError("dhash column not found — run backfill_geneva --gpu --columns dhash first")

    # Sample flagged frames
    rows = (
        tbl.search()
        .where("is_duplicate = true")
        .select(["image_id", "image_bytes", "dhash"])
        .limit(pool)
        .to_arrow()
    )

    if len(rows) == 0:
        print("No duplicate frames found.")
        return

    print(f"Found {len(rows)} flagged frames (pool={pool}), saving {min(n, len(rows))} pairs …\n")

    distance_threshold = 1.0 - threshold
    saved = 0

    for i in range(len(rows)):
        if saved >= n:
            break

        iid         = rows["image_id"][i].as_py()
        image_bytes = rows["image_bytes"][i].as_py()
        h           = rows["dhash"][i].as_py()

        # Find nearest non-self neighbour via dHash L2 search
        result = (
            tbl.search(h, vector_column_name="dhash")
            .metric("l2")
            .where(f"image_id != '{iid}'")
            .select(["image_id", "image_bytes", "_distance"])
            .limit(1)
            .to_arrow()
        )

        if len(result) == 0:
            continue

        # _distance is squared L2 = Hamming distance for binary vectors
        hamming = result["_distance"][0].as_py()
        if hamming > threshold:
            continue
        neighbour_bytes = result["image_bytes"][0].as_py()
        neighbour_id    = result["image_id"][0].as_py()

        left  = _label(_to_pil(image_bytes),      f"{iid}",         (60, 60, 140))
        right = _label(_to_pil(neighbour_bytes),  f"{neighbour_id}", (60, 60, 140))

        pair = _hstack(left, right)

        # Add a thin Hamming distance banner at the bottom
        banner = Image.new("RGB", (pair.width, 24), (30, 30, 30))
        ImageDraw.Draw(banner).text(
            (8, 4),
            f"hamming distance: {int(hamming)}  (threshold: {threshold})",
            fill=(200, 200, 200),
        )
        out = Image.new("RGB", (pair.width, pair.height + 24))
        out.paste(pair, (0, 0))
        out.paste(banner, (0, pair.height))

        path = output_dir / f"pair_{saved:02d}_hamming{int(hamming)}.jpg"
        out.save(path, quality=90)
        print(f"  {path}  hamming={int(hamming)}")
        saved += 1

    print(f"\nSaved {saved} pairs to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Visualize near-duplicate frame pairs.")
    p.add_argument("--db",          default=DEFAULT_DB)
    p.add_argument("--output-dir",  default="viz/dedup", type=Path)
    p.add_argument("--n",           type=int, default=10,
                   help="Number of pairs to save (default: 10)")
    p.add_argument("--pool",        type=int, default=100,
                   help="Number of flagged frames to sample from (default: 100)")
    p.add_argument("--threshold",   type=int, default=10,
                   help="Hamming distance threshold used during backfill (default: 10)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    visualize_dedup(
        db_path=args.db,
        output_dir=args.output_dir,
        n=args.n,
        pool=args.pool,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
