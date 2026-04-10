"""
Visualize edge-case examples from the three BDD100K failure modes.

For each failure mode, samples N frames from the val view, runs the pretrained
COCO model and (optionally) the fine-tuned checkpoint, then saves side-by-side
comparison images: ground truth · pretrained · fine-tuned.

Usage
-----
# Ground truth + pretrained only (no fine-tuned checkpoints yet):
python -m object_detection.visualize --output-dir viz/

# Full before/after with fine-tuned checkpoints:
python -m object_detection.visualize \
    --rider-ckpt        checkpoints/rider/fasterrcnn_bdd_finetuned.pt \
    --nighttime-ckpt    checkpoints/nighttime_person/fasterrcnn_bdd_finetuned.pt \
    --distant-ckpt      checkpoints/distant_person/fasterrcnn_bdd_finetuned.pt \
    --output-dir viz/
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Optional

import lancedb
import torch
import torchvision.io as tio
from PIL import Image, ImageDraw, ImageFont

from object_detection.dataloader import BDD_LABEL_MAP
from object_detection.train_detector import build_model
from object_detection.schema import NUM_CLASSES

# Reverse map: COCO label id → name (subset used in BDD)
_COCO_NAMES: dict[int, str] = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
    6: "bus", 7: "train", 8: "truck", 10: "traffic light",
}

# Colours for each panel
_GT_COLOR        = (0,   200,  0)    # green  — ground truth
_PRETRAINED_COLOR = (220,  50, 50)   # red    — COCO pretrained
_FINETUNED_COLOR  = (50,  120, 220)  # blue   — fine-tuned

DEFAULT_DB = "data/bdd100k/lancedb"

# The three failure-mode val views and their display labels
SLICES = [
    ("bdd100k_rider_val",            "rider"),
    ("bdd100k_nighttime_person_val", "nighttime_person"),
    ("bdd100k_distant_person_val",   "distant_person"),
]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _to_pil(image_bytes: bytes) -> Image.Image:
    buf = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
    t = tio.decode_image(buf, tio.ImageReadMode.RGB)
    return Image.fromarray(t.permute(1, 2, 0).numpy())


def _draw_boxes(
    img: Image.Image,
    boxes: list[list[float]],
    labels: list[int],
    scores: Optional[list[float]],
    color: tuple[int, int, int],
    score_thresh: float = 0.3,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if scores is not None and scores[i] < score_thresh:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        name = _COCO_NAMES.get(label, str(label))
        text = f"{name} {scores[i]:.2f}" if scores is not None else name
        draw.text((x1 + 2, max(y1 - 14, 0)), text, fill=color)
    return img


def _label_panel(img: Image.Image, title: str, color: tuple[int, int, int]) -> Image.Image:
    """Add a coloured title bar above the image."""
    bar_h = 28
    bar = Image.new("RGB", (img.width, bar_h), color)
    draw = ImageDraw.Draw(bar)
    draw.text((6, 5), title, fill=(255, 255, 255))
    out = Image.new("RGB", (img.width, img.height + bar_h))
    out.paste(bar, (0, 0))
    out.paste(img, (0, bar_h))
    return out


def _hstack(images: list[Image.Image], gap: int = 8) -> Image.Image:
    h = max(im.height for im in images)
    w = sum(im.width for im in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, h), (40, 40, 40))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict(model, image_bytes: bytes, device: torch.device) -> dict:
    buf = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
    img_t = tio.decode_image(buf, tio.ImageReadMode.RGB).float().div(255.0).to(device)
    pred = model([img_t])[0]
    return {
        "boxes":  pred["boxes"].cpu().tolist(),
        "labels": pred["labels"].cpu().tolist(),
        "scores": pred["scores"].cpu().tolist(),
    }


def _load_model(checkpoint: Optional[str], device: torch.device) -> torch.nn.Module:
    model = build_model(NUM_CLASSES, pretrained=True, replace_head=False).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize(
    db_path: str,
    output_dir: Path,
    n: int,
    pool: int,
    score_thresh: float,
    rider_ckpt: Optional[str],
    nighttime_ckpt: Optional[str],
    distant_ckpt: Optional[str],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpts = {
        "rider":            rider_ckpt,
        "nighttime_person": nighttime_ckpt,
        "distant_person":   distant_ckpt,
    }

    db = lancedb.connect(db_path)
    pretrained = _load_model(None, device)

    for view_name, slice_key in SLICES:
        finetuned_ckpt = ckpts[slice_key]
        finetuned = _load_model(finetuned_ckpt, device) if finetuned_ckpt else None

        try:
            tbl = db.open_table(view_name)
        except Exception:
            print(f"  [skip] view '{view_name}' not found")
            continue

        rows = (
            tbl.search()
            .select(["image_bytes", "ann_categories", "ann_bboxes"])
            .limit(pool)
            .to_arrow()
        )
        print(f"\n[{slice_key}] {len(rows)} frames from '{view_name}' → saving {n}")

        for rank in range(min(n, len(rows))):
            i = rank
            image_bytes = rows["image_bytes"][i].as_py()
            categories  = rows["ann_categories"][i].as_py() or []
            bboxes      = rows["ann_bboxes"][i].as_py() or []

            gt_boxes, gt_labels = [], []
            for cat, box in zip(categories, bboxes):
                lid = BDD_LABEL_MAP.get(cat)
                if lid is not None:
                    gt_boxes.append(box)
                    gt_labels.append(lid)

            pre_pred = _predict(pretrained, image_bytes, device)
            ft_pred  = _predict(finetuned,  image_bytes, device) if finetuned else None
            gt_img = _to_pil(image_bytes)
            gt_img = _draw_boxes(gt_img, gt_boxes, gt_labels, None, _GT_COLOR)
            gt_img = _label_panel(gt_img, "ground truth", _GT_COLOR)

            pre_img = _to_pil(image_bytes)
            pre_img = _draw_boxes(pre_img, pre_pred["boxes"], pre_pred["labels"],
                                  pre_pred["scores"], _PRETRAINED_COLOR, score_thresh)
            pre_img = _label_panel(pre_img, "pretrained (COCO)", _PRETRAINED_COLOR)

            panels = [gt_img, pre_img]

            if ft_pred is not None:
                ft_img = _to_pil(image_bytes)
                ft_img = _draw_boxes(ft_img, ft_pred["boxes"], ft_pred["labels"],
                                     ft_pred["scores"], _FINETUNED_COLOR, score_thresh)
                ft_img = _label_panel(ft_img, "fine-tuned", _FINETUNED_COLOR)
                panels.append(ft_img)

            out_img = _hstack(panels)
            out_path = output_dir / f"{slice_key}_{rank:02d}.jpg"
            out_img.save(out_path, quality=90)
            print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Visualize edge-case examples with pretrained vs fine-tuned predictions."
    )
    p.add_argument("--db",              default=DEFAULT_DB)
    p.add_argument("--output-dir",      default="viz", type=Path)
    p.add_argument("--n",               type=int, default=5,
                   help="Number of examples to save per failure mode (default: 5)")
    p.add_argument("--pool",            type=int, default=50,
                   help="Frames to sample before ranking by improvement (default: 50)")
    p.add_argument("--score-thresh",    type=float, default=0.3)
    p.add_argument("--rider-ckpt",      default=None,
                   help="Fine-tuned checkpoint for rider slice")
    p.add_argument("--nighttime-ckpt",  default=None,
                   help="Fine-tuned checkpoint for nighttime pedestrian slice")
    p.add_argument("--distant-ckpt",    default=None,
                   help="Fine-tuned checkpoint for distant pedestrian slice")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    visualize(
        db_path=args.db,
        output_dir=args.output_dir,
        n=args.n,
        pool=args.pool,
        score_thresh=args.score_thresh,
        rider_ckpt=args.rider_ckpt,
        nighttime_ckpt=args.nighttime_ckpt,
        distant_ckpt=args.distant_ckpt,
    )


if __name__ == "__main__":
    main()
