"""
Evaluation utilities for Faster R-CNN on BDD100K.

Computes:
  - precision @ IoU 0.5
  - recall    @ IoU 0.5
  - mAP       @ IoU 0.5  (averaged over classes)

The evaluate() function is also called from train_detector.py so both the
baseline and fine-tuned checkpoints are scored on the same held-out slice.

Usage
-----
# Evaluate a saved checkpoint:
python -m object_detection.eval \\
    --checkpoint checkpoints/fasterrcnn_bdd_finetuned.pt \\
    --db data/bdd100k/lancedb \\
    --table bdd100k \\
    --split val

# Evaluate on the red-ambulance curated slice:
python -m object_detection.eval \\
    --checkpoint checkpoints/fasterrcnn_bdd_finetuned.pt \\
    --db data/bdd100k/lancedb \\
    --table bdd100k_red_ambulance
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from object_detection.dataloader import make_detection_loader
from object_detection.schema import NUM_CLASSES

IOU_THRESH = 0.5


# ---------------------------------------------------------------------------
# IoU & matching helpers
# ---------------------------------------------------------------------------

def _box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes [N,4] and [M,4]."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _match_predictions(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_thresh: float = IOU_THRESH,
) -> tuple[list[int], list[int], list[float]]:
    """
    Match predictions to ground-truth boxes by IoU threshold.

    Returns (tp_flags, fp_flags, scores) where each element corresponds to
    one predicted box sorted by descending score.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        tp = [0] * len(pred_scores)
        fp = [1] * len(pred_scores)
        return tp, fp, pred_scores.tolist()

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    pred_labels = pred_labels[order]
    pred_scores = pred_scores[order]

    iou = _box_iou(pred_boxes, gt_boxes)  # [P, G]
    matched_gt = torch.full((len(gt_boxes),), False, dtype=torch.bool)

    tp, fp = [], []
    for i in range(len(pred_boxes)):
        best_iou, best_j = iou[i].max(0) if len(gt_boxes) else (torch.tensor(0.0), -1)
        if (
            best_iou >= iou_thresh
            and not matched_gt[best_j]
            and pred_labels[i] == gt_labels[best_j]
        ):
            tp.append(1)
            fp.append(0)
            matched_gt[best_j] = True
        else:
            tp.append(0)
            fp.append(1)

    return tp, fp, pred_scores.tolist()


# ---------------------------------------------------------------------------
# Per-class precision / recall / AP
# ---------------------------------------------------------------------------

def _compute_ap(tp_list, fp_list, n_gt: int) -> float:
    """Compute average precision from TP/FP flags sorted by descending score."""
    if n_gt == 0:
        return 0.0

    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls = []

    for tp, fp in zip(tp_list, fp_list):
        tp_cum += tp
        fp_cum += fp
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt)

    # Trapezoidal AP
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    score_thresh: float = 0.3,
) -> dict[str, float]:
    """
    Run inference on ``loader`` and compute mAP@0.5, precision, recall.

    Returns a dict with keys: map_50, precision, recall.
    """
    model.eval()

    # Accumulate per-class TP/FP lists and GT counts
    class_tp: dict[int, list[int]] = defaultdict(list)
    class_fp: dict[int, list[int]] = defaultdict(list)
    class_scores: dict[int, list[float]] = defaultdict(list)
    class_n_gt: dict[int, int] = defaultdict(int)

    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        predictions = model(images_dev)

        for pred, tgt in zip(predictions, targets):
            gt_boxes = tgt["boxes"]
            gt_labels = tgt["labels"]

            # Filter predictions by score threshold
            keep = pred["scores"] >= score_thresh
            p_boxes = pred["boxes"][keep]
            p_labels = pred["labels"][keep]
            p_scores = pred["scores"][keep]

            # Per-class matching
            for cls_id in gt_labels.unique().tolist():
                gt_mask = gt_labels == cls_id
                p_mask = p_labels == cls_id

                tp, fp, scores = _match_predictions(
                    p_boxes[p_mask],
                    p_labels[p_mask],
                    p_scores[p_mask],
                    gt_boxes[gt_mask],
                    gt_labels[gt_mask],
                )
                class_tp[cls_id].extend(tp)
                class_fp[cls_id].extend(fp)
                class_scores[cls_id].extend(scores)
                class_n_gt[cls_id] += int(gt_mask.sum())

    # Compute per-class AP then macro-average
    ap_values = []
    total_tp = total_fp = total_gt = 0

    for cls_id in class_n_gt:
        # Sort by descending score before computing AP
        pairs = sorted(
            zip(class_scores[cls_id], class_tp[cls_id], class_fp[cls_id]),
            key=lambda x: -x[0],
        )
        if pairs:
            _, tp_sorted, fp_sorted = zip(*pairs)
        else:
            tp_sorted, fp_sorted = [], []

        ap = _compute_ap(tp_sorted, fp_sorted, class_n_gt[cls_id])
        ap_values.append(ap)
        total_tp += sum(tp_sorted)
        total_fp += sum(fp_sorted)
        total_gt += class_n_gt[cls_id]

    map_50 = sum(ap_values) / len(ap_values) if ap_values else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0

    return {"map_50": map_50, "precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a Faster R-CNN checkpoint on LanceDB data.")
    p.add_argument("--checkpoint", required=True, help="Path to saved model state dict (.pt)")
    p.add_argument("--db", default="data/bdd100k/lancedb")
    p.add_argument("--table", default="bdd100k")
    p.add_argument("--split", default=None, help="SQL filter on the 'split' column, e.g. 'val'")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--score-thresh", type=float, default=0.3)
    return p.parse_args(argv)


def main(argv=None):
    from object_detection.train_detector import build_model

    args = _parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(NUM_CLASSES, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    where = f"split = '{args.split}'" if args.split else None
    loader = make_detection_loader(
        uri=args.db,
        table_name=args.table,
        where=where,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Evaluating on {len(loader.dataset)} samples …")

    metrics = evaluate(model, loader, device, score_thresh=args.score_thresh)
    print(f"\nmAP@0.5  : {metrics['map_50']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
