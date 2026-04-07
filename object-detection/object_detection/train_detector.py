"""
Fine-tune Faster R-CNN (ResNet50 FPN v2) on a Lance-curated BDD100K split.

The script compares two runs:
  baseline  — frozen COCO pretrained weights, evaluated on the Geneva-filtered
              validation slice (no training)
  fine-tune — one full training pass on the Geneva-curated train split, then
              evaluated on the same filtered val slice

Both runs use the same Lance dataloader, demonstrating that LanceDB removes
the data pipeline bottleneck: the curated training subset is a WHERE filter,
not a separate export step.

Usage
-----
# Evaluate the pretrained COCO checkpoint on the val split (no training):
python -m object_detection.train_detector --mode baseline \\
    --db data/bdd100k/lancedb --val-table bdd100k

# Fine-tune on a curated red-ambulance subset and evaluate:
python -m object_detection.train_detector --mode finetune \\
    --db data/bdd100k/lancedb \\
    --train-table bdd100k_red_ambulance \\
    --val-table bdd100k \\
    --val-where "vehicle_label = 'red_ambulance' AND vehicle_bbox_area_pct > 5.0" \\
    --epochs 5 \\
    --batch-size 4 \\
    --lr 0.005

# Quick smoke test with a tiny batch:
python -m object_detection.train_detector --mode finetune \\
    --db data/bdd100k/lancedb --train-table bdd100k \\
    --train-where "split = 'train'" \\
    --val-where "split = 'val'" \\
    --limit-train 200 --limit-val 50 \\
    --epochs 1 --batch-size 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lancedb
import torch
import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from object_detection.dataloader import make_detection_loader
from object_detection.eval import evaluate
from object_detection.schema import NUM_CLASSES


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(num_classes: int, pretrained: bool = True, replace_head: bool = True) -> torch.nn.Module:
    """
    Return Faster R-CNN ResNet50 FPN v2.

    pretrained=True  loads COCO weights.
    replace_head=True  swaps the 91-class COCO head for a fresh num_classes head
                       (required for fine-tuning on a custom class set).
    replace_head=False keeps the original COCO head intact — use this for
                       baseline evaluation so pretrained weights are not destroyed.
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    if replace_head:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, loader, device, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Skip empty batches (frames with no annotations)
        if all(t["labels"].numel() == 0 for t in targets):
            continue

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += losses.item()
        n_batches += 1

        if batch_idx % 20 == 0:
            print(
                f"  epoch {epoch}  batch {batch_idx:4d}  "
                f"loss {losses.item():.4f}"
            )

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Log the exact table version used — full provenance for checkpoint reproducibility
    db = lancedb.connect(args.db)
    source_table = args.train_table if args.mode == "finetune" else args.val_table
    tbl = db.open_table(source_table)
    print(f"Table '{tbl.name}'  version={tbl.version}  rows={len(tbl)}")

    # --- validation loader (always needed) ---
    val_loader = make_detection_loader(
        uri=args.db,
        table_name=args.val_table,
        where=args.val_where,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_desc = f"'{args.val_table}'" + (f" WHERE {args.val_where}" if args.val_where else "")
    print(f"Val loader: {len(val_loader.dataset)} samples from {val_desc}")

    # Baseline uses the intact COCO head (replace_head=False) so pretrained
    # weights are not destroyed.  Fine-tune swaps in a fresh head afterwards.
    model = build_model(NUM_CLASSES, pretrained=True, replace_head=False).to(device)

    # --- baseline: just evaluate the COCO pretrained checkpoint ---
    print("\n=== Baseline (pretrained COCO checkpoint) ===")
    t0 = time.time()
    baseline_metrics = evaluate(model, val_loader, device)
    print(f"  map@0.5: {baseline_metrics['map_50']:.4f}  "
          f"({time.time() - t0:.1f}s)")

    if args.mode == "baseline":
        return

    # Replace head for fine-tuning on BDD class IDs
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # --- fine-tune ---
    train_loader = make_detection_loader(
        uri=args.db,
        table_name=args.train_table,
        where=args.train_where,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    train_desc = f"'{args.train_table}'" + (f" WHERE {args.train_where}" if args.train_where else "")
    print(f"\nTrain loader: {len(train_loader.dataset)} samples from {train_desc}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print(f"\n=== Fine-tuning for {args.epochs} epoch(s) ===")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()
        print(f"  epoch {epoch}  avg_loss {avg_loss:.4f}  ({time.time() - t0:.1f}s)")

    print("\n=== Fine-tuned checkpoint ===")
    finetuned_metrics = evaluate(model, val_loader, device)
    print(f"  map@0.5: {finetuned_metrics['map_50']:.4f}")

    # --- summary table ---
    print("\n--- Results ---")
    print(f"{'metric':<20}  {'baseline':>10}  {'fine-tuned':>10}  {'delta':>10}")
    print("-" * 55)
    for k in ["map_50", "precision", "recall"]:
        b = baseline_metrics.get(k, 0.0)
        f = finetuned_metrics.get(k, 0.0)
        print(f"  {k:<18}  {b:>10.4f}  {f:>10.4f}  {f - b:>+10.4f}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "fasterrcnn_bdd_finetuned.pt")
        print(f"\nCheckpoint saved to {out}/fasterrcnn_bdd_finetuned.pt")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train/evaluate Faster R-CNN on LanceDB BDD100K.")
    p.add_argument("--mode", choices=["baseline", "finetune"], default="finetune")
    p.add_argument("--db", default="data/bdd100k/lancedb")
    p.add_argument("--train-table", default="bdd100k")
    p.add_argument("--val-table", default="bdd100k")
    p.add_argument("--train-where", default=None,
                   help="SQL filter on the training table, e.g. \"split='train' AND vehicle_label='red_ambulance'\"")
    p.add_argument("--val-where", default=None,
                   help="SQL filter on the validation table, e.g. \"split='val'\"")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
