"""
PyTorch dataloader for BDD100K object detection using LanceDB Permutation API.

Follows the pattern from examples/ViT/mfu_bench_fp16/dataloaders.py:
  - Dataset stores connection params; each worker reopens its own Permutation
  - __getitems__ returns a pa.RecordBatch (with_format="arrow")
  - collate_fn decodes the whole batch at once

Usage
-----
from object_detection.dataloader import make_detection_loader

# Pass the right table for each split — views are already filtered
train_loader = make_detection_loader("data/bdd100k/lancedb", "bdd100k_nighttime_person_train", batch_size=32, num_workers=8, shuffle=True)
val_loader   = make_detection_loader("data/bdd100k/lancedb", "bdd100k_nighttime_person_val",   batch_size=32, num_workers=8)
"""

from __future__ import annotations

import lancedb
import pyarrow as pa
import torch
import torchvision.io as tio
from lancedb.permutation import Permutation


# BDD100K → COCO class ID mapping.
# Torchvision Faster R-CNN is pretrained on 91-class COCO; we must use the
# same IDs so baseline and fine-tuned evaluation are comparable.
#
# BDD has 10 classes; 8 map directly to COCO:
#   rider       — BDD-specific (person straddling a bike/motorcycle). No COCO
#                 equivalent, but semantically a person, so mapped to person (1).
#                 This gives the model more person training signal in rider frames.
#   traffic sign — COCO only has stop sign (13), not a generic traffic sign class.
#                 Dropped to avoid label noise.
BDD_LABEL_MAP: dict[str, int] = {
    "person":        1,   # COCO: person
    "rider":         1,   # BDD-specific → treated as person for detection purposes
    "bicycle":       2,   # COCO: bicycle
    "car":           3,   # COCO: car
    "motorcycle":    4,   # COCO: motorcycle
    "bus":           6,   # COCO: bus
    "train":         7,   # COCO: train
    "truck":         8,   # COCO: truck
    "traffic light": 10,  # COCO: traffic light
}

_DETECTION_COLS = ["image_bytes", "ann_categories", "ann_bboxes"]


def _decode_image(raw: bytes) -> torch.Tensor:
    buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return tio.decode_image(buf, tio.ImageReadMode.RGB).float().div(255.0)


def _decode_target(categories: list, bboxes: list) -> dict:
    valid_boxes, valid_labels = [], []
    for cat, box in zip(categories, bboxes):
        lid = BDD_LABEL_MAP.get(cat)
        if lid is None:
            continue
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        valid_boxes.append(box)
        valid_labels.append(lid)

    if not valid_labels:
        return {
            "boxes":  torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,),   dtype=torch.int64),
        }
    return {
        "boxes":  torch.tensor(valid_boxes, dtype=torch.float32),
        "labels": torch.tensor(valid_labels, dtype=torch.int64),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LanceDetectionDataset(torch.utils.data.Dataset):
    """
    Detection dataset backed by a LanceDB table via the Permutation API.

    Stores only connection params — each DataLoader worker reopens its own
    Permutation handle lazily, exactly like the ViT benchmark example.
    """

    def __init__(self, uri: str, table_name: str):
        self.uri        = uri
        self.table_name = table_name
        self._perm      = None

        db = lancedb.connect(uri)
        self.length = len(db.open_table(table_name))

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict:
        # Permutation holds Rust async state — zero it so each worker reopens its own
        state = self.__dict__.copy()
        state["_perm"] = None
        return state

    def _ensure_open(self) -> None:
        if self._perm is None:
            db = lancedb.connect(self.uri)
            self._perm = (
                Permutation.identity(db.open_table(self.table_name))
                .select_columns(_DETECTION_COLS)
                .with_format("arrow")
            )

    def __getitem__(self, idx: int):
        self._ensure_open()
        return self._perm[idx]

    def __getitems__(self, indices: list[int]):
        # Returns a pa.RecordBatch — collate_fn processes the whole batch at once
        self._ensure_open()
        return self._perm.__getitems__(indices)


# ---------------------------------------------------------------------------
# Collate — receives pa.RecordBatch from __getitems__
# ---------------------------------------------------------------------------

def _detection_collate(batch: pa.RecordBatch):
    images, targets = [], []
    for raw, cats, bboxes in zip(
        batch.column("image_bytes").to_pylist(),
        batch.column("ann_categories").to_pylist(),
        batch.column("ann_bboxes").to_pylist(),
    ):
        images.append(_decode_image(raw))
        targets.append(_decode_target(cats or [], bboxes or []))
    return images, targets


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_detection_loader(
    uri: str,
    table_name: str,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = False,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """
    Return a DataLoader backed by a LanceDB table via the Permutation API.

    Pass the correct table for each split — Geneva materialized views are
    already split into train/val, so no filtering is needed here.
    """
    dataset = LanceDetectionDataset(uri=uri, table_name=table_name)

    sampler = None
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(dataset, generator=g)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_detection_collate,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
