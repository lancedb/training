"""
PyTorch dataloader for BDD100K object detection using LanceDB materialized views.

Each training table is a small Geneva materialized view (~700–1200 rows).
The dataset fetches all matching rows as an Arrow table once per DataLoader
worker (lazy load on first access), then serves items by index.  Calling
to_arrow() on a few hundred rows is fine — the concern is never doing it on
the full 25k-row parent table.

Torchvision detection API expects each sample to be a tuple of:
  (image_tensor, target_dict)
where target_dict has keys: boxes (FloatTensor[N,4]), labels (Int64Tensor[N]).

Usage
-----
from object_detection.dataloader import make_detection_loader

train_loader = make_detection_loader(
    uri="data/bdd100k/lancedb",
    table_name="bdd100k_rider",   # Geneva materialized view
    where="split='train'",
    batch_size=4,
    num_workers=2,
)

for images, targets in train_loader:
    # images : list[Tensor[3, H, W]]
    # targets: list[{boxes: Tensor[N,4], labels: Tensor[N]}]
    ...
"""

from __future__ import annotations

import io

import lancedb
import pyarrow as pa
import torch
from PIL import Image
from torchvision import transforms

_to_tensor = transforms.ToTensor()

# BDD100K 10-class label map (index 0 = background, matching torchvision convention)
BDD_LABEL_MAP: dict[str, int] = {
    "car": 1, "truck": 2, "bus": 3, "person": 4, "rider": 5,
    "bicycle": 6, "motorcycle": 7, "traffic light": 8, "traffic sign": 9, "train": 10,
}


def _decode_image(raw: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return _to_tensor(img)


def _decode_target(categories: list[str], bboxes: list[list[float]]) -> dict[str, torch.Tensor]:
    """Convert annotation lists from Lance into a torchvision-compatible target dict.

    FasterRCNN requires label ids >= 1 (0 is background).  Categories not in
    BDD_LABEL_MAP are dropped rather than mapped to 0, which would corrupt the
    classification loss.
    """
    valid_boxes, valid_labels = [], []
    for cat, box in zip(categories, bboxes):
        lid = BDD_LABEL_MAP.get(cat)
        if lid is None:
            continue
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:   # skip degenerate boxes
            continue
        valid_boxes.append(box)
        valid_labels.append(lid)

    if not valid_labels:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    return {
        "boxes": torch.tensor(valid_boxes, dtype=torch.float32),
        "labels": torch.tensor(valid_labels, dtype=torch.int64),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LanceArrowDetectionDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset backed by a LanceDB table (or Geneva materialized view).

    On first access, fetches all matching rows as an Arrow table and caches
    them in memory.  This is intentionally simple: our training tables are
    Geneva materialized views with a few hundred to ~1200 rows, so caching
    fits comfortably in RAM.  Do not point this at the full 25k-row parent
    table without a WHERE filter.

    Parameters
    ----------
    uri        : LanceDB database directory
    table_name : Lance table name (typically a Geneva materialized view)
    where      : optional SQL filter, e.g. "split = 'train'"
    """

    def __init__(self, uri: str, table_name: str, where: str | None = None):
        self.uri = uri
        self.table_name = table_name
        self.where = where
        self._data: pa.Table | None = None  # loaded lazily on first access

        db = lancedb.connect(uri)
        tbl = db.open_table(table_name)
        self.length = tbl.count_rows(filter=where) if where else tbl.count_rows()

    def _load(self) -> None:
        """Fetch all matching rows from LanceDB into an Arrow table."""
        db = lancedb.connect(self.uri)
        tbl = db.open_table(self.table_name)
        q = tbl.search().select(["image_bytes", "ann_categories", "ann_bboxes"])
        if self.where:
            q = q.where(self.where)
        self._data = q.limit(self.length).to_arrow()

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict:
        # Drop the cached Arrow table when pickling across DataLoader workers.
        # Each worker reloads on its first access — Arrow tables are not
        # shareable across forked processes.
        state = self.__dict__.copy()
        state["_data"] = None
        return state

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        if self._data is None:
            self._load()
        row = self._data.slice(idx, 1)
        return _parse_batch(row)[0]

    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, dict]]:
        if self._data is None:
            self._load()
        batch = self._data.take(pa.array(indices, type=pa.int64()))
        return _parse_batch(batch)


def _parse_batch(batch: pa.Table) -> list[tuple[torch.Tensor, dict]]:
    results = []
    for raw, cats, bboxes in zip(
        batch.column("image_bytes").to_pylist(),
        batch.column("ann_categories").to_pylist(),
        batch.column("ann_bboxes").to_pylist(),
    ):
        results.append((_decode_image(raw), _decode_target(cats or [], bboxes or [])))
    return results


# ---------------------------------------------------------------------------
# Collate — detection models receive a list of variable-size tensors
# ---------------------------------------------------------------------------

def _detection_collate(
    batch: list[tuple[torch.Tensor, dict]],
) -> tuple[list[torch.Tensor], list[dict]]:
    """Standard torchvision detection collate: keep as lists (variable image size)."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_detection_loader(
    uri: str,
    table_name: str,
    where: str | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = False,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """
    Return a DataLoader backed by a LanceDB table or Geneva materialized view.

    Parameters
    ----------
    uri        : LanceDB database directory
    table_name : Lance table name (typically a Geneva materialized view)
    where      : SQL filter, e.g. "split='train'"
    batch_size : samples per batch
    num_workers: DataLoader worker processes (0 = main process only)
    shuffle    : shuffle sample order each epoch
    seed       : random seed for shuffle
    """
    dataset = LanceArrowDetectionDataset(uri=uri, table_name=table_name, where=where)

    sampler = None
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(dataset, generator=g)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,          # handled by sampler above
        num_workers=num_workers,
        collate_fn=_detection_collate,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
