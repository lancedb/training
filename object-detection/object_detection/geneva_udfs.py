"""
Geneva UDFs for BDD100K feature engineering.

Three tiers:

  Tier 1 — Annotation-derived (CPU, fast, no image decoding):
    has_person, has_rider, scene_description, scene_has_crossroad,
    scene_has_mountain, white_balance.

  Tier 2 — GPU inference (Faster R-CNN ResNet50 FPN v2):
    person_bbox_area_pct — area of the largest detected person as a percentage
    of the total frame area.  Use this to find frames where pedestrians are
    prominent (close to camera, large in frame) vs. distant background figures.

    CPU fallback (SSDLite320) available for local dev — selected by default;
    pass --gpu to backfill_geneva.py to use Faster R-CNN on a GPU cluster.

  Tier 3 — Dedup: dHash perceptual hash (GPU) + is_duplicate flag (CPU).

    Two-step process:
      1. backfill_geneva --gpu --columns dhash   → GPU dHash forward pass
      2. dedup --action index                    → IVF L2 index on dhash
      3. backfill_geneva --columns is_duplicate  → per-row vector search

    dHash encodes each frame as a 64-bit binary vector (0.0/1.0 floats).
    For binary vectors, L2² = Hamming distance, so Hamming ≤ 10 corresponds
    to L2 distance ≤ ~3.16.  is_duplicate = True when the nearest non-self
    neighbour has Hamming distance ≤ threshold (default 10).

Stateful GPU UDF pattern (Geneva docs):
  - @udf decorator goes on the CLASS
  - __init__ runs on the driver — keep it cheap (no model loading)
  - __call__ runs on the Ray worker; model is loaded lazily and then reused
    for every subsequent batch that worker processes
  - input_columns receive pa.Array batches; return a pa.Array
  - cuda=True routes tasks to GPU workers; num_gpus=1 allocates the resource
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pyarrow as pa
import torch
from PIL import Image

from geneva.transformer import udf

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# COCO person class index
_PERSON_COCO = 1


def _decode_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _bbox_area_pct(bbox: list[float], w: int, h: int) -> float:
    x1, y1, x2, y2 = bbox
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    frame_area = w * h
    return float(box_area / frame_area * 100) if frame_area > 0 else 0.0


def _largest_person_bbox(predictions, score_thresh: float = 0.3) -> list[float]:
    """Return the bbox of the highest-area person detection above threshold, or zeros."""
    boxes  = predictions["boxes"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()

    mask = (scores >= score_thresh) & (labels == _PERSON_COCO)
    if not mask.any():
        return [0.0, 0.0, 0.0, 0.0]

    person_boxes = boxes[mask]
    areas = (person_boxes[:, 2] - person_boxes[:, 0]) * (person_boxes[:, 3] - person_boxes[:, 1])
    best = int(np.argmax(areas))
    return person_boxes[best].tolist()


# ---------------------------------------------------------------------------
# Tier 2a — CPU fallback: SSDLite320 + MobileNetV3 (no GPU required)
# ---------------------------------------------------------------------------

_ssd_model: Optional[torch.nn.Module] = None


def _get_ssd_model() -> torch.nn.Module:
    global _ssd_model
    if _ssd_model is None:
        from torchvision.models.detection import (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights,
        )
        _ssd_model = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        ).eval()
    return _ssd_model


def _run_ssd_person(image_bytes: bytes, width: int, height: int) -> float:
    from torchvision.transforms.functional import to_tensor
    img = _decode_image(image_bytes)
    with torch.no_grad():
        preds = _get_ssd_model()(to_tensor(img).unsqueeze(0))[0]
    bbox = _largest_person_bbox(preds)
    return _bbox_area_pct(bbox, width, height)


@udf(data_type=pa.float32(), input_columns=["image_bytes", "width", "height"])
def _person_bbox_area_pct_cpu(image_bytes: bytes, width: int, height: int) -> float:
    """Largest detected person bbox as % of frame area (SSDLite — CPU)."""
    return _run_ssd_person(image_bytes, width, height)


# ---------------------------------------------------------------------------
# Tier 2b — GPU: Faster R-CNN ResNet50 FPN v2 (accurate, GPU recommended)
#
# Stateful class-based UDF: __init__ on driver (cheap), __call__ on worker.
# Geneva calls __call__ once per checkpoint batch (one GPU forward pass).
# ---------------------------------------------------------------------------

class _FRCNNBase:
    """Shared model lifecycle for Faster R-CNN GPU UDFs."""

    def __init__(self) -> None:
        # NOT loaded here — __init__ runs on the driver which has no GPU.
        # Model is loaded lazily in __call__ on the Ray worker.
        self.model: Optional[torch.nn.Module] = None
        self.device: Optional[torch.device] = None

    def _load(self) -> None:
        if self.model is not None:
            return
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights,
        )
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        ).eval().half().cuda()
        self.device = next(self.model.parameters()).device

    def _infer(self, image_bytes: pa.Array):

        import torchvision.io as tvio
        self._load()
        tensors = []
        for b in image_bytes:
            raw = b.as_py()
            byte_t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            # decode_jpeg with device='cuda' uses nvjpeg → decoded tensor on GPU
            img_t = tvio.decode_jpeg(byte_t, device=self.device).half() / 255.0
            tensors.append(img_t)
        with torch.no_grad():
            predictions = self.model(tensors)
        return predictions


@udf(data_type=pa.float32(), input_columns=["image_bytes", "width", "height"],
     num_gpus=1, num_cpus=1, cuda=True)
class _PersonBboxAreaPctGPU(_FRCNNBase):
    """
    Largest detected person bbox as % of frame area (Faster R-CNN — GPU).

    Returns 0.0 when no person is detected above the score threshold.
    High values (>5%) indicate pedestrians close to the camera — useful for
    curating training splits focused on prominent, close-range pedestrians.
    """

    def __call__(
        self, image_bytes: pa.Array, width: pa.Array, height: pa.Array
    ) -> pa.Array:
        predictions = self._infer(image_bytes)
        results = []
        for pred, w, h in zip(predictions, width, height):
            bbox = _largest_person_bbox(pred)
            results.append(_bbox_area_pct(bbox, w.as_py(), h.as_py()))
        # Release GPU memory immediately so concurrent workers don't accumulate
        # fragmented allocations across batches.
        torch.cuda.empty_cache()
        return pa.array(results, type=pa.float32())


# ---------------------------------------------------------------------------
# White balance  (gray-world colour temperature estimate)
# ---------------------------------------------------------------------------

@udf(data_type=pa.float32(), input_columns=["image_bytes"])
def white_balance(image_bytes: bytes) -> float:
    """
    Estimate correlated colour temperature (CCT) via gray-world assumption.

    Returns a rough CCT in Kelvin.  Warm (yellow/red) scenes → low values
    (~3000 K); cool (blue/overcast) scenes → high values (~7000 K).
    """
    img = _decode_image(image_bytes)
    arr = np.array(img, dtype=np.float32)
    r_mean = arr[:, :, 0].mean()
    b_mean = arr[:, :, 2].mean()
    rb_ratio = r_mean / (b_mean + 1e-6)
    return float(np.clip(6500.0 / (rb_ratio + 1e-6), 2500.0, 10000.0))


# ---------------------------------------------------------------------------
# Scene context
# ---------------------------------------------------------------------------

@udf(data_type=pa.bool_(), input_columns=["ann_categories"])
def has_person(ann_categories) -> bool:
    """True if any annotation in this frame is a 'person'."""
    if ann_categories is None:
        return False
    return "person" in list(ann_categories)


@udf(data_type=pa.bool_(), input_columns=["ann_categories"])
def has_rider(ann_categories) -> bool:
    """True if any annotation in this frame is a 'rider' (person on bicycle/motorcycle)."""
    if ann_categories is None:
        return False
    return "rider" in list(ann_categories)


@udf(data_type=pa.bool_(), input_columns=["scene"])
def scene_has_crossroad(scene: str) -> bool:
    """True if the BDD100K scene label suggests a crossroad / intersection."""
    return "street" in scene.lower() or "intersection" in scene.lower()


@udf(data_type=pa.bool_(), input_columns=["scene"])
def scene_has_mountain(scene: str) -> bool:
    """True if the BDD100K scene label suggests a mountain / rural road."""
    return "mountain" in scene.lower() or "rural" in scene.lower()


@udf(data_type=pa.string(), input_columns=["weather", "scene", "timeofday"])
def scene_description(weather: str, scene: str, timeofday: str) -> str:
    """Short human-readable description combining BDD100K frame attributes."""
    return f"{timeofday} {weather} scene on a {scene}"


# ---------------------------------------------------------------------------
# Tier 3 — Dedup: dHash perceptual hash (GPU) + is_duplicate flag (CPU)
#
# Two-step process:
#   1. backfill_geneva --gpu --columns dhash   → GPU dHash forward pass
#   2. dedup --action index                    → IVF L2 index on dhash
#   3. backfill_geneva --columns is_duplicate  → per-row vector search
#
# dHash encodes each frame as a 64-bit binary vector (0.0/1.0 floats).
# For binary vectors, L2² = Hamming distance — so Hamming ≤ 10 corresponds
# to L2 ≤ ~3.16.  is_duplicate = True when the nearest non-self neighbour
# has Hamming distance ≤ threshold (default 10).
# ---------------------------------------------------------------------------

_DEDUP_DB_PATH = "data/bdd100k/lancedb"
_DEDUP_HAMMING_THRESHOLD = 10


@udf(data_type=pa.list_(pa.float32(), 64), input_columns=["image_bytes"],
     num_gpus=1, num_cpus=1, cuda=True)
class _DHashGPU:
    """
    Perceptual dHash (64-bit) computed on GPU — GPU UDF.

    Pipeline per image:
      1. Decode JPEG on GPU via nvjpeg (torchvision.io.decode_jpeg device='cuda')
      2. Convert to grayscale using ITU-R BT.601 luma coefficients
      3. Resize to 9×8 via bilinear interpolation on GPU
      4. Compare each pixel to its right neighbour → 8×8 = 64 bits

    Result stored as list<float32>[64] with values in {0.0, 1.0}.
    For binary vectors, L2² = Hamming distance, making LanceDB L2 search
    a direct proxy for Hamming distance without any extra conversion.
    """

    def __init__(self) -> None:
        self.device: Optional[torch.device] = None

    def _load(self) -> None:
        if self.device is not None:
            return
        self.device = torch.device("cuda")

    def __call__(self, image_bytes: pa.Array) -> pa.Array:
        import torchvision.io as tvio
        self._load()
        results = []
        for b in image_bytes:
            byte_t = torch.frombuffer(bytearray(b.as_py()), dtype=torch.uint8)
            # Decode JPEG on GPU using nvjpeg — full resolution, no CPU copy
            img_t = tvio.decode_jpeg(byte_t, device=self.device).float()  # (3, H, W)
            # Grayscale: ITU-R BT.601 luma
            gray = 0.299 * img_t[0] + 0.587 * img_t[1] + 0.114 * img_t[2]  # (H, W)
            # Resize to 9×8 for 64-bit hash (9 cols → 8 left-right comparisons per row)
            gray = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            gray = torch.nn.functional.interpolate(gray, size=(8, 9), mode="bilinear", align_corners=False)
            gray = gray.squeeze()  # (8, 9)
            # dHash: 1 where left pixel is brighter than right neighbour
            bits = (gray[:, :-1] > gray[:, 1:]).float()  # (8, 8)
            results.append(bits.flatten().cpu().tolist())
        torch.cuda.empty_cache()
        return pa.array(results, type=pa.list_(pa.float32(), 64))


@udf(data_type=pa.bool_(), input_columns=["image_id", "dhash"])
class _IsDuplicateCPU:
    """
    Per-row duplicate flag — CPU UDF.

    True when the nearest non-self neighbour has Hamming distance ≤ threshold.
    For binary vectors (0/1), L2² = Hamming distance, so we use LanceDB's L2
    metric and compare _distance (squared L2) directly against the threshold.

    Requires an L2 vector index on bdd100k.dhash
    (run ``dedup --action index`` before backfilling this column).
    """

    def __init__(self, db_path: str = _DEDUP_DB_PATH, hamming_threshold: int = _DEDUP_HAMMING_THRESHOLD) -> None:
        self.db_path = db_path
        self.hamming_threshold = hamming_threshold
        self._tbl = None

    def __call__(self, image_id: pa.Array, dhash: pa.Array) -> pa.Array:
        if self._tbl is None:
            import lancedb as _ldb
            self._tbl = _ldb.connect(self.db_path).open_table("bdd100k")
        results = []
        for iid, h in zip(image_id.to_pylist(), dhash.to_pylist()):
            if h is None:
                results.append(None)
                continue
            result = (
                self._tbl.search(h, vector_column_name="dhash")
                .metric("l2")
                .where(f"image_id != '{iid}'")
                .limit(1)
                .to_arrow()
            )
            # _distance for L2 metric is squared L2, which equals Hamming for binary vectors
            is_dup = len(result) > 0 and result["_distance"][0].as_py() <= self.hamming_threshold
            results.append(is_dup)
        return pa.array(results, type=pa.bool_())


# ---------------------------------------------------------------------------
# Registry  — used by backfill_geneva.py
# ---------------------------------------------------------------------------

#: CPU person UDF — SSDLite320 + MobileNetV3 (no GPU required).
CPU_PERSON_UDFS: dict[str, object] = {
    "person_bbox_area_pct": _person_bbox_area_pct_cpu,
}

#: GPU person UDF — Faster R-CNN ResNet50 FPN v2 (GPU recommended).
#: Pass --gpu to backfill_geneva.py to select this variant.
GPU_PERSON_UDFS: dict[str, object] = {
    "person_bbox_area_pct": _PersonBboxAreaPctGPU(),
}

#: GPU dHash UDF — 64-bit perceptual hash. Requires --gpu flag.
GPU_DHASH_UDFS: dict[str, object] = {
    "dhash": _DHashGPU(),
}

#: Tier 1 — annotation-derived and image statistics, no detector.
METADATA_UDFS: dict[str, object] = {
    "white_balance":       white_balance,
    "scene_has_crossroad": scene_has_crossroad,
    "scene_has_mountain":  scene_has_mountain,
    "scene_description":   scene_description,
    "has_person":          has_person,
    "has_rider":           has_rider,
}

#: All UDFs keyed by column name.  Person column resolves to CPU variant
#: by default; backfill_geneva.py --gpu swaps it to GPU_PERSON_UDFS.
#: is_duplicate UDF is instantiated in backfill_geneva.py so it can receive
#: the correct db_path from the CLI arg.
ALL_UDFS: dict[str, object] = {**CPU_PERSON_UDFS, **METADATA_UDFS}
