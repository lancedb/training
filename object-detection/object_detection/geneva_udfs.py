"""
Geneva UDFs for BDD100K emergency-vehicle feature engineering.

Each UDF adds one flat scalar column to the Lance table so it stays directly
queryable without hitting LanceDB's nested-struct query bug.

Two detector variants are provided:
  - vehicle_light_*  : SSDLite320 + MobileNetV3 (fast, CPU-only, for local dev)
  - vehicle_*        : Faster R-CNN ResNet50 FPN v2 (accurate, GPU recommended)

The lightweight variant is what you run locally when iterating on the pipeline.
Swap to the heavy variant on a GPU cluster for the final backfill.

UDF colour-enrichment logic
---------------------------
Both detectors operate on COCO classes (car, truck, bus, etc. — no "ambulance").
We enrich the label by inspecting HSV statistics inside the top bounding box:
  - (car | truck | bus) + dominant hue in red range  → "red_ambulance"
  - (car | truck | bus) + dominant hue in yellow range → "yellow_ambulance"
  - "traffic light" stays as "traffic_light"
  - everything else keeps its COCO label

This is intentionally heuristic — the goal is to surface candidate frames for
the Geneva-filtered training split, not to be a ground-truth detector.
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

# COCO class index → label string (91-class COCO, torchvision convention)
_COCO_LABELS: dict[int, str] = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
    5: "airplane", 6: "bus", 7: "train", 8: "truck",
    9: "boat", 10: "traffic light", 11: "fire hydrant",
    13: "stop sign", 14: "parking meter", 15: "bench",
    # … only vehicle-relevant ones matter for us
}
_VEHICLE_COCO = {3, 6, 8}   # car, bus, truck
_TRAFFIC_LIGHT_COCO = {10}


def _decode_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _dominant_hsv(img: Image.Image, bbox: list[float]) -> tuple[float, float, float]:
    """Return mean HSV of the pixels inside `bbox` = [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    crop = img.crop((x1, y1, x2, y2))
    if crop.width < 2 or crop.height < 2:
        return 0.0, 0.0, 0.0
    hsv = crop.convert("HSV")
    arr = np.array(hsv, dtype=np.float32)
    return float(arr[:, :, 0].mean()), float(arr[:, :, 1].mean()), float(arr[:, :, 2].mean())


def _enrich_label(coco_idx: int, h: float, s: float, v: float) -> str:
    """Map a COCO detection + HSV stats to an enriched emergency-vehicle label."""
    if coco_idx in _TRAFFIC_LIGHT_COCO:
        return "traffic_light"

    if coco_idx in _VEHICLE_COCO and s > 60:
        # OpenCV-style HSV: hue in [0, 180] when stored as uint8
        # Pillow HSV: hue in [0, 255]
        hue_norm = h / 255.0  # normalise to [0, 1]
        if hue_norm < 0.05 or hue_norm > 0.95:   # red wraps at 0/1
            return "red_ambulance"
        if 0.10 < hue_norm < 0.17:               # yellow ~25-45° of 360°
            return "yellow_ambulance"

    return _COCO_LABELS.get(coco_idx, "other")


def _bbox_area_pct(bbox: list[float], w: int, h: int) -> float:
    x1, y1, x2, y2 = bbox
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    frame_area = w * h
    return float(box_area / frame_area * 100) if frame_area > 0 else 0.0


def _top_detection(predictions, score_thresh: float = 0.3):
    """Return (label_idx, score, bbox) for the highest-scoring detection above threshold."""
    boxes = predictions["boxes"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()

    mask = scores >= score_thresh
    if not mask.any():
        return None, 0.0, [0.0, 0.0, 0.0, 0.0]

    best = int(np.argmax(scores[mask]))
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]
    bbox = filtered_boxes[best].tolist()
    return int(filtered_labels[best]), float(filtered_scores[best]), bbox


# ---------------------------------------------------------------------------
# Lightweight detector  (SSDLite320, CPU-friendly)
# ---------------------------------------------------------------------------

_ssd_model: Optional[torch.nn.Module] = None


def _get_ssd_model():
    global _ssd_model
    if _ssd_model is None:
        from torchvision.models.detection import (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights,
        )
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        _ssd_model = ssdlite320_mobilenet_v3_large(weights=weights)
        _ssd_model.eval()
    return _ssd_model


def _run_ssd(image_bytes: bytes):
    """Decode image, run SSDLite, return (img, label_idx, score, bbox)."""
    from torchvision.transforms.functional import to_tensor
    img = _decode_image(image_bytes)
    tensor = to_tensor(img).unsqueeze(0)
    with torch.no_grad():
        preds = _get_ssd_model()(tensor)[0]
    label_idx, score, bbox = _top_detection(preds)
    return img, label_idx, score, bbox


@udf(data_type=pa.string(), input_columns=["image_bytes"], batch_size=32)
def vehicle_light_label(image_bytes: bytes) -> str:
    """Enriched label from the lightweight SSDLite detector."""
    img, label_idx, _, bbox = _run_ssd(image_bytes)
    if label_idx is None:
        return "no_detection"
    h, s, v = _dominant_hsv(img, bbox)
    return _enrich_label(label_idx, h, s, v)


@udf(data_type=pa.float32(), input_columns=["image_bytes"], batch_size=32)
def vehicle_light_confidence(image_bytes: bytes) -> float:
    """Detection confidence from the lightweight SSDLite detector."""
    _, _, score, _ = _run_ssd(image_bytes)
    return score


@udf(data_type=pa.float32(), input_columns=["image_bytes", "width", "height"], batch_size=32)
def vehicle_light_bbox_area_pct(image_bytes: bytes, width: int, height: int) -> float:
    """Bounding-box area of the top SSDLite detection as a % of frame area."""
    _, _, _, bbox = _run_ssd(image_bytes)
    return _bbox_area_pct(bbox, width, height)


# ---------------------------------------------------------------------------
# Heavy detector  (Faster R-CNN ResNet50 FPN v2, GPU recommended)
# ---------------------------------------------------------------------------

_frcnn_model: Optional[torch.nn.Module] = None


def _get_frcnn_model():
    global _frcnn_model
    if _frcnn_model is None:
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights,
        )
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        _frcnn_model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        _frcnn_model.eval()
        if torch.cuda.is_available():
            _frcnn_model = _frcnn_model.cuda()
    return _frcnn_model


def _run_frcnn(image_bytes: bytes):
    img = _decode_image(image_bytes)
    model = _get_frcnn_model()
    from torchvision.transforms.functional import to_tensor
    tensor = to_tensor(img)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        preds = model([tensor])[0]
    label_idx, score, bbox = _top_detection(preds)
    return img, label_idx, score, bbox


@udf(data_type=pa.string(), input_columns=["image_bytes"], num_gpus=0.25, batch_size=32)
def vehicle_label(image_bytes: bytes) -> str:
    """Enriched label from the heavy Faster R-CNN detector (GPU recommended).

    Used to filter the ambulance materialized view:
      vehicle_label = 'red_ambulance'
    """
    img, label_idx, _, bbox = _run_frcnn(image_bytes)
    if label_idx is None:
        return "no_detection"
    h, s, v = _dominant_hsv(img, bbox)
    return _enrich_label(label_idx, h, s, v)


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
    g_mean = arr[:, :, 1].mean()
    b_mean = arr[:, :, 2].mean()

    eps = 1e-6
    rb_ratio = r_mean / (b_mean + eps)

    # Simple heuristic mapping rb_ratio → CCT:
    # rb_ratio > 1  (more red than blue)  → warm  ~3000 K
    # rb_ratio < 1  (more blue than red)  → cool  ~7000 K
    cct = float(np.clip(6500.0 / (rb_ratio + eps), 2500.0, 10000.0))
    return cct


# ---------------------------------------------------------------------------
# Scene context  (lightweight keyword heuristic — swap for a real classifier
#                 on GPU cluster)
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
# Registry  — used by backfill_geneva.py
# ---------------------------------------------------------------------------

#: Lightweight UDFs — SSDLite-based, use GPU when available, fall back to CPU.
LIGHT_UDFS: dict[str, object] = {
    "vehicle_light_label": vehicle_light_label,
    "vehicle_light_confidence": vehicle_light_confidence,
    "vehicle_light_bbox_area_pct": vehicle_light_bbox_area_pct,
    "white_balance": white_balance,
    "scene_has_crossroad": scene_has_crossroad,
    "scene_has_mountain": scene_has_mountain,
    "scene_description": scene_description,
    "has_person": has_person,
    "has_rider":  has_rider,
}

#: Heavy UDFs that need a GPU for reasonable throughput.
HEAVY_UDFS: dict[str, object] = {
    "vehicle_label": vehicle_label,
}

ALL_UDFS: dict[str, object] = {**LIGHT_UDFS, **HEAVY_UDFS}
