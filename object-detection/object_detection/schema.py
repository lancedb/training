"""
Lance schema for BDD100K detection dataset.

Design principle: no nested structs — LanceDB has a known query bug with
nested struct columns.  Bboxes are stored as list<fixed_size_list<float32>[4]>
([x1, y1, x2, y2] per annotation) and other annotation fields use parallel
list columns.  Geneva UDF outputs are individual scalar columns.
"""

import pyarrow as pa

# ---------------------------------------------------------------------------
# Base table schema (written by ingest_bdd.py)
# ---------------------------------------------------------------------------

BDD_SCHEMA = pa.schema(
    [
        # --- identity ---
        pa.field("image_id", pa.string()),           # filename stem, e.g. "b1c66a42-6f7d68ca"
        pa.field("split", pa.string()),               # "train" | "val" | "test"

        # --- image ---
        pa.field("image_bytes", pa.large_binary()),  # raw JPEG bytes
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),

        # --- scene metadata (from BDD100K frame attributes) ---
        pa.field("weather", pa.string()),             # clear | overcast | rainy | snowy | …
        pa.field("scene", pa.string()),               # city street | highway | parking lot | …
        pa.field("timeofday", pa.string()),           # daytime | night | dawn/dusk | undefined
        pa.field("timestamp", pa.int64()),            # milliseconds

        # --- detection annotations (parallel lists, one element per box) ---
        pa.field("ann_categories", pa.list_(pa.string())),
        # each bbox is [x1, y1, x2, y2] in pixels
        pa.field("ann_bboxes", pa.list_(pa.list_(pa.float32()))),
        pa.field("ann_occluded", pa.list_(pa.bool_())),
        pa.field("ann_truncated", pa.list_(pa.bool_())),
        pa.field("ann_traffic_light_colors", pa.list_(pa.string())),
        pa.field("num_annotations", pa.int32()),
    ]
)

# ---------------------------------------------------------------------------
# Geneva UDF output columns (added by backfill_geneva.py)
#
# All flat scalars — no structs — so they stay directly queryable with SQL.
#
# vehicle_light_*  : lightweight SSDLite detector (CPU-friendly, for local dev)
# vehicle_*        : full Faster R-CNN detector (GPU recommended)
# white_balance    : estimated colour temperature (K)
# scene_*          : lightweight scene classifier
# ---------------------------------------------------------------------------

# Number of detection classes (10 BDD categories + 1 background)
NUM_CLASSES = 11

GENEVA_UDF_COLUMNS = [
    # lightweight detector (SSDLite)
    pa.field("vehicle_light_label", pa.string()),
    pa.field("vehicle_light_confidence", pa.float32()),
    pa.field("vehicle_light_bbox_area_pct", pa.float32()),

    # heavy detector (Faster R-CNN)
    pa.field("vehicle_label", pa.string()),
    pa.field("vehicle_confidence", pa.float32()),
    pa.field("vehicle_bbox_area_pct", pa.float32()),
    pa.field("vehicle_bbox_hsv_h", pa.float32()),
    pa.field("vehicle_bbox_hsv_s", pa.float32()),
    pa.field("vehicle_bbox_hsv_v", pa.float32()),

    # white balance
    pa.field("white_balance", pa.float32()),

    # scene context
    pa.field("scene_has_crossroad", pa.bool_()),
    pa.field("scene_has_mountain", pa.bool_()),
    pa.field("scene_description", pa.string()),

    # annotation presence flags (derived from ann_categories — no image needed)
    pa.field("has_person", pa.bool_()),
    pa.field("has_rider",  pa.bool_()),
]
