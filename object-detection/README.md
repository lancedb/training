# BDD100K Rare-Class Detection with LanceDB + Geneva

How LanceDB fits into your model training **lifecycle** — targeted fine-tuning on
underrepresented classes, with curated datasets that stay fresh as fleet data grows.

---

## The Problem

A COCO-pretrained detector works well on common objects but degrades on classes that are
rare or structurally different in your deployment domain:

| Class | Why COCO fails |
|---|---|
| **Riders** (person on bike/motorcycle) | COCO labels `person` and `bicycle` separately; BDD combines them into `rider` |
| **Nighttime pedestrians** | COCO is heavily daytime-biased |
| **Nighttime riders** | Both shifts compounded |
| **Emergency vehicles** | No ambulance class in COCO; colour/shape varies across fleets |

The fix is targeted fine-tuning on curated slices. The harder question is *maintenance*:
when your AV fleet ingests new footage every day, how do you keep your training splits
current without manual curation work?

---

## Lifecycle Overview

```
                ┌─────────────────────────────────────┐
                │         Production AV Fleet          │
                │   new footage arriving continuously  │
                └────────────────┬────────────────────┘
                                 │  ingest_bdd.py (append mode)
                                 ▼
                     ┌───────────────────────┐
                     │  bdd100k (parent)      │
                     │  LanceDB table         │
                     │  80k → 85k → ... rows  │
                     └──────────┬────────────┘
                                │  backfill_geneva.py
                                │  Tier 1: has_person, has_rider   (CPU, fast)
                                │  Tier 2: vehicle_label, ...      (GPU, Faster R-CNN)
                                ▼
                     ┌───────────────────────┐
                     │  bdd100k + UDF cols    │
                     │  (enriched, queryable) │
                     └──────────┬────────────┘
                                │  manage_views.py --action curate
                                │  manage_views.py --action curate-vehicle
                                │  manage_views.py --action refresh  (after new data)
                                ▼
          ┌──────────────────────────────────────────────────────┐
          │              Geneva Materialized Views                │
          │                                                        │
          │  bdd100k_nighttime_person_train / _val                │
          │  bdd100k_rider_train / _val                           │
          │  bdd100k_nighttime_rider_train / _val                 │
          │  bdd100k_emergency_vehicle_train / _val               │
          │                                                        │
          │  Split is baked into each view — training script      │
          │  just opens the table, no WHERE clause needed.        │
          │  mv.refresh() keeps all views in sync automatically.  │
          └──────────────────────┬───────────────────────────────┘
                                 │  train_detector.py --train-table bdd100k_rider_train
                                 ▼
                     ┌───────────────────────┐
                     │  Faster R-CNN          │
                     │  fine-tuned checkpoint │
                     │  logs table version ✓  │
                     └───────────────────────┘
```

**Key insight**: the curation definition lives in Geneva, not in your training command.
When new footage is ingested and backfilled, one `manage_views.py --action refresh`
updates every split. Training scripts just open a named view — they don't know or care
when it last changed.

---

## Prerequisites

### 1. Download BDD100K

The ingestion script handles this automatically — skip ahead to **Step 1**.

### 2. Python environment

```bash
cd object-detection/
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install lancedb geneva torch torchvision pyarrow pillow
```

### 3. Geneva setup

```bash
export GENEVA_PIPELINE_STALL_TIMEOUT_S=7200  # required on all platforms, especially GPU machines
sudo chmod a+rw /tmp/.geneva_zip_setup        # fixes permission error on first run (macOS)
```

> **GPU machines**: the default stall timeout is 600 s — the GPU backfill easily exceeds
> this while loading Faster R-CNN weights on the first batch. Always export the env var
> before running any `backfill_geneva` or `manage_views` command.

---

## Full Pipeline (step by step)

All commands run from the `object-detection/` directory.

```bash
cd object-detection/
export GENEVA_PIPELINE_STALL_TIMEOUT_S=7200
```

### Step 1 — Ingest

Downloads BDD100K automatically (~6.6 GB images + ~190 MB labels) on first run.

```bash
# Full dataset (~80k frames):
python -m object_detection.ingest_bdd --splits train val --overwrite

# Subset for local dev:
python -m object_detection.ingest_bdd --splits train val --limit 5000 --overwrite

# Synthetic data — verify the pipeline works without downloading anything:
python -m object_detection.ingest_bdd --synthetic 500 --overwrite
```

> **Important**: tables are created with `new_table_enable_stable_row_ids=true`.
> This is required for `mv.refresh()` to work after data is appended.

### Step 2 — Backfill UDF columns

Two tiers, matching the two classes of features:

**Tier 1 — Annotation-based** (fast, CPU-only, no image decoding):

```bash
python -m object_detection.backfill_geneva --columns has_person has_rider
```

**Tier 2 — Model-inference-based** (requires GPU for practical runtimes):

```bash
# CPU fallback (SSDLite — slow but works locally):
python -m object_detection.backfill_geneva \
    --columns vehicle_label vehicle_confidence vehicle_bbox_area_pct

# GPU (Faster R-CNN — recommended):
python -m object_detection.backfill_geneva --gpu \
    --columns vehicle_label vehicle_confidence vehicle_bbox_area_pct
```

Both variants write the same three columns (`vehicle_label`, `vehicle_confidence`,
`vehicle_bbox_area_pct`). Backfill is incremental — re-running skips already-computed rows.
Add `--overwrite` to reset and restart from scratch.

### Step 3 — EDA

```bash
python -m object_detection.spec_queries
```

```
Spec counts — table 'bdd100k'  (80000 rows total)

  nighttime_person      6,431 rows
  rider                 4,105 rows
  nighttime_rider         851 rows
  emergency_vehicle     1,756 rows   ← red + yellow combined
  red_ambulance            11 rows
  yellow_ambulance      1,745 rows
  traffic_light         2,461 rows
  daytime_clear        14,241 rows
```

### Step 4 — Create materialized views

```bash
# Tier 1 — annotation-based (run after Tier 1 backfill):
python -m object_detection.manage_views --action curate

# Tier 2 — emergency vehicles (run after GPU backfill):
python -m object_detection.manage_views --action curate-vehicle
```

Views created and their filters:

| View | Filter |
|---|---|
| `bdd100k_nighttime_person_train` | `timeofday='night' AND has_person=true AND split='train'` |
| `bdd100k_nighttime_person_val` | `timeofday='night' AND has_person=true AND split='val'` |
| `bdd100k_rider_train` | `has_rider=true AND split='train'` |
| `bdd100k_rider_val` | `has_rider=true AND split='val'` |
| `bdd100k_nighttime_rider_train` | `timeofday='night' AND has_rider=true AND split='train'` |
| `bdd100k_nighttime_rider_val` | `timeofday='night' AND has_rider=true AND split='val'` |
| `bdd100k_emergency_vehicle_train` | `vehicle_label IN ('red_ambulance','yellow_ambulance') AND vehicle_bbox_area_pct>5.0 AND split='train'` |
| `bdd100k_emergency_vehicle_val` | `vehicle_label IN ('red_ambulance','yellow_ambulance') AND vehicle_bbox_area_pct>5.0 AND split='val'` |

For one-off custom views, `--action add` accepts any SQL filter:

```bash
python -m object_detection.manage_views --action add \
    --name bdd100k_foggy \
    --filter "weather = 'foggy' AND split = 'train'"
```

### Step 5 — Train

The split is already baked into each view — just point `--train-table` and `--val-table`
at the right pair. The script evaluates the COCO pretrained baseline first, then fine-tunes
and prints a before/after comparison.

```
=== Baseline (pretrained COCO checkpoint) ===
  map@0.5: 0.1820

=== Fine-tuning for 10 epoch(s) ===
  ...

--- Results ---
metric                baseline  fine-tuned       delta
map_50                  0.1820      0.4076      +0.2256
precision               0.3210      0.5188      +0.1978
recall                  0.4900      0.6493      +0.1593
```

**Riders** (annotation-based, runs on CPU):

```bash
python -m object_detection.train_detector \
    --train-table bdd100k_rider_train \
    --val-table bdd100k_rider_val \
    --epochs 10 --batch-size 8 --num-workers 4 \
    --output-dir checkpoints/rider
```

**Nighttime pedestrians** (annotation-based, runs on CPU):

```bash
python -m object_detection.train_detector \
    --train-table bdd100k_nighttime_person_train \
    --val-table bdd100k_nighttime_person_val \
    --epochs 10 --batch-size 8 --num-workers 4 \
    --output-dir checkpoints/nighttime_person
```

**Emergency vehicles** (requires GPU backfill + `--action curate-vehicle`):

```bash
python -m object_detection.train_detector \
    --train-table bdd100k_emergency_vehicle_train \
    --val-table bdd100k_emergency_vehicle_val \
    --epochs 10 --batch-size 8 --num-workers 4 \
    --output-dir checkpoints/emergency_vehicle
```

The same pattern works for any view. Create a view, point `--train-table` at it, done.

Training logs the **table version** for provenance:
```
Table 'bdd100k_rider_train'  version=4  rows=3284
```

Checkpoint ↔ exact data snapshot. If you retrain after a refresh, the version increments.

### Step 6 — New data arrives → refresh views

```bash
# 1. Append new footage:
python -m object_detection.ingest_bdd --synthetic 500

# 2. Incremental backfill — only newly added rows are processed:
python -m object_detection.backfill_geneva --columns has_person has_rider
python -m object_detection.backfill_geneva --gpu \
    --columns vehicle_label vehicle_confidence vehicle_bbox_area_pct

# 3. Refresh all views — one command, every split updates:
python -m object_detection.manage_views --action refresh
```

```
Parent 'bdd100k': 80500 rows (version 21)
[bdd100k_nighttime_person_train]     5145 →  5178 rows  (+33)  version 5
[bdd100k_nighttime_person_val]       1286 →  1293 rows   (+7)  version 5
[bdd100k_rider_train]                3284 →  3311 rows  (+27)  version 5
[bdd100k_rider_val]                   821 →   827 rows   (+6)  version 5
[bdd100k_emergency_vehicle_train]    1404 →  1416 rows  (+12)  version 5
[bdd100k_emergency_vehicle_val]       352 →   355 rows   (+3)  version 5
```

Retrain with the same command — same view name, new version number, more data.

### Check status any time

```bash
python -m object_detection.manage_views --action status
```

```
table                                    rows   version
----------------------------------------------------------------
  bdd100k                               80000        19  (source)
  bdd100k_nighttime_person_train         5145         4
  bdd100k_nighttime_person_val           1286         4
  bdd100k_rider_train                    3284         4
  bdd100k_rider_val                       821         4
  bdd100k_nighttime_rider_train           681         4
  bdd100k_nighttime_rider_val             170         4
  bdd100k_emergency_vehicle_train        1404         2
  bdd100k_emergency_vehicle_val           352         2
```

---

## Results

The training script always evaluates the COCO pretrained checkpoint first (baseline),
then fine-tunes, and prints a side-by-side delta. No separate baseline run needed.

### CPU baseline (1 epoch, 15k subset, Apple Silicon)

| Class | Metric | Baseline (COCO) | Fine-tuned | Δ |
|---|---|---|---|---|
| **Nighttime pedestrian** | mAP@0.5 | 0.4007 | **0.5002** | **+0.100** |
| | Precision | 0.4722 | **0.5104** | **+0.038** |
| | Recall | 0.5919 | **0.7624** | **+0.171** |
| **Rider** | mAP@0.5 | 0.5295 | **0.6370** | **+0.108** |
| | Precision | 0.5670 | 0.5435 | -0.024 |
| | Recall | 0.6828 | **0.7922** | **+0.109** |

### Emergency vehicle *(GPU required — results pending)*

Requires Tier 2 GPU backfill and `--action curate-vehicle`. Results pending.

---

## Reproducing the CPU baseline

macOS, Apple Silicon, no GPU, Python 3.13. Uses a 15k subset to keep runtimes manageable.

```bash
cd object-detection/
export GENEVA_PIPELINE_STALL_TIMEOUT_S=7200

python -m object_detection.ingest_bdd --splits train val --limit 15000 --overwrite
python -m object_detection.backfill_geneva --columns has_person has_rider
python -m object_detection.manage_views --action curate

python -m object_detection.train_detector \
    --train-table bdd100k_nighttime_person_train \
    --val-table bdd100k_nighttime_person_val \
    --epochs 1 --batch-size 4 --num-workers 0

python -m object_detection.train_detector \
    --train-table bdd100k_rider_train \
    --val-table bdd100k_rider_val \
    --epochs 1 --batch-size 4 --num-workers 0
```

> **CPU runtime**: FasterRCNN runs ~15s/image. Training 400 frames ≈ 100 min per experiment.
> Use GPU with `--num-workers 4 --batch-size 8` for practical iteration.

---

## Project Structure

```
object-detection/
├── object_detection/
│   ├── schema.py            # Lance schema + GENEVA_UDF_COLUMNS
│   ├── ingest_bdd.py        # BDD100K → LanceDB (streaming RecordBatch ingestion)
│   ├── geneva_udfs.py       # Geneva UDF functions (Tier 1 annotation + Tier 2 GPU inference)
│   ├── backfill_geneva.py   # Geneva backfill runner (incremental, checkpointed)
│   ├── manage_views.py      # create / refresh / status of materialized views  ← lifecycle
│   ├── dataloader.py        # LanceDetectionDataset + make_detection_loader (Permutation API)
│   ├── train_detector.py    # Faster R-CNN fine-tune (logs table version)
│   ├── eval.py              # mAP evaluation
│   └── spec_queries.py      # SQL filter specs + EDA helpers
├── notebooks/
│   └── eda_bdd100k.ipynb    # EDA walkthrough
├── e2e_verify.py            # integration test against real BDD100K data
└── data/bdd100k/lancedb/    # Lance tables (gitignored)
    ├── bdd100k.lance
    ├── bdd100k_nighttime_person_train.lance
    ├── bdd100k_nighttime_person_val.lance
    ├── bdd100k_rider_train.lance
    ├── bdd100k_rider_val.lance
    ├── bdd100k_nighttime_rider_train.lance
    ├── bdd100k_nighttime_rider_val.lance
    ├── bdd100k_emergency_vehicle_train.lance   # Tier 2 — requires GPU backfill
    └── bdd100k_emergency_vehicle_val.lance
```

---

## Key LanceDB Patterns

**Stable row IDs** — required for Geneva materialized view refresh across table versions:
```python
db.create_table(name, data=reader, schema=schema,
                storage_options={"new_table_enable_stable_row_ids": "true"})
```

**Streaming ingestion** — never call `table.add()` in a loop:
```python
reader = pa.RecordBatchReader.from_batches(schema, batch_generator())
db.create_table(name, data=reader, schema=schema)
# append: tbl.add(reader)
```

**Incremental Geneva backfill** — only processes NULL rows, safe to re-run:
```bash
python -m object_detection.backfill_geneva --columns has_person has_rider
```

**Materialized view refresh** — one call, all views stay current:
```python
gconn = geneva.connect("data/bdd100k/lancedb")
mv = gconn.open_table("bdd100k_rider_train")
mv.refresh()
```

**Training provenance** — `train_detector.py` logs `table.version` automatically:
```
Table 'bdd100k_rider_train'  version=4  rows=3284
```

**Flat schema only** — no nested structs (LanceDB SQL query limitation):
```python
# ✓  ann_bboxes: list<list<float32>>
# ✗  ann_bboxes: list<struct<x1, y1, x2, y2>>
```

**Geneva local runs** require:
```bash
export GENEVA_PIPELINE_STALL_TIMEOUT_S=7200
sudo chmod a+rw /tmp/.geneva_zip_setup   # macOS only
```
