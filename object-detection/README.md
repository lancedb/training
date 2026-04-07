# BDD100K Rare-Class Detection with LanceDB + Geneva

How LanceDB fits into your model training **lifecycle** — not just getting to training fast,
but keeping your curated datasets fresh as fleet data grows.

---

## The Problem

COCO-pretrained models degrade on domain shifts underrepresented in COCO:

- **Nighttime pedestrians** — COCO is heavily daytime-biased
- **Riders** (person on bicycle/motorcycle) — COCO separates `person` + `bicycle`; BDD labels the combined `rider` class
- **Nighttime riders** — both shifts compounded

The fix is targeted fine-tuning on those exact slices. The harder question is *maintenance*:
when your AV fleet ingests new footage every day, how do you keep your training splits current
without manual curation work?

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
                         │  25k → 30k → ... rows  │
                         └──────────┬────────────┘
                                    │  backfill_geneva.py
                                    │  Geneva UDFs: has_person,
                                    │  has_rider, vehicle_*
                                    ▼
                         ┌───────────────────────┐
                         │  bdd100k + UDF columns │
                         │  (enriched, queryable) │
                         └──────────┬────────────┘
                                    │  manage_views.py --action curate  (run once)
                                    │  manage_views.py --action refresh (after new data)
                                    ▼
              ┌─────────────────────────────────────────────────┐
              │           Geneva Materialized Views              │
              │  bdd100k_nighttime_person_train / _val           │
              │  bdd100k_rider_train / _val                      │
              │  bdd100k_nighttime_rider_train / _val            │
              │                                                   │
              │  mv.refresh() keeps them in sync automatically   │
              └──────────────────────┬──────────────────────────┘
                                     │  train_detector.py --train-table bdd100k_rider_train
                                     ▼
                         ┌───────────────────────┐
                         │  Faster R-CNN          │
                         │  fine-tuned checkpoint │
                         │  logs table version ✓  │
                         └───────────────────────┘
```

The key insight: **the curation definition lives in Geneva, not in your training command.**
When new footage is ingested and backfilled, `manage_views.py --action refresh` updates all
curated splits. Your training script just reads from `bdd100k_nighttime_rider` — it doesn't
know or care when that view last changed.

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
export GENEVA_PIPELINE_STALL_TIMEOUT_S=7200  # prevents timeout on slow runs — required on GPU machines too
sudo chmod a+rw /tmp/.geneva_zip_setup        # fixes permission error on first run (macOS)
```

> **GPU machines**: export `GENEVA_PIPELINE_STALL_TIMEOUT_S=7200` before the GPU backfill step.
> The default is 600 s, which times out before Faster R-CNN finishes loading weights and
> processing its first batch.

---

## Full Pipeline (step by step)

All commands run from the `object-detection/` directory.

```bash
cd object-detection/
export GENEVA_PIPELINE_STALL_TIMEOUT_S=7200
```

`--output` defaults to `data/bdd100k/lancedb` and `--table-name` defaults to `bdd100k`,
so those flags are omitted below.

### Step 1 — Ingest

Downloads BDD100K automatically (~6.6 GB images + ~190 MB labels) on first run if `data/bdd100k/` is empty.

```bash
# Full dataset (GPU training — ~80k frames):
python -m object_detection.ingest_bdd --splits train val --overwrite

# Subset for local dev (--limit caps frames per split):
python -m object_detection.ingest_bdd --splits train val --limit 5000 --overwrite
```

> **Want to skip the download?** Use `--synthetic N` to generate fake frames and verify
> the full pipeline works first:
> ```bash
> python -m object_detection.ingest_bdd --synthetic 500 --overwrite
> ```

> **Important**: the table is created with `new_table_enable_stable_row_ids=true`.
> This is required for Geneva materialized view refresh to work across table versions.
> Without stable row IDs, `mv.refresh()` fails once any new data has been appended.

### Step 2 — Geneva backfill

There are two tiers of backfill depending on which training narrative you want to run.

**Tier 1 — Annotation-based** (fast, CPU-friendly, needed for the pedestrian/rider narrative):

```bash
python -m object_detection.backfill_geneva --columns has_person has_rider
```

**Tier 2 — Model-inference-based** (needed for the ambulance narrative):

```bash
# CPU (SSDLite — no GPU required, good for local dev):
python -m object_detection.backfill_geneva \
    --columns vehicle_label vehicle_confidence vehicle_bbox_area_pct

# GPU (Faster R-CNN — more accurate, recommended for final backfill):
python -m object_detection.backfill_geneva --gpu \
    --columns vehicle_label vehicle_confidence vehicle_bbox_area_pct
```

Both variants write to the same `vehicle_label`, `vehicle_confidence`, and
`vehicle_bbox_area_pct` columns.  The only difference is which model runs under the hood.

To restart a stuck backfill job, add `--overwrite` — it drops and re-adds the column from scratch:

```bash
python -m object_detection.backfill_geneva --columns vehicle_label --overwrite
```

Backfill is incremental — re-running without `--overwrite` only processes newly added rows.

### Step 3 — EDA (optional but recommended)

```bash
python -m object_detection.spec_queries
# or: jupyter notebook notebooks/eda_bdd100k.ipynb
```

### Step 4 — Create materialized views

```bash
# Tier 1 — built-in views (pedestrian + rider narrative, needs Tier 1 backfill):
python -m object_detection.manage_views --action curate

# Tier 2 — ambulance views (requires vehicle_label backfill first):
python -m object_detection.manage_views --action curate-ambulance
```

`--action curate` creates pre-split train/val pairs:

| View | Filter |
|---|---|
| `bdd100k_nighttime_person_train` | `timeofday='night' AND has_person=true AND split='train'` |
| `bdd100k_nighttime_person_val` | `timeofday='night' AND has_person=true AND split='val'` |
| `bdd100k_rider_train` | `has_rider=true AND split='train'` |
| `bdd100k_rider_val` | `has_rider=true AND split='val'` |
| `bdd100k_nighttime_rider_train` | `timeofday='night' AND has_rider=true AND split='train'` |
| `bdd100k_nighttime_rider_val` | `timeofday='night' AND has_rider=true AND split='val'` |

`--action curate-ambulance` adds:

| View | Filter |
|---|---|
| `bdd100k_ambulance_train` | `vehicle_label='red_ambulance' AND vehicle_bbox_area_pct>5.0 AND split='train'` |
| `bdd100k_ambulance_val` | `vehicle_label='red_ambulance' AND vehicle_bbox_area_pct>5.0 AND split='val'` |

For one-off custom views, `--action add --name <name> --filter "<SQL>"` works for any
SQL filter over any backfilled column.

### Step 5 — Train

The script always evaluates the COCO pretrained checkpoint first, then fine-tunes and
prints a before/after comparison — no separate baseline run needed:

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

**Narrative A — Pedestrian / Rider** (annotation-based curation, works on CPU too):

```bash
# Nighttime pedestrian
python -m object_detection.train_detector \
    --train-table bdd100k_nighttime_person_train \
    --val-table bdd100k_nighttime_person_val \
    --epochs 10 --batch-size 8 --num-workers 4 \
    --output-dir checkpoints/nighttime_person

# Rider
python -m object_detection.train_detector \
    --train-table bdd100k_rider_train \
    --val-table bdd100k_rider_val \
    --epochs 10 --batch-size 8 --num-workers 4 \
    --output-dir checkpoints/rider
```

**Narrative B — Ambulance** (requires Tier 2 backfill + `--action curate-ambulance`):

```bash
python -m object_detection.train_detector \
    --train-table bdd100k_ambulance_train \
    --val-table bdd100k_ambulance_val \
    --epochs 10 --batch-size 4 --lr 0.002 --num-workers 4 \
    --output-dir checkpoints/ambulance
```

The same pattern extends to any class you can express as a SQL filter over a backfilled
column. Create a view, point `--train-table` at it, done.

Training logs the **table version** for provenance:
```
Table 'bdd100k_rider'  version=4  rows=1269
```

Checkpoint ↔ exact data snapshot. If you retrain after a refresh, the version number increments.

### Step 6 — New data arrives → refresh views

```bash
# 1. Append new footage (omit --overwrite to append, not replace):
python -m object_detection.ingest_bdd --synthetic 500

# 2. Incremental backfill — only processes the newly added rows:
python -m object_detection.backfill_geneva --columns has_person has_rider

# 3. Refresh all views — one command, all splits update:
python -m object_detection.manage_views --action refresh
```

Output:
```
Parent 'bdd100k': 25700 rows (version 21)
[bdd100k_nighttime_person_train]   932 →  984 rows  (+52)  version 5
[bdd100k_nighttime_person_val]     233 →  246 rows  (+13)  version 5
[bdd100k_rider_train]              598 →  634 rows  (+36)  version 5
[bdd100k_rider_val]                149 →  159 rows  (+10)  version 5
[bdd100k_nighttime_rider_train]    111 →  122 rows  (+11)  version 5
[bdd100k_nighttime_rider_val]       28 →   30 rows   (+2)  version 5
```

Retrain — same command, different data, new version number in the logs.

### Check status any time

```bash
python -m object_detection.manage_views --action status
```

```
table                                rows   version
------------------------------------------------------------
  bdd100k                           25200        19  (source)
  bdd100k_nighttime_person_train     1620         4
  bdd100k_nighttime_person_val        404         4
  bdd100k_rider_train                1016         4
  bdd100k_rider_val                   253         4
  bdd100k_nighttime_rider_train       198         4
  bdd100k_nighttime_rider_val          49         4
  bdd100k_ambulance_train             250         2
  bdd100k_ambulance_val                62         2
```

Custom views show up automatically — `refresh` and `status` discover all views in the DB.

---

## Results (1 epoch, CPU baseline — run on GPU for production)

The training script always evaluates the COCO pretrained checkpoint first (baseline),
then fine-tunes, and prints a side-by-side delta. No separate baseline run needed.

### Narrative A — Pedestrian / Rider (CPU baseline, 1 epoch, 15k subset)

| metric | baseline (COCO) | curated fine-tune | Δ |
|---|---|---|---|
| **Nighttime pedestrian** | | | |
| mAP@0.5 | 0.4007 | **0.5002** | **+0.100** |
| Precision | 0.4722 | **0.5104** | **+0.038** |
| Recall | 0.5919 | **0.7624** | **+0.171** |
| **Rider** | | | |
| mAP@0.5 | 0.5295 | **0.6370** | **+0.108** |
| Precision | 0.5670 | 0.5435 | -0.024 |
| Recall | 0.6828 | **0.7922** | **+0.109** |

### Narrative B — Ambulance *(GPU required, results pending)*

Requires Tier 2 backfill (`vehicle_label`). Results pending after GPU run.

---

## Reproducing the CPU baseline

Environment: macOS, Apple Silicon, no GPU, Python 3.13. Uses a 15k/10k subset to keep
runtimes manageable. GPU runs should use the full dataset (drop `--limit`).

```bash
cd object-detection/
export GENEVA_PIPELINE_STALL_TIMEOUT_S=7200

# 1. Ingest subset
python -m object_detection.ingest_bdd --splits train val --limit 15000 --overwrite

# 2. Backfill (Tier 1 — fast, no image decoding)
python -m object_detection.backfill_geneva --columns has_person has_rider

# 3. Create views
python -m object_detection.manage_views --action curate

# 4a. Nighttime pedestrian
python -m object_detection.train_detector \
    --train-table bdd100k_nighttime_person_train \
    --val-table bdd100k_nighttime_person_val \
    --epochs 1 --batch-size 4 --num-workers 0

# 4b. Rider
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
│   ├── geneva_udfs.py       # all Geneva UDF functions
│   ├── backfill_geneva.py   # Geneva backfill runner (incremental, checkpointed)
│   ├── manage_views.py      # create / refresh / status of materialized views  ← lifecycle
│   ├── dataloader.py        # LanceDetectionDataset + make_detection_loader (Permutation API)
│   ├── train_detector.py    # Faster R-CNN fine-tune (logs table version)
│   ├── eval.py              # torchmetrics mAP evaluation
│   └── spec_queries.py      # SQL filter specs + EDA / FTS helpers
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
    ├── bdd100k_ambulance_train.lance    # Tier 2 — requires GPU backfill
    └── bdd100k_ambulance_val.lance
```

---

## Key LanceDB Patterns

**Stable row IDs** — required for Geneva materialized view refresh across table versions:
```python
db.create_table(name, data=reader, schema=schema,
                storage_options={"new_table_enable_stable_row_ids": "true"})
```
Without this, `mv.refresh()` raises a `RuntimeError` once any new data has been appended.

**Streaming ingestion** — never call `table.add()` in a loop:
```python
reader = pa.RecordBatchReader.from_batches(schema, batch_generator())
db.create_table(name, data=reader, schema=schema)
# append: tbl.add(reader)
```

**Incremental Geneva backfill** — only processes NULL rows, safe to re-run:
```bash
python -m object_detection.backfill_geneva --db data/bdd100k/lancedb \
    --columns has_person has_rider vehicle_label vehicle_confidence vehicle_bbox_area_pct
```

**Materialized view refresh** — one call, all views stay current:
```python
gconn = geneva.connect("data/bdd100k/lancedb")
mv = gconn.open_table("bdd100k_rider_train")
mv.refresh()   # picks up any new rows matching the view's WHERE clause
```

**Training provenance** — `train_detector.py` logs `table.version` automatically:
```
Table 'bdd100k_rider_train'  version=4  rows=1016
```

**Flat schema only** — no nested structs (LanceDB SQL query bug):
```python
# ✓  ann_bboxes: list<list<float32>>
# ✗  ann_bboxes: list<struct<x1, y1, x2, y2>>
```

**Disk hygiene after multi-step updates**:
```python
from datetime import timedelta
tbl.cleanup_old_versions(older_than=timedelta(seconds=0), delete_unverified=True)
tbl.compact_files()
```

**Geneva local runs** require:
```bash
export GENEVA_PIPELINE_STALL_TIMEOUT_S=7200  # default 600s times out on CPU
sudo chmod a+rw /tmp/.geneva_zip_setup        # often root-owned on macOS
```
