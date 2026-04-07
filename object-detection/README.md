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
                                    │  has_rider, vehicle_light_*
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
              │  bdd100k_nighttime_person  (WHERE night+person)  │
              │  bdd100k_rider             (WHERE has_rider)      │
              │  bdd100k_nighttime_rider   (WHERE night+rider)    │
              │                                                   │
              │  mv.refresh() keeps them in sync automatically   │
              └──────────────────────┬──────────────────────────┘
                                     │  train_detector.py --train-table bdd100k_rider
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

**Dataset**: Download BDD100K Detection 2020 from https://bdd-data.berkeley.edu/portal.html.
You need:
- `bdd100k/images/100k/train/` and `bdd100k/images/100k/val/` — JPEG frames
- `bdd100k/labels/det_20/train/` and `bdd100k/labels/det_20/val/` — per-frame JSON annotations

Place them under `object-detection/data/bdd100k/`.

**Python environment**: Requires `lancedb`, `geneva`, `torch`, `torchvision`, `pyarrow`.

```bash
cd object-detection/
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install lancedb geneva torch torchvision pyarrow pillow
```

**Geneva local setup** (macOS):
```bash
export GENEVA_PIPELINE_STALL_TIMEOUT_S=7200  # prevents timeout on slow CPU runs
sudo chmod a+rw /tmp/.geneva_zip_setup        # fixes permission error on first run
```

---

## Full Pipeline (step by step)

All commands run from the `object-detection/` directory.

```bash
cd object-detection/
export DB=data/bdd100k/lancedb
export TABLE=bdd100k
export GENEVA_PIPELINE_STALL_TIMEOUT_S=7200
```

### Step 1 — Ingest

```bash
# Smoke test — no download needed:
python -m object_detection.ingest_bdd \
    --synthetic 1000 \
    --output $DB --table-name $TABLE --overwrite

# Real BDD100K (25k frames used in these experiments):
python -m object_detection.ingest_bdd \
    --splits train val \
    --image-root data/bdd100k/images \
    --annotation-root data/bdd100k/labels \
    --output $DB --table-name $TABLE --overwrite
```

> **Important**: the table is created with `new_table_enable_stable_row_ids=true`.
> This is required for Geneva materialized view refresh to work across table versions
> (i.e. after appending new footage). Without stable row IDs, `mv.refresh()` fails
> once any new data has been appended.

### Step 2 — Geneva backfill

```bash
# Annotation-presence flags (fast — no image decoding):
python -m object_detection.backfill_geneva \
    --db $DB --table $TABLE \
    --columns has_person has_rider \
    --min-checkpoint-size 25200 --max-checkpoint-size 25200

# Vehicle detector labels (SSDLite, ~30 min CPU):
python -m object_detection.backfill_geneva \
    --db $DB --table $TABLE \
    --columns vehicle_light_label vehicle_light_confidence vehicle_light_bbox_area_pct \
    --min-checkpoint-size 25200 --max-checkpoint-size 25200

# Heavy Faster R-CNN columns (GPU recommended):
python -m object_detection.backfill_geneva \
    --db $DB --table $TABLE \
    --columns vehicle_label vehicle_confidence vehicle_bbox_area_pct \
    --concurrency 8
```

Geneva's backfill is **incremental** — re-running after new ingests only processes rows
where the column is still `NULL`. No re-work on existing data.

### Step 3 — EDA (optional but recommended)

Explore what the Geneva columns reveal before committing to curation filters:

```bash
# Row counts by spec — pure metadata, no image loading:
python -m object_detection.spec_queries --db $DB --table $TABLE

# Or open the EDA notebook:
jupyter notebook notebooks/eda_bdd100k.ipynb
```

The notebook shows: frame distributions, annotation counts by spec, FTS search on
`scene_description`, white-balance CCT distributions (night vs daytime), and a sample
image preview. All queries use `count_rows(filter=)` or `search().limit()` — the full
table is never loaded into memory.

### Step 4 — Create materialized views

```bash
python -m object_detection.manage_views --action curate --db $DB
```

This creates three Geneva materialized views as child tables:

| View | Filter | Train rows | Val rows |
|---|---|---|---|
| `bdd100k_nighttime_person` | `timeofday='night' AND has_person=true` | 1,165 | 859 |
| `bdd100k_rider` | `has_rider=true` | 747 | 522 |
| `bdd100k_nighttime_rider` | `timeofday='night' AND has_rider=true` | 139 | 108 |

The curation logic — the WHERE clause — lives in `manage_views.py`, not in training scripts.

### Step 5 — Train

The training script auto-detects GPU. If `torch.cuda.is_available()` is true it uses CUDA
automatically — no extra flags needed. On an A100 each experiment takes ~10–20 min.

**Curated runs** (main experiments):

```bash
# Exp 1 — Nighttime pedestrian  (~1165 train frames, ~20 min on A100)
python -m object_detection.train_detector \
    --mode finetune \
    --db $DB \
    --train-table bdd100k_nighttime_person \
    --val-table $TABLE \
    --train-where "split='train'" \
    --val-where "split='val' AND timeofday='night' AND has_person=true" \
    --epochs 10 --batch-size 8 --num-workers 4 \
    --output-dir checkpoints/nighttime_person_frcnn

# Exp 2 — Rider  (~747 train frames, ~10 min on A100)
python -m object_detection.train_detector \
    --mode finetune \
    --db $DB \
    --train-table bdd100k_rider \
    --val-table $TABLE \
    --train-where "split='train'" \
    --val-where "split='val' AND has_rider=true" \
    --epochs 10 --batch-size 8 --num-workers 4 \
    --output-dir checkpoints/rider_frcnn

# Exp 3 — Nighttime rider  (~139 train frames, GPU only — too slow on CPU)
# Lower LR because the slice is tiny; use all available frames, no subsampling
python -m object_detection.train_detector \
    --mode finetune \
    --db $DB \
    --train-table bdd100k_nighttime_rider \
    --val-table $TABLE \
    --train-where "split='train'" \
    --val-where "split='val' AND timeofday='night' AND has_rider=true" \
    --epochs 10 --batch-size 4 --lr 0.001 --num-workers 4 \
    --output-dir checkpoints/nighttime_rider_frcnn
```

**Random baseline** (to reproduce the results comparison):

```bash
# Exp 1 random baseline — train on full table, eval on the same curated slice:
python -m object_detection.train_detector \
    --mode finetune \
    --db $DB \
    --train-table $TABLE \
    --val-table $TABLE \
    --train-where "split='train' AND num_annotations > 0" \
    --val-where "split='val' AND timeofday='night' AND has_person=true" \
    --epochs 10 --batch-size 8 --num-workers 4 \
    --output-dir checkpoints/nighttime_person_random

# Exp 2 random baseline:
python -m object_detection.train_detector \
    --mode finetune \
    --db $DB \
    --train-table $TABLE \
    --val-table $TABLE \
    --train-where "split='train' AND num_annotations > 0" \
    --val-where "split='val' AND has_rider=true" \
    --epochs 10 --batch-size 8 --num-workers 4 \
    --output-dir checkpoints/rider_random
```

Training logs the **table version** for provenance:
```
Table 'bdd100k_rider'  version=4  rows=1269
```

Checkpoint ↔ exact data snapshot. If you retrain after a refresh, the version number increments.

### Step 6 — New data arrives → refresh views

```bash
# 1. Append new footage (no --overwrite = append mode):
python -m object_detection.ingest_bdd \
    --synthetic 500 --output $DB --table-name $TABLE

# 2. Incremental backfill (Geneva skips already-computed rows):
TOTAL=$(python -c "import lancedb; print(lancedb.connect('$DB').open_table('$TABLE').count_rows())")
python -m object_detection.backfill_geneva \
    --db $DB --table $TABLE \
    --columns has_person has_rider \
    --min-checkpoint-size $TOTAL --max-checkpoint-size $TOTAL

# 3. Refresh all views — one command, all splits update:
python -m object_detection.manage_views --action refresh --db $DB
```

Output:
```
Parent 'bdd100k': 25700 rows (version 21)
[bdd100k_nighttime_person]   1165 → 1230 rows  (+65)  version 5
[bdd100k_rider]               747 →  793 rows  (+46)  version 5
[bdd100k_nighttime_rider]     139 →  152 rows  (+13)  version 5
```

Retrain — same command, different data, new version number in the logs.

### Check status any time

```bash
python -m object_detection.manage_views --action status --db $DB
```

```
table                                rows   version  filter
------------------------------------------------------------------------------------------
  bdd100k                           25200        19  (source)
  bdd100k_nighttime_person           2024         4  WHERE timeofday = 'night' AND has_person = true
  bdd100k_rider                      1269         4  WHERE has_rider = true
  bdd100k_nighttime_rider             247         4  WHERE timeofday = 'night' AND has_rider = true
```

---

## Results (1 epoch, CPU baseline — run on GPU for production)

Each experiment compares two models trained from the same COCO pretrained weights
with the same data budget (400 frames). Only the *selection* of those 400 frames differs.

**Val set**: the Geneva-curated slice for that experiment (not general val).

### Experiment 1 — Nighttime pedestrian detection

| metric | random 400 | curated 400 | Δ |
|---|---|---|---|
| mAP@0.5 | 0.2569 | 0.2509 | -0.006 |
| Precision | 0.3288 | **0.4288** | **+0.100** |
| Recall | 0.5887 | 0.5875 | ~0 |

### Experiment 2 — Rider detection ✓ best result

| metric | random 400 | curated 400 | Δ |
|---|---|---|---|
| mAP@0.5 | 0.3101 | **0.4076** | **+0.098** |
| Precision | 0.4523 | **0.5188** | **+0.067** |
| Recall | 0.6157 | **0.6493** | **+0.034** |

Clean sweep — Geneva curation beats random sampling on all three metrics.

### Experiment 3 — Nighttime rider detection *(GPU required)*

Only 139 curated train frames — too few for a meaningful CPU run. See Step 5 for the
GPU command (`--lr 0.001`, lower because the slice is small).

*Results pending — fill in after GPU run.*

---

## Reproducing the CPU baseline results

Environment used: macOS, Apple Silicon, no GPU, Python 3.13 (uv venv).

```bash
cd object-detection/
export DB=data/bdd100k/lancedb
export TABLE=bdd100k
export GENEVA_PIPELINE_STALL_TIMEOUT_S=7200

# 1. Ingest 25k frames (15k train + 10.2k val)
python -m object_detection.ingest_bdd \
    --splits train val \
    --image-root data/bdd100k/images \
    --annotation-root data/bdd100k/labels \
    --output $DB --table-name $TABLE --overwrite

# 2. Backfill annotation-presence flags (fast — no image decoding)
python -m object_detection.backfill_geneva \
    --db $DB --table $TABLE \
    --columns has_person has_rider \
    --min-checkpoint-size 25200 --max-checkpoint-size 25200

# 3. Backfill vehicle light labels (SSDLite, ~30 min CPU)
python -m object_detection.backfill_geneva \
    --db $DB --table $TABLE \
    --columns vehicle_light_label vehicle_light_confidence vehicle_light_bbox_area_pct \
    --min-checkpoint-size 25200 --max-checkpoint-size 25200

# 4. Create materialized views
python -m object_detection.manage_views --action curate --db $DB

# 5a. Exp 1 — Nighttime pedestrian (train on curated view, eval on nighttime person slice)
python -m object_detection.train_detector \
    --mode finetune --db $DB \
    --train-table bdd100k_nighttime_person --val-table $TABLE \
    --train-where "split='train'" \
    --val-where "split='val' AND timeofday='night' AND has_person=true" \
    --epochs 1 --batch-size 4 --num-workers 0

# 5b. Exp 2 — Rider (best result)
python -m object_detection.train_detector \
    --mode finetune --db $DB \
    --train-table bdd100k_rider --val-table $TABLE \
    --train-where "split='train'" \
    --val-where "split='val' AND has_rider=true" \
    --epochs 1 --batch-size 4 --num-workers 0
```

> **Note on CPU runtime**: FasterRCNN on CPU is ~15s/image. Training 400 frames ≈ 100 min,
> eval 150 frames ≈ 30 min per experiment. Use GPU (`--num-workers 4 --batch-size 8`) for
> practical iteration.

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
│   ├── dataloader.py        # LanceArrowDetectionDataset + make_detection_loader
│   ├── train_detector.py    # Faster R-CNN fine-tune (logs table version)
│   ├── eval.py              # torchmetrics mAP evaluation
│   └── spec_queries.py      # SQL filter specs + EDA / FTS helpers
├── notebooks/
│   └── eda_bdd100k.ipynb    # EDA walkthrough
├── e2e_verify.py            # integration test against real BDD100K data
└── data/bdd100k/lancedb/    # Lance tables (gitignored)
    ├── bdd100k.lance
    ├── bdd100k_nighttime_person.lance
    ├── bdd100k_rider.lance
    └── bdd100k_nighttime_rider.lance
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
python -m object_detection.backfill_geneva --db $DB --table $TABLE \
    --columns has_person has_rider vehicle_light_label \
    --min-checkpoint-size $TOTAL_ROWS --max-checkpoint-size $TOTAL_ROWS
```

**Materialized view refresh** — one call, all views stay current:
```python
gconn = geneva.connect("data/bdd100k/lancedb")
mv = gconn.open_table("bdd100k_rider")
mv.refresh()   # picks up any new rows matching the view's WHERE clause
```

**Training provenance** — `train_detector.py` logs `table.version` automatically:
```
Table 'bdd100k_rider'  version=4  rows=1269
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
