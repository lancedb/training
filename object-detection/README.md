# BDD100K Targeted Fine-Tuning with LanceDB + Geneva

Targeted fine-tuning on curated failure-mode slices — using LanceDB as the data backbone at every stage of the ML pipeline, from raw video frames to trained checkpoints.

---

## The Problem

A COCO-pretrained Faster R-CNN degrades on three deployment-critical failure modes:

| Failure mode | Root cause | Curation signal |
|---|---|---|
| **Riders** | COCO separates `person` and `bicycle`; BDD combines them — the model never saw that silhouette | `has_rider = true` (annotation flag) |
| **Nighttime pedestrians** | COCO training data is heavily daytime-biased | `timeofday = 'night' AND has_person = true` |
| **Distant pedestrians** | Small far-away people are underrepresented; a miss at distance has the highest real-world consequence | `person_bbox_area_pct < 30%` (GPU inference) |

The fix is targeted fine-tuning on curated slices. The harder question is *maintenance*: as fleet footage arrives continuously, how do you keep training splits current without manual work — and how do you ensure you're not wasting compute on near-duplicate frames?

---

## Workflow Overview

```
  Fix three failure modes the COCO-pretrained model misses on real fleet footage:
  riders · nighttime pedestrians · distant pedestrians

┌──────────────────────────────────────────────────────┐
│              Fleet footage (BDD100K)                 │
└─────────────────────────┬────────────────────────────┘
                          │
                          ▼  ingest_bdd.py
                          │  Multimodal lakehouse: raw JPEG bytes, scene metadata,
                          │  and parallel-list annotations in one Lance table.
                          │  Stable row IDs enable incremental view refreshes.
                          │
┌─────────────────────────┴────────────────────────────┐
│  bdd100k  [Lance table]  ← append as footage arrives │
│  image_bytes · weather · timeofday · ann_* · split   │
└─────────────────────────┬────────────────────────────┘
                          │
                          ▼  backfill_geneva.py  (Geneva UDFs — incremental, checkpointed)
                          │  Tier 1: has_person, has_rider            (CPU, annotation-derived)
                          │  Tier 2: person_bbox_area_pct             (GPU, Faster R-CNN)
                          │  Tier 3: embedding [512-d ResNet18]       (GPU)
                          │          → dedup --action index            (IVF-PQ cosine index)
                          │          → backfill is_duplicate           (CPU, vector search)
                          │
                          ▼  manage_views.py  (Materialized views)
                          │
            ┌─────────────┼──────────────┐
            ▼             ▼              ▼
  bdd100k_rider  bdd100k_nighttime  bdd100k_distant
  _{train,val}   _person_{t,v}      _person_{t,v}
  [Materialized  [Materialized      [Materialized
     view]          view]              view]
            └─────────────┼──────────────┘
                          │
                          ▼  train_detector.py
                          │  LanceDB Permutation API · PyTorch DataLoader · AMP
                          │
                 ┌────────┴────────┐
                 │ fine-tuned      │
                 │ checkpoint      │
                 └────────┬────────┘
                          │
                 New footage arrives?
                 → ingest → backfill → manage_views --action refresh
                 → same view names · new versions · more data
```

---

## Setup

```bash
cd object-detection/
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install lancedb geneva torch torchvision pyarrow pillow tqdm

export GENEVA_PIPELINE_STALL_TIMEOUT_S=7200
sudo chmod a+rw /tmp/.geneva_zip_setup   # macOS only
```

---

## Running the Pipeline

### 1 · Ingest

Raw JPEG bytes, structured metadata (weather, scene, time of day), and parallel-list annotations (one element per bounding box) land in a single Lance table with an enforced schema — no separate preprocessing step. `stable_row_ids` ensures materialized views can be refreshed incrementally after footage is appended.

```bash
python -m object_detection.ingest_bdd --splits train val --overwrite
```

### 2 · Backfill feature columns

Instead of writing frames to disk and running a separate preprocessing job, Geneva UDFs run directly against the Lance table and write results back as queryable columns. Backfill is incremental (`WHERE col IS NULL`), stateful (class-based UDFs with lazy model loading), and checkpointed — safe to re-run as new footage arrives.

```bash
# Tier 1 — CPU, annotation-derived (minutes on full dataset)
python -m object_detection.backfill_geneva --columns has_person has_rider

# Tier 2 — GPU Faster R-CNN: largest detected person bbox as % of frame area
# <30% → pedestrian is distant or small; used to curate the hard-cases split
python -m object_detection.backfill_geneva --gpu --columns person_bbox_area_pct
```

### 3 · Explore

With all signals stored as flat scalar columns, standard SQL and full-text search work directly on the table — no export, no joining with external manifests.

```bash
python -m object_detection.spec_queries
```

```python
tbl.count_rows(filter="timeofday = 'night' AND has_person = true")          # 6,431 frames
tbl.count_rows(filter="has_rider = true")                                   # 4,105 frames
tbl.count_rows(filter="has_person = true AND person_bbox_area_pct < 30.0")  # ~3,200 frames

# FTS on the Geneva-generated scene_description column
tbl.search("rainy night city pedestrian", query_type="fts").limit(10).to_pandas()
```

### 4 · Create training views

A training split is a named SQL materialized view, not a CSV manifest or a directory of files. The filter definition lives in `manage_views.py`; the training script opens the view by name and never needs a WHERE clause. The training script logs `table.version` with every checkpoint — a direct link between weights and the exact data snapshot that produced them.

```bash
python -m object_detection.manage_views --action curate          # Tier 1 views
python -m object_detection.manage_views --action curate-person  # Tier 2 distant-pedestrian views
```

### 5 · Train

LanceDB's Permutation API provides random-access indexing over Lance tables. Workers open their own connections lazily and read directly from the materialized view — no copy to local storage, no intermediate file format. Combined with `pin_memory`, `persistent_workers`, and AMP, this keeps A100 utilisation above 85% on 1280×720 driving footage.

```bash
python -m object_detection.train_detector \
    --train-table bdd100k_rider_train --val-table bdd100k_rider_val \
    --epochs 10 --batch-size 64 --lr 0.04 --num-workers 14 --output-dir checkpoints/rider

python -m object_detection.train_detector \
    --train-table bdd100k_nighttime_person_train --val-table bdd100k_nighttime_person_val \
    --epochs 10 --batch-size 64 --lr 0.04 --num-workers 14 --output-dir checkpoints/nighttime_person

python -m object_detection.train_detector \
    --train-table bdd100k_distant_person_train --val-table bdd100k_distant_person_val \
    --epochs 10 --batch-size 64 --lr 0.04 --num-workers 14 --output-dir checkpoints/distant_person
```

### 6 · Deduplicate

BDD100K is dashcam footage at ~10 Hz. Consecutive frames within a clip are nearly identical pixels — training on them wastes compute and over-represents specific scenes. Dedup follows the same Geneva backfill pattern as every other feature: `embedding` (ResNet18, 512-d, L2-normalised) is added as a column on `bdd100k`, an IVF-PQ cosine index is built on it, then `is_duplicate` is backfilled by querying the index for each row's nearest non-self neighbour (~1ms per query at 80k scale). Since `manage_views` already filters `is_duplicate = false`, views automatically exclude near-duplicates once the column is populated — no extra step.

> **Why ResNet18 and not CLIP?** Dedup targets pixel-level similarity — consecutive frames from the same clip that barely differ. ResNet18's ImageNet features capture edges, textures, and colour distributions, exactly what changes (slightly) between near-duplicate frames. CLIP captures semantic meaning and would over-cluster visually distinct frames from different clips that happen to share a scene type, undermining the dedup signal.

```bash
# Backfill ResNet18 embeddings as a column on bdd100k (GPU)
python -m object_detection.backfill_geneva --gpu --columns embedding

# Build IVF-PQ cosine index on the embedding column
python -m object_detection.dedup --action index

# Backfill is_duplicate: True when nearest-neighbour similarity >= 0.85
python -m object_detection.backfill_geneva --columns is_duplicate

# Optional — show duplicate rate from the backfilled column
python -m object_detection.dedup --action stats

# Recreate views — is_duplicate filter is added automatically
python -m object_detection.manage_views --action curate
python -m object_detection.manage_views --action curate-person
```

### 7 · Refresh after new footage

When new footage is appended, one call per step increments every view's version and makes the extra data immediately available to the training script — no view definitions change, no manifests to regenerate.

```bash
python -m object_detection.ingest_bdd --synthetic 500
python -m object_detection.backfill_geneva --columns has_person has_rider
python -m object_detection.backfill_geneva --gpu --columns person_bbox_area_pct
python -m object_detection.manage_views --action refresh
```

---

## Results

GPU runs on full BDD100K (80k frames), 10 epochs, A100, batch size 64, AMP enabled. Baseline is the pretrained COCO checkpoint evaluated on each curated val split.

| Failure mode | Metric | Baseline (COCO) | Fine-tuned | Δ |
|---|---|---|---|---|
| **Nighttime pedestrian** | mAP@0.5 | 0.4025 | **0.5260** | **+0.1235** |
| | Precision | 0.4739 | **0.5569** | **+0.0830** |
| | Recall | 0.5923 | **0.7579** | **+0.1656** |
| **Rider** | mAP@0.5 | 0.5563 | **0.6565** | **+0.1002** |
| | Precision | 0.5872 | **0.6016** | **+0.0145** |
| | Recall | 0.6788 | **0.7834** | **+0.1046** |
| **Distant pedestrian** | mAP@0.5 | 0.4746 | **0.5810** | **+0.1064** |
| | Precision | 0.5847 | **0.6363** | **+0.0517** |
| | Recall | 0.6794 | **0.8038** | **+0.1244** |

Recall improvement dominates across all three failure modes — the model catches significantly more of the objects it was previously missing. Nighttime pedestrian shows the strongest lift (consistent with it being the largest distribution shift from COCO's daytime-heavy data). All improvements use the same COCO pretrained weights as starting point; no external data was added.

---

## Results — After Deduplication

Dedup removes near-identical consecutive frames (cosine similarity ≥ 0.97 on ResNet18 embeddings) from training splits only. Val splits are unchanged so baseline numbers are identical.

**Training data reduction per split**

| Split | Before dedup | After dedup | Removed |
|---|---|---|---|
| rider_train | 3,590 | 3,150 | 440 (12.3%) |
| nighttime_person_train | 5,594 | 3,022 | 2,572 (46.0%) |
| distant_person_train | 22,092 | 19,004 | 3,088 (14.0%) |

Nighttime is hit hardest — long monotonous highway clips with barely-changing frames. Rider and distant person are more episodic, so less redundancy.

**Training results**

| Failure mode | Metric | Baseline (COCO) | No dedup | Deduped | Δ vs no-dedup |
|---|---|---|---|---|---|
| **Rider** | mAP@0.5 | 0.5563 | 0.6565 | **0.6598** | **+0.0033** |
| | Precision | 0.5872 | 0.6016 | 0.5983 | -0.0033 |
| | Recall | 0.6788 | 0.7834 | **0.7829** | -0.0005 |
| **Nighttime pedestrian** | mAP@0.5 | 0.4025 | 0.5260 | 0.5154 | -0.0106 |
| | Precision | 0.4739 | 0.5569 | 0.5483 | -0.0086 |
| | Recall | 0.5923 | 0.7579 | 0.7527 | -0.0052 |
| **Distant pedestrian** | mAP@0.5 | 0.4746 | 0.5810 | **0.5803** | -0.0007 |
| | Precision | 0.5847 | 0.6363 | **0.6374** | +0.0011 |
| | Recall | 0.6794 | 0.8038 | **0.8023** | -0.0015 |

---

## Project Structure

```
object-detection/
├── object_detection/
│   ├── schema.py            # Lance schema + GENEVA_UDF_COLUMNS
│   ├── ingest_bdd.py        # BDD100K → LanceDB (streaming RecordBatch ingestion)
│   ├── geneva_udfs.py       # Tier 1 annotation UDFs + Tier 2 GPU inference UDFs
│   ├── backfill_geneva.py   # Geneva backfill runner (incremental, checkpointed)
│   ├── manage_views.py      # Create / refresh / status materialized views
│   ├── dedup.py             # Vector index build + duplicate rate stats
│   ├── dataloader.py        # Permutation API + PyTorch DataLoader
│   ├── train_detector.py    # Faster R-CNN fine-tune with AMP (logs table version)
│   ├── eval.py              # mAP@0.5 evaluation
│   └── spec_queries.py      # SQL filter specs + EDA helpers
└── data/bdd100k/lancedb/    # Lance tables (gitignored)
    ├── bdd100k.lance                          # source table (+ embedding, is_duplicate cols)
    ├── bdd100k_rider_{train,val}.lance
    ├── bdd100k_nighttime_person_{train,val}.lance
    ├── bdd100k_nighttime_rider_{train,val}.lance
    └── bdd100k_distant_person_{train,val}.lance
```
