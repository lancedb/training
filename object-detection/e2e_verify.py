"""
End-to-end verification on a real BDD100K subset.

Runs the full pipeline against an already-ingested + backfilled LanceDB table:
  1. Table sanity checks (row counts, schema, Geneva UDF columns)
  2. Spec counts for all curated slices
  3. FTS search on scene_description
  4. Permutation split (80/20)
  5. Training smoke test (1 epoch, 100 train / 50 val, nighttime+person slice)

Usage
-----
cd training/
PYTHONPATH=object-detection/src .venv/bin/python object-detection/e2e_verify.py
"""

import sys
import time
from pathlib import Path

import lancedb
import torch

ROOT    = Path(__file__).parent
DB_PATH = str(ROOT / "data/bdd100k/lancedb")
TABLE   = "bdd100k"

# ── 1. Table sanity ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1 — Table sanity checks")
print("="*60)

db  = lancedb.connect(DB_PATH)
tbl = db.open_table(TABLE)

total = tbl.count_rows()
print(f"Total rows : {total:,}")
print(f"Version    : {tbl.version}")

n_train = tbl.count_rows(filter="split = 'train'")
n_val   = tbl.count_rows(filter="split = 'val'")
print(f"Train      : {n_train:,}  Val: {n_val:,}")

# Verify key Geneva UDF columns are present and populated
required_cols = ["has_person", "has_rider", "vehicle_label",
                 "white_balance", "scene_description"]
missing = [c for c in required_cols if c not in tbl.schema.names]
if missing:
    print(f"WARNING: Geneva columns not yet backfilled: {missing}")
    print("Run: python -m object_detection.backfill_geneva --columns " + " ".join(missing))
else:
    print(f"Geneva columns present: {required_cols}")

    # Quick null check on a small sample (no full scan)
    sample = (tbl.search()
                 .select(required_cols)
                 .limit(100)
                 .to_pandas())
    null_counts = sample.isnull().sum()
    if null_counts.any():
        print(f"WARNING: NULLs in sample: {null_counts[null_counts > 0].to_dict()}")
    else:
        print("No NULLs in 100-row sample — backfill looks complete")

# ── 2. Spec counts ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2 — Spec counts")
print("="*60)

from object_detection.spec_queries import SPEC_FILTERS

for spec, filter_template in SPEC_FILTERS.items():
    try:
        sql = filter_template.format(bbox_pct=5.0, min_confidence=0.5)
    except KeyError:
        sql = filter_template
    n = tbl.count_rows(filter=sql)
    print(f"  {spec:<25}  {n:>6,} rows")

# ── 3. FTS search ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 — FTS search on scene_description")
print("="*60)

from object_detection.spec_queries import make_fts_index, fts_search

make_fts_index(tbl, replace=True)
results = fts_search(tbl, "night city street rider", limit=5)
print(f"FTS 'night city street rider' → {len(results)} hits")
print(results[["image_id", "scene_description", "has_rider", "timeofday"]].to_string(index=False))

# ── 4. Permutation split ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4 — Permutation split (nighttime+rider, 80/20)")
print("="*60)

from object_detection.spec_queries import make_split

perm_tbl = make_split(tbl, train_ratio=0.8, seed=42, spec="nighttime_rider")
n_tr = perm_tbl.count_rows(filter="split_id = 0")
n_vl = perm_tbl.count_rows(filter="split_id = 1")
print(f"train={n_tr}  val={n_vl}  (seed=42, reproducible)")

# ── 5. Training smoke test ────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5 — Training smoke (1 epoch, nighttime+person, 100 train / 50 val)")
print("="*60)

from torch.utils.data import DataLoader, Subset
from object_detection.dataloader import LanceArrowDetectionDataset, _detection_collate
from object_detection.train_detector import build_model, train_one_epoch
from object_detection.eval import evaluate
from object_detection.schema import NUM_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_ds = LanceArrowDetectionDataset(
    DB_PATH, TABLE, where="split = 'train' AND timeofday = 'night' AND has_person = true"
)
val_ds = LanceArrowDetectionDataset(
    DB_PATH, TABLE, where="split = 'val' AND timeofday = 'night' AND has_person = true"
)
print(f"Nighttime+person — train: {len(train_ds):,}  val: {len(val_ds):,}")

train_loader = DataLoader(Subset(train_ds, range(100)), batch_size=2, collate_fn=_detection_collate)
val_loader   = DataLoader(Subset(val_ds,   range(50)),  batch_size=2, collate_fn=_detection_collate)

model = build_model(NUM_CLASSES, pretrained=True).to(device)

t0 = time.time()
avg_loss = train_one_epoch(model, torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9
), train_loader, device, epoch=1)
print(f"avg_loss={avg_loss:.4f}  ({time.time()-t0:.0f}s)")

metrics = evaluate(model, val_loader, device)
print(f"mAP@0.5={metrics['map_50']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")

print("\n" + "="*60)
print("E2E COMPLETE")
print("="*60)
