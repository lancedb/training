# leWorldModel × LanceDB

End-to-end training of [leWorldModel](https://github.com/lucas-maes/le-wm) (a JEPA-based world model) backed by LanceDB instead of HDF5.

```
examples/leWorldModel/
├── create_data.py          # HDF5 → LanceDB conversion + Geneva embedding backfill
├── train.py                # Trainer: LanceDB loaders, identical LeWM model/loss
├── eda_analysis.py         # EDA, quality scan, splits, vector search
├── config/
│   └── lewm_pusht.yaml     # Example config (copy and edit per dataset)
└── lewm_lance/
    ├── dataset.py          # LeWMLanceDataset — temporal window sampler
    └── dataloaders.py      # make_train_val_loaders() factory
```

---

## What is leWorldModel?

LeWM is a Joint Embedding Predictive Architecture (JEPA) that learns a world model from raw pixels with two losses:

- **Next-embedding prediction** — MSE between predicted and actual next latent state
- **SIGReg** (Sketch Isotropic Gaussian Regularizer) — keeps the latent space well-shaped

The model is ~15M parameters: a ViT-tiny encoder, an autoregressive predictor, and an action embedder. It trains stably on a single GPU.

The paper evaluates on four datasets independently — they are not mixed during training:

| Dataset | Env | Modalities | Config |
|---------|-----|-----------|--------|
| DMControl Reacher | `reacher` | pixels, action, observation | `lewm_reacher.yaml` |
| OGBench Cube | `cube_single_expert` | pixels, action, observation | `lewm_cube.yaml` |
| PushT | `pusht_expert_train` | pixels, action, proprio, state | `lewm_pusht.yaml` |
| TwoRoom | `tworoom` | pixels, action, proprio | `lewm_tworoom.yaml` |

---

## Hardware

LeWM is intentionally small (~15M params) and trains on a single GPU. These are practical recommendations:

| GPU | VRAM | batch_size | Notes |
|-----|------|-----------|-------|
| RTX 3090 / 4090 | 24 GB | 128 | Matches paper. ~4–6 hrs per dataset at 100 epochs. |
| A100 40 GB | 40 GB | 256 | 2× faster than 3090. Use if available. |
| A100 80 GB / H100 | 80 GB | 512 | Overkill for LeWM alone; useful if running multiple seeds in parallel. |
| RTX 3080 / 4070 | 10–12 GB | 64 | Reduce `batch_size` and `num_workers` to fit. Scale `lr` linearly. |

Training uses `bf16-mixed` precision throughout. If your GPU does not support bf16 (pre-Ampere), change `precision: "16-mixed"` in the config.

For the DataLoader, `num_workers=6` works well with a local LanceDB store. With S3-backed storage, increase to `num_workers=8–12` to overlap network I/O with GPU compute.

---

## Reproducing the paper

### Step 1 — Convert datasets

Download the datasets via `stable-worldmodel` and convert each to a LanceDB table.
All four datasets can share a single LanceDB store (each gets its own table).

```bash
# All four datasets into one local store
python create_data.py --dataset all --lance-uri ./lewm_lance

# Or one at a time
python create_data.py --dataset reacher  --lance-uri ./lewm_lance
python create_data.py --dataset cube     --lance-uri ./lewm_lance
python create_data.py --dataset pusht    --lance-uri ./lewm_lance
python create_data.py --dataset tworoom  --lance-uri ./lewm_lance

# S3-backed store (credentials via env or --aws-* flags)
python create_data.py --dataset all --lance-uri s3://my-bucket/lewm
```

This creates four tables: `lewm_reacher`, `lewm_cube`, `lewm_pusht`, `lewm_tworoom`.

### Step 2 — EDA and data quality check

Run this **before** training to catch any data issues and understand each dataset.
Uses DINOv2 embeddings (no training needed — frozen foundation model).

```bash
# Quality scan + statistics on each dataset
python eda_analysis.py --table lewm_pusht   --section quality
python eda_analysis.py --table lewm_reacher --section quality
python eda_analysis.py --table lewm_cube    --section quality
python eda_analysis.py --table lewm_tworoom --section quality

# Add DINOv2 embeddings for pre-training EDA (semantic search, clustering, dedup)
# Requires LanceDB Enterprise (Geneva) — skip if unavailable
python create_data.py --dataset all --embed --embedding-model dinov2

# Explore with embeddings
python eda_analysis.py --table lewm_pusht --section vector_search --emb-col emb_dinov2
python eda_analysis.py --table lewm_pusht --section entropy        # find diverse episodes
python eda_analysis.py --table lewm_pusht --section stats
```

### Step 3 — Train on each dataset

Each dataset is trained independently. Create a config per dataset by copying
`config/lewm_pusht.yaml` and updating `data.table_name` and `data.columns`.

```bash
# PushT
python train.py --config config/lewm_pusht.yaml

# Reacher — override table and columns directly without a separate config
python train.py --config config/lewm_pusht.yaml \
  --table-name lewm_reacher \
  --columns pixels action observation

# Cube
python train.py --config config/lewm_pusht.yaml \
  --table-name lewm_cube \
  --columns pixels action observation

# TwoRoom
python train.py --config config/lewm_pusht.yaml \
  --table-name lewm_tworoom \
  --columns pixels action proprio

# S3-backed store with explicit credentials
python train.py --config config/lewm_pusht.yaml \
  --lance-uri s3://my-bucket/lewm \
  --aws-region us-east-1 \
  --aws-access-key-id AKIA... \
  --aws-secret-access-key ...

# With credentials in environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION)
python train.py --config config/lewm_pusht.yaml --lance-uri s3://my-bucket/lewm
```

### Step 4 — Post-training analysis

After training, add LeWM encoder embeddings to the table for world-model-specific analysis.
These reflect what the trained model considers semantically similar — different from DINOv2.

```bash
python create_data.py \
  --dataset pusht \
  --embed \
  --embedding-model lewm \
  --checkpoint ./checkpoints/lewm_pusht_lewm_epoch_99_object.ckpt

# Compare DINOv2 vs LeWM similarity structure
python eda_analysis.py --table lewm_pusht --section vector_search --emb-col emb_lewm
python eda_analysis.py --table lewm_pusht --section retrieval     --emb-col emb_lewm
```

---

## Benchmarking dataloader throughput

`bench.py` measures raw dataloader throughput (samples/sec) for three backends
independently of GPU compute, so differences are purely from data loading.

```bash
# LanceDB S3 vs HDF5 local (put the HDF5 file in /dev/shm for best-case comparison)
python bench.py \
  --lance-uri s3://my-bucket/lewm \
  --table-name lewm_pusht \
  --hdf5-local /dev/shm/pusht.hdf5

# Add HDF5-from-S3 via s3fs (reads HDF5 directly from S3, no download)
python bench.py \
  --lance-uri s3://my-bucket/lewm \
  --table-name lewm_pusht \
  --hdf5-local /dev/shm/pusht.hdf5 \
  --hdf5-s3-key hdf5/pusht.hdf5 \
  --s3-bucket my-bucket

# Credentials via env vars (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION)
python bench.py --lance-uri s3://my-bucket/lewm --table-name lewm_pusht
```

Why the gap:

- **LanceDB S3 vs HDF5 local**: HDF5 serializes workers through a POSIX file lock — effective parallelism is ~1 worker regardless of `num_workers`. LanceDB workers hold independent connections with no locking.
- **HDF5 s3fs**: HDF5 makes many small random seeks per batch. Each seek over S3 becomes a separate HTTP range request. For temporal window reads (T=4 rows × multiple columns), this is dozens of round-trips per batch.

---

## How the temporal window sampler works

LeWM needs **contiguous T=4 frame windows** from the same episode per training sample.

`LeWMLanceDataset` precomputes valid window positions at init time:

1. Loads only `(episode_idx, step_idx)` into memory (~16 bytes/row — negligible even at millions of steps)
2. Checks all consecutive row pairs for same-episode + sequential step constraints
3. Stores the resulting `_window_starts` array (int64 numpy)

At training time, `__getitems__(window_indices)` fetches all **B×T rows in one `Permutation.__getitems__`** call and splits into per-sample dicts. No N×B individual lookups.

---

## Multi-worker safety

The LanceDB `Permutation` holds Rust async state that cannot be pickled. Each DataLoader worker gets a zeroed-out copy and lazily rebuilds its own connection:

```python
def __getstate__(self):
    state = self.__dict__.copy()
    state["_perm"] = None
    return state

def _ensure_open(self):
    if self._perm is None:
        db = lancedb.connect(self.uri, **self.connect_kwargs)
        self._perm = Permutation.identity(db.open_table(...))...
```

Combined with `multiprocessing_context="spawn"` and `persistent_workers=True`.

---

## LanceDB vs HDF5

| Feature | LanceDB | HDF5 |
|---------|---------|------|
| Multi-process reads | Yes (per-worker connection) | No (POSIX file lock) |
| Columnar partial reads | Native Arrow | Compound datasets only |
| Vector / ANN search | Built-in IVF-PQ | Not supported |
| SQL-like episode filters | `episode_idx = 42` | Loop + mask in Python |
| Cloud-native (S3/GCS) | Native, parallel | Download first |
| Schema evolution | Add columns in-place | Limited |
| Versioning / time-travel | Yes | No |
| Embedding storage | Native vector column | Separate dataset |
| Train/val split | Filter query, zero copy | Copy or index arrays |
| JPEG pixel compression | ~13× smaller than raw uint8 | Raw arrays only |
| Concurrent writers | Yes | No |

The key practical difference for LeWM training: HDF5 serializes all DataLoader workers through a single file lock, limiting effective parallelism to ~1–2 workers regardless of how many you spawn. LanceDB workers each hold their own connection with no contention.

---

## What else you can do with multimodal robotics data in LanceDB

1. **Pre-training data curation**: use DINOv2 ANN search to deduplicate near-identical episodes before spending GPU hours on them.
2. **Curriculum learning**: rank episodes by action entropy (`eda_analysis.py --section entropy`) and present easy→hard schedules during training.
3. **Goal-conditioned retrieval**: encode a goal frame, search `emb_lewm` to find the K nearest observed states — useful for reward shaping.
4. **Offline RL data mixing**: union multiple datasets in one table with a `dataset_name` column, filter at training time with no file management.
5. **Reward relabeling**: append a `reward` column after collection without rewriting pixel data.
6. **Active data collection**: stream new rollout episodes into the table while training runs — LanceDB concurrent writes are safe.
7. **Embedding visualization**: dump `emb_lewm` → UMAP to inspect the latent space structure the world model has learned.
