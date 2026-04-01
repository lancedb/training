# Dataset Format: HDF5 vs LanceDB

This document describes the original leWorldModel HDF5 dataset format, what we
store in LanceDB, and why the two differ in specific ways.

---

## Original HDF5 Format

leWorldModel datasets are produced by the `stable-worldmodel` package and
downloaded into `$STABLEWM_HOME` (default: `~/.stable-wm/`).  Each dataset is
a **single monolithic HDF5 file** (e.g. `pusht_expert_train.hdf5`).

### Structure

HDF5 stores data as **flat arrays at the file root** — one row per timestep,
all episodes concatenated sequentially.  There is no nesting.

```
pusht_expert_train.hdf5
├── pixels          (N, C, H, W)   uint8        — raw pixel frames
├── action          (N, A)         float32      — continuous action vectors
├── proprio         (N, P)         float32      — proprioceptive state
├── state           (N, S)         float32      — full simulator state
├── episode_idx     (N,)           int32        — which episode each row belongs to
└── step_idx        (N,)           int32        — step counter within episode
```

`N` = total number of timesteps across all episodes. Episodes are contiguous:
all rows for episode 0 come first, then episode 1, etc.

`episode_idx` and `step_idx` are index columns, not observation data — they
exist purely to let you reconstruct episode boundaries.

### Per-dataset column map

| Dataset file                  | pixels shape      | Columns present                           |
|-------------------------------|-------------------|-------------------------------------------|
| `reacher.hdf5`                | (N, 3, H, W)      | pixels, action, observation               |
| `cube_single_expert.hdf5`     | (N, 3, H, W)      | pixels, action, observation               |
| `pusht_expert_train.hdf5`     | (N, 3, H, W)      | pixels, action, proprio, state            |
| `tworoom.hdf5`                | (N, 3, H, W)      | pixels, action, proprio                   |

`H` and `W` vary by dataset but are typically 64 or 84 pixels in the raw files;
`train.py` resizes to 224×224 at load time.

### Episode boundaries

The `stable_worldmodel.data.HDF5Dataset` class reconstructs episode windows by
scanning `episode_idx` and building a mapping from `(episode, local_step)` to
the global row index `i`.  Training samples T consecutive rows within the same
episode:

```
global_row = episode_start[ep] + local_step
window     = [global_row, global_row+1, ..., global_row+T-1]
```

---

## LanceDB Format

### What changes — and what stays the same

The row granularity is **identical**: one LanceDB row = one HDF5 row = one
timestep.  We do not reshape, aggregate, or split the data differently.

What does change:

| Aspect | HDF5 | LanceDB |
|--------|------|---------|
| **Storage layout** | Monolithic `.h5` file | Columnar Lance fragments (local or S3) |
| **Pixel encoding** | Raw uint8 (C, H, W) | JPEG-compressed binary (quality=95) |
| **Vector columns** | float32 ndarrays | `fixed_size_list<float32>` in Arrow schema |
| **Index columns** | `episode_idx`, `step_idx` arrays | Same, stored as int32 Arrow columns |
| **Reads** | `h5py.File[col][i]` — single-threaded | `Permutation.__getitems__` — parallel, multi-worker safe |
| **Partial column reads** | Requires loading the full compound dataset | Native: `.select_columns(["action"])` reads only that column |

### Schema (example: pusht)

```
episode_idx   int32
step_idx      int32
pixels        binary                       ← JPEG bytes (not raw uint8)
pixels_h      int16                        ← stored so decode knows H
pixels_w      int16                        ← stored so decode knows W
action        fixed_size_list<float32>[A]
proprio       fixed_size_list<float32>[P]
state         fixed_size_list<float32>[S]
```

After running `create_data.py --embed`, embedding columns are appended:

```
emb_dinov2    fixed_size_list<float32>[384]  ← DINOv2 ViT-S/14 (pre-training EDA)
emb_clip      fixed_size_list<float32>[512]  ← CLIP ViT-B/32    (text queries)
emb_lewm      fixed_size_list<float32>[192]  ← Trained LeWM CLS token (post-training)
```

None of these columns are in the original HDF5 files.

### Why JPEG for pixels, and does it hurt training speed?

**The storage case** is clear: raw uint8 RGB at 224×224 is 150 KB per frame.
At JPEG quality=95 the same frame compresses to ~10-15 KB — a 10-15× reduction
with negligible perceptual quality loss for vision model training.

**The compute tradeoff** is real but net-positive when combined with
LanceDB's multi-worker DataLoader:

| | Raw uint8 in HDF5 | JPEG binary in LanceDB |
|--|--|--|
| Bytes transferred from storage per frame | 150 KB (raw float32: 600 KB) | 12 KB |
| I/O time (NVMe, 3 GB/s) per 1000 frames | ~50 ms | ~4 ms |
| JPEG decode time per 1000 frames (CPU) | — | ~30 ms (libturbo) |
| Decode happens in | main process | DataLoader worker (parallel) |

The decode cost (~30 ms/1000 frames on a single CPU core) is hidden because:

1. **Parallel decoding across workers.** With `num_workers=6` each worker
   decodes its own subset of JPEG frames.  The GPU training step and worker
   decoding overlap in time — decoding is not on the critical path.

2. **I/O dominates for large datasets, especially on S3.** When data lives on
   S3 (which is the point of using LanceDB), the network round-trip to fetch
   150 KB vs 12 KB per frame is 12× slower.  JPEG decode is cheap compared to
   cross-AZ network I/O.

3. **HDF5 single-threaded read negates its "no decode" advantage.** HDF5 reads
   are serialized through a file lock.  Even though there's no decode step, all
   8 DataLoader workers queue behind the same lock.  In practice the leWorldModel
   HDF5 pipeline runs roughly 2-3 workers effectively regardless of how many you
   spawn.  LanceDB workers never contend with each other.

**When raw uint8 would be better:** if your entire dataset fits in RAM and you
use a RAM disk, raw storage avoids the decode cost.  For datasets that fit in
`/dev/shm` (tens of GB) this is a valid strategy.  LanceDB supports this too —
you can store raw `pa.binary()` frames and handle encode/decode yourself.

**Bottom line:** for typical leWorldModel dataset sizes (gigabytes to tens of
gigabytes), training on JPEG-compressed LanceDB tables matches or exceeds the
throughput of HDF5, because the 8× I/O reduction and parallelism gains exceed
the added decode cost.  The ViT MFU benchmarks in `examples/ViT/` confirm this:
LanceDB with JPEG achieves 37-39% MFU on H200 vs 13% for raw S3 object storage.

### Why `fixed_size_list` instead of a flat float column?

Arrow `fixed_size_list<float32>[D]` lets you:
- Read an entire column as a 2D numpy array in one zero-copy call
- Use it as a vector column for ANN indexing after adding embeddings
- Keep schema self-describing (dimension D is part of the type)

A plain `list<float32>` (variable-length) would also work but is slower to
convert to numpy because Arrow cannot guarantee contiguous memory layout.

---

## Episode Boundary Handling

Both HDF5 and LanceDB store episodes as contiguous row ranges.

`LeWMLanceDataset` reconstructs valid window positions at init time by loading
`(episode_idx, step_idx)` into two numpy arrays (~16 bytes/row, trivial even at
1M steps) and checking:

```python
same_ep = episode_idx[i+offset] == episode_idx[i]        # same episode
consec  = step_idx[i+offset] == step_idx[i] + offset     # consecutive steps
valid[i] = all(same_ep and consec for offset in 1..T)
```

This is equivalent to what `HDF5Dataset` does internally, but exposed
explicitly so the dataset object knows all valid start rows upfront.

The precomputed `_window_starts` array is a plain int64 numpy array stored on
the dataset object — it is pickled safely to DataLoader workers.

---

## What is NOT changed

- **Training sample format**: each sample is still `{pixels: (T,C,H,W), action: (T,A), ...}`
- **Normalization**: z-score per column, computed on train episodes only (same as `get_column_normalizer` in `le-wm/utils.py`)
- **Image preprocessing**: ImageNet normalization + resize to 224×224 (same as `get_img_preprocessor`)
- **Episode-level train/val split**: 90/10 by default with fixed seed
- **NaN handling**: `nan_to_num(nan=0.0)` on action boundaries (same as original)

The LanceDB pipeline is a direct structural equivalent of the HDF5 pipeline
with a different I/O backend.
