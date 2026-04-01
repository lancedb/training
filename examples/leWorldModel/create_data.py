"""
Convert leWorldModel HDF5 datasets to LanceDB tables.

Each LanceDB row = one timestep (same granularity as the source HDF5).
See dataset.md for full format documentation.

COLLECTING THE DATASETS
-----------------------
Three datasets (reacher, pusht, tworoom) must be collected locally using the
stable-worldmodel expert scripts before converting.  The cube dataset is
downloaded automatically from HuggingFace (ogbench/cube_single_expert).

  # Collect reacher (~30 min on a single GPU with mujoco)
  python scripts/data/collect_dmc.py

  # Collect pusht
  python scripts/data/collect_pusht_fov.py   # or collect_weak_pusht.py

  # Collect tworoom
  python scripts/data/collect_tworooms.py

  # cube is auto-downloaded from HuggingFace when you run create_data.py

HDF5 files are written to $STABLEWM_HOME (default: ~/.stable_worldmodel/).

CONVERTING TO LANCEDB
---------------------
  # Convert all datasets to a local LanceDB store
  python create_data.py --dataset all --lance-uri ./lewm_lance

  # Convert a single dataset to S3-backed LanceDB
  python create_data.py --dataset pusht --lance-uri s3://my-bucket/lewm

  # Overwrite an existing table
  python create_data.py --dataset pusht --overwrite

  # Convert + back-fill embeddings for vector search
  python create_data.py --dataset pusht --embed --embedding-model dinov2
"""

import argparse
import io
import os
from collections.abc import Iterator

import h5py
import hdf5plugin  # noqa: F401 — registers HDF5 decompression filters (Blosc, Zstd, etc.)
import lancedb
import numpy as np
import pyarrow as pa
from PIL import Image
from stable_worldmodel.data.utils import get_cache_dir, load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    # swm_name is a HuggingFace repo id (owner/repo).
    # stable_worldmodel.data.load_dataset() downloads and caches the archive
    # automatically on first run — no manual download needed.
    # HF collection: https://huggingface.co/collections/quentinll/lewm
    "reacher": {
        "swm_name": "quentinll/lewm-reacher",
        "table_name": "lewm_reacher",
        "columns": ["pixels", "action", "observation"],
    },
    "cube": {
        "swm_name": "quentinll/lewm-cube",
        "table_name": "lewm_cube",
        "columns": ["pixels", "action", "observation"],
    },
    "pusht": {
        "swm_name": "quentinll/lewm-pusht",
        "table_name": "lewm_pusht",
        "columns": ["pixels", "action", "proprio", "state"],
    },
    "tworoom": {
        "swm_name": "quentinll/lewm-tworooms",
        "table_name": "lewm_tworoom",
        "columns": ["pixels", "action", "proprio"],
    },
}

JPEG_QUALITY = 95       # 95 → ~13× smaller than raw uint8, negligible quality loss
BATCH_ROWS = 1000       # rows per RecordBatch yielded to LanceDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_jpeg_bytes(frame: np.ndarray) -> bytes:
    """(C,H,W) or (H,W,C) uint8 ndarray → JPEG-compressed bytes."""
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))          # (C,H,W) → (H,W,C)
    buf = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8)).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _infer_episode_step(f: h5py.File, total: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (episode_idx_arr, step_idx_arr) as int32 arrays from an open HDF5 file.

    The stable-worldmodel HDF5 format stores per-episode metadata:
      ep_len    — int array of shape (n_episodes,) with the length of each episode
      ep_offset — int array of shape (n_episodes,) with the global start row of each episode

    These are expanded into per-row arrays for the LanceDB schema.
    """
    ep_len    = np.array(f["ep_len"],    dtype=np.int32)
    ep_offset = np.array(f["ep_offset"], dtype=np.int32)

    episode_idx = np.zeros(total, dtype=np.int32)
    step_idx    = np.zeros(total, dtype=np.int32)
    for i, (off, length) in enumerate(zip(ep_offset.tolist(), ep_len.tolist())):
        episode_idx[off : off + length] = i
        step_idx[off : off + length]    = np.arange(length, dtype=np.int32)

    return episode_idx, step_idx


def _build_schema(columns: list[str], dims: dict[str, int]) -> pa.Schema:
    """Build a PyArrow schema for a leWorldModel table."""
    fields = [
        pa.field("episode_idx", pa.int32()),
        pa.field("step_idx",    pa.int32()),
        pa.field("pixels",      pa.binary()),
        pa.field("pixels_h",    pa.int16()),
        pa.field("pixels_w",    pa.int16()),
    ]
    for col in columns:
        if col == "pixels":
            continue
        fields.append(pa.field(col, pa.list_(pa.float32(), dims[col])))
    return pa.schema(fields)


def _record_batch_reader(
    f: h5py.File,
    columns: list[str],
    episode_arr: np.ndarray,
    step_arr: np.ndarray,
    h: int,
    w: int,
    schema: pa.Schema,
) -> pa.RecordBatchReader:
    """
    Return a pa.RecordBatchReader that streams the HDF5 file in BATCH_ROWS chunks.

    Using a RecordBatchReader rather than repeated table.add() calls lets
    LanceDB write all data in a single pass through the file without accumulating
    large in-memory lists, and avoids creating many small Lance fragments.
    """
    total = len(episode_arr)
    non_pixel_cols = [c for c in columns if c != "pixels"]

    def _generate() -> Iterator[pa.RecordBatch]:
        # Buffers — reset every BATCH_ROWS rows
        ep_buf:   list[int]   = []
        st_buf:   list[int]   = []
        px_buf:   list[bytes] = []
        ph_buf:   list[int]   = []
        pw_buf:   list[int]   = []
        col_bufs: dict[str, list[list[float]]] = {c: [] for c in non_pixel_cols}

        for idx in tqdm(range(total), desc="  Converting", unit="step"):
            ep_buf.append(int(episode_arr[idx]))
            st_buf.append(int(step_arr[idx]))
            px_buf.append(_to_jpeg_bytes(np.array(f["pixels"][idx])))
            ph_buf.append(h)
            pw_buf.append(w)
            for col in non_pixel_cols:
                col_bufs[col].append(np.array(f[col][idx], dtype=np.float32).flatten().tolist())

            if len(ep_buf) == BATCH_ROWS:
                yield _make_batch(ep_buf, st_buf, px_buf, ph_buf, pw_buf, col_bufs, schema)
                ep_buf, st_buf, px_buf, ph_buf, pw_buf = [], [], [], [], []
                col_bufs = {c: [] for c in non_pixel_cols}

        if ep_buf:
            yield _make_batch(ep_buf, st_buf, px_buf, ph_buf, pw_buf, col_bufs, schema)

    return pa.RecordBatchReader.from_batches(schema, _generate())


def _make_batch(
    ep_buf:   list[int],
    st_buf:   list[int],
    px_buf:   list[bytes],
    ph_buf:   list[int],
    pw_buf:   list[int],
    col_bufs: dict[str, list[list[float]]],
    schema:   pa.Schema,
) -> pa.RecordBatch:
    arrays = [
        pa.array(ep_buf,  type=pa.int32()),
        pa.array(st_buf,  type=pa.int32()),
        pa.array(px_buf,  type=pa.binary()),
        pa.array(ph_buf,  type=pa.int16()),
        pa.array(pw_buf,  type=pa.int16()),
    ]
    for col in col_bufs:
        field_type = schema.field(col).type       # fixed_size_list<float32>[D]
        arrays.append(pa.array(col_bufs[col], type=field_type))
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_dataset(
    dataset_name: str,
    lance_uri: str,
    overwrite: bool = False,
    connect_kwargs: dict | None = None,
):
    cfg        = DATASETS[dataset_name]
    swm_name   = cfg["swm_name"]
    table_name = cfg["table_name"]
    columns    = cfg["columns"]
    connect_kwargs = connect_kwargs or {}

    # load_dataset() resolves HuggingFace repo ids: downloads the .tar.zst archive,
    # extracts it, caches the .h5 file under $STABLEWM_HOME, and returns an
    # HDF5Dataset with the resolved .h5_path.  Nothing to do manually.
    print(f"\n{'=' * 60}")
    print(f"Dataset : {dataset_name}  (swm_name={swm_name!r})")
    print(f"  Resolving HDF5 path via stable_worldmodel...")
    ds = load_dataset(swm_name)
    hdf5_path = ds.h5_path

    print(f"HDF5    : {hdf5_path}")
    print(f"Lance   : {lance_uri}  (table={table_name})")
    print(f"{'=' * 60}")

    db = lancedb.connect(lance_uri, **connect_kwargs)

    if table_name in db.table_names():
        if overwrite:
            print(f"  Dropping existing table '{table_name}'...")
            db.drop_table(table_name)
        else:
            print(f"  Table '{table_name}' already exists. Use --overwrite to recreate.")
            return

    with h5py.File(hdf5_path, "r") as f:
        total = len(f["pixels"])
        episode_arr, step_arr = _infer_episode_step(f, total)
        n_episodes = int(episode_arr.max()) + 1

        print(f"  Steps     : {total:,}")
        print(f"  Episodes  : {n_episodes:,}")

        # Determine vector dims from first row
        dims: dict[str, int] = {}
        for col in columns:
            if col == "pixels":
                continue
            dims[col] = int(np.array(f[col][0], dtype=np.float32).flatten().shape[0])
            print(f"  {col:<14}: dim={dims[col]}")

        # Determine pixel dimensions
        sample_frame = np.array(f["pixels"][0])
        if sample_frame.ndim == 3 and sample_frame.shape[0] in (1, 3, 4):
            _, h, w = sample_frame.shape
        else:
            h, w = sample_frame.shape[:2]
        print(f"  pixels      : ({h} × {w}) → JPEG quality={JPEG_QUALITY}")

        schema = _build_schema(columns, dims)
        reader = _record_batch_reader(f, columns, episode_arr, step_arr, h, w, schema)

        # Single create_table call — LanceDB reads from the reader in streaming fashion.
        # This produces one Lance fragment per BATCH_ROWS rows, then compacts.
        db.create_table(table_name, data=reader, schema=schema)

    final_count = len(db.open_table(table_name))
    print(f"  Done!  {final_count:,} rows written to '{table_name}'.")
    assert final_count == total, f"Row count mismatch: wrote {final_count}, expected {total}"


# ---------------------------------------------------------------------------
# Embedding back-fill via LanceDB Geneva (Enterprise feature engineering)
# ---------------------------------------------------------------------------
#
# WHEN to generate embeddings and WHICH model to use:
#
#   PRE-TRAINING (before training LeWM — for EDA, clustering, quality filtering):
#     Use a frozen foundation model: DINOv2 or CLIP.
#     The encoder requires no training — embeddings are semantically meaningful
#     from day one.  Good for:
#       - Clustering frames to discover sub-behaviours
#       - Detecting near-duplicate / degenerate episodes before wasting GPU time
#       - Curriculum design: embed → cluster → order episodes by difficulty
#       - Goal-state retrieval using natural-language queries (CLIP only)
#
#   POST-TRAINING (after training LeWM — for analysis of the learned world model):
#     Use the trained LeWM encoder (CLS token of the ViT).  The latent space now
#     reflects the dynamics your model has learned, not just visual similarity.
#     Good for:
#       - Validating that the encoder separates distinct behaviours
#       - ANN retrieval of states that "look the same to the world model"
#       - Debugging: find states the model consistently mispredicts
#
#   Using LeWM embeddings BEFORE training gives meaningless results —
#   the encoder is randomly initialised.
#
# HOW: LanceDB Geneva (LanceDB Enterprise)
#   Geneva's UDF API replaces the manual encode-loop-then-merge pattern with:
#     1. Define a stateful GPU UDF (@udf class with setup() + __call__())
#     2. Register it as a column:  tbl.add_columns({"emb_X": MyUDF()})
#     3. Backfill:                  tbl.backfill("emb_X", batch_size=32)
#        or async with progress:    tbl.backfill_async("emb_X", concurrency=4)
#   Geneva handles batching, GPU process concurrency, partial commits, and
#   incremental re-runs (where="emb_X is null") automatically.
#
# ---------------------------------------------------------------------------

def add_embeddings_geneva(
    lance_uri: str,
    table_name: str,
    model_name: str = "dinov2",
    checkpoint: str | None = None,
    batch_size: int = 32,
    img_size: int = 224,
    concurrency: int = 1,
    connect_kwargs: dict | None = None,
):
    """
    Add a frame embedding column to a LanceDB table using Geneva UDFs.

    Requires LanceDB Enterprise with the `geneva` package installed.
    Geneva handles batching, GPU concurrency, partial commits, and incremental
    re-runs — no manual encode loop needed.

    Args:
        model_name:   "dinov2" | "clip" | "lewm"
                      dinov2/clip → pre-training EDA (frozen foundation model)
                      lewm        → post-training analysis (requires checkpoint)
        checkpoint:   Path to a trained LeWM .ckpt file (only for model_name="lewm").
        batch_size:   Frames per UDF call (tune to fit GPU VRAM).
        concurrency:  Number of parallel GPU worker processes for backfill.
    """
    import geneva
    import pyarrow as pa

    connect_kwargs = connect_kwargs or {}
    conn = geneva.connect(lance_uri, **connect_kwargs)
    tbl  = conn.open_table(table_name)

    col_name = f"emb_{model_name}"
    if col_name in tbl.schema.names:
        print(f"  '{col_name}' column already present. Skipping.")
        return

    # Build the UDF class for the chosen model
    udf_cls = _make_embedding_udf(model_name, checkpoint, img_size)

    print(f"  Registering '{col_name}' UDF ({model_name})...")
    tbl.add_columns({col_name: udf_cls()})

    print(f"  Starting backfill (concurrency={concurrency}, batch_size={batch_size})...")
    fut = tbl.backfill(
        col_name,
        batch_size=batch_size,
        concurrency=concurrency,
        # Only process rows that don't have embeddings yet — safe to re-run
        where=f"{col_name} IS NULL",
    )

    print(f"  Backfill complete. Building IVF-PQ vector index on '{col_name}'...")
    tbl.create_index(
        column=col_name,
        index_type="IVF_PQ",
        num_partitions=64,
        num_sub_vectors=16,
    )
    print(f"  Done! ANN search: tbl.search(vec, vector_column_name='{col_name}').limit(10)")


def _make_embedding_udf(model_name: str, checkpoint: str | None, img_size: int):
    """
    Return a Geneva UDF class that encodes JPEG-binary frames with the chosen model.

    Geneva UDFs are stateful classes:
      setup()      — called once per worker process to load model weights
      __call__()   — called per row (or per batch if batch_size > 1)

    The UDF is decorated with @geneva.udf(data_type=...) to declare the output
    Arrow type so Geneva can build the schema before any data is processed.
    """
    import geneva
    import io as _io
    import numpy as _np
    from PIL import Image as _Image
    from torchvision import transforms as _transforms

    _transform = _transforms.Compose([
        _transforms.Resize((img_size, img_size)),
        _transforms.ToTensor(),
        _transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if model_name == "dinov2":
        EMBED_DIM = 384

        @geneva.udf(data_type=pa.list_(pa.float32(), EMBED_DIM))
        class DINOv2Embedder:
            def setup(self):
                import timm, torch
                self.model = timm.create_model(
                    "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0
                ).cuda().eval()
                self.torch = torch

            def __call__(self, pixel_bytes: bytes) -> list[float]:
                img = _Image.open(_io.BytesIO(pixel_bytes)).convert("RGB")
                t = _transform(img).unsqueeze(0).cuda()
                with self.torch.no_grad():
                    return self.model(t)[0].cpu().tolist()

        return DINOv2Embedder

    if model_name == "clip":
        EMBED_DIM = 512

        @geneva.udf(data_type=pa.list_(pa.float32(), EMBED_DIM))
        class CLIPEmbedder:
            def setup(self):
                import clip, torch
                self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
                self.model.eval()
                self.torch = torch

            def __call__(self, pixel_bytes: bytes) -> list[float]:
                img = _Image.open(_io.BytesIO(pixel_bytes)).convert("RGB")
                t = self.preprocess(img).unsqueeze(0).cuda()
                with self.torch.no_grad():
                    return self.model.encode_image(t)[0].cpu().float().tolist()

        return CLIPEmbedder

    if model_name == "lewm":
        assert checkpoint, "--checkpoint is required for --embedding-model lewm"
        _ckpt = checkpoint

        @geneva.udf(data_type=pa.list_(pa.float32(), 192))   # ViT-tiny embed_dim
        class LeWMEmbedder:
            def setup(self):
                import torch
                model = torch.load(_ckpt, map_location="cuda")
                model.eval()
                self.encoder = model.encoder
                self.torch = torch

            def __call__(self, pixel_bytes: bytes) -> list[float]:
                img = _Image.open(_io.BytesIO(pixel_bytes)).convert("RGB")
                t = _transform(img).unsqueeze(0).cuda()
                with self.torch.no_grad():
                    out = self.encoder(t)
                    return out[0, 0, :].cpu().tolist()   # CLS token

        return LeWMEmbedder

    raise ValueError(f"Unknown model_name: {model_name!r}. Choose dinov2 | clip | lewm")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert leWorldModel HDF5 datasets to LanceDB tables"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset to convert (default: all)",
    )
    parser.add_argument(
        "--lance-uri",
        default="./lewm_lance",
        help="LanceDB URI — local path or s3://bucket/prefix (default: ./lewm_lance)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Drop and recreate existing tables",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Add an embedding column for vector search after conversion",
    )
    parser.add_argument(
        "--embedding-model",
        choices=["clip", "dinov2", "lewm"],
        default="dinov2",
        help=(
            "Which vision model to use for embeddings.\n"
            "  dinov2 — Meta DINOv2 ViT-S/14, pre-trained (best for pre-training EDA)\n"
            "  clip   — OpenAI CLIP ViT-B/32, pre-trained (supports text queries)\n"
            "  lewm   — Trained LeWM encoder (post-training analysis only, needs --checkpoint)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        metavar="CKPT",
        help="Path to a trained LeWM object checkpoint (required when --embedding-model=lewm)",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=256,
        help="Frames per GPU forward pass during embedding generation (default: 256)",
    )
    args = parser.parse_args()

    datasets_to_run = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for ds_name in datasets_to_run:
        convert_dataset(
            dataset_name=ds_name,
            lance_uri=args.lance_uri,
            overwrite=args.overwrite,
        )

    if args.embed:
        for ds_name in datasets_to_run:
            add_embeddings_geneva(
                lance_uri=args.lance_uri,
                table_name=DATASETS[ds_name]["table_name"],
                model_name=args.embedding_model,
                checkpoint=args.checkpoint,
                batch_size=args.embed_batch_size,
            )

    print("\nAll done!")


if __name__ == "__main__":
    main()
