"""
LanceDB-backed PyTorch Dataset for leWorldModel temporal sequences.

leWorldModel trains on windows of T consecutive frames from the same episode:
  T = history_size (3) + num_preds (1) = 4  by default

Each dataset item is a dict of tensors:
  "pixels"  : (T, C, H, W)  float32  ImageNet-normalized
  "action"  : (T, A)         float32  z-score normalized, NaN→0
  "proprio" : (T, P)         float32  z-score normalized      [if present]
  ...

Design:
  - One LanceDB row  = one timestep.
  - Window index is precomputed at __init__ so __getitems__ only does I/O.
  - Permutation object (Rust state) is zeroed before pickling; each worker
    lazily reopens its own connection inside _ensure_open().
  - __getitems__ batches the full (B*T) row fetch in a single Permutation call,
    then splits into per-sample dicts — same pattern as ViT dataloaders.py.
"""

import io

import lancedb
import numpy as np
import pyarrow as pa
import torch
from lancedb.permutation import Permutation
from PIL import Image
from torchvision import transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _build_img_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _jpeg_to_tensor(jpeg_bytes: bytes, transform: transforms.Compose) -> torch.Tensor:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return transform(img)


def compute_normalizers(
    uri: str,
    table_name: str,
    columns: list[str],
    **connect_kwargs,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Compute per-column (mean, std) arrays for z-score normalization.

    Reads only the requested non-pixel columns using LanceDB's column projection
    so no pixel data is loaded.  For large datasets this streams in batches via
    the Arrow scanner rather than materializing the full table at once.

    Returns:
      {col: (mean_array, std_array)}  — each array has shape (D,).
    """
    db = lancedb.connect(uri, **connect_kwargs)
    tbl = db.open_table(table_name)
    non_pixel = [c for c in columns if c != "pixels"]
    if not non_pixel:
        return {}

    # Use column-projected Arrow read — loads only the requested columns.
    # episode_idx + step_idx columns are tiny; the vector columns are float32
    # lists already in Arrow format, so this is as efficient as possible.
    arrow = tbl.to_arrow(columns=non_pixel)
    normalizers = {}
    for col in non_pixel:
        data = np.stack([row.as_py() for row in arrow[col]], axis=0).astype(np.float32)
        valid = ~np.isnan(data).any(axis=1)
        data = data[valid]
        normalizers[col] = (data.mean(axis=0), data.std(axis=0))
    return normalizers


class LeWMLanceDataset(torch.utils.data.Dataset):
    """
    Temporal-window dataset backed by a LanceDB table.

    Args:
        uri:          LanceDB URI (local path or s3://…).
        table_name:   Name of the table created by create_data.py.
        columns:      List of column names to return, e.g. ["pixels","action","proprio"].
        num_steps:    Window length T (= history_size + num_preds).
        img_size:     Target image size after resize (square).
        normalizers:  Output of compute_normalizers(); used to z-score non-pixel columns.
        **connect_kwargs: Passed to lancedb.connect() (api_key, host_override, region, …).
    """

    def __init__(
        self,
        uri: str,
        table_name: str,
        columns: list[str],
        num_steps: int = 4,
        img_size: int = 224,
        normalizers: dict | None = None,
        **connect_kwargs,
    ):
        self.uri = uri
        self.table_name = table_name
        self.columns = columns
        self.num_steps = num_steps
        self.img_size = img_size
        self.normalizers = normalizers or {}
        self.connect_kwargs = connect_kwargs

        # Rust Permutation — zeroed before pickling, rebuilt per-worker
        self._perm: Permutation | None = None
        self._transform: transforms.Compose | None = None

        # ------------------------------------------------------------------
        # Eagerly load the episode/step index to precompute valid windows.
        # These are only two int32 columns — ~8 bytes/row regardless of
        # dataset size, so loading them fully is fine.
        # ------------------------------------------------------------------
        db = lancedb.connect(uri, **connect_kwargs)
        tbl = db.open_table(table_name)
        idx_arrow = tbl.to_arrow(columns=["episode_idx", "step_idx"])
        self._ep   = idx_arrow["episode_idx"].to_numpy().astype(np.int32)
        self._step = idx_arrow["step_idx"].to_numpy().astype(np.int32)
        self._n_rows = len(self._ep)

        # Precompute valid window start rows.
        # A window starting at row i is valid iff rows i..i+T-1 are all in
        # the same episode and have consecutive step indices.
        T = num_steps
        N = self._n_rows - T + 1
        valid = np.ones(N, dtype=bool)
        for offset in range(1, T):
            same_ep = self._ep[offset : N + offset] == self._ep[:N]
            consec  = self._step[offset : N + offset] == self._step[:N] + offset
            valid &= same_ep & consec

        # _window_starts[i] = absolute row index for the i-th valid window
        self._window_starts = np.where(valid)[0].astype(np.int64)

    # ---------------------------------------------------------------------- #
    # PyTorch Dataset protocol
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._window_starts)

    def __getstate__(self) -> dict:
        """Zero out Rust state before the object is pickled for a worker process."""
        state = self.__dict__.copy()
        state["_perm"] = None
        state["_transform"] = None
        return state

    def _ensure_open(self):
        """Lazily open DB connection + Permutation once per worker process."""
        if self._perm is None:
            db = lancedb.connect(self.uri, **self.connect_kwargs)
            tbl = db.open_table(self.table_name)
            fetch_cols = ["pixels"] + [c for c in self.columns if c != "pixels"]
            self._perm = (
                Permutation.identity(tbl)
                .select_columns(fetch_cols)
                .with_format("arrow")
            )
            self._transform = _build_img_transform(self.img_size)

    # ---------------------------------------------------------------------- #
    # Internal: convert a RecordBatch of T rows into a sample dict
    # ---------------------------------------------------------------------- #

    def _rows_to_sample(self, batch: pa.RecordBatch) -> dict[str, torch.Tensor]:
        T = self.num_steps
        assert len(batch) == T

        # Decode JPEG pixels → (T, C, H, W)
        jpeg_list = batch["pixels"].to_pylist()
        frames = torch.stack([_jpeg_to_tensor(b, self._transform) for b in jpeg_list])
        sample: dict[str, torch.Tensor] = {"pixels": frames}

        for col in self.columns:
            if col == "pixels":
                continue
            data = np.array([batch[col][t].as_py() for t in range(T)], dtype=np.float32)
            if col in self.normalizers:
                mean, std = self.normalizers[col]
                data = (data - mean) / (std + 1e-8)
            data = np.nan_to_num(data, nan=0.0)
            sample[col] = torch.from_numpy(data)

        return sample

    # ---------------------------------------------------------------------- #
    # Single-item access (used when num_workers=0)
    # ---------------------------------------------------------------------- #

    def __getitem__(self, window_idx: int) -> dict[str, torch.Tensor]:
        self._ensure_open()
        start = int(self._window_starts[window_idx])
        rows = list(range(start, start + self.num_steps))
        batch = self._perm.__getitems__(rows)
        return self._rows_to_sample(batch)

    # ---------------------------------------------------------------------- #
    # Batch access — called by DataLoader with num_workers > 0.
    # Fetches all B*T rows in ONE Permutation call instead of B calls.
    # ---------------------------------------------------------------------- #

    def __getitems__(self, window_indices: list[int]) -> list[dict[str, torch.Tensor]]:
        self._ensure_open()
        T = self.num_steps
        starts = self._window_starts[window_indices]          # (B,)

        # Build flat list: [w0_t0, w0_t1, …, w1_t0, w1_t1, …]
        all_rows: list[int] = []
        for s in starts:
            all_rows.extend(range(int(s), int(s) + T))

        big_batch: pa.RecordBatch = self._perm.__getitems__(all_rows)  # (B*T, cols)

        samples = []
        for b in range(len(window_indices)):
            row_slice = big_batch.slice(b * T, T)
            samples.append(self._rows_to_sample(row_slice))
        return samples
