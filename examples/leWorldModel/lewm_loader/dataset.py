"""
LanceDB-backed PyTorch Dataset for leWorldModel temporal sequences.

leWorldModel trains on windows of T frames with a configurable frameskip (default 5,
matching the original le-wm paper):
  T    = history_size (3) + num_preds (1) = 4 frames
  span = T × frameskip = 20 raw rows per window

Frameskip mirrors the original HDF5Dataset behaviour:
  - Pixels:  sampled at stride frameskip → (T, C, H, W)
  - Actions: ALL span rows kept, reshaped to (T, frameskip × action_dim)
             This matches le-wm's effective_act_dim = frameskip × action_dim.
  - Other columns (proprio, state, observation): sampled at stride frameskip → (T, D)

Each dataset item is a dict of tensors:
  "pixels"  : (T, C, H, W)           float32  ImageNet-normalized
  "action"  : (T, frameskip×A)        float32  NaN→0
  "proprio" : (T, P)                  float32  [if present]
  ...

Design:
  - One LanceDB row = one raw timestep.
  - Window index is precomputed at __init__ so __getitems__ only does I/O.
  - Permutation object (Rust state) is zeroed before pickling; each worker
    lazily reopens its own connection inside _ensure_open().
  - __getitems__ batches the full (B × span) row fetch in a single Permutation
    call, then splits into per-sample dicts.
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


class LeWMLanceDataset(torch.utils.data.Dataset):
    """
    Temporal-window dataset backed by a LanceDB table.

    Args:
        uri:          LanceDB URI (local path or s3://…).
        table_name:   Name of the table created by create_data.py.
        columns:      List of column names to return.
        num_steps:    Window length T (= history_size + num_preds).
        frameskip:    Stride between sampled frames. Matches le-wm default of 5.
                      With frameskip=5 and T=4, each window spans 20 raw rows.
                      action is kept at full resolution and reshaped to
                      (T, frameskip × action_dim); all other columns are strided.
        img_size:     Target image size after resize.
        **connect_kwargs: Passed to lancedb.connect().
    """

    def __init__(
        self,
        uri: str,
        table_name: str,
        columns: list[str],
        num_steps: int = 4,
        frameskip: int = 5,
        img_size: int = 224,
        **connect_kwargs,
    ):
        self.uri         = uri
        self.table_name  = table_name
        self.columns     = columns
        self.num_steps   = num_steps
        self.frameskip   = frameskip
        self.img_size    = img_size
        self.connect_kwargs = connect_kwargs
        self._span = num_steps * frameskip   # raw rows per window

        self._perm:      Permutation | None      = None
        self._transform: transforms.Compose | None = None

        # Load only the two int32 index columns to precompute valid windows.
        # Pixels and all other data columns are never touched here.
        db  = lancedb.connect(uri, **connect_kwargs)
        tbl = db.open_table(table_name)
        idx = tbl.to_lance().to_table(columns=["episode_idx", "step_idx"])
        self._ep   = idx["episode_idx"].to_numpy().astype(np.int32)
        self._step = idx["step_idx"].to_numpy().astype(np.int32)
        self._n_rows = len(self._ep)

        # A window starting at row i is valid iff all span rows are in the same
        # episode with consecutive step indices.
        span = self._span
        N    = self._n_rows - span + 1
        valid = np.ones(N, dtype=bool)
        for offset in range(1, span):
            valid &= (self._ep[offset : N + offset]   == self._ep[:N])
            valid &= (self._step[offset : N + offset] == self._step[:N] + offset)
        self._window_starts = np.where(valid)[0].astype(np.int64)

    def __len__(self) -> int:
        return len(self._window_starts)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_perm"]      = None
        state["_transform"] = None
        return state

    def _ensure_open(self):
        if self._perm is None:
            db  = lancedb.connect(self.uri, **self.connect_kwargs)
            tbl = db.open_table(self.table_name)
            fetch_cols = ["pixels"] + [c for c in self.columns if c != "pixels"]
            self._perm = (
                Permutation.identity(tbl)
                .select_columns(fetch_cols)
                .with_format("arrow")
            )
            self._transform = _build_img_transform(self.img_size)

    def _rows_to_sample(self, batch: pa.RecordBatch) -> dict[str, torch.Tensor]:
        """
        Convert a RecordBatch of `span` raw rows into one training sample.

        Pixels and non-action columns: take every frameskip-th row → T frames.
        Action: keep all span rows, reshape to (T, frameskip × action_dim).
        """
        T         = self.num_steps
        frameskip = self.frameskip
        assert len(batch) == self._span

        # Pixels: stride by frameskip → T frames
        jpeg_list = batch["pixels"].to_pylist()
        frames = torch.stack(
            [_jpeg_to_tensor(jpeg_list[t * frameskip], self._transform) for t in range(T)]
        )
        sample: dict[str, torch.Tensor] = {"pixels": frames}

        for col in self.columns:
            if col == "pixels":
                continue

            if col == "action":
                # Keep all span rows, reshape to (T, frameskip × action_dim).
                # This matches le-wm's effective_act_dim = frameskip × raw_action_dim.
                data = np.array(batch.column(col).to_pylist(), dtype=np.float32)
                data = np.nan_to_num(data, nan=0.0)
                data = data.reshape(T, -1)
            else:
                # Proprio, state, observation: stride by frameskip → (T, D)
                data = np.array(batch.column(col).to_pylist(), dtype=np.float32)
                data = data[::frameskip]
                data = np.nan_to_num(data, nan=0.0)

            sample[col] = torch.from_numpy(data)

        return sample

    def __getitem__(self, window_idx: int) -> dict[str, torch.Tensor]:
        self._ensure_open()
        start = int(self._window_starts[window_idx])
        rows  = list(range(start, start + self._span))
        batch = self._perm.__getitems__(rows)
        return self._rows_to_sample(batch)

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]]:
        """
        Fetch an entire DataLoader batch in one round trip.

        Permutation.__getitems__ deduplicates row indices, so we cannot pass
        all B*span rows directly (overlapping windows would silently drop rows).
        Instead we:
          1. Collect the exact row ranges for each window.
          2. Deduplicate ourselves → sorted unique row list.
          3. Single Permutation fetch for those unique rows.
          4. Reconstruct each window by indexing into the fetched result.
        This reduces S3 round trips from B (one per sample) to 1 per batch.
        """
        self._ensure_open()

        # Step 1 — row ranges per window
        window_rows = [
            list(range(int(self._window_starts[i]), int(self._window_starts[i]) + self._span))
            for i in indices
        ]

        # Step 2 — unique sorted rows + reverse mapping
        all_rows   = sorted(set(r for rows in window_rows for r in rows))
        row_to_pos = {r: pos for pos, r in enumerate(all_rows)}

        # Step 3 — single fetch
        fetched = self._perm.__getitems__(all_rows)

        # Step 4 — reconstruct each window
        return [
            self._rows_to_sample(fetched.take([row_to_pos[r] for r in rows]))
            for rows in window_rows
        ]
