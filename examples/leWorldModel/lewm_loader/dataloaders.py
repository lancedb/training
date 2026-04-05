"""
DataLoader factories for leWorldModel LanceDB-backed training.

Two public functions:
  make_lewm_lance_loader()    – single loader (no split)
  make_train_val_loaders()    – random window train/val split, returns two loaders
"""

import lancedb
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import LeWMLanceDataset


def _update_running_stats(entry: dict, data: np.ndarray) -> None:
    """Numerically stable running mean/std update (per-dimension)."""

    if data.size == 0:
        return

    batch_count = data.shape[0]
    batch_mean = data.mean(axis=0, dtype=np.float64)
    batch_m2 = ((data - batch_mean) ** 2).sum(axis=0, dtype=np.float64)

    if entry["count"] == 0:
        entry["count"] = batch_count
        entry["mean"] = batch_mean
        entry["m2"] = batch_m2
        return

    total = entry["count"] + batch_count
    delta = batch_mean - entry["mean"]
    entry["mean"] = entry["mean"] + delta * (batch_count / total)
    entry["m2"] = (
        entry["m2"]
        + batch_m2
        + (delta**2) * entry["count"] * batch_count / total
    )
    entry["count"] = total


def _compute_column_normalizers(
    uri: str,
    table_name: str,
    columns: list[str],
    train_episodes: set[int] | None,
    connect_kwargs: dict,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute per-column (mean,std) stats on selected episodes (or all)."""

    norm_cols = [c for c in columns if c != "pixels"]
    if not norm_cols:
        return {}

    db = lancedb.connect(uri, **connect_kwargs)
    tbl = db.open_table(table_name)
    lance_ds = tbl.to_lance()
    scanner = lance_ds.scanner(
        columns=["episode_idx", *norm_cols],
        batch_size=8192,
    )

    stats = {col: {"count": 0, "mean": None, "m2": None} for col in norm_cols}
    episode_ids = (
        np.array(sorted(train_episodes), dtype=np.int32)
        if train_episodes is not None
        else None
    )

    for batch in scanner.to_batches():
        ep = np.array(batch["episode_idx"].to_pylist(), dtype=np.int32)
        if episode_ids is None:
            mask = np.ones_like(ep, dtype=bool)
        else:
            mask = np.isin(ep, episode_ids)
        if not mask.any():
            continue

        for col in norm_cols:
            arr = np.array(batch[col].to_pylist(), dtype=np.float32)
            arr = arr[mask]
            if arr.ndim == 1:
                arr = arr[:, None]
            arr = arr[~np.isnan(arr).any(axis=1)]
            if arr.size == 0:
                continue
            _update_running_stats(stats[col], arr)

    normalizers: dict[str, dict[str, np.ndarray]] = {}
    for col, entry in stats.items():
        if entry["count"] == 0:
            continue
        mean = entry["mean"].astype(np.float32)
        if entry["count"] > 1:
            var = entry["m2"] / (entry["count"] - 1)
        else:
            var = np.ones_like(mean, dtype=np.float64)
        std = np.sqrt(var).astype(np.float32)
        std = np.where(std > 1e-6, std, np.ones_like(std))
        normalizers[col] = {"mean": mean, "std": std}

    return normalizers


# ---------------------------------------------------------------------------
# Collate: list[{key: (T,...) tensor}] → {key: (B,T,...) tensor}
# ---------------------------------------------------------------------------

def _lewm_collate(samples: list[dict]) -> dict:
    keys = samples[0].keys()
    return {k: torch.stack([s[k] for s in samples], dim=0) for k in keys}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_lewm_lance_loader(
    uri: str,
    table_name: str,
    columns: list[str],
    batch_size: int,
    num_steps: int = 4,
    frameskip: int = 5,
    img_size: int = 224,
    num_workers: int = 6,
    prefetch_factor: int = 3,
    shuffle: bool = False,
    **connect_kwargs,
) -> DataLoader:
    """
    Build a single DataLoader over a LanceDB leWorldModel table.

    frameskip=5 matches the le-wm paper default. With T=4 and frameskip=5,
    each window spans 20 raw rows; action is reshaped to (T, 5×action_dim).
    """
    dataset = LeWMLanceDataset(
        uri=uri,
        table_name=table_name,
        columns=columns,
        num_steps=num_steps,
        frameskip=frameskip,
        img_size=img_size,
        **connect_kwargs,
    )
    return _build_loader(dataset, batch_size, num_workers, prefetch_factor, shuffle=shuffle)


def make_train_val_loaders(
    uri: str,
    table_name: str,
    columns: list[str],
    batch_size: int,
    num_steps: int = 4,
    frameskip: int = 5,
    img_size: int = 224,
    num_workers: int = 6,
    prefetch_factor: int = 3,
    val_fraction: float = 0.1,
    seed: int = 42,
    **connect_kwargs,
) -> tuple[DataLoader, DataLoader]:
    """
    Random window train/val split (matches le-wm Hydra config).

    Returns:
        (train_loader, val_loader)
    """

    print("  Computing column normalizers (all episodes)...")
    normalizers = _compute_column_normalizers(
        uri=uri,
        table_name=table_name,
        columns=columns,
        train_episodes=None,
        connect_kwargs=connect_kwargs,
    )
    for col, stats in normalizers.items():
        print(f"    {col}: mean={stats['mean'].tolist()}, std={stats['std'].tolist()}")

    base_ds = LeWMLanceDataset(
        uri,
        table_name,
        columns,
        num_steps,
        frameskip,
        img_size,
        normalizers=normalizers,
        **connect_kwargs,
    )

    total_windows = len(base_ds)
    n_val = max(1, int(total_windows * val_fraction))
    n_train = total_windows - n_val
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_windows)
    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train:])

    train_ds = base_ds
    train_ds._window_starts = train_ds._window_starts[train_idx]

    val_ds = LeWMLanceDataset(
        uri,
        table_name,
        columns,
        num_steps,
        frameskip,
        img_size,
        normalizers=normalizers,
        **connect_kwargs,
    )
    val_ds._window_starts = val_ds._window_starts[val_idx]

    print(f"  Windows: {len(train_ds):,} train, {len(val_ds):,} val")

    return (
        _build_loader(train_ds, batch_size, num_workers, prefetch_factor, shuffle=True),
        _build_loader(val_ds,   batch_size, num_workers, prefetch_factor, shuffle=False),
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _build_loader(
    dataset: LeWMLanceDataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_lewm_collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
