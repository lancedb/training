"""
DataLoader factories for leWorldModel LanceDB-backed training.

Two public functions:
  make_lewm_lance_loader()    – single loader (no split)
  make_train_val_loaders()    – episode-level train/val split, returns two loaders

Episode-level split (not random-row split) avoids data leakage:
  all timesteps of a given episode go entirely to train or entirely to val.
"""

import lancedb
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import LeWMLanceDataset


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
    Episode-level train/val split.

    val_fraction of episodes (randomly sampled, seeded) are held out for
    validation. All timesteps within an episode go entirely to one split —
    no row-level leakage between train and val.

    Returns:
        (train_loader, val_loader)
    """
    db  = lancedb.connect(uri, **connect_kwargs)
    tbl = db.open_table(table_name)

    # Only reads one int32 column — negligible memory even at millions of rows
    ep_arr = tbl.to_lance().to_table(columns=["episode_idx"])["episode_idx"].to_numpy()
    all_episodes = np.unique(ep_arr)

    rng = np.random.default_rng(seed)
    rng.shuffle(all_episodes)
    n_val = max(1, int(len(all_episodes) * val_fraction))
    val_episodes   = set(all_episodes[:n_val].tolist())
    train_episodes = set(all_episodes[n_val:].tolist())

    print(f"  Split: {len(train_episodes)} train episodes, {len(val_episodes)} val episodes")

    # Build full datasets then restrict _window_starts by episode membership.
    # Both datasets share the same table — no data is copied.
    train_ds = LeWMLanceDataset(uri, table_name, columns, num_steps, frameskip, img_size, **connect_kwargs)
    val_ds   = LeWMLanceDataset(uri, table_name, columns, num_steps, frameskip, img_size, **connect_kwargs)

    train_ep_mask = np.isin(train_ds._ep[train_ds._window_starts], list(train_episodes))
    val_ep_mask   = np.isin(val_ds._ep[val_ds._window_starts],   list(val_episodes))

    train_ds._window_starts = train_ds._window_starts[train_ep_mask]
    val_ds._window_starts   = val_ds._window_starts[val_ep_mask]

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
