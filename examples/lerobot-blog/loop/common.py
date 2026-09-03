"""Shared pieces for the curation loop: dataset construction, policy loading, batched scoring.

Everything reads the Lance dataset through lerobot's own ``LeRobotDataset`` (storage_format
"lance"), so the frames scored here are exactly the frames the trainer sees.
"""
from __future__ import annotations

import json
import os
import time

MP_CTX = os.environ.get("MP_CTX", "spawn")  # DataLoader worker start method; fork deadlocks with Lance threads

import numpy as np
import torch

REPO_ID = "lerobot/droid_1.0.1"


def env_root() -> str:
    root = os.environ.get("LANCE_ROOT")
    if not root:
        raise SystemExit("set LANCE_ROOT to the dataset root (local dir or s3:// URI)")
    return root


def load_rename_map() -> dict:
    return json.load(open(os.environ.get("RENAME_MAP", "config/rename_map.json")))


def load_policy(ckpt_path: str, meta, rename_map: dict, device: str = "cuda"):
    """Policy + preprocessor + delta timestamps for one checkpoint (a ``pretrained_model`` dir)."""
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401  (registers the config)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(ckpt_path)
    cfg.pretrained_path = ckpt_path
    delta_ts = resolve_delta_timestamps(cfg, meta, rename_map)
    pre, _ = make_pre_post_processors(cfg, pretrained_path=ckpt_path)
    policy = make_policy(cfg, ds_meta=meta, rename_map=rename_map).to(device).eval()
    return policy, pre, delta_ts, cfg


def action_stats(ckpt_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Action normalization statistics baked into a checkpoint's preprocessor."""
    import glob

    from safetensors.torch import load_file

    files = glob.glob(os.path.join(ckpt_path, "*normalizer_processor.safetensors"))
    if not files:
        raise SystemExit(f"no normalizer file under {ckpt_path}")
    st = load_file(files[0])
    if "action.mean" not in st:
        raise SystemExit(f"{ckpt_path} carries no `action.mean`; it was not trained on this dataset")
    return st["action.mean"].float().numpy().ravel(), st["action.std"].float().numpy().ravel()


def assert_same_stats(paths: list[str]) -> None:
    ref = action_stats(paths[0])
    for p in paths[1:]:
        m, s = action_stats(p)
        if not (np.allclose(m, ref[0]) and np.allclose(s, ref[1])):
            raise SystemExit(f"ABORT: action normalization differs between {paths[0]} and {p}; "
                             "MAEs would be in different units")


def open_dataset(root: str, episodes: list[int], delta_ts: dict, tolerance_s: float = 5e-3):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(REPO_ID, root=root, episodes=sorted(episodes), delta_timestamps=delta_ts,
                          return_uint8=True, tolerance_s=tolerance_s)


def load_meta(root: str):
    from lerobot.datasets.storage import load_dataset_metadata

    return load_dataset_metadata(REPO_ID, root=root)


def episode_lengths(meta) -> dict[int, int]:
    d = meta.episodes.data
    frm = d.column("dataset_from_index").to_numpy()
    to = d.column("dataset_to_index").to_numpy()
    eps = d.column("episode_index").to_numpy()
    return {int(e): int(b - a) for e, a, b in zip(eps, frm, to)}


def relative_indices(ds, episodes: list[int], stride: int = 1, per_episode: int | None = None) -> list[int]:
    """Dataset-relative indices covering ``episodes`` at ``stride`` (or ``per_episode`` spread points)."""
    reader = ds._reader if hasattr(ds, "_reader") else ds.reader
    rel_to_abs = reader._rel_to_abs
    abs_to_rel = reader._absolute_to_relative_idx
    frm = reader._ep_from
    to = reader._ep_to
    out = []
    for ep in sorted(episodes):
        a, b = int(frm[ep]), int(to[ep])
        if per_episode is not None:
            n = min(per_episode, b - a)
            absi = np.linspace(a, b - 1, n).astype(int)
        else:
            absi = np.arange(a, b, stride)
        out.extend(int(abs_to_rel[int(x)]) for x in absi)
    assert rel_to_abs is not None
    return out


class BatchScorer:
    """Deterministic per-sample action error of one policy on preprocessed batches."""

    def __init__(self, policy, cfg, device: str = "cuda"):
        self.policy, self.cfg, self.device = policy, cfg, device

    @torch.no_grad()
    def __call__(self, batch: dict, batch_seed: int) -> dict[str, np.ndarray]:
        pol = self.policy
        pol.reset()
        bsz = batch["action"].shape[0]
        g = torch.Generator(device=self.device).manual_seed(int(batch_seed))
        noise = torch.randn((bsz, self.cfg.chunk_size, self.cfg.max_action_dim),
                            generator=g, device=self.device, dtype=torch.float32)
        pred = pol.predict_action_chunk(dict(batch), noise=noise).float()   # (B, n_action_steps, D)
        gt = batch["action"].float()                                          # (B, chunk, D) normalized
        T = min(pred.shape[1], gt.shape[1])
        pred, gt = pred[:, :T], gt[:, :T]
        err = (pred - gt).abs()                                               # (B, T, D)
        pad = batch.get("action_is_pad")
        if pad is not None:
            valid = (~pad[:, :T]).float().unsqueeze(-1)
            chunk = (err * valid).sum((1, 2)) / (valid.sum((1, 2)) * err.shape[-1]).clamp(min=1)
        else:
            chunk = err.mean((1, 2))
        return {
            "err_chunk_mae": chunk.cpu().numpy().astype(np.float32),
            "err_next_mae": err[:, 0].mean(-1).cpu().numpy().astype(np.float32),
            "err_gripper_next": err[:, 0, -1].cpu().numpy().astype(np.float32),
        }


def score_indices(ds, rel_indices: list[int], policy, pre, cfg, rename_map: dict,
                  batch_size: int = 64, num_workers: int = 8, device: str = "cuda",
                  log_every: int = 50, label: str = "") -> dict[str, np.ndarray]:
    """Run the policy over ``rel_indices`` of ``ds`` and return per-frame errors plus keys."""
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    scorer = BatchScorer(policy, cfg, device)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, sampler=rel_indices, num_workers=num_workers,
        multiprocessing_context=MP_CTX if num_workers else None, persistent_workers=False,
        prefetch_factor=4 if num_workers else None, pin_memory=True, drop_last=False)
    cols = {k: [] for k in ("index", "episode_index", "frame_index",
                            "err_chunk_mae", "err_next_mae", "err_gripper_next")}
    t0 = time.perf_counter()
    for bi, raw in enumerate(loader):
        idx = raw["index"].numpy().astype(np.int64)
        ep = raw["episode_index"].numpy().astype(np.int64)
        fi = raw["frame_index"].numpy().astype(np.int64)
        batch = _preprocess_dataset_batch(raw, ds.meta.camera_keys, rename_map, pre)
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = scorer(batch, batch_seed=int(idx[0]))
        cols["index"].append(idx); cols["episode_index"].append(ep); cols["frame_index"].append(fi)
        for k, v in out.items():
            cols[k].append(v)
        if log_every and (bi + 1) % log_every == 0:
            n = sum(len(x) for x in cols["index"])
            print(f"{label} {n:,}/{len(rel_indices):,} frames  {n / (time.perf_counter() - t0):.0f} fr/s",
                  flush=True)
    return {k: (np.concatenate(v) if v else np.array([], dtype=np.float32)) for k, v in cols.items()}
