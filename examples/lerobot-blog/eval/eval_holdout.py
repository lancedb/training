#!/usr/bin/env python3
"""Aggregate open-loop MAE on held-out DROID episodes: base SmolVLA vs the 10k Lance run.

Shards episodes across GPUs. Records per-episode motion so improvement can be read
against how much the arm actually moves.
"""
import os
import argparse, json, time
import numpy as np, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("LANCE_ROOT", "./data/droid_lance"))
    ap.add_argument("--episodes", required=True, help="comma-separated")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    a = ap.parse_args()

    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.storage import load_dataset_metadata
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    eps = [int(x) for x in a.episodes.split(",")]
    rmap = json.load(open(os.environ.get("RENAME_MAP", "config/rename_map.json")))
    meta = load_dataset_metadata("lerobot/droid_1.0.1", root=a.root)
    out = {}

    for pair in a.checkpoints:
        label, path = pair.split("=", 1)
        cfg = PreTrainedConfig.from_pretrained(path); cfg.pretrained_path = path
        dt = resolve_delta_timestamps(cfg, meta, rmap)
        pre, post = make_pre_post_processors(cfg, pretrained_path=path)
        pol = None
        for ep in eps:
            ds = LeRobotDataset("lerobot/droid_1.0.1", root=a.root, episodes=[ep],
                                delta_timestamps=dt, return_uint8=True, tolerance_s=5e-3)
            if pol is None:
                pol = make_policy(cfg, ds_meta=ds.meta, rename_map=rmap).to("cuda").eval()
            n = min(a.steps, len(ds))
            idx = np.linspace(0, len(ds) - 1, n).astype(int)   # spread over the episode
            preds, gts = [], []
            with torch.no_grad():
                for i in idx:
                    raw = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v])
                           for k, v in ds[int(i)].items()}
                    batch = _preprocess_dataset_batch(raw, ds.meta.camera_keys, rmap, pre)
                    batch = {k: (v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v)
                             for k, v in batch.items()}
                    pol.reset()
                    preds.append(pol.predict_action_chunk(dict(batch))[0, 0].float().cpu().numpy())
                    gts.append(batch["action"][0, 0].float().cpu().numpy())
            P, G = np.stack(preds), np.stack(gts)
            motion = float(np.abs(np.diff(G, axis=0)).sum(axis=1).mean())
            out.setdefault(str(ep), {})[label] = {
                "mae": float(np.abs(P - G).mean()), "motion": motion, "n": int(n)}
            print(f"{label} ep{ep}: mae={np.abs(P-G).mean():.4f} motion={motion:.4f}", flush=True)
        del pol; torch.cuda.empty_cache()

    json.dump(out, open(a.out, "w"))
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
