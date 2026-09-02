#!/usr/bin/env python3
"""Held-out loss + action MAE for two checkpoints of the same fine-tune.

Both checkpoints MUST share action normalization statistics, or the two losses are
computed in different units and the comparison is meaningless. That is asserted, not
assumed -- an earlier version of this evaluation compared the fine-tune against
`lerobot/smolvla_base`, whose stats cover the SO-100 arm in degrees and contain no
DROID entry at all, which made every reported "before" number noise.
"""
import os
import argparse, json, time
import numpy as np, torch


def stats_of(path):
    from safetensors.torch import load_file
    import glob, os
    f = os.path.join(path, "policy_preprocessor_step_5_normalizer_processor.safetensors")
    st = load_file(f)
    if "action.mean" not in st:
        raise SystemExit(f"ABORT: {path} has no `action.mean` -- it was not trained on this "
                         f"dataset; its keys are {[k for k in st if 'action' in k][:4]}")
    return (st["action.mean"].float().numpy().ravel(), st["action.std"].float().numpy().ravel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("LANCE_ROOT", "./data/droid_lance"))
    ap.add_argument("--holdout", default=os.environ.get("HOLDOUT", "config/cur_holdout.json"))
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
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

    pairs = [p.split("=", 1) for p in a.checkpoints]
    ref = stats_of(pairs[0][1])
    for label, path in pairs[1:]:
        m, s = stats_of(path)
        if not (np.allclose(m, ref[0]) and np.allclose(s, ref[1])):
            raise SystemExit(f"ABORT: {label} action stats differ from {pairs[0][0]} -- "
                             f"losses would be in different units")
    print(f"normalization verified identical across {len(pairs)} checkpoints", flush=True)

    hold = json.load(open(a.holdout))
    rmap = json.load(open(os.environ.get("RENAME_MAP", "config/rename_map.json")))
    meta = load_dataset_metadata("lerobot/droid_1.0.1", root=a.root)
    res = {"holdout_episodes": len(hold), "batches": a.batches, "batch_size": a.batch_size}

    for label, path in pairs:
        cfg = PreTrainedConfig.from_pretrained(path); cfg.pretrained_path = path
        dt = resolve_delta_timestamps(cfg, meta, rmap)
        ds = LeRobotDataset("lerobot/droid_1.0.1", root=a.root, episodes=hold,
                            delta_timestamps=dt, return_uint8=True, tolerance_s=5e-3)
        pol = make_policy(cfg, ds_meta=ds.meta, rename_map=rmap).to("cuda").eval()
        pre, _ = make_pre_post_processors(cfg, pretrained_path=path)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                                         num_workers=8, generator=torch.Generator().manual_seed(0))
        losses, maes = [], []
        t0 = time.perf_counter()
        with torch.no_grad():
            for bi, raw in enumerate(dl):
                if bi >= a.batches: break
                batch = _preprocess_dataset_batch(raw, ds.meta.camera_keys, rmap, pre)
                batch = {k: (v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                loss, _ = pol.forward(dict(batch))
                losses.append(float(loss))
                pol.reset()
                pred = pol.predict_action_chunk(dict(batch))
                gt = batch["action"]
                T = min(pred.shape[1], gt.shape[1])
                maes.append(float((pred[:, :T] - gt[:, :T]).abs().mean()))
        L, M = np.array(losses), np.array(maes)
        res[label] = {"path": path, "loss": round(float(L.mean()), 4),
                      "loss_sd": round(float(L.std()), 4),
                      "action_chunk_mae": round(float(M.mean()), 4),
                      "batches": len(L), "samples": len(L) * a.batch_size,
                      "seconds": round(time.perf_counter() - t0, 1)}
        print(f"{label}: loss={L.mean():.4f}+-{L.std():.4f}  mae={M.mean():.4f}  "
              f"({len(L)} batches)", flush=True)
        del pol; torch.cuda.empty_cache()

    a_, b_ = pairs[0][0], pairs[-1][0]
    res["delta"] = {
        "loss_reduction_pct": round(100 * (1 - res[b_]["loss"] / res[a_]["loss"]), 1),
        "mae_reduction_pct": round(100 * (1 - res[b_]["action_chunk_mae"]
                                          / res[a_]["action_chunk_mae"]), 1)}
    json.dump(res, open(a.out, "w"), indent=1)
    print("WROTE", a.out, json.dumps(res["delta"]), flush=True)


if __name__ == "__main__":
    main()
