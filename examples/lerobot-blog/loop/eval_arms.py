#!/usr/bin/env python
"""Per-episode action error of several checkpoints on the held-out episodes.

Every checkpoint is scored on the same frames (a fixed spread of points per episode) with the
same noise, so differences are the policies, not the sampling. Shard episodes across GPUs:

  for r in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$r python loop/eval_arms.py --rank $r --world 4 \
      --checkpoints base=... mined=... random=... & done; wait
"""
import argparse
import json
import os

import numpy as np

from common import (assert_same_stats, env_root, load_meta, load_policy, load_rename_map, open_dataset,
                    relative_indices, score_indices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="config/loop_sets.json")
    ap.add_argument("--per-episode", type=int, default=60)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir", default="out/eval")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="label=<pretrained_model dir>")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    pairs = [p.split("=", 1) for p in a.checkpoints]
    assert_same_stats([p for _, p in pairs])
    holdout = sorted(json.load(open(a.sets))["slices"]["all_holdout"])
    shard = holdout[a.rank::a.world]
    root, rmap = env_root(), load_rename_map()
    meta = load_meta(root)

    ds = None
    result = {}
    for label, path in pairs:
        policy, pre, delta_ts, cfg = load_policy(path, meta, rmap)
        if ds is None:
            ds = open_dataset(root, shard, delta_ts)
            rel = relative_indices(ds, shard, per_episode=a.per_episode)
            print(f"[rank {a.rank}] {len(shard)} holdout episodes, {len(rel):,} frames", flush=True)
        cols = score_indices(ds, rel, policy, pre, cfg, rmap, batch_size=a.batch_size,
                             num_workers=a.num_workers, label=f"[rank {a.rank}] {label}")
        per_ep = {}
        for e in np.unique(cols["episode_index"]):
            m = cols["episode_index"] == e
            per_ep[str(int(e))] = {k: round(float(cols[k][m].mean()), 5)
                                   for k in ("err_chunk_mae", "err_next_mae", "err_gripper_next")}
            per_ep[str(int(e))]["n"] = int(m.sum())
        result[label] = {"path": path, "per_episode": per_ep,
                         "mean_chunk_mae": round(float(cols["err_chunk_mae"].mean()), 5),
                         "mean_next_mae": round(float(cols["err_next_mae"].mean()), 5)}
        print(f"[rank {a.rank}] {label}: chunk MAE {result[label]['mean_chunk_mae']:.4f}  "
              f"next MAE {result[label]['mean_next_mae']:.4f}", flush=True)
        del policy
        import torch
        torch.cuda.empty_cache()

    out = os.path.join(a.out_dir, f"shard_{a.rank:02d}.json")
    json.dump(result, open(out, "w"))
    print(f"[rank {a.rank}] WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
