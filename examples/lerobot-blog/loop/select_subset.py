#!/usr/bin/env python
"""Pick the episode subset the loop works on and write it to config/loop_subset.json.

  holdout : the 200 uniformly sampled episodes in config/scope_holdout_random.json. They are
            excluded from every training run in this experiment and used only for evaluation.
  pool    : N training-eligible episodes sampled uniformly at random from the rest. This is the
            slice we embed and score; the fine-tuning sets are drawn from it.

Both lists are frozen here so every later step reads the same subset.
"""
import argparse
import json
import os

import numpy as np

from common import env_root, episode_lengths, load_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=os.environ.get("HOLDOUT", "config/scope_holdout_random.json"))
    ap.add_argument("--pool-size", type=int, default=2000)
    ap.add_argument("--min-frames", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="config/loop_subset.json")
    a = ap.parse_args()

    meta = load_meta(env_root())
    lengths = episode_lengths(meta)
    holdout = sorted(int(e) for e in json.load(open(a.holdout)))
    hold = set(holdout)
    eligible = [e for e, n in lengths.items() if n >= a.min_frames and e not in hold]
    rng = np.random.default_rng(a.seed)
    pool = sorted(int(e) for e in rng.choice(eligible, size=a.pool_size, replace=False))
    out = {"holdout": holdout, "pool": pool,
           "holdout_frames": int(sum(lengths[e] for e in holdout)),
           "pool_frames": int(sum(lengths[e] for e in pool)),
           "total_episodes": int(meta.total_episodes), "seed": a.seed}
    json.dump(out, open(a.out, "w"))
    print(f"holdout {len(holdout)} episodes / {out['holdout_frames']:,} frames; "
          f"pool {len(pool)} episodes / {out['pool_frames']:,} frames  -> {a.out}")


if __name__ == "__main__":
    main()
