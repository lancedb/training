#!/usr/bin/env python
"""Turn the base model's error column into fine-tuning sets, with the index doing the work.

Reads the columns merged by merge_columns.py straight off the frames table and writes
config/loop_sets.json with four training arms of K episodes each, drawn from the pool:

  mined  : take the frames where the base model is worst, one per episode, keep a diverse set
           of them as seeds, then ask the vector index "where else does this happen?" and
           collect the nearest distinct episodes. Error -> column -> query -> training set.
  hard   : the K pool episodes with the highest mean base error. Error-based curation with no
           index at all, for comparison.
  text   : keyword expansion of the seed instructions over language_instruction. What you
           could do with full-text search alone.
  random : K pool episodes drawn uniformly. The control.

And two evaluation slices of the held-out episodes (never trained on by any arm):

  hard_holdout          : top quartile of holdout episodes by base error
  mined_similar_holdout : holdout episodes nearest to the seed frames in embedding space
"""
import argparse
import collections
import json
import re
import time

import lance
import lancedb
import numpy as np

from common import env_root

STOP = set("the a an to of in on and from with into onto it its up down left right off out put move "
           "take pick place open close grab hold then your this that for at by over under".split())


def farthest_point(E: np.ndarray, k: int, start: int = 0) -> list[int]:
    """Greedy farthest-point sampling in cosine distance."""
    chosen = [start]
    d = 1 - E @ E[start]
    while len(chosen) < min(k, len(E)):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, 1 - E @ E[nxt])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="config/loop_subset.json")
    ap.add_argument("--k", type=int, default=300, help="episodes per training arm")
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--slice-size", type=int, default=50)
    ap.add_argument("--err-col", default="err_chunk_mae_base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="config/loop_sets.json")
    ap.add_argument("--root", default=None, help="dataset root to query (default $LANCE_ROOT); may be s3://")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    sub = json.load(open(a.subset))
    pool, hold = set(sub["pool"]), set(sub["holdout"])
    root = a.root or env_root()
    ds = lance.dataset(f"{root}/frames.lance")
    t0 = time.perf_counter()
    T = ds.to_table(columns=["index", "episode_index", "frame_index", a.err_col, "err_next_mae_base",
                             "err_gripper_next_base", "language_instruction", "is_episode_successful"],
                    filter=f"{a.err_col} IS NOT NULL").to_pandas()
    print(f"{len(T):,} scored frames read in {time.perf_counter() - t0:.1f}s (version {ds.version})")

    per_ep = T.groupby("episode_index").agg(err=(a.err_col, "mean"), err_next=("err_next_mae_base", "mean"),
                                            frames=("index", "size"),
                                            task=("language_instruction", "first"),
                                            ok=("is_episode_successful", "first"))
    pool_ep = per_ep[per_ep.index.isin(pool)]
    hold_ep = per_ep[per_ep.index.isin(hold)]
    print(f"pool: {len(pool_ep)} episodes, mean err {pool_ep.err.mean():.4f}; "
          f"holdout: {len(hold_ep)} episodes, mean err {hold_ep.err.mean():.4f}")

    # --- seeds: worst frames, one per episode, then a diverse subset of those ----------------------
    P = T[T.episode_index.isin(pool)]
    thresh = P[a.err_col].quantile(0.98)
    worst = (P[P[a.err_col] >= thresh].sort_values(a.err_col, ascending=False)
              .drop_duplicates("episode_index"))
    print(f"{len(worst)} candidate seed frames above p98 error {thresh:.4f}")
    cand_idx = worst["index"].to_numpy()
    E = ds.to_table(columns=["index", "emb_siglip2"],
                    filter=f"index IN ({','.join(map(str, cand_idx))})").to_pandas()
    E = E.set_index("index").loc[cand_idx]
    embs = np.stack(E["emb_siglip2"].to_numpy()).astype(np.float32)
    pick = farthest_point(embs, a.n_seeds)
    seeds = worst.iloc[pick]
    seed_vecs = embs[pick]
    seed_eps = [int(e) for e in seeds.episode_index]
    print("seeds:")
    for (_, r) in seeds.head(10).iterrows():
        print(f"  ep {int(r.episode_index):>6} f {int(r.frame_index):>4} err {r[a.err_col]:.3f}  "
              f"{(r.language_instruction or '')[:60]}")

    # --- mined: nearest distinct episodes to each seed, round-robin until K -----------------------
    db = lancedb.connect(root)
    tbl = db.open_table("frames")
    pool_sql = ",".join(map(str, sorted(pool)))
    hold_sql = ",".join(map(str, sorted(hold)))
    per_seed, per_seed_hold = [], []
    qms = []
    for v, sep in zip(seed_vecs, seed_eps):
        t = time.perf_counter()
        rows = (tbl.search(v, vector_column_name="emb_siglip2").metric("cosine")
                .where(f"episode_index IN ({pool_sql}) AND emb_siglip2 IS NOT NULL", prefilter=True)
                .select(["episode_index", "frame_index", "_distance"]).limit(600).to_list())
        qms.append(1000 * (time.perf_counter() - t))
        seen, order = set(), []
        for r in rows:
            e = int(r["episode_index"])
            if e not in seen:
                seen.add(e); order.append((e, int(r["frame_index"]), float(r["_distance"])))
        per_seed.append(order)
        rows = (tbl.search(v, vector_column_name="emb_siglip2").metric("cosine")
                .where(f"episode_index IN ({hold_sql}) AND emb_siglip2 IS NOT NULL", prefilter=True)
                .select(["episode_index", "_distance"]).limit(300).to_list())
        hseen, horder = set(), []
        for r in rows:
            e = int(r["episode_index"])
            if e not in hseen:
                hseen.add(e); horder.append(e)
        per_seed_hold.append(horder)
    print(f"{2 * len(seed_eps)} vector queries, {np.mean(qms):.1f} ms mean, {np.max(qms):.1f} ms max")

    mined, mined_hits = [], []
    ptr = [0] * len(per_seed)
    while len(mined) < a.k and any(ptr[i] < len(per_seed[i]) for i in range(len(per_seed))):
        for i in range(len(per_seed)):
            while ptr[i] < len(per_seed[i]):
                e, f, d = per_seed[i][ptr[i]]; ptr[i] += 1
                if e not in mined:
                    mined.append(e); mined_hits.append({"seed": seed_eps[i], "ep": e, "frame": f, "dist": round(d, 4)})
                    break
            if len(mined) >= a.k:
                break
    mined_similar_holdout = []
    ptr = [0] * len(per_seed_hold)
    while len(mined_similar_holdout) < a.slice_size and any(ptr[i] < len(per_seed_hold[i]) for i in range(len(ptr))):
        for i in range(len(per_seed_hold)):
            while ptr[i] < len(per_seed_hold[i]):
                e = per_seed_hold[i][ptr[i]]; ptr[i] += 1
                if e not in mined_similar_holdout:
                    mined_similar_holdout.append(e); break
            if len(mined_similar_holdout) >= a.slice_size:
                break

    # --- hard: highest mean error, no index -------------------------------------------------------
    hard = [int(e) for e in pool_ep.sort_values("err", ascending=False).index[: a.k]]

    # --- text: keyword expansion of the seed instructions ----------------------------------------
    words = collections.Counter()
    for s in seeds.language_instruction.fillna(""):
        words.update(w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 3 and w not in STOP)
    keywords = [w for w, _ in words.most_common(15)]
    scores = {}
    for e, task in pool_ep.task.fillna("").items():
        low = task.lower()
        n = sum(1 for w in keywords if w in low)
        if n:
            scores[int(e)] = n
    text = [e for e, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))][: a.k]
    print(f"keywords {keywords}; {len(scores)} pool episodes match, taking {len(text)}")

    # --- random ----------------------------------------------------------------------------------
    random_arm = sorted(int(e) for e in rng.choice(sorted(pool), size=a.k, replace=False))

    hard_holdout = [int(e) for e in hold_ep.sort_values("err", ascending=False).index[: a.slice_size]]

    def frames_of(eps):
        return int(per_ep.loc[[e for e in eps if e in per_ep.index], "frames"].sum())

    arms = {"mined": sorted(mined), "hard": sorted(hard), "text": sorted(text), "random": random_arm}
    overlap = {f"{x}&{y}": len(set(arms[x]) & set(arms[y])) for x in arms for y in arms if x < y}
    out = {
        "k": a.k, "err_col": a.err_col, "arms": arms,
        "arm_frames_scored": {k: frames_of(v) for k, v in arms.items()},
        "arm_mean_base_err": {k: round(float(per_ep.loc[v, "err"].mean()), 4) for k, v in arms.items()},
        "overlap": overlap,
        "seeds": [{"ep": int(r.episode_index), "frame": int(r.frame_index), "err": round(float(r[a.err_col]), 4),
                   "task": r.language_instruction} for _, r in seeds.iterrows()],
        "mined_hits": mined_hits, "keywords": keywords,
        "slices": {"all_holdout": sorted(hold), "hard_holdout": sorted(hard_holdout),
                   "mined_similar_holdout": sorted(mined_similar_holdout)},
        "episode_error": {str(int(e)): round(float(r.err), 5) for e, r in per_ep.iterrows()},
        "query_ms": {"mean": round(float(np.mean(qms)), 1), "max": round(float(np.max(qms)), 1)},
        "table_version": ds.version,
    }
    json.dump(out, open(a.out, "w"))
    print(json.dumps({k: out[k] for k in ("arm_frames_scored", "arm_mean_base_err", "overlap", "query_ms")}, indent=1))
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
