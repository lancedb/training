#!/usr/bin/env python3
"""Data mining by example: one frame in, every episode that contains it out.

Curation asks "which of my data should I keep?". Mining asks "where else does THIS happen?",
and that is the question that needs an index. A seed frame becomes a vector query against the
embedding column, constrained by SQL on the same row.

Two details that matter for an honest demo:
  * Frame-level nearest neighbours are dominated by TEMPORAL near-duplicates -- the 12 closest
    frames to any frame are usually its own neighbours a few timesteps away. Mining wants
    distinct episodes, so we over-fetch and keep the best frame per episode.
  * The query is pixels only. Whether the returned episodes share the seed's *task string* is
    then a real check on the embedding, not something we asked for.
"""
import argparse, json, os, time
import numpy as np
import lancedb

SEL = ["episode_index", "frame_index", "jerk_score", "language_instruction",
       "is_episode_successful"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("LANCE_FE_ROOT", "./data/droid_fe_lance"))
    ap.add_argument("--table", default="frames")
    ap.add_argument("--k", type=int, default=8, help="distinct episodes to return")
    ap.add_argument("--fetch", type=int, default=600, help="over-fetch before per-episode dedupe")
    ap.add_argument("--out", default="out/mining.json")
    a = ap.parse_args()

    tbl = lancedb.connect(a.root).open_table(a.table)
    print(f"table: {tbl.count_rows():,} frames, embeddings + scores + text in one place", flush=True)

    # A seed worth querying: the arm is moving AND the episode has a task string.
    seed = (tbl.search().where(
        "jerk_score > 1.0 AND length(language_instruction) > 25", prefilter=True)
        .select(SEL + ["emb_siglip2"]).limit(1).to_list())[0]
    sep, sfr = int(seed["episode_index"]), int(seed["frame_index"])
    stask = (seed.get("language_instruction") or "").strip()
    print(f"\nseed frame -> episode {sep}, frame {sfr}, jerk {seed['jerk_score']:.2f}")
    print(f'  its task: "{stask[:78]}"', flush=True)
    vec = np.asarray(seed["emb_siglip2"], dtype=np.float32)

    res = {"seed": {"episode": sep, "frame": sfr, "task": stask,
                    "jerk": round(float(seed["jerk_score"]), 3)}, "queries": {}}

    cases = [
        ("find this situation in other episodes", f"episode_index != {sep}"),
        ("...restricted to successful episodes",
         f"episode_index != {sep} AND is_episode_successful = true"),
        ("...and only where the arm is moving hard (jerk > 1.0)",
         f"episode_index != {sep} AND is_episode_successful = true AND jerk_score > 1.0"),
    ]
    for label, where in cases:
        t0 = time.perf_counter()
        rows = (tbl.search(vec, vector_column_name="emb_siglip2")
                   .where(where, prefilter=True).select(SEL).limit(a.fetch).to_list())
        best = {}
        for r in rows:                        # keep the nearest frame per episode
            e = int(r["episode_index"])
            if e not in best:
                best[e] = r
        hits = list(best.values())[:a.k]
        ms = 1000 * (time.perf_counter() - t0)
        same = sum(1 for h in hits if (h.get("language_instruction") or "").strip() == stask)
        res["queries"][label] = {
            "where": where, "ms": round(ms, 1),
            "scanned": len(rows), "distinct_episodes": len(best), "returned": len(hits),
            "same_task_as_seed": same,
            "hits": [{"ep": int(h["episode_index"]), "frame": int(h["frame_index"]),
                      "jerk": round(float(h.get("jerk_score") or 0), 3),
                      "task": (h.get("language_instruction") or "").strip()[:64]} for h in hits],
        }
        print(f"\n{label}")
        print(f"  {ms:6.1f} ms | {len(best)} distinct episodes in the top {len(rows)} frames | "
              f"{same}/{len(hits)} share the seed's task string")
        for h in res["queries"][label]["hits"][:5]:
            mark = "=" if h["task"] == stask[:64] else " "
            print(f"   {mark} ep {h['ep']:>5} frame {h['frame']:>5}  jerk {h['jerk']:>5.2f}  {h['task'][:52]}")

    json.dump(res, open(a.out, "w"), indent=1)
    print("\nWROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
