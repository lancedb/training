"""Equivalence test: detect.py's whole-table numpy maths vs the per-episode UDF version.

1. read the chosen episodes' frames from the S3 frames table (same columns detect.py reads)
2. score them the old way (one numpy pass, contiguous slices)
3. score them the new way, twice: calling the UDF classes directly, and through geneva
   add_columns + backfill on a local `episodes` table
4. compare every number, then apply the flag rules and compare the kept-episode lists
"""
import argparse, os, sys, time, shutil
import numpy as np, pyarrow as pa, lance, lancedb

sys.path.insert(0, os.path.dirname(__file__))
from old_math import score_all, goal_dist_all, robust_z
from new_udfs import make_udfs, make_goal_dist_udf

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--region", default="us-east-2")
ap.add_argument("--episodes", type=int, default=0, help="sample this many episodes (0 = all)")
ap.add_argument("--action-dims", type=int, default=6)
ap.add_argument("--pos-dims", type=int, default=3)
ap.add_argument("--geneva", action="store_true")
ap.add_argument("--local", default="./eq_local_db")
a = ap.parse_args()
so = {"region": a.region}

t0 = time.perf_counter()
frames = lance.dataset(f"{a.root}/frames.lance", storage_options=so)
cols = ["index", "episode_index", "task_index", "frame_index", "action", "observation_state"]
if a.episodes:
    eps_all = frames.to_table(columns=["episode_index"]).column("episode_index").unique().to_numpy()
    rng = np.random.default_rng(0)
    chosen = np.sort(rng.choice(eps_all, size=a.episodes, replace=False))
    T = frames.to_table(columns=cols, filter=f"episode_index IN ({','.join(map(str, chosen))})")
else:
    T = frames.to_table(columns=cols)
T = T.sort_by("index")
print(f"read {T.num_rows:,} frames, {len(T.column('episode_index').unique()):,} episodes from S3 in {time.perf_counter()-t0:.1f}s")

# ---- old way -------------------------------------------------------------------------------------
jerk_frame, old = score_all(T, a.action_dims, a.pos_dims)
episodes, task = old["episodes"], old["task"]
rng = np.random.default_rng(1)
E = rng.normal(size=(len(episodes), 16)).astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)              # stand-in for the final-frame embeddings
old_goal = goal_dist_all(E, task)

# ---- the episodes table ---------------------------------------------------------------------------
shutil.rmtree(a.local, ignore_errors=True)
db = lancedb.connect(a.local)
ep_tbl = db.create_table("episodes", pa.table({
    "episode_index": pa.array(episodes, pa.int64()),
    "task_index": pa.array(task, pa.int64()),
    "goal_emb": pa.FixedSizeListArray.from_arrays(pa.array(E.ravel()), 16),
}))

EpisodeJerk, ActLag = make_udfs(a.root, so, a.action_dims, a.pos_dims)
GoalDist = make_goal_dist_udf(f"{a.local}/episodes.lance")

# ---- new way, direct calls -----------------------------------------------------------------------
t0 = time.perf_counter()
jerk_u, lag_u, goal_u = EpisodeJerk(), ActLag(), GoalDist()
new_jerk = np.array([jerk_u(int(e)) for e in episodes], dtype=np.float32)
lag_out = [lag_u(int(e)) for e in episodes]
new_goal = np.array([goal_u(int(e), int(t), E[i]) for i, (e, t) in enumerate(zip(episodes, task))], dtype=np.float32)
print(f"new way (direct): {len(episodes)} episodes scored in {time.perf_counter()-t0:.1f}s, "
      f"{(time.perf_counter()-t0)/len(episodes)*1000:.0f} ms per episode incl. S3 read")

def report(label, new_jerk, new_agree, new_lag, new_gain, new_goal):
    ok = True
    for name, o, n, tol in [("jerk_score", old["jerk_ep"], new_jerk, 1e-5), ("act_lag.agree", old["agree"], new_agree, 1e-5),
                            ("act_lag.lag", old["lag_best"], new_lag, 0), ("act_lag.gain", old["lag_gain"], new_gain, 1e-5),
                            ("goal_dist", old_goal, new_goal, 1e-6)]:
        d = np.max(np.abs(np.asarray(o, dtype=np.float64) - np.asarray(n, dtype=np.float64)))
        good = d <= tol
        ok &= bool(good)
        print(f"  {label:8s} {name:14s} max |old-new| = {d:.2e}  {'OK' if good else 'MISMATCH'}")
    # the flag rules, applied to both
    def flags(jerk_ep, lag, gain, goal):
        zj = np.zeros(len(episodes)); zg = np.zeros(len(episodes))
        for t in np.unique(task):
            m = task == t
            zj[m] = robust_z(jerk_ep[m]); zg[m] = robust_z(goal[m])
        return (zj > 3.0) | ((lag >= 2) & (gain > 0.05)) | (zg > 3.0)
    fo = flags(old["jerk_ep"], old["lag_best"], old["lag_gain"], old_goal)
    fn = flags(np.asarray(new_jerk), np.asarray(new_lag), np.asarray(new_gain), np.asarray(new_goal))
    same = np.array_equal(fo, fn)
    print(f"  {label:8s} flagged {int(fo.sum())} of {len(episodes)} episodes (old) vs {int(fn.sum())} (new); "
          f"kept lists identical: {same}")
    return ok and same

ok_direct = report("direct", new_jerk, [d["agree"] for d in lag_out], [d["lag"] for d in lag_out],
                   [d["gain"] for d in lag_out], new_goal)

# ---- new way, through geneva ---------------------------------------------------------------------
ok_geneva = None
if a.geneva:
    import geneva
    gdb = geneva.connect(a.local)
    gt = gdb.open_table("episodes")
    t0 = time.perf_counter()
    gt.add_columns({"jerk_score": EpisodeJerk(), "act_lag": ActLag(), "goal_dist": GoalDist()})
    print(f"geneva add_columns: {time.perf_counter()-t0:.3f}s (nothing computed yet), version {gt.version}")
    with gdb.local_ray_context():
        for c in ("jerk_score", "act_lag", "goal_dist"):
            t0 = time.perf_counter()
            gt.backfill(c, concurrency=2)
            print(f"geneva backfill {c}: {time.perf_counter()-t0:.1f}s, version {gt.version}")
    r = lancedb.connect(a.local).open_table("episodes").search().limit(1_000_000).to_arrow().sort_by("episode_index")
    order = np.argsort(episodes)
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    # r is sorted by episode_index; old arrays are in row order of first appearance -> align
    r_eps = r.column("episode_index").to_numpy()
    assert np.array_equal(r_eps, np.sort(episodes))
    pos = {e: i for i, e in enumerate(r_eps)}
    sel = [pos[e] for e in episodes]
    lagcol = r.column("act_lag").combine_chunks()
    ok_geneva = report("geneva", r.column("jerk_score").to_numpy()[sel],
                       lagcol.field("agree").to_numpy()[sel], lagcol.field("lag").to_numpy()[sel],
                       lagcol.field("gain").to_numpy()[sel], r.column("goal_dist").to_numpy()[sel])
    # the filter the trainer needs, on the episodes table, with a struct field in the predicate
    n_bad = lancedb.connect(a.local).open_table("episodes").search() \
        .where("act_lag.lag >= 2 AND act_lag.gain > 0.05", prefilter=True).select(["episode_index"]).limit(1_000_000).to_arrow().num_rows
    print(f"  SQL on a struct field works: {n_bad} episodes with act_lag.lag >= 2 AND gain > 0.05")

print("\nRESULT direct:", "PASS" if ok_direct else "FAIL", "| geneva:", {None: "skipped", True: "PASS", False: "FAIL"}[ok_geneva])
