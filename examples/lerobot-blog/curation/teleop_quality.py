#!/usr/bin/env python3
"""Per-episode teleoperation quality metrics over all of DROID, from scalar columns only.

The metric set is the one robot-data QA papers converge on (DQAF; Siemens 2026 arXiv:2605.26349
lists jerk / action-range saturation / gripper chatter / stalling, and three independent groups
use spectral smoothness). All are O(T) over columns we already have -- no video decode.

The question worth answering, which nobody has published on DROID: do these automated quality
scores actually predict the human success label?
"""
import os
import json, sys, time, collections
import numpy as np, lance

URI = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("LANCE_ROOT", "./data/droid_lance"), "frames.lance")
OUT = sys.argv[2] if len(sys.argv) > 2 else "out/teleop_quality.json"
FPS = 15.0
COLS = ["episode_index", "action_joint_velocity", "action_joint_position",
        "action_gripper_position", "is_episode_successful", "building", "collector_id"]

def ldj(vel):
    """Log dimensionless jerk. Standard smoothness metric from human motor control
    (Balasubramanian, IEEE TBME 2012); more negative = jerkier."""
    T = len(vel)
    if T < 4:
        return None
    dt = 1.0 / FPS
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    dur = T * dt
    vpk = np.abs(vel).max()
    if vpk <= 0:
        return None
    integral = np.sum(np.sum(jerk ** 2, axis=1)) * dt
    if integral <= 0:
        return None
    return float(-np.log((dur ** 3) / (vpk ** 2) * integral))

ds = lance.dataset(URI)
t0 = time.time()
cur, buf = None, collections.defaultdict(list)
eps = {}

def flush(e, b):
    if e is None or len(b["jv"]) < 4:
        return
    jv = np.asarray(b["jv"], dtype=np.float32)
    jp = np.asarray(b["jp"], dtype=np.float32)
    gp = np.asarray(b["gp"], dtype=np.float32).ravel()
    T = len(jv)
    # gripper chatter: mean absolute change of the binarised gripper command
    gb = (gp > (gp.min() + gp.max()) / 2).astype(np.float32)
    chatter = float(np.abs(np.diff(gb)).mean()) if T > 1 else 0.0
    # stalling: frames whose commanded joint velocity is ~zero
    stall = float((np.abs(jv).max(axis=1) < 1e-3).mean())
    # saturation proxy: frames within 2% of this episode's own joint range extremes
    lo, hi = jp.min(axis=0), jp.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    near = ((jp - lo) / span < 0.02) | ((hi - jp) / span < 0.02)
    sat = float(near.any(axis=1).mean())
    eps[e] = dict(frames=T, ldj=ldj(jv), chatter=chatter, stall=stall, sat=sat,
                  ok=bool(b["ok"][0]), building=b["bld"][0], collector=b["col"][0])

n = 0
for batch in ds.to_batches(columns=COLS, batch_size=65536):
    d = batch.to_pydict()
    for i in range(len(d["episode_index"])):
        e = d["episode_index"][i]
        if e != cur:
            flush(cur, buf); buf = collections.defaultdict(list); cur = e
        buf["jv"].append(d["action_joint_velocity"][i])
        buf["jp"].append(d["action_joint_position"][i])
        buf["gp"].append(d["action_gripper_position"][i])
        if not buf["ok"]:
            buf["ok"].append(d["is_episode_successful"][i])
            buf["bld"].append(d["building"][i]); buf["col"].append(d["collector_id"][i])
    n += len(d["episode_index"])
flush(cur, buf)
el = time.time() - t0
print(f"{n:,} frames, {len(eps):,} episodes in {el:.0f}s (scalar columns only, no video decoded)", flush=True)

def arr(k, sub=None):
    return np.array([v[k] for v in eps.values()
                     if v[k] is not None and (sub is None or v["ok"] == sub)], dtype=float)

res = {"frames": n, "episodes": len(eps), "seconds": round(el, 1)}
for k in ("ldj", "chatter", "stall", "sat"):
    a = arr(k)
    res[k] = {"p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)),
              "p90": float(np.percentile(a, 90))}
    # does the metric separate human-labelled success from failure?
    s, f = arr(k, True), arr(k, False)
    if len(s) > 100 and len(f) > 100:
        # AUROC via rank statistic
        allv = np.concatenate([s, f]); lab = np.concatenate([np.ones(len(s)), np.zeros(len(f))])
        order = np.argsort(allv); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
        auc = (ranks[lab == 1].sum() - len(s) * (len(s) + 1) / 2) / (len(s) * len(f))
        res[k].update(success_mean=float(s.mean()), failure_mean=float(f.mean()),
                      auroc_vs_success=round(float(auc), 3))
json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps(res, indent=2)[:1800], flush=True)
