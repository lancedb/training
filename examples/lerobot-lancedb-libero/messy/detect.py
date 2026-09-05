#!/usr/bin/env python
"""Find the messy episodes with columns on the frames table. No labels, no manifest: three
checks that each need only what the table already holds, written back as columns, then one
query per check.

  jerk_score          per frame, |d action / dt| over the arm dims. Flags jittery teleop.
  act_state_agree     per episode, correlation between commanded translation and the change in
                      end-effector position one frame later. Flags an action stream that does
                      not line up with its observations.
  goal_dist           per episode, cosine distance between the final-frame embedding and the
                      median final-frame embedding of every other episode with the same label.
                      Flags a label that does not match what the pictures show the robot did.

Thresholds are robust z-scores (median / MAD), 3.5 by default, computed within task where that
makes sense. `messy_manifest.json`, if present next to the dataset, is used only to GRADE the
flags (precision / recall per defect type); it never feeds the detectors.
"""
import argparse
import json
import os
import time

import lance
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import torch

EMB_MODEL = "google/siglip2-base-patch16-224"
EMB_DIM = 768
MAX_LAG = 10                      # frames (1 s at 10 fps)
CAM = "observation.images.image"     # agentview


def fsl(col):
    col = col.combine_chunks()
    return col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-9
    return (x - med) / mad


def embed_final_frames(root: str, episodes: np.ndarray, ep_from: np.ndarray, ep_to: np.ndarray, n_last: int, device="cuda") -> dict:
    """SigLIP2 embedding of the mean of the last `n_last` agentview frames of every episode."""
    from transformers import AutoModel, AutoProcessor

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("HuggingFaceVLA/libero", root=root, return_uint8=True, tolerance_s=1e-4)
    reader = ds.reader
    model = AutoModel.from_pretrained(EMB_MODEL, dtype=torch.float16).to(device).eval()
    proc = AutoProcessor.from_pretrained(EMB_MODEL)
    out = {}
    idxs = [(int(e), int(ep_to[i] - 1 - j)) for i, e in enumerate(episodes) for j in range(n_last)]
    B = 64
    t0 = time.perf_counter()
    for s in range(0, len(idxs), B):
        chunk = idxs[s:s + B]
        items = reader.get_items([fi for _, fi in chunk])
        imgs = [it[CAM].permute(1, 2, 0).numpy() for it in items]
        with torch.no_grad():
            px = proc(images=imgs, return_tensors="pt")
            px = {k: v.to(device, dtype=torch.float16 if v.is_floating_point() else None) for k, v in px.items()}
            e = model.get_image_features(**px)
            if not torch.is_tensor(e):
                e = e.pooler_output
            e = torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy()
        for (ep, fi), v in zip(chunk, e):
            out.setdefault(ep, []).append(v)
        if (s // B) % 20 == 0:
            print(f"  embedded {s + len(chunk):,}/{len(idxs):,} frames  {(s + len(chunk)) / (time.perf_counter() - t0):.0f} fr/s", flush=True)
    return {ep: np.mean(vs, axis=0) / (np.linalg.norm(np.mean(vs, axis=0)) + 1e-9) for ep, vs in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Lance dataset root (the messy one)")
    ap.add_argument("--z", type=float, default=3.5)
    ap.add_argument("--n-last", type=int, default=3)
    ap.add_argument("--min-lag", type=int, default=2, help="flag misaligned if the best lag is at least this")
    ap.add_argument("--lag-gain", type=float, default=0.05, help="... and beats lag 0 by this much correlation")
    ap.add_argument("--out", default=None, help="curated episode list + report (default <root>/curation.json)")
    ap.add_argument("--no-write", action="store_true", help="do not merge columns into the table")
    a = ap.parse_args()
    out_path = a.out or os.path.join(a.root, "curation.json")

    uri = os.path.join(a.root, "frames.lance")
    ds = lance.dataset(uri)
    v0 = ds.version
    t0 = time.perf_counter()
    T = ds.to_table(columns=["index", "episode_index", "frame_index", "task_index", "action", "observation_state"])
    idx = T.column("index").to_numpy(); ep = T.column("episode_index").to_numpy()
    task = T.column("task_index").to_numpy(); act = fsl(T.column("action")); st = fsl(T.column("observation_state"))
    episodes, first = np.unique(ep, return_index=True)
    order = np.argsort(first); episodes = episodes[order]
    ep_from = first[order]; ep_to = np.append(ep_from[1:], len(ep))
    print(f"{len(idx):,} frames, {len(episodes):,} episodes read in {time.perf_counter() - t0:.1f}s (version {v0})")

    # --- 1. jerk_score, per frame -----------------------------------------------------------------
    jerk = np.zeros(len(idx), dtype=np.float32)
    agree = np.zeros(len(episodes), dtype=np.float32)
    lag_best = np.zeros(len(episodes), dtype=np.int64); lag_gain = np.zeros(len(episodes), dtype=np.float32)
    ep_task = np.zeros(len(episodes), dtype=np.int64)
    for i, (s, e) in enumerate(zip(ep_from, ep_to)):
        A = act[s:e]
        d = np.abs(np.diff(A[:, :6], axis=0)).sum(axis=1)
        jerk[s + 1:e] = d; jerk[s] = d[0] if len(d) else 0
        # --- 2. action / state alignment: at which lag does the commanded translation best explain
        #        the end-effector displacement? Clean logs peak at lag 0 (the next frame moves as
        #        commanded); a stream logged ahead of its observations peaks at the offset.
        if e - s > MAX_LAG + 5:
            dx = st[s + 1:e, :3] - st[s:e - 1, :3]          # displacement after frame t
            cors = []
            for lag in range(MAX_LAG + 1):
                cs = []
                for k in range(3):
                    x, y = A[: len(dx) - lag, k], dx[lag:, k]
                    if x.std() > 1e-6 and y.std() > 1e-6:
                        cs.append(np.corrcoef(x, y)[0, 1])
                cors.append(float(np.mean(cs)) if cs else 0.0)
            agree[i] = cors[0]
            lag_best[i] = int(np.argmax(cors))
            lag_gain[i] = max(cors) - cors[0]
        ep_task[i] = task[s]
    ep_jerk = np.array([jerk[s:e].mean() for s, e in zip(ep_from, ep_to)], dtype=np.float32)

    # --- 3. goal_dist: does the end state look like the labelled task's end state? ----------------
    print("embedding final frames ...", flush=True)
    emb = embed_final_frames(a.root, episodes, ep_from, ep_to, a.n_last)
    E = np.stack([emb[int(e)] for e in episodes])
    goal_dist = np.zeros(len(episodes), dtype=np.float32)
    for t in np.unique(ep_task):
        m = np.where(ep_task == t)[0]
        for i in m:
            others = E[m[m != i]]
            ref = np.median(others, axis=0); ref /= (np.linalg.norm(ref) + 1e-9)
            goal_dist[i] = 1.0 - float(E[i] @ ref)

    # --- flags: robust z within task for jerk and goal_dist, global for agreement ------------------
    z_jerk = np.zeros(len(episodes)); z_goal = np.zeros(len(episodes))
    for t in np.unique(ep_task):
        m = ep_task == t
        z_jerk[m] = robust_z(ep_jerk[m]); z_goal[m] = robust_z(goal_dist[m])
    z_agree = robust_z(agree)
    flag_noise = z_jerk > a.z
    flag_misaligned = (lag_best >= a.min_lag) & (lag_gain > a.lag_gain)
    flag_label = z_goal > a.z
    flagged = flag_noise | flag_misaligned | flag_label
    keep = sorted(int(e) for e, f in zip(episodes, flagged) if not f)

    per_ep = pd.DataFrame({"episode_index": episodes, "task_index": ep_task, "jerk_score_ep": ep_jerk,
                           "act_state_agree": agree, "act_lag": lag_best, "act_lag_gain": lag_gain, "goal_dist": goal_dist,
                           "z_jerk": z_jerk, "z_agree": z_agree, "z_goal": z_goal,
                           "flag_noise": flag_noise, "flag_misaligned": flag_misaligned, "flag_label": flag_label})
    report = {"root": a.root, "table_version_before": v0, "z": a.z, "episodes": int(len(episodes)),
              "flagged": int(flagged.sum()), "kept": len(keep),
              "flagged_by": {"noise": int(flag_noise.sum()), "misaligned": int(flag_misaligned.sum()), "label": int(flag_label.sum())},
              "rules": {"noise": f"robust z of episode jerk within task > {a.z}", "misaligned": f"best action-to-motion lag >= {a.min_lag} frames with correlation gain > {a.lag_gain}", "label": f"robust z of final-frame distance to same-label median within task > {a.z}"},
              "curated_episodes": keep}
    print(f"flagged {flagged.sum()} / {len(episodes)} episodes: noise {flag_noise.sum()}, misaligned {flag_misaligned.sum()}, label {flag_label.sum()}")

    # --- grade against the manifest, if we have one ----------------------------------------------
    mpath = os.path.join(a.root, "messy_manifest.json")
    if os.path.exists(mpath):
        man = json.load(open(mpath))
        truth = {int(e): kind for kind, eps in man["groups"].items() for e in eps}
        y = np.array([truth.get(int(e), "clean") for e in episodes])
        grade = {}
        for kind, fl in (("action_noise", flag_noise), ("misaligned", flag_misaligned), ("label_swap", flag_label)):
            tp = int(((y == kind) & fl).sum()); fp = int(((y != kind) & fl).sum()); fn = int(((y == kind) & ~fl).sum())
            grade[kind] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)}
        bad = y != "clean"
        tp = int((bad & flagged).sum()); fp = int((~bad & flagged).sum()); fn = int((bad & ~flagged).sum())
        grade["any_defect"] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
                               "clean_episodes_dropped": fp, "corrupted_episodes_kept": fn}
        report["grade"] = grade
        per_ep["truth"] = y
        print(json.dumps(grade, indent=1))

    # --- write the columns back onto the table ---------------------------------------------------
    if not a.no_write:
        t0 = time.perf_counter()
        # LanceDB Table.merge is a left join on the key column: it appends the new columns and rewrites
        # nothing else, so readers pinned to the previous version keep working.
        frames = lancedb.connect(a.root).open_table("frames")
        drop = [c for c in ("jerk_score", "act_state_agree", "act_lag", "goal_dist", "quality_flag") if c in frames.schema.names]
        if drop:
            frames.drop_columns(drop)
        frames.merge(pa.table({"index": idx, "jerk_score": jerk}), left_on="index")
        ep_tbl = pa.table({"episode_index": episodes.astype(np.int64), "act_state_agree": agree, "act_lag": lag_best, "goal_dist": goal_dist,
                           "quality_flag": pa.array([("noise" if n else "") + ("misaligned" if m else "") + ("label" if l else "") or "ok"
                                                     for n, m, l in zip(flag_noise, flag_misaligned, flag_label)])})
        frames.merge(ep_tbl, left_on="episode_index")
        report["table_version_after"] = frames.version
        report["merge_s"] = round(time.perf_counter() - t0, 1)
        print(f"columns merged in {report['merge_s']}s: version {v0} -> {frames.version}")

    json.dump(report, open(out_path, "w"), indent=1)
    per_ep.to_csv(out_path.replace(".json", "_episodes.csv"), index=False)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
