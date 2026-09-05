#!/usr/bin/env python
"""Find the messy episodes with columns on an `episodes` table. No labels, no manifest.

Three detectors, each a per-episode feature declared with `add_columns` and computed with
`backfill`. Every UDF takes one episode_index and reads that episode's frames from the frames
table, the same way the DROID embedding UDF reads the videos table. Nothing is merged into the
frames table and nothing is broadcast onto frames.

  jerk_score    mean over the episode of |d action| over the arm dims. Flags jittery teleop.
  act_lag       {lag, gain, agree}: the frame lag at which the commanded translation best
                explains the observed end-effector displacement. Clean logs peak at lag 0.
  goal_emb      SigLIP2 embedding of the last `--n-last` agentview frames (GPU UDF).
  goal_dist     cosine distance of goal_emb to the median goal_emb of every OTHER episode with
                the same label. Flags a label that does not match what the pictures show.
  quality_flag  'ok', or the name of the defect. Thresholds are robust z-scores (median / MAD)
                within task; the lag rule is absolute.

The training set is then one filter: quality_flag = 'ok'.

`messy_manifest.json`, if present next to the dataset, is used only to GRADE the flags. It never
feeds the detectors. If results/curation_episodes.csv from a previous run exists, the kept
episode list is compared against it, so a re-run can prove it reproduces the published numbers.
"""
import argparse
import json
import os
import time

import geneva
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
from geneva import udf

EMB_MODEL = "google/siglip2-base-patch16-224"
EMB_DIM = 768
MAX_LAG = 10                      # frames (1 s at 10 fps)
CAM = "observation.images.image"  # agentview
N_ACTION_DIMS = 6                 # arm dims; the 7th is the gripper
N_POS_DIMS = 3                    # end-effector xyz at the front of observation_state


def fsl(col):
    col = col.combine_chunks()
    return col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-9
    return (x - med) / mad


# --- the per-episode UDFs ------------------------------------------------------------------------

class _EpisodeReader:
    """Opens the frames table once per worker and returns one episode's actions and states in order."""

    def __init__(self, root):
        self.root, self._tbl = root, None

    def episode(self, episode_index):
        if self._tbl is None:
            self._tbl = lancedb.connect(self.root).open_table("frames")
        t = (self._tbl.search().where(f"episode_index = {int(episode_index)}", prefilter=True)
             .select(["frame_index", "action", "observation_state"]).limit(1_000_000).to_arrow())
        t = t.sort_by("frame_index")
        return fsl(t.column("action")), fsl(t.column("observation_state"))


def make_udfs(root: str, n_last: int):
    @udf(data_type=pa.float32(), input_columns=["episode_index"])
    class EpisodeJerk:
        def __init__(self):
            self.reader = _EpisodeReader(root)

        def __call__(self, episode_index: int) -> float:
            A, _ = self.reader.episode(episode_index)
            d = np.abs(np.diff(A[:, :N_ACTION_DIMS], axis=0)).sum(axis=1)
            if len(d) == 0:
                return 0.0
            return float((d.sum() + d[0]) / len(A))   # first frame carries d[0], then the mean

    @udf(data_type=pa.struct([("lag", pa.int64()), ("gain", pa.float32()), ("agree", pa.float32())]),
         input_columns=["episode_index"])
    class ActLag:
        def __init__(self):
            self.reader = _EpisodeReader(root)

        def __call__(self, episode_index: int) -> dict:
            A, S = self.reader.episode(episode_index)
            if len(A) <= MAX_LAG + 5:
                return {"lag": 0, "gain": 0.0, "agree": 0.0}
            dx = S[1:, :N_POS_DIMS] - S[:-1, :N_POS_DIMS]           # displacement after frame t
            cors = []
            for lag in range(MAX_LAG + 1):
                cs = []
                for k in range(N_POS_DIMS):
                    x, y = A[: len(dx) - lag, k], dx[lag:, k]
                    if x.std() > 1e-6 and y.std() > 1e-6:
                        cs.append(np.corrcoef(x, y)[0, 1])
                cors.append(float(np.mean(cs)) if cs else 0.0)
            return {"lag": int(np.argmax(cors)), "gain": float(max(cors) - cors[0]), "agree": float(cors[0])}

    @udf(data_type=pa.list_(pa.float32(), EMB_DIM), num_gpus=1, input_columns=["last_index"])
    class GoalEmb:
        """SigLIP2 embedding of the mean of the last n_last agentview frames. Model loads once per worker."""

        def __init__(self):
            self.model = None

        def _load(self):
            import torch
            from transformers import AutoModel, AutoProcessor
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            self.torch = torch
            self.reader = LeRobotDataset("HuggingFaceVLA/libero", root=root, return_uint8=True, tolerance_s=1e-4).reader
            self.model = AutoModel.from_pretrained(EMB_MODEL, dtype=torch.float16).to("cuda").eval()
            self.proc = AutoProcessor.from_pretrained(EMB_MODEL)

        def __call__(self, last_index: int) -> list:
            if self.model is None:
                self._load()
            items = self.reader.get_items([int(last_index) - j for j in range(n_last)])
            imgs = [it[CAM].permute(1, 2, 0).numpy() for it in items]
            with self.torch.no_grad():
                px = self.proc(images=imgs, return_tensors="pt")
                px = {k: v.to("cuda", dtype=self.torch.float16 if v.is_floating_point() else None) for k, v in px.items()}
                e = self.model.get_image_features(**px)
                if not self.torch.is_tensor(e):
                    e = e.pooler_output
                e = self.torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy()
            v = e.mean(axis=0)
            return (v / (np.linalg.norm(v) + 1e-9)).tolist()

    return EpisodeJerk, ActLag, GoalEmb


def make_second_stage(episodes_uri: str, z: float, min_lag: int, lag_gain: float):
    """UDFs that need the whole episodes table: per-task reference sets, loaded once per worker."""

    def _episodes(cols):
        import lance
        return lance.dataset(episodes_uri).to_table(columns=cols)

    @udf(data_type=pa.float32(), input_columns=["episode_index", "task_index", "goal_emb"])
    class GoalDist:
        def __init__(self):
            self._by_task = None

        def __call__(self, episode_index: int, task_index: int, goal_emb) -> float:
            if self._by_task is None:
                t = _episodes(["episode_index", "task_index", "goal_emb"])
                E, ep, task = fsl(t.column("goal_emb")), t.column("episode_index").to_numpy(), t.column("task_index").to_numpy()
                self._by_task = {int(k): (ep[task == k], E[task == k]) for k in np.unique(task)}
            eps, E = self._by_task[int(task_index)]
            ref = np.median(E[eps != int(episode_index)], axis=0)
            ref /= (np.linalg.norm(ref) + 1e-9)
            return float(1.0 - np.asarray(goal_emb, dtype=np.float32) @ ref)

    @udf(data_type=pa.string(), input_columns=["task_index", "jerk_score", "act_lag", "goal_dist"])
    class QualityFlag:
        """Robust z within task for jerk and goal_dist; the lag rule is absolute."""

        def __init__(self):
            self._stats = None

        def __call__(self, task_index: int, jerk_score: float, act_lag: dict, goal_dist: float) -> str:
            if self._stats is None:
                t = _episodes(["task_index", "jerk_score", "goal_dist"]).to_pandas()
                self._stats = {}
                for k, g in t.groupby("task_index"):
                    self._stats[int(k)] = {c: (float(np.median(g[c])), float(np.median(np.abs(g[c] - np.median(g[c]))) * 1.4826 + 1e-9))
                                           for c in ("jerk_score", "goal_dist")}
            s = self._stats[int(task_index)]
            noise = (jerk_score - s["jerk_score"][0]) / s["jerk_score"][1] > z
            misaligned = act_lag["lag"] >= min_lag and act_lag["gain"] > lag_gain
            label = (goal_dist - s["goal_dist"][0]) / s["goal_dist"][1] > z
            return ("noise" if noise else "") + ("misaligned" if misaligned else "") + ("label" if label else "") or "ok"

    return GoalDist, QualityFlag


# --- driver --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Lance dataset root (the messy one); local or s3://")
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--n-last", type=int, default=3)
    ap.add_argument("--min-lag", type=int, default=2, help="flag misaligned if the best lag is at least this")
    ap.add_argument("--lag-gain", type=float, default=0.05, help="... and beats lag 0 by this much correlation")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=None, help="curated episode list + report (default <root>/curation.json)")
    ap.add_argument("--compare", default=os.path.join(os.path.dirname(__file__), "results", "curation_episodes.csv"),
                    help="previous per-episode results to compare the kept list against")
    a = ap.parse_args()
    out_path = a.out or os.path.join(a.root, "curation.json")

    # 1. the episodes table: one row per episode, built from the frames table's tabular columns
    t0 = time.perf_counter()
    db = lancedb.connect(a.root)
    frames = db.open_table("frames")
    f = frames.search().select(["index", "episode_index", "task_index"]).limit(100_000_000).to_arrow().to_pandas()
    g = f.groupby("episode_index", sort=True)
    ep_tbl = pa.table({
        "episode_index": pa.array(g["episode_index"].first().values, pa.int64()),
        "task_index": pa.array(g["task_index"].first().values, pa.int64()),
        "num_frames": pa.array(g.size().values, pa.int64()),
        "last_index": pa.array(g["index"].max().values, pa.int64()),   # global index of the last frame
    })
    db.create_table("episodes", ep_tbl, mode="overwrite")
    print(f"episodes table: {ep_tbl.num_rows:,} rows from {len(f):,} frames in {time.perf_counter()-t0:.1f}s")

    # 2. declare the features, then compute them
    gdb = geneva.connect(a.root)
    episodes = gdb.open_table("episodes")
    EpisodeJerk, ActLag, GoalEmb = make_udfs(a.root, a.n_last)
    GoalDist, QualityFlag = make_second_stage(os.path.join(a.root, "episodes.lance"), a.z, a.min_lag, a.lag_gain)
    v0 = episodes.version
    timings = {}
    with gdb.local_ray_context():
        # first the three detectors, which read the frames table ...
        episodes.add_columns({"jerk_score": EpisodeJerk(), "act_lag": ActLag(), "goal_emb": GoalEmb()})
        print(f"declared 3 columns, version {v0} -> {episodes.version}, nothing computed yet")
        for col in ("jerk_score", "act_lag", "goal_emb"):
            t0 = time.perf_counter()
            episodes.backfill(col, concurrency=a.concurrency)
            timings[col] = round(time.perf_counter() - t0, 1)
            print(f"backfill {col}: {timings[col]}s, version {episodes.version}")
        # ... then the two that compare an episode with the other episodes of its task
        episodes.add_columns({"goal_dist": GoalDist(), "quality_flag": QualityFlag()})
        for col in ("goal_dist", "quality_flag"):
            t0 = time.perf_counter()
            episodes.backfill(col, concurrency=a.concurrency)
            timings[col] = round(time.perf_counter() - t0, 1)
            print(f"backfill {col}: {timings[col]}s, version {episodes.version}")

    # 3. the training set is one filter
    ep = lancedb.connect(a.root).open_table("episodes")
    keep = sorted(ep.search().where("quality_flag = 'ok'", prefilter=True)
                  .select(["episode_index"]).limit(1_000_000).to_arrow().column("episode_index").to_pylist())
    per_ep = ep.search().limit(1_000_000).to_arrow().sort_by("episode_index").to_pandas()
    per_ep["act_lag_lag"] = per_ep["act_lag"].map(lambda d: d["lag"])
    per_ep["act_lag_gain"] = per_ep["act_lag"].map(lambda d: d["gain"])
    per_ep = per_ep.drop(columns=["goal_emb", "act_lag"])
    flagged = per_ep["quality_flag"] != "ok"
    counts = {k: int(per_ep["quality_flag"].str.contains(k).sum()) for k in ("noise", "misaligned", "label")}
    report = {"root": a.root, "episodes_table_version": ep.version, "z": a.z, "episodes": int(len(per_ep)),
              "flagged": int(flagged.sum()), "kept": len(keep), "flagged_by": counts, "backfill_seconds": timings,
              "rules": {"noise": f"robust z of episode jerk within task > {a.z}",
                        "misaligned": f"best action-to-motion lag >= {a.min_lag} frames with correlation gain > {a.lag_gain}",
                        "label": f"robust z of final-frame distance to same-label median within task > {a.z}"},
              "curated_episodes": keep}
    print(f"flagged {int(flagged.sum())} / {len(per_ep)} episodes: {counts}; kept {len(keep)}")

    # 4. grade against the manifest, if we have one
    mpath = os.path.join(a.root, "messy_manifest.json")
    if os.path.exists(mpath):
        man = json.load(open(mpath))
        truth = {int(e): kind for kind, eps in man["groups"].items() for e in eps}
        y = np.array([truth.get(int(e), "clean") for e in per_ep["episode_index"]])
        grade = {}
        for kind, key in (("action_noise", "noise"), ("misaligned", "misaligned"), ("label_swap", "label")):
            fl = per_ep["quality_flag"].str.contains(key).values
            tp = int(((y == kind) & fl).sum()); fp = int(((y != kind) & fl).sum()); fn = int(((y == kind) & ~fl).sum())
            grade[kind] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)}
        bad = y != "clean"; fl = flagged.values
        tp = int((bad & fl).sum()); fp = int((~bad & fl).sum()); fn = int((bad & ~fl).sum())
        grade["any_defect"] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
                               "clean_episodes_dropped": fp, "corrupted_episodes_kept": fn}
        report["grade"] = grade
        per_ep["truth"] = y
        print(json.dumps(grade, indent=1))

    # 5. does this reproduce the previous run's kept list?
    if a.compare and os.path.exists(a.compare):
        prev = pd.read_csv(a.compare)
        prev_keep = sorted(int(e) for e, n, m, l in zip(prev["episode_index"], prev["flag_noise"], prev["flag_misaligned"], prev["flag_label"])
                           if not (n or m or l))
        same = prev_keep == keep
        report["matches_previous_kept_list"] = same
        print(f"kept list {'IDENTICAL to' if same else 'DIFFERS from'} {a.compare}: "
              f"{len(set(keep) - set(prev_keep))} newly kept, {len(set(prev_keep) - set(keep))} newly dropped")

    json.dump(report, open(out_path, "w"), indent=1)
    per_ep.to_csv(out_path.replace(".json", "_episodes.csv"), index=False)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
