#!/usr/bin/env python
"""Make a LIBERO Lance dataset messy in three realistic, known ways.

LIBERO is clean: every episode succeeds, every label is right, every action lines up with its
frame. Real collections are not. This script injects three defects seen in real teleop data,
each on a disjoint random 10% of episodes, and records which episode got what in a manifest
that lives OUTSIDE the table (in the real world you do not have it; here it grades the detectors):

  label_swap    the instruction attached to the episode is another task's from the same suite
                (a mislabeled demonstration)
  action_noise  jittery actions: Gaussian noise on the arm dims plus occasional spikes and
                gripper flips (a shaky or lagging teleop device)
  misaligned    the action stream runs ahead of the observations by a fixed offset
                (a logging / timestamp bug)

Only the tabular frames table is rewritten, in the same row order. Videos, meta and stats are
copied as-is, because that is what a real messy dataset looks like: the pixels are fine.
"""
import argparse
import json
import os
import shutil
import time

import lance
import numpy as np
import pyarrow as pa


def fsl_to_np(col: pa.ChunkedArray) -> np.ndarray:
    col = col.combine_chunks()
    return col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)


def np_to_fsl(arr: np.ndarray, like: pa.DataType) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(arr.reshape(-1), like.value_type), arr.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="clean Lance dataset root")
    ap.add_argument("--dst", required=True, help="messy copy to create")
    ap.add_argument("--frac", type=float, default=0.10, help="fraction of episodes per defect type")
    ap.add_argument("--noise-std", type=float, default=0.35, help="noise std as a multiple of each action dim's std")
    ap.add_argument("--spike-frac", type=float, default=0.04)
    ap.add_argument("--shift", type=int, default=6, help="frames the action stream runs ahead (0.6 s at 10 fps)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    if os.path.exists(a.dst):
        raise SystemExit(f"{a.dst} exists; remove it first")
    t0 = time.time()
    os.makedirs(a.dst)
    for name in os.listdir(a.src):
        if name == "frames.lance":
            continue
        s, d = os.path.join(a.src, name), os.path.join(a.dst, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, copy_function=os.link)   # videos: hardlinks, no bytes copied
        else:
            shutil.copy2(s, d)
    print(f"copied meta + videos (hardlinks) in {time.time() - t0:.1f}s")

    ds = lance.dataset(os.path.join(a.src, "frames.lance"))
    T = ds.to_table()
    n = T.num_rows
    ep = T.column("episode_index").to_numpy()
    task = T.column("task_index").to_numpy().copy()
    act = fsl_to_np(T.column("action"))
    episodes = np.unique(ep)
    print(f"{n:,} frames, {len(episodes):,} episodes, action dims {act.shape[1]}")

    # suites: LIBERO's 40 tasks come in four blocks of ten
    ep_task = {int(e): int(task[ep == e][0]) for e in episodes}
    ep_suite = {e: t // 10 for e, t in ep_task.items()}

    k = int(round(a.frac * len(episodes)))
    perm = rng.permutation(episodes)
    groups = {"label_swap": sorted(int(e) for e in perm[:k]),
              "action_noise": sorted(int(e) for e in perm[k:2 * k]),
              "misaligned": sorted(int(e) for e in perm[2 * k:3 * k])}
    act_std = act[:, :6].std(axis=0)
    new_task = {}
    for e in groups["label_swap"]:
        same_suite = [t for t in range(10 * ep_suite[e], 10 * ep_suite[e] + 10) if t != ep_task[e]]
        new_task[e] = int(rng.choice(same_suite))
        task[ep == e] = new_task[e]
    for e in groups["action_noise"]:
        m = np.where(ep == e)[0]
        act[m, :6] += rng.normal(0, a.noise_std * act_std, size=(len(m), 6)).astype(np.float32)
        spikes = m[rng.random(len(m)) < a.spike_frac]
        dims = rng.integers(0, 6, size=len(spikes))
        act[spikes, dims] = rng.choice([-1.0, 1.0], size=len(spikes)).astype(np.float32)
        flips = m[rng.random(len(m)) < 0.02]
        act[flips, 6] = -act[flips, 6]
        act[m, :6] = np.clip(act[m, :6], -1.0, 1.0)
    for e in groups["misaligned"]:
        m = np.where(ep == e)[0]              # rows of this episode, in time order
        src = np.minimum(np.arange(len(m)) + a.shift, len(m) - 1)
        act[m] = act[m][src]

    cols = {name: T.column(name) for name in T.column_names}
    cols["task_index"] = pa.array(task, type=T.schema.field("task_index").type)
    cols["action"] = np_to_fsl(act, T.schema.field("action").type)
    out = pa.table(cols, schema=T.schema)
    lance.write_dataset(out, os.path.join(a.dst, "frames.lance"), mode="create", max_rows_per_file=n)
    chk = lance.dataset(os.path.join(a.dst, "frames.lance"))
    assert chk.count_rows() == n and chk.schema == T.schema
    print(f"wrote messy frames table: {n:,} rows, same schema, same order")

    manifest = {"src": a.src, "seed": a.seed, "frac_per_type": a.frac,
                "params": {"noise_std_x": a.noise_std, "spike_frac": a.spike_frac, "shift_frames": a.shift},
                "groups": groups, "label_swap_new_task": new_task,
                "n_episodes": int(len(episodes)), "n_corrupted": int(3 * k)}
    json.dump(manifest, open(os.path.join(a.dst, "messy_manifest.json"), "w"), indent=1)
    print(f"{3 * k} of {len(episodes)} episodes corrupted ({k} per type); manifest -> {a.dst}/messy_manifest.json")


if __name__ == "__main__":
    main()
