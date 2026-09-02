#!/usr/bin/env python
"""The score column, as a Geneva column.

This is the thesis of the post made executable. Every curation method in the current
literature -- delete the bad rows, or keep them and condition on a quality token -- needs a
derived score computed from the trajectory and stored next to it. Here that score IS a column
on the table the trainer reads, declared once and materialised by a checkpointed job.

No pixels, no GPU, no second store. The inputs are scalar columns already in the table.
"""
import os
import argparse, json, time
import numpy as np, pyarrow as pa
from geneva import connect, udf


@udf(data_type=pa.float32(), input_columns=["action_joint_velocity"],
     max_checkpoint_size=8192)
def jerk_score(action_joint_velocity: pa.Array) -> pa.Array:
    """Per-frame motion roughness: |d(joint velocity)/dt|, summed over joints.

    The frame-level ingredient of the log-dimensionless-jerk metric that robot-data QA work
    uses as a quality gate -- the one we measured as INVERTED against DROID's success labels.
    Having it as a column is what let us check that in 63 seconds.
    """
    v = np.asarray(action_joint_velocity.to_pylist(), dtype=np.float32)
    if v.ndim == 1:
        v = v.reshape(len(v), -1)
    d = np.zeros(len(v), dtype=np.float32)
    if len(v) > 1:
        d[1:] = np.abs(np.diff(v, axis=0)).sum(axis=1)
        d[0] = d[1]
    return pa.array(d, pa.float32())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lance-root", required=True)
    p.add_argument("--table", default="frames")
    p.add_argument("--column", default="jerk_score")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default="out/geneva_score.json")
    a = p.parse_args()

    db = connect(a.lance_root)
    tbl = db.open_table(a.table)
    res = {"rows": tbl.count_rows(), "column": a.column, "concurrency": a.concurrency}
    if a.column in [f.name for f in tbl.schema]:
        tbl.drop_columns([a.column])

    t = time.perf_counter()
    tbl.add_columns({a.column: jerk_score})      # a declaration: schema changes, no compute
    res["declare_s"] = round(time.perf_counter() - t, 3)
    print(f"column declared in {res['declare_s']}s", flush=True)

    t = time.perf_counter()
    tbl.backfill(a.column, concurrency=a.concurrency)
    res["backfill_s"] = round(time.perf_counter() - t, 1)
    res["non_null"] = tbl.count_rows(f"{a.column} IS NOT NULL")
    res["rows_per_s"] = round(res["non_null"] / max(res["backfill_s"], 1e-9))
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
