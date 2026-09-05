# Old maths vs per-episode UDFs

`old_math.py` is the scoring exactly as the earlier `detect.py` computed it: one numpy pass over the
whole frames table, per-episode slices by contiguous row ranges. `new_udfs.py` is the per-episode UDF
form now used by `../detect.py` (the SigLIP2 embedding UDF is not part of this test; `goal_dist` is
tested on stand-in unit vectors). `run_eq.py` reads a dataset's tabular columns, scores it both ways,
also runs the UDFs through `geneva` `add_columns` + `backfill` on a local `episodes` table, and compares
every number and the flagged-episode lists.

```bash
# all 206 pusht episodes, direct calls and geneva backfill (pusht actions and states are 2-D)
python run_eq.py --root s3://<bucket>/lerobot/pusht-lance --action-dims 2 --pos-dims 2 --geneva
# a 20-episode DROID sample, direct calls
python run_eq.py --root s3://<bucket>/lerobot/droid_1.0.1-lance --episodes 20
```

Result on 2026-09-05: pusht PASS on both paths (max |old - new| <= 1.5e-8, flagged lists identical);
DROID sample PASS for `jerk_score` and `act_lag`. `goal_dist` is NaN in both implementations on the
DROID sample because a random 20-episode sample leaves most tasks with a single episode, so the
"median of the other episodes" is empty; this is the same behaviour, not a difference.

Note for DROID-scale tables: `episode_index = N` inside a UDF scans the frames table unless there is a
scalar index on `episode_index`. On the 27.6M-row DROID table without one it took ~9 s per episode;
build the index first (`frames.create_scalar_index("episode_index")`).
