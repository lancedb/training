# Pitfalls

Every item here is a **wrong number this harness actually produced**, not a hypothetical.
Most were caught only because a result looked too good. If you add a backend or change the
read pattern, re-read this list.

### 1. `shuffle=True` is not comparable across dataset styles
PyTorch **refuses** `shuffle=True` on an `IterableDataset`. So the naive harness gives the
map-style backend a *global shuffle over every frame* and the streaming backend its internal
reservoir window (default 1,000 frames), then reports the ratio as a throughput difference.
Much of that gap is a difference in what was asked for, not in how fast it was served.

Use `--no-shuffle` to match access patterns when comparing raw IO, and report **shuffle
coverage as its own column** rather than folding it into samples/s.

### 2. Page cache makes local runs superlinear
A worker sweep once showed 4.27x scaling from 8 to 16 workers. Impossible. `vmtouch` showed
60-64% of the dataset resident and *growing across the sweep*, so later points were reading
RAM and being credited to the loader. Drop caches before **every** local run:
`sync; echo 3 > /proc/sys/vm/drop_caches`.

### 3. NUMA topology dominates the map-style upstream reader
On a 4-socket box (4x28 cores), unpinned upstream showed **36.5%** data wait; pinned to one
NUMA node, **0.9%** — same code, same data. `taskset -c 0-31` is **not** equivalent: it lands
on node 0 and looks like a free win. Use `numactl --cpunodebind --membind`, and always state
which you used. The Lance reader was insensitive (~1% either way) because it plans whole
batches instead of fetching one frame per call.

### 4. The memory tracker can silently under-report
`RSSTracker` originally shelled out to `ps --ppid`. Under memory pressure that subprocess
fails, the `except` branch falls back to self-only, and the tracker reports a **smaller**
number for a **bigger** workload — 0.61 GB at buffer=64,000 vs 39.85 GB at buffer=1,000. It
now reads `/proc/*/smaps_rollup` transitively and also records `sys_mem_consumed_gb`, a
`MemAvailable` delta that per-process accounting cannot break. **Trust that floor.**

### 5. Streaming throughput can be fast because it is reading the WRONG frames
`StreamingLeRobotDataset` used a dataset-global frame index as an in-file timestamp. With a
camera in `delta_timestamps` it **clamps to episode bounds**, silently returning a boundary
frame and collapsing distinct requests onto one cached decode — which *inflates* samples/s.
Fixed by huggingface/lerobot#4316 (also 7-19% faster).

The bug is inert iff `max|dataset_from_index/fps - videos/<cam>/from_timestamp| ~ 0` — true for
single-file datasets like pusht, false for koch / berkeley / droid.
**Never report a streaming number without verifying the frames are the right ones.**

### 6. Parallel launches contaminate each other
Four separate contaminations, one root cause: independently launched scripts coordinating
through advisory gates — a process-absence check that passes during a `sleep`, an `OR` that
should have been an `AND`, `SIGSTOP` not stopping already-spawned children, orphaned children
surviving a parent kill. **Gates are not a serialisation primitive.** Run phases sequentially
in one process.

Kill by PID or process group, never `pkill -f <pattern>`: the pattern matches your own shell's
command line, which killed two of my own runs (exit 144).

### 7. float32 timestamps run out of precision on long videos
DROID's within-video timestamps reach ~10,896 s, where float32 spacing is 9.77e-4 — only 1.02x
headroom under `tolerance_s=1e-3`. Pass `--tolerance-s 0.005` on **both** sides, or upstream
fails for a reason unrelated to what you are measuring.

### 8. An episode allowlist changes the dataset it returns
Passing `episodes=[...]` (which is what `--dataset.exclude_episodes` becomes) is **not** a
transparent filter. Excluding 100 of DROID's 95,658 episodes made the upstream map-style
dataset report **39,966,958 frames where the unfiltered set has 27,630,375** — 44% *larger*
after removing episodes — and training then died within 150-270 s on
`IndexError: Invalid frame index=... must be less than ...`, reproducibly across 5 seeds.
The same flag on the Lance reader gave the expected length (-28,505 frames).

If you use a train/holdout split, **assert the resulting length and spot-check that no held-out
episode is reachable** before believing any number from that run. Not fully diagnosed; treat
any filtered-dataset result as suspect until it is.

### Data hazard: DROID has out-of-range frame indices
Independently of #8, DROID contains frames whose index exceeds their video's stream length. A
reader that converts a timestamp into a position can reach them under random access. Without an
episode filter this is **probabilistic** — 1 of 3 seeds died at 267 s, the other two ran 25 min
clean — so do not claim a reader "cannot finish". `toto`, `berkeley_rpt` and
`berkeley_autolab_ur5` have a worse version: they cannot be shuffled at all, identically on
both readers. That is a source-data defect, not a format one, and it affected over a third of
the public datasets tried.

### 9. `lerobot-train` abbreviates step counters, and the naive regex silently drops most lines
Logs print `step:10K` and `smpl:2M`. A `step:(\d+)` pattern matches the **10** of "10K", so a
`step >= 500` warmup filter kept 3 of 40 lines and computed "steady state" from steps 500-750 —
early training, when the loader is still spinning up. That reported 519 samples/s / 1.5% data
wait where the real steady-state values were 495 / 1.7%.

Use `parse_train_log.py`, which resolves K/M/G suffixes and cuts warmup **by step number**
rather than by row count. Runs shorter than 1,000 steps are unaffected, which is why the
per-step grids were right while the 10,000-step summary was not.

Cross-check every rate against wall clock: `steps x batch x gpus / seconds`. For the runs above
that gives 487.7 and 354.5 samples/s against steady-state 495 and 361 — the gap is startup and
checkpointing, and if the two disagree by more than that, the log parse is wrong.
