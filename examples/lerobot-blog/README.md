# LeRobot x LanceDB blog — reproduction scripts

The scripts behind the launch blog: feature engineering with Geneva, search and curation on
the same table the trainer reads, held-out evaluation, and the Foxglove recordings.

**Throughput benchmarking lives in `../lerobot-loader-benchmark/`** — that is a separate,
self-contained harness with its own pitfalls guide. Nothing here measures throughput.

```
curation/   feature engineering, indexing, search, dataset EDA
eval/       held-out loss and action MAE, before/after traces
viz/        Foxglove servers + headless recorder
config/     rename map, held-out episode list
```

## Setup

```bash
pip install lerobot lancedb geneva "lerobot-lancedb>=0.3.0"
pip install nvidia-npp-cu12          # torchcodec links libnppicc.so.12 but doesn't declare it
export LD_LIBRARY_PATH=$VENV/lib/python3.12/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH

export LANCE_ROOT=/data/droid_lance  # or s3://your-bucket/droid_1.0.1-lance
export RENAME_MAP=config/rename_map.json
export HOLDOUT=config/cur_holdout.json
mkdir -p out
```

Every script writes to `out/` and reads `LANCE_ROOT`. No paths are hardcoded.

## Feature engineering — Geneva

Geneva adds a column to a Lance table by registering a UDF as a *virtual* column, then running
a distributed, checkpointed backfill. It works **directly on the `.lance` table** — no
DataLoader, no dataset wrapper.

```bash
python curation/geneva_score.py    # scalar UDF: per-frame jerk from action_joint_velocity
python curation/geneva_embed_blob.py --concurrency 2   # GPU UDF: SigLIP2 frame embeddings
```

`geneva_score.py` declares in 0.016 s and backfills 610,403 rows in 16.7 s (36,551 rows/s).
`geneva_embed_blob.py` is the interesting one: a stateful GPU UDF that pulls mp4 bytes straight
out of the blob column with `take_blobs("video_bytes", indices=[...])`, decodes with torchcodec
and embeds — 712 frames/s on 2 GPUs.

Two things that will bite you, both learned the hard way:

- **`take_blobs` is `take_blobs(column, indices=...)`**, not `(indices, column)`. Getting it
  backwards inside a `try/except` produced a run that reported "12,208 frames/s, all non-null"
  and was **100% zero vectors**. Check embedding norms, not null counts.
- **Don't wrap a `LeRobotDataset` inside a Geneva UDF.** It stalls silently in a Ray actor —
  model resident, 0% GPU, no error. Read the table directly.

## Search and curation

```bash
python curation/index_search.py    # build IVF-PQ vector index + FTS index
python curation/hybrid_search.py   # semantic + SQL predicate in one query
python curation/teleop_quality.py  # per-episode teleoperation quality EDA
```

The point of the blog section: the embedding, the jerk score and the raw frames sit in **one
table**, so a curation query is a single hybrid vector+SQL call (32.0 / 27.4 / 69.0 ms across
three cases) rather than a join across a feature store, a vector database and object storage.

## Evaluation

```bash
python eval/eval_loss.py --out out/loss.json \
  --checkpoints early=runs/ckpt/000080/pretrained_model \
                trained=runs/ckpt/010000/pretrained_model

python eval/eval_holdout.py --episodes 226,39,130 --steps 60 --out out/per_ep.json \
  --checkpoints early=... trained=...
```

### Why the baseline is two checkpoints of one run

Do not use a pretrained base checkpoint as "before". `lerobot/smolvla_base` carries action
normalization for the SO-100 arm **in degrees** (mean ~125, std ~39) and has no DROID entry at
all (DROID is mean ~0.01, std ~0.33); `make_policy(cfg, ds_meta=droid_meta)` then silently
re-initializes its action head to DROID's 8 dims. Each policy normalizes `action` with its own
statistics, so the two MAEs are not even in the same units — you are scoring a random head
against wrongly scaled targets. That mistake invalidated four numbers here before it was caught.

`eval_loss.py` therefore **asserts** both checkpoints' `action.mean`/`action.std` are identical
and aborts otherwise; `before_after_traces.py` aborts if the two see different ground truth.

Measured, 80 steps vs 10,000 steps of the same run, held-out episodes only:

| metric | 80 steps | 10,000 steps | |
|---|---|---|---|
| held-out loss | 0.4777 +- 0.0438 | 0.2165 +- 0.0257 | 54.7% lower |
| action-chunk MAE | 0.5130 | 0.3706 | 27.8% lower |
| next-action MAE | 0.3259 | 0.1004 | 69.2% lower |
| episodes improved | | 48 / 48 | |

**Pick evaluation episodes by motion, not convenience.** Improvement correlates *negatively*
with how much the arm moves (r = -0.45): a near-static episode flatters the finetune badly.
Quartiles ran 70.5 / 71.2 / 71.4 / 64.0%. Report where your chosen episode sits — the one in
the blog video (226) is at the 98th percentile of motion, i.e. harder than average.

## Foxglove recordings

```bash
# dataset playback: cameras + state/action traces, streamed from LANCE_ROOT
python viz/fox_record.py --episode-index 7 --seconds 30 --name dataset

# before/after: camera + one panel per action dim, three series overlaid
python eval/before_after_traces.py --episode 226 --start 339 --steps 300 \
  --out out/traces.json --checkpoints early=... trained=...
python viz/fox_record.py --layout fox_layout_ba --ws-port 8766 --seconds 32 --name before_after \
  --server-cmd "python viz/fox_before_after.py --fps 10 --loops 2 --dims 0,5,6"
```

Four ways these plots come up **blank**, all of which cost me a re-record:

1. No `schema` on the channel — Lichtblick cannot resolve the plot path. Reuse lerobot's
   `_SCALARS_SCHEMA`.
2. The path must be `"<topic>.scalars[<i>].value"`. Using `scalars[:].value` is *legal and
   plotted*, but it splices every dimension into one zigzag series and is meaningless.
3. The y-axis defaults to 0..1, which hides negative actions entirely. Set
   `minYValue`/`maxYValue` from the data.
4. `followingViewWidth: 0` is a zero-width live window. This one shipped in a published video
   whose caption claimed traces that were not visible.

Recording is a headless screenshot loop, so capture runs at ~1.5 fps regardless of
`--fps`. Encode the captured frames at a rate that matches wall time or the result is an
unreadable timelapse.

## Caveat on train/holdout splits

`--dataset.exclude_episodes` (equivalently `episodes=[...]`) is **not** a transparent filter on
the upstream map-style reader: excluding 100 of DROID's 95,658 episodes reported 39,966,958
frames where the unfiltered dataset has 27,630,375, and training died within 150-270 s across
5 seeds. The Lance reader gave the expected length (-28,505 frames). Not fully diagnosed. If
you use a split, assert the resulting length before trusting anything downstream.
