# LLM Pretraining on a Single LanceDB Table

Pretrain a GPT on a text corpus where **one LanceDB table is the entire data
layer** — corpus, curation signals, token ids, EDA, retrieval, and the
training dataloader all operate on the same table. No webdataset shards, no
tokenized parquet copies, no pre-shuffled manifests.

```
             ┌──────────────────────────────────────────────────────────┐
             │                    ONE LANCE TABLE                       │
             │  id │ text │ source │ score │ is_dup │ input_ids │ ...   │
             └──────────────────────────────────────────────────────────┘
   ingest.py ──▲          ▲            ▲                ▲          │
   (raw text)  │          │            │                │          ▼
               │   curate.py     curate.py       tokenize_data.py │ train.py
               │   SQL EDA +     zero-copy       zero-copy column │ StreamingDataset
               │   FTS search    `is_dup` col    `input_ids`      │ elastic + resumable
               │                                                  ▼
               │                                     torchrun, 1..64+ GPUs
               └── same table serves retrieval / data forensics after training
```

The classic pretraining data pipeline materializes the dataset several times:
raw → cleaned copy → tokenized copy → pre-shuffled shards. With Lance's
zero-copy schema evolution each derived signal is **a new column on the same
table** — adding `input_ids` to a multi-TB corpus writes only the new column's
bytes and rewrites nothing. Curation predicates (`NOT is_dup AND score >= 2`)
become SQL prefilters on the dataloader instead of new dataset copies, so
every ablation ("what if we train only on score ≥ 3?") is a flag change, not
a preprocessing job.

## Scripts

| Script | Purpose |
|---|---|
| `ingest.py` | Stream a corpus (FineWeb-Edu or synthetic) into a Lance table |
| `curate.py` | SQL EDA, BM25 full-text search, dedup flag as a zero-copy column |
| `tokenize_data.py` | Tokenize once; store `input_ids` as a zero-copy column |
| `geneva_backfill.py` | Distributed, checkpointed tokenization via Geneva (Ray) |
| `train.py` | torchrun-ready pretraining with `lancedb.streaming.StreamingDataset` |
| `bench_loader.py` | Loader-only throughput probe (tune `read_batch_size`/splits) |
| `sample.py` | Generate text from a trained checkpoint |
| `forensics.py` | Vector index, hybrid search, generation attribution, near-dups |
| `bench_parquet.py` | The standard-workflow control: Parquet export + S3 read benches |
| `build_packed_datasets.py` | Identical pre-packed blocks -> Parquet, pre-shuffled Parquet, MDS shards, Lance table (8 parallel packers) |
| `blocks_loaders.py` | Parquet random-take and pre-shuffled-sequential block loaders (local + s3://), the A/B controls |
| `mosaic_compare.py` | MosaicML Streaming A/B: elastic, resume, throughput |
| `elastic_pack_check.py` | Packed elastic determinism + cross-topology (`merge_state_dicts`) resume on the real table |
| `LOADER_TUNING.md` | Measured loader knob sweep, profile, and the settings the 8-GPU runs use |
| `verify_e2e.py` | Offline CPU verification of the whole pipeline (~2 min) |

## Setup

```bash
cd examples/llm_pretraining
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e .          # add `-e .[hf]` for FineWeb-Edu + HF tokenizers
```

Requires `lancedb >= 0.38` — the first release line with loader-native sequence
packing (`pack_sequences`, lancedb/lancedb#3920) and `StreamingDataLoader`
(consumer-committed checkpoints for worker processes). Until 0.38.0 is on PyPI,
build the Python package from a `v0.38.0-beta.*` tag (`rustup`, `protoc`, then
`cd python && maturin build --release`; ~30 min on 112 cores).

## Quickstart (offline, CPU)

```bash
python verify_e2e.py
```

This runs the full pipeline on a synthetic corpus and asserts the two claims
that make this loader different, using real training runs:

```
  [+] ingest: row count
  [+] curate: is_dup column exists
  [+] curate: duplicates flagged  141 dups
  [+] tokenize: input_ids column exists
  [+] tokenize: ids round-trip
  [+] elastic: ws=1 == ws=2 global batches  563 steps
  [+] elastic: filter honored, full coverage  2252/2253 rows (1 dropped by split rounding)
  [+] train: completes  val_loss=4.6852
  [+] resume: matches uninterrupted run  4.685200 vs 4.685200
  [+] pack: deterministic block stream  40 blocks bit-identical across fresh iterations
  [+] pack: 100% utilization vs pad/truncate  packed=100% real tokens; pad/truncate sees 17% of corpus tokens at seq_len=256
  [+] pack: train completes  val_loss=4.9157
  [+] pack: mid-epoch resume matches uninterrupted  4.915700 vs 4.915700
  [+] pack: checkpoint pins pad/eos/epoch  pad_id mismatch raised
  [+] pack: elastic ws=1 == ws=2 global blocks  30 steps
  [+] pack: cross-topology merge resume ws2 -> ws1  merged rank states continue exactly

============================================================
E2E COMPLETE: 16 passed  0 failed
============================================================
```

A run killed at step 12 and resumed lands on the **bit-identical** validation
loss as the uninterrupted run; the samples that form each global step are
identical whether the job runs on one process or two — and, new with the
merged packing API, that now holds for *packed* blocks too: per-rank packed
checkpoints merge (`StreamingDataset.merge_state_dicts`) into a
topology-independent checkpoint that resumes exactly on a different world size.

## Real corpus

```bash
python ingest.py --source fineweb --rows 2400000        # streams FineWeb-Edu via `datasets`
python ingest.py --source fineweb-parquet --sample 10BT --files 4 --rows 2400000   # same docs, ~10x faster: parallel shard download + row-group ingest
python curate.py
python tokenize_data.py --tokenizer hf:gpt2             # or geneva_backfill.py for the distributed version
```

Zero-copy evolution in action (synthetic 3K-row corpus shown; the mechanics
are identical at any scale):

```
DEDUP -> zero-copy `is_dup` column
flagged 141 duplicate rows
data files: 1 -> 2, bytes: 853,344 -> 854,014 (+670 for the new column; nothing rewritten)

TOKENIZE (byte) -> zero-copy `input_ids` column
tokenized 3000 docs -> 4,504,547 tokens in 0.9s
data files: 2 -> 3, bytes: 854,014 -> 6,230,499 (+5,376,485 for token columns; nothing rewritten)
table version: 4 (previous versions still readable)
```

Every stage bumps the table version, so a training run can be pinned to the
exact data version it saw (`db.open_table(name).checkout(v)`), and the raw
`text` column is still right there for FTS/vector retrieval, data forensics,
and eval-set inspection after training.

This example backfills columns with `tbl.to_lance().add_columns(udf)` to
stay zero-extra-dependency and offline-runnable. For real feature
engineering — GPU UDFs, checkpointed distributed backfills, laptop-Ray to
KubeRay with the same code — use [Geneva](https://github.com/lancedb/geneva);
the [object-detection example](../../object-detection/) shows that flow.


## Results: 8x H100, lancedb 0.38 (merged sequence packing)

The pipeline below was re-run end to end on an 8x H100 node (112 cores, virtio
disk) with the *released* packing API, replacing the earlier 4x H100 numbers
that used a pre-merge branch. Same corpus (first 2.4M FineWeb-Edu docs),
same model, same 512 x 1024-token global batch.

| Stage | Wall time | What happened |
|---|---|---|
| Ingest 2.4M docs (11.4B chars) | **2m 06s** | 4 parquet shards -> 4.8GB Lance table (was 3m00s via the `datasets` iterator) |
| Curate: EDA + FTS + dedup | **4m 10s** | 22,558 dups; `is_dup` col = +306,604 bytes on a 5.1GB table, byte-identical to the previous run |
| Geneva tokenize (32 Ray workers, geneva 0.15) | **4m 54s** | 2.43B GPT-2 tokens + `n_tokens` as zero-copy columns; table 12GB, v13 |
| **Train GPT-2 124M, 1 epoch = 2.43B tokens** | **14m 08s** | 8x H100, **3.18M tok/s, 34.5% MFU** flat from step 100; val loss 3.236 (4x H100: ~50 min, 1.60M tok/s, 35%, val 3.230) |
| Upload table to S3 us-east-2 (from Norway) | 1m 47s | 11.9GB, `aws s3 sync` |
| **Train from `s3://` across the Atlantic** | 500 steps | **3.16M tok/s / 34.4% MFU — identical to local disk** |
| Pre-pack 2.37M blocks into 4 standard formats | 4m 55s (+2 min derived) | 8 parallel packers; was 33 min single-process |
| Loader A/B, local + S3 (below) | 300-500 steps each | every loader but Parquet random-take sits at the same 34.5% compute ceiling |
| Packed elastic + cross-topology resume | 3m 24s (CPU) | ws=8 == ws=4 global steps; 8-rank checkpoint merged and resumed on 4 ranks, bit-identical |

Raw text to training-ready: **11 minutes**. To a Chinchilla-optimal GPT-2 124M:
**25 minutes** including prep.

```
epoch 0 step 1000/4635 | loss 3.9628 | 3,180,409 tok/s | mfu 34.5% | q 24264/49928/41176/31640 | fetch 168.3s tx 15.1s
  val loss @ step 1500: 3.5971
  val loss @ step 3000: 3.3201
  val loss @ step 4500: 3.2363
final: opt_step=4636 tokens_seen(rank0)=302,999,960 val_loss=3.2361
```

### What changed with the merged API — and what we had to tune

The released packer differs from the pre-merge branch in ways that matter at
8 ranks. Measured loader-only (one rank's shape, packed, local disk; full sweep
and profile in [`LOADER_TUNING.md`](LOADER_TUNING.md)):

| loader config | tok/s (one rank shape) |
|---|---|
| library defaults (`io_queue_depth=4`, `transform_parallelism=cpu_count`) | 158k |
| `io_queue_depth=1`, `transform_parallelism=2` | 795k |
| same, 16 splits per rank (`--num-splits 128` on 8 ranks) | 960k |
| 8 concurrent rank processes, aggregate | ~4.8M |

Two root causes: thread oversubscription (8 ranks x (128 I/O + 112 transform)
threads on 112 cores, all contending for the GIL) and the packer snapshotting
every owned split's token buffer after every block (`_commit_pack_state`, 42%
of main-thread time). `train.py` now exposes `--io-queue-depth` (1),
`--transform-parallelism` (2) and `--num-splits`.

Even tuned, packing on the training thread costs GPU time: the in-process run
held **28% MFU while I/O was in flight and 34% after the table was fully read**.
Moving iteration into worker processes with lancedb's `StreamingDataLoader`
(`--num-workers 2`, forkserver) gives **34.5% from step 100 onward** — the same
ceiling as pre-packed blocks (below), i.e. the loader is fully hidden.
Checkpoints stay exact: the loader commits worker state only for batches the
trainer has received (a `num_workers=2` run killed and resumed with
`num_workers=0` reproduces the uninterrupted val loss to 4 decimals).

### Loader A/B: identical blocks, five loaders, local and S3

`build_packed_datasets.py` writes the *same* 2,373,376 packed 1024-token blocks
as Parquet (stream order), pre-shuffled Parquet, MDS shards and a Lance table.
`train.py --blocks-mode ...` trains the same GPT-2 124M on them; only the
loader changes. The Lance corpus row is the on-the-fly path (pack + shuffle
from the raw table, no pre-pack at all).

| loader (GPT-2 124M, 8x H100, 512 x 1024 tok/step) | local disk | S3 us-east-2 from Norway | extra copies | prep |
|---|---|---|---|---|
| **Lance corpus table** — pack + shuffle on the fly | 3.16M tok/s / 34.3% | **3.16M / 34.4%** | 0 | 0 |
| Lance blocks table (pre-packed) | 3.17M / 34.5% | 3.16M / 34.3% | 1 (9.7GB) | 5 min |
| MosaicML Streaming (MDS) | 3.17M / 34.5% | 3.18M / 34.6% between shard fetches, ~0.75M during each (mean 2.87M / 31.2% over steps 125-500); fills an 8.9GB shard cache per node | 1 (9.7GB) + per-node cache | 5 min |
| Parquet, pre-shuffled shards read sequentially | 3.17M / 34.4% | 3.18M / 34.5% | 2 (5.0GB + 5.0GB) | 5 min + reshuffle per epoch/filter |
| Parquet, random-take (global shuffle over row groups) | 2.02M / 21.9% | **74k / 0.8%** (43x slower than the Lance table from the same bucket) | 1 (5.0GB) | 5 min |

GPU utilisation sampled by `nvidia-smi` sits at 93-100% for every row except
Parquet random-take. Sequential streaming is fast everywhere — that was never
the problem. The difference is workflow: the Parquet and MDS rows each start
with a 5-minute materialisation that bakes in the tokenizer, `seq_len`, filter
and dedup decisions (and, for the pre-shuffled copy, one shuffle order), while
the Lance corpus row changes any of them by editing a SQL string or a seed.

MosaicML's guarantees tie with Lance on the same blocks
(`mosaic_compare.py`): elastic determinism ws1==ws2==ws4 and ws4->ws2 resume
both pass for both loaders.

### Packed runs are now elastic

The pre-merge packer could only resume at the same world size. With
`blocks_per_epoch` fixed per split (exact budget computed from `n_tokens`),
the packed stream is topology-independent:

```
$ python elastic_pack_check.py --db ~/runs/small/db --num-splits 128 --ws 8 4
elastic: 30 global steps ws=8 == ws=4: True
resume: ws=8 for 12 steps -> merge 8 rank states -> ws=4: next 18 global steps match: True
blocks_per_epoch=2,373,376 (18,542 per split)
```

And on real GPUs (`runs/resume_demo.sh`): an 8-GPU packed run checkpointed
at step 200 and `kill -9`'d at step 260 was resumed **on 4 GPUs** (batch 64 x
accum 2 keeps the 512-sequence global step) from the eight per-rank loader
states, and finished the 400-step budget at val loss 4.9718 vs 4.9704 for the
uninterrupted 8-GPU reference (the two runs shard the eval set differently;
the training batches are provably identical per the check above).

```
resumed from .../ckpt/step_00000200.pt (epoch 0, opt step 200)
epoch 0 step 300/400 | loss 5.3989 | 1,639,864 tok/s | mfu 35.6% | ...     # 4 GPUs
final: opt_step=400 tokens_seen(rank0)=26,214,400 val_loss=4.9718
```

### Scale run: 17.5M docs, GPT-2 medium (354M) on a 7.0B-token budget

Same commands, 7x the corpus: 24 shards of FineWeb-Edu `sample-100BT`.

| Stage (17.48M docs) | Wall time | Notes |
|---|---|---|
| Ingest (`--source fineweb-parquet --files 24`) | **14m 08s** | 45GB table, v1 |
| Curate + dedup | **25m 45s** | 1,140,626 dups (6.5%) flagged as a zero-copy column |
| Geneva tokenize (64 Ray workers) | **22m 57s** | 18.06B GPT-2 tokens (16.75B post-filter); table 81GB, v13 |
| **Train GPT-2 medium 354M, 7.0B tokens = 13,351 steps** | **1h 37m** | 1.34M tok/s / 41.1% MFU steady (mean 1.26M / 38.6% — see GC note in `LOADER_TUNING.md`); **val loss 2.841** (4x H100: 3h 06m, 684k tok/s, val 2.840) |

Pad/truncate at `seq_len=1024` would see only **61.9%** of this corpus's
tokens; packing sees all of them. The permutation build over 16.2M filtered
rows needs `LANCEDB_PERM_BUILDER_MEMORY_LIMIT=8589934592` (the DataFusion sort
pool defaults to 100MB and fails fast with a clear error otherwise).

```
  val loss @ step 2000: 3.4359
  val loss @ step 4000: 3.1417      <- already below the 124M's final 3.236
  val loss @ step 8000: 2.9297
  val loss @ step 12000: 2.8452
final: opt_step=13351 val_loss=2.8410
```

TBD_MEDIUM_AB

### Known rough edges (upstream notes)
- lancedb 0.38 is beta-only at the time of writing: build the wheel from a tag.
- Packed `state_dict()` requires an aligned cycle (every owned split has
  emitted the same block count); checkpoint on optimizer-step boundaries where
  `batch_size x grad_accum` is a multiple of the rank's split count (and, with
  workers, `ckpt_every x grad_accum` a multiple of `num_workers`).
- `StreamingDataLoader` workers: use `forkserver`/`spawn`, never `fork`, inside
  CUDA + DDP ranks. If workers die at start-up with
  `SemLock._rebuild -> FileNotFoundError`, check `systemd-logind RemoveIPC`
  on the host (see [`ISSUE_streaming_workers.md`](ISSUE_streaming_workers.md)).
- Interpreter exit can hang after a worker-process run (Rust runtime + torch
  teardown); `train.py` exits with `os._exit(0)` once checkpoints and the final
  eval are written.
- geneva 0.15 installs cleanly with lancedb 0.37 in its own venv (the 0.14
  `lancedb==0.34` pin is gone).

## Training

Single GPU / debug:

```bash
python train.py --model small --batch-size 16 --epochs 1
```

One 8×GPU node (the configuration used for the results above):

```bash
torchrun --nproc-per-node 8 train.py --model small --tokenizer hf:gpt2 \
    --pack --compile --batch-size 32 --grad-accum 2 --seq-len 1024 --epochs 1 \
    --num-splits 128 --read-batch-size 8 --io-queue-depth 1 --transform-parallelism 2 \
    --num-workers 2 --ckpt-every 1000 --eval-every 1500
```

4 or 8 H200 nodes (same command on every node; set `MASTER_ADDR`):

```bash
torchrun --nnodes 4 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint $MASTER_ADDR:29500 \
    train.py --model large --seq-len 1024 --batch-size 4 \
    --num-splits 256 --grad-accum 4 --ckpt-every 500
```

Sample log — the `q A/B/C/D` column is the loader's built-in observability
(unscanned / raw / transformed / consumed rows). Prefetch queue full +
raw queue empty = the model, not the data, is the bottleneck — exactly what
you want to see:

```
model=tiny params=859,264 world_size=1 global_batch=8 num_splits=8
train rows (post-filter): 2253  steps/epoch: 281  target steps: 40
epoch 0 step 10/40 | loss 5.0289 |  5,660 tok/s | q 0/0/2168/80  | fetch 6.1s tx 5.7s
epoch 0 step 20/40 | loss 4.5420 | 20,108 tok/s | q 0/0/2088/160 | fetch 6.1s tx 5.7s
  val loss @ step 20: 4.5093
epoch 0 step 40/40 | loss 4.2343 | 24,815 tok/s | q 0/0/1928/320 | fetch 6.1s tx 5.7s
  val loss @ step 40: 4.2359
```

### Elasticity: lose a node, keep the run

The loader partitions the table into `num_splits` fixed splits; ranks own
contiguous blocks of splits. Any world size that divides `num_splits` yields
**the same global batches in the same order**, and checkpoints record
consumption per split — so a checkpoint taken on 32 GPUs resumes correctly
on 24:

```bash
# 4 nodes... one dies at step 8000. Restart on 3 nodes:
torchrun --nnodes 3 --nproc-per-node 8 ... train.py ... --resume auto
```

With `--num-splits 256` you can run on 1, 2, 4, 8, 16, 32, or 64 GPUs and
step N always trains on the same data.

### Ablations without preprocessing jobs

The `filter` is a SQL prefilter — rejected rows are never read from storage:

```bash
python train.py --min-score 3.0          # high-quality subset
python train.py --min-score 0.0          # everything incl. junk, same table
```

## Configuration notes

- **`--num-splits`** must be divisible by every world size you plan to use,
  and the global batch (`--batch-size × world_size`) must be a multiple of
  it — `train.py` enforces this because elastic determinism silently breaks
  otherwise. Default: one split per global-batch slot.
- **`num_workers=0`** (the default) keeps everything in-process: the loader
  parallelizes I/O and transforms on threads internally. It is simplest and
  exact, but the GIL-bound packer then shares the training thread — use
  `--num-workers` for the last ~6 MFU points.
- **`--pack` (sequence packing)** concatenates documents with an EOS
  separator and slices fixed `seq_len` blocks, so every trained position is
  a real token. This is what makes the loader token-competitive with
  memmapped-.bin pipelines: on a long-document corpus, pad/truncate at
  `seq_len=256` trains on only ~17% of corpus tokens (truncation discards
  the rest), packing trains on 100%. Packing happens per logical split with
  a fixed per-split block budget (`blocks_per_epoch`, computed exactly from
  the `n_tokens` column), so packed runs are deterministic and resumable
  **across world sizes**: each rank checkpoints its own splits and
  `StreamingDataset.merge_state_dicts` combines them on resume.
- **`--num-workers N`** moves iteration (I/O, transform, packing) into `N`
  worker processes per rank via lancedb's `StreamingDataLoader`, whose
  checkpoints are committed only for batches the trainer has received.
  `num_splits` must be divisible by `world_size x N`; use `--mp-context
  forkserver` (default) or `spawn`, never fork, inside CUDA ranks.
- **Loader threads**: `--io-queue-depth 1 --transform-parallelism 2
  --read-batch-size 8` on local disk; see `LOADER_TUNING.md`.
- The tokenizer choice only needs to be consistent between
  `tokenize_data.py` and `train.py` (`--tokenizer` sets the vocab size).

## Project structure

```
llm_pretraining/
  common.py         # table helpers, byte tokenizer, val/train SQL filters
  ingest.py         # corpus -> Lance table (streamed RecordBatches)
  curate.py         # EDA + FTS + zero-copy is_dup column
  tokenize_data.py  # zero-copy input_ids column
  model.py          # compact GPT (tiny/small/medium/large presets)
  train.py          # elastic, resumable pretraining loop
  verify_e2e.py     # offline end-to-end verification
```
