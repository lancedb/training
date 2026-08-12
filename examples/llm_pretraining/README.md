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
| `build_packed_datasets.py` | Identical pre-packed blocks -> MDS shards + Lance table |
| `mosaic_compare.py` | MosaicML Streaming A/B: elastic, resume, throughput |
| `verify_e2e.py` | Offline CPU verification of the whole pipeline (~2 min) |

## Setup

```bash
cd examples/llm_pretraining
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e .          # add `-e .[hf]` for FineWeb-Edu + HF tokenizers
```

Requires `lancedb >= 0.36` (first release with `lancedb.streaming`).

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
  [+] pack: refuses cross-topology resume  ValueError raised

============================================================
E2E COMPLETE: 14 passed  0 failed
============================================================
```

Note the last two lines: a run killed at step 12 and resumed lands on the
**bit-identical** validation loss as the uninterrupted run, and the samples
that form each global step are identical whether the job runs on one process
or two.

## Real corpus

```bash
python ingest.py --source fineweb --rows 2000000        # streams FineWeb-Edu
python curate.py
python tokenize_data.py --tokenizer hf:Qwen/Qwen2.5-0.5B
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


## Results: GPT-2 124M, Chinchilla-scale, 4x H100

One epoch over FineWeb-Edu with everything — corpus, curation, tokens,
training, retrieval — on a single Lance table. `--pack` (loader-native
sequence packing from the `ayush/seq-packing` branch) + `--compile`.

| Stage | Wall time | What happened |
|---|---|---|
| Ingest 2.4M docs (11.4B chars) | **3m 00s** | HF stream -> 4.8GB Lance table |
| Curate: EDA + FTS + dedup | **3m 02s** | 22,558 dups flagged; `is_dup` col = +306KB on a 5.1GB table, nothing rewritten |
| Geneva tokenize (32 Ray workers) | **4m 53s** | 2.43B GPT-2 tokens as a zero-copy column (+`n_tokens` 1m 24s) |
| Train 124M, 1 epoch = 2.43B tokens | **~50 min** | 4x H100, 1.6M tok/s sustained, 35% MFU |
| Upload table to S3 + train from s3:// | 33s upload | **same 1.60M tok/s / 34.7% MFU streaming over WAN** |
| Geneva GPU embeddings (4 workers) + IVF-PQ index | 11m 10s + 13s | 2.4M x 384-d; hybrid search + generation attribution |
| Parquet control (bench_parquet.py) | 30s export + 145s pre-shuffle copy | random S3 reads: 46k tok/s, 474x amplification vs Lance 660k tok/s, 0 extra copies |
| Scale run: full 10BT sample, GPT-2 medium 354M | prep 51m; train **3h06m** | 9.67M docs (4.2% dups), 7.0B tokens, **684k tok/s / 42.0% MFU flat**, val 2.840 |
| Loader-knob MFU sweep (354M, trainer frozen) | - | read_batch_size 64/16/8 -> 17% / 40% / 42% MFU; identical losses (determinism) |
| Mosaic Streaming A/B (identical 2.37M packed blocks) | 33m pre-pack (required by MDS flow) | elastic ws1/2/4 + ws4->ws2 resume: both pass; GPU training: lance 1.619M tok/s / 35.2% vs mosaic 1.618M / 35.1% |

Raw text to training-ready: **~12 minutes**. Total to a Chinchilla-optimal
GPT-2: about an hour.

```
epoch 0 step 1500/4635 | loss 3.7020 | 1,602,377 tok/s | mfu 34.8% | q 0/0/397844/190572
  val loss @ step 1500: 3.5902
  val loss @ step 3000: 3.3134
  val loss @ step 4500: 3.2295
```

- **Packing vs pad/truncate at seq 1024**: pad/truncate trains on only
  **62.1%** of corpus tokens (truncation discards the rest); packed blocks
  are ~100% real tokens.
- **Kill -9 at step 1500, resume from step-1000 checkpoint**: per-rank
  loader states restore exactly (`resumed from ... opt step 1000`), loss
  rejoins the curve, same topology.
- **Loader tuning matters**: `read_batch_size` 64 -> 8 took one loader
  process from 102k to 419k tok/s on local NVMe (per-take cost grows
  super-linearly with rows per take). `bench_loader.py` finds this in a
  minute; 4 ranks at rb=8 gave a ~1.7M tok/s ceiling, ahead of the model.
- **Data forensics on the same table**: after training, FTS-query the
  corpus for what the model actually saw (`curate.py`-built index), or
  `checkout` any earlier table version (ingest was v1; this run trained
  against v13).

Sample from the final checkpoint (temperature 0.8):

> Photosynthesis is the process by which plants produce energy, the
> process by which they convert it into energy. [...] The carbon dioxide
> present in the atmosphere is released into the atmosphere, which is why
> it is referred to as the "Carbon Cycle."

124M-grade prose: fluent, on-topic, confidently wrong — exactly on spec.

### Known rough edges (upstream notes)
- geneva 0.14 needs `lancedb==0.34.x` in its own venv (its `>=0.34.0b4`
  pin admits incompatible newer versions).
- A shutdown-time `PyGILState_Release`/SIGABRT can fire at process exit
  after successful completion (mixed Rust-runtime + torch teardown); data
  and checkpoints are unaffected.
- Packed resume requires the same world_size (padding and checkpoint
  state are per-rank); per-rank checkpoint files handle multi-GPU.

## Training

Single GPU / debug:

```bash
python train.py --model small --batch-size 16 --epochs 1
```

One 8×GPU node:

```bash
torchrun --nproc-per-node 8 train.py --model medium --batch-size 8
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
- **`num_workers=0`** on the DataLoader is deliberate: the loader
  parallelizes I/O and transforms on threads internally, and the loader's
  `state_dict()` used for checkpointing reflects consumption only in the
  process that iterates.
- **`--pack` (sequence packing)** concatenates documents with an EOS
  separator and slices fixed `seq_len` blocks, so every trained position is
  a real token. This is what makes the loader token-competitive with
  memmapped-.bin pipelines: on a long-document corpus, pad/truncate at
  `seq_len=256` trains on only ~17% of corpus tokens (truncation discards
  the rest), packing trains on 100%. Trade-off: block boundaries follow the
  per-rank stream, so packed runs are deterministic and exactly resumable
  **at a fixed world size** (the packer refuses cross-topology resume),
  while the default pad/truncate mode keeps token-exact elasticity across
  world sizes. Pick per run: elasticity experiments → default; production
  token throughput → `--pack`.
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
