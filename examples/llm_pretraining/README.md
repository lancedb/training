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
| `train.py` | torchrun-ready pretraining with `lancedb.streaming.StreamingDataset` |
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

============================================================
E2E COMPLETE: 9 passed  0 failed
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
- **Fixed-length rows** (truncate/pad per document) keep elastic determinism
  exact at the token level. Packed sequences (concat-and-chunk) squeeze out
  the padding but tie packing boundaries to the per-rank stream, so global
  token batches would no longer be topology-independent — pick per run.
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
