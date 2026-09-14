# LLM pretraining on one LanceDB table

Pretrain a GPT where **one Lance table is the whole data layer**: raw text,
curation flags, token ids, the training dataloader and post-training
retrieval all work on the same table. No tokenized copies, no pre-shuffled
shards. Sequence packing and shuffling happen inside the loader
(`lancedb.streaming.StreamingDataset`, lancedb >= 0.38).

```
             ┌──────────────────────────────────────────────────────────┐
             │                    ONE LANCE TABLE                       │
             │  id │ text │ source │ score │ is_dup │ input_ids │ ...   │
             └──────────────────────────────────────────────────────────┘
   ingest.py ──▲          ▲            ▲                ▲          │
   (raw text)  │          │            │                │          ▼
               │   curate.py     curate.py       tokenize_data.py │ train.py
               │   SQL EDA +     zero-copy       zero-copy column │ StreamingDataset
               │   FTS search    `is_dup` col    `input_ids`      │ pack + shuffle on the fly
               │                                                  ▼
               │                                     torchrun, 1..64+ GPUs
               └── same table serves retrieval / data forensics after training
```

Every derived signal is a new column written without rewriting existing data;
every curation rule is a SQL filter the loader evaluates. Numbers and the
write-up are in the blog; raw run outputs are in [`runs/results/`](runs/results/).

## What's here

| File | Purpose |
|---|---|
| `ingest.py` | FineWeb-Edu (or a synthetic corpus) -> Lance table |
| `curate.py` | SQL EDA, full-text index, exact-dedup flag as a new column |
| `tokenize_data.py` | Token ids + token counts as new columns (single process) |
| `geneva_backfill.py` | Same columns as distributed, checkpointed [Geneva](https://github.com/lancedb/geneva) backfills, plus a GPU embedding column |
| `train.py` | torchrun-ready trainer; `--pack` for loader-side sequence packing; `--blocks-mode` for the loader A/B |
| `model.py`, `common.py`, `sample.py` | Compact GPT, shared helpers, text sampling |
| `verify_e2e.py` | Offline CPU check of the whole pipeline (~2 min) |
| `elastic_pack_check.py` | Packed runs give identical global steps at any world size, and resume across world sizes |
| `bench_loader.py` | Loader-only throughput for one setting |
| `build_packed_datasets.py`, `blocks_loaders.py`, `mosaic_compare.py` | The A/B controls: identical pre-packed blocks as Parquet, pre-shuffled Parquet, MDS shards and a Lance table; their loaders; Mosaic determinism and resume checks |
| `forensics.py` | Vector index, hybrid search, generation attribution, near-duplicates on the training table |
| `LOADER_TUNING.md` | Loader settings the 8-GPU runs use, the measured sweep, and why the knobs matter |
| `runs/` | The exact 8x H100 commands and their raw outputs |

## Setup

```bash
cd examples/llm_pretraining
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e .          # add `-e .[hf]` for FineWeb-Edu + HF tokenizers
```

Geneva bundles Ray and lives in its own environment:
`uv venv .venv-geneva --python 3.12 && uv pip install geneva "transformers>=4.40"`.

## Quickstart (offline, CPU)

```bash
python verify_e2e.py
```

Runs every stage on a synthetic corpus and asserts, with real training runs:
zero-copy `is_dup` and `input_ids` columns; identical global batches at
world size 1 and 2; a killed-and-resumed run matching the uninterrupted one
to four decimals; packed blocks deterministic, 100% real tokens against ~17%
for pad/truncate at `seq_len=256`, exact mid-epoch packed resume, and packed
elasticity plus cross-world-size resume via `merge_state_dicts`. 16 checks.

## Real corpus

```bash
python ingest.py --source fineweb-parquet --sample 10BT --files 4 --rows 2400000
python curate.py
python tokenize_data.py --tokenizer hf:gpt2                       # or:
.venv-geneva/bin/python geneva_backfill.py --tokenizer hf:gpt2 --concurrency 32
```

Train GPT-2 124M on 8 GPUs, one epoch, packed (the configuration behind the
reported numbers):

```bash
torchrun --nproc-per-node 8 train.py --model small --tokenizer hf:gpt2 \
    --pack --compile --batch-size 32 --grad-accum 2 --seq-len 1024 --epochs 1 \
    --num-splits 128 --read-batch-size 8 --io-queue-depth 1 --transform-parallelism 2 \
    --num-workers 2 --ckpt-every 1000 --eval-every 1500
```

Resume on any world size that divides `num_splits` (keep the global batch the
same, e.g. 4 GPUs at `--batch-size 64`):

```bash
torchrun --nproc-per-node 4 train.py ... --batch-size 64 --resume auto
```

Loader A/B on identical pre-packed blocks (local paths or `s3://`):

```bash
python build_packed_datasets.py --db ./lance_pretrain_db --out ./blocks --workers 8
torchrun --nproc-per-node 8 train.py --blocks-mode mosaic        --blocks-path ./blocks/mds_blocks ...
torchrun --nproc-per-node 8 train.py --blocks-mode parquet-random --blocks-path ./blocks/blocks_parquet ...
python elastic_pack_check.py --db ./lance_pretrain_db --num-splits 128 --ws 8 4
```

## Headline results (8x H100, lancedb 0.38, 2.4M FineWeb-Edu docs)

| Stage | Wall time |
|---|---|
| Ingest 2.4M docs -> 4.8GB table | 2m 06s |
| Curate (EDA, FTS index, dedup flag: +306KB on a 5.1GB table) | 4m 10s |
| Tokenize with Geneva, 32 Ray workers, 2.43B tokens | 4m 54s |
| Train GPT-2 124M, one epoch, 3.18M tok/s, 34.5% MFU, val 3.236 | 14m 08s |
| Same run reading the table from S3 on another continent | 3.16M tok/s, 34.4% MFU |

Loader A/B (same GPT-2 124M, same 2,373,376 blocks, tok/s and MFU):

| Loader | Local disk | S3 (trans-Atlantic) | Extra copies |
|---|---|---|---|
| Lance corpus table, pack + shuffle on the fly | 3.16M / 34.3% | 3.16M / 34.4% | 0 |
| Lance blocks table | 3.17M / 34.5% | 3.16M / 34.3% | 1 |
| MosaicML Streaming | 3.17M / 34.5% | 2.87M / 31.2% mean, 0.75M during shard downloads | 1 + per-node cache |
| Parquet, pre-shuffled, sequential | 3.17M / 34.4% | 3.18M / 34.5% | 2 |
| Parquet, random reads | 2.02M / 21.9% | 74k / 0.8% | 1 |

The same pipeline on 17.5M docs trained GPT-2 medium (354M) on 7.0B tokens
in 1h 37m at 1.34M tok/s / 41.1% MFU, val 2.841. Details: `runs/results/`.

## Known rough edges

- Packed `state_dict()` needs every owned split at the same block count:
  checkpoint on optimizer steps where `batch_size x grad_accum` is a multiple
  of the rank's split count (with workers, `ckpt_every x grad_accum` a
  multiple of `num_workers`).
- `--num-workers` uses `forkserver` (or `spawn`), never `fork`, inside CUDA
  ranks. If workers die at start-up with `SemLock._rebuild ->
  FileNotFoundError`, the host's `systemd-logind` has `RemoveIPC=yes` and is
  wiping `/dev/shm`; set `RemoveIPC=no` in `/etc/systemd/logind.conf.d/`.
- Building the permutation over ~16M filtered rows needs
  `LANCEDB_PERM_BUILDER_MEMORY_LIMIT` raised from its 100MB default.
- Interpreter exit can hang after a worker-process run; `train.py` calls
  `os._exit(0)` once checkpoints and the final eval are written.
