# LLM pretraining on one LanceDB table

Pretrain a GPT where one Lance table is the whole data layer: raw text,
curation flags, token ids, the training dataloader and post-training
retrieval all work on the same table. Derived signals are new columns
(nothing existing is rewritten), curation rules are SQL filters, and
sequence packing and shuffling happen inside the loader
(`lancedb.streaming.StreamingDataset`, lancedb >= 0.38).

Numbers and the write-up are in the blog post.

| File | Purpose |
|---|---|
| `ingest.py` | FineWeb-Edu (or a synthetic corpus) -> Lance table |
| `curate.py` | SQL EDA, full-text index, exact-dedup flag as a new column |
| `tokenize_data.py` | Token ids + counts as new columns (single process) |
| `geneva_backfill.py` | Same columns as distributed, checkpointed [Geneva](https://github.com/lancedb/geneva) backfills, plus a GPU embedding column |
| `train.py` | torchrun-ready trainer; `--pack` for loader-side packing; `--resume auto` on any world size dividing `num_splits`; `--blocks-mode` for the loader A/B |
| `model.py`, `common.py`, `sample.py` | Compact GPT, shared helpers, text sampling |
| `build_packed_datasets.py`, `blocks_loaders.py` | A/B controls: identical pre-packed blocks as Parquet, pre-shuffled Parquet, MDS shards and a Lance table, plus the Parquet loaders |
| `forensics.py` | Vector index, hybrid search, generation attribution, near-duplicates on the training table |
| `loader_gil_repro.py` | Standalone, CPU-only reproduction of why fewer loader threads are faster |

## Setup

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e .
uv venv .venv-geneva --python 3.12 && uv pip install --python .venv-geneva/bin/python geneva transformers sentence-transformers
```

## Run

Offline smoke test on a laptop (synthetic corpus, byte tokenizer, tiny model):

```bash
python ingest.py --source synthetic --rows 5000
python curate.py
python tokenize_data.py --tokenizer byte
python train.py --model tiny --pack --seq-len 256 --steps 40
```

Real corpus and the 8-GPU configuration behind the reported numbers:

```bash
python ingest.py --source fineweb-parquet --sample 10BT --files 4 --rows 2400000
python curate.py
.venv-geneva/bin/python geneva_backfill.py --tokenizer hf:gpt2 --concurrency 32   # or tokenize_data.py

torchrun --nproc-per-node 8 train.py --model small --tokenizer hf:gpt2 \
    --pack --compile --batch-size 32 --grad-accum 2 --seq-len 1024 --epochs 1 \
    --num-splits 128 --read-batch-size 8 --io-queue-depth 1 --transform-parallelism 2 \
    --transform-queue-depth 16 --num-workers 2 --ckpt-every 1000 --eval-every 1500

torchrun --nproc-per-node 4 train.py ... --batch-size 64 --resume auto        # same global batch, half the GPUs

python build_packed_datasets.py --db ./lance_pretrain_db --out ./blocks --workers 8
torchrun --nproc-per-node 8 train.py --blocks-mode mosaic --blocks-path ./blocks/mds_blocks ...
```

Loader settings: `--io-queue-depth 1 --transform-parallelism 2` and 16 splits
per rank. The library defaults spawn hundreds of threads per rank that take
the interpreter lock from the single packer thread; `loader_gil_repro.py`
shows the effect in two minutes on any machine.

## Known rough edges

- Packed `state_dict()` needs every owned split at the same block count:
  checkpoint on optimizer steps where `batch_size x grad_accum` is a multiple
  of the rank's split count (with workers, `ckpt_every x grad_accum` a
  multiple of `num_workers`).
- `--num-workers` uses `forkserver` or `spawn`, never `fork`, inside CUDA
  ranks. Workers dying at start-up with `SemLock._rebuild ->
  FileNotFoundError` means the host's `systemd-logind` has `RemoveIPC=yes`;
  set `RemoveIPC=no` in `/etc/systemd/logind.conf.d/`.
- Building the permutation over ~16M filtered rows needs
  `LANCEDB_PERM_BUILDER_MEMORY_LIMIT` raised from its 100MB default.
- Interpreter exit can hang after a worker-process run; `train.py` calls
  `os._exit(0)` once checkpoints and the final eval are written.
