"""Tokenize the corpus with Geneva: distributed, checkpointed UDF backfill.

The production-scale alternative to tokenize_data.py.  Geneva registers the
token columns on the same Lance table and populates them with a Ray-parallel
backfill job that checkpoints as it goes — safe to re-run (already-computed
rows are skipped) and safe to kill mid-job.

Runs in its own environment (Geneva bundles Ray):
    uv venv .venv-geneva --python 3.12
    uv pip install geneva "transformers>=4.40"

Usage
-----
.venv-geneva/bin/python geneva_backfill.py --tokenizer hf:gpt2 --concurrency 16
.venv-geneva/bin/python geneva_backfill.py --tokenizer byte      # offline smoke
"""

from __future__ import annotations

import argparse
import time

import pyarrow as pa

import geneva
from geneva.transformer import udf

DEFAULT_DB = "./lance_pretrain_db"
DEFAULT_TABLE = "corpus"


@udf(data_type=pa.list_(pa.int32()), input_columns=["text"])
class _TokenizeHF:
    """Stateful tokenizer UDF: the tokenizer loads once per Ray worker."""

    def __init__(self, model_name: str):
        self.model_name = model_name  # driver-side: keep cheap, no loading
        self._tok = None

    def __call__(self, text: pa.Array) -> pa.Array:
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
        encoded = self._tok(text.to_pylist())["input_ids"]
        return pa.array(encoded, type=pa.list_(pa.int32()))


@udf(data_type=pa.list_(pa.int32()), input_columns=["text"])
def _tokenize_bytes(text: pa.Array) -> pa.Array:
    # Offline byte-level tokenizer (matches common.ByteTokenizer: BOS=256,
    # EOS=257) so the pipeline smoke-tests without network access.
    encoded = [[256, *t.encode("utf-8"), 257] for t in text.to_pylist()]
    return pa.array(encoded, type=pa.list_(pa.int32()))


@udf(data_type=pa.int32(), input_columns=["input_ids"])
def _n_tokens(input_ids: pa.Array) -> pa.Array:
    import pyarrow.compute as pc

    return pc.cast(pc.list_value_length(input_ids), pa.int32())


@udf(
    data_type=pa.list_(pa.float32(), 384),
    input_columns=["text"],
    cuda=True,
    num_gpus=1,
    batch_size=1024,
)
class _EmbedGPU:
    """384-d MiniLM sentence embedding; model loads once per GPU worker."""

    def __init__(self):
        self._model = None

    def __call__(self, text: pa.Array) -> pa.Array:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device="cuda"
            )
        vecs = self._model.encode(
            text.to_pylist(),
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return pa.array(vecs.tolist(), type=pa.list_(pa.float32(), 384))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--tokenizer", default="hf:gpt2", help="'byte' or 'hf:<model>'")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--task-size", type=int, default=4096)
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["input_ids", "n_tokens"],
        help="which columns to backfill (input_ids, n_tokens, embedding)",
    )
    args = parser.parse_args(argv)

    if args.tokenizer == "byte":
        tokenize_udf = _tokenize_bytes
    elif args.tokenizer.startswith("hf:"):
        tokenize_udf = _TokenizeHF(args.tokenizer[3:])
    else:
        raise SystemExit(f"unknown tokenizer {args.tokenizer!r}")

    conn = geneva.connect(args.db)
    tbl = conn.open_table(args.table)
    print(f"table '{args.table}': {tbl.count_rows():,} rows, v{tbl.version}")

    registry = {"input_ids": tokenize_udf, "n_tokens": _n_tokens, "embedding": _EmbedGPU()}
    registry = {c: registry[c] for c in args.columns}
    missing = {c: u for c, u in registry.items() if c not in set(tbl.schema.names)}
    if missing:
        print(f"registering columns: {list(missing)}")
        tbl.add_columns(missing)

    with conn.local_ray_context():
        for col, u in registry.items():
            t0 = time.perf_counter()
            job = tbl.backfill(
                col,
                udf=u,
                concurrency=args.concurrency,
                task_size=args.task_size,
            )
            print(
                f"[backfill] {col}: {time.perf_counter() - t0:,.1f}s (job={job})"
            )

    tbl = conn.open_table(args.table)
    print(f"done — columns now: {tbl.schema.names}, v{tbl.version}")


if __name__ == "__main__":
    main()
