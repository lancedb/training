"""Shared helpers for the LLM pretraining example.

Usage
-----
Imported by ingest.py, curate.py, tokenize_data.py, train.py, and
verify_e2e.py.  Not a script.
"""

from __future__ import annotations

import os

import lancedb

DEFAULT_DB = os.environ.get("LANCE_PRETRAIN_DB", "./lance_pretrain_db")
DEFAULT_TABLE = "corpus"

# Held-out validation slice: deterministic, insert-stable, and expressible as a
# SQL filter that StreamingDataset can prefilter on.
VAL_FILTER = "id % 100 = 0"
TRAIN_FILTER = "id % 100 != 0"


def connect_table(db_uri: str, table_name: str) -> "lancedb.table.Table":
    """Open a table, always at the latest version.

    Each pipeline stage opens the table fresh: columns added by a previous
    stage (possibly out-of-band through ``tbl.to_lance()``) are only visible
    on a newly opened handle.
    """
    db = lancedb.connect(db_uri)
    return db.open_table(table_name)


def data_file_stats(db_uri: str, table_name: str) -> tuple[int, int]:
    """(file_count, total_bytes) of the table's data files.

    Used to demonstrate zero-copy column addition: adding a column writes new
    files for the new column only and never rewrites existing ones.
    """
    data_dir = os.path.join(db_uri, f"{table_name}.lance", "data")
    n, total = 0, 0
    for dirpath, _dirnames, filenames in os.walk(data_dir):
        for f in filenames:
            n += 1
            total += os.path.getsize(os.path.join(dirpath, f))
    return n, total


def banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


class ByteTokenizer:
    """Trivial offline tokenizer: UTF-8 bytes + BOS/EOS.

    Exists so the whole pipeline (including training) runs without network
    access or model downloads.  Real runs should use ``--tokenizer hf:<name>``
    (e.g. ``hf:Qwen/Qwen2.5-0.5B``) which loads a HuggingFace tokenizer.
    """

    BOS = 256
    EOS = 257
    PAD = 258

    vocab_size = 259
    pad_token_id = PAD
    eos_token_id = EOS

    def encode(self, text: str) -> list[int]:
        return [self.BOS, *text.encode("utf-8"), self.EOS]


def load_tokenizer(spec: str):
    """Load a tokenizer from a spec string: ``byte`` or ``hf:<model_name>``."""
    if spec == "byte":
        return ByteTokenizer()
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(spec[3:])
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        class _HFAdapter:
            vocab_size = len(tok)
            pad_token_id = tok.pad_token_id
            eos_token_id = tok.eos_token_id

            @staticmethod
            def encode(text: str) -> list[int]:
                return tok.encode(text)

        return _HFAdapter()
    raise ValueError(f"Unknown tokenizer spec: {spec!r} (use 'byte' or 'hf:<name>')")
