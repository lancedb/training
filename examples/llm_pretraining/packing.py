"""Sequence packing over a StreamingDataset document stream.

Concat-and-chunk (GPT-style) packing: token streams from consecutive
documents are joined with an EOS separator and sliced into fixed
``seq_len`` blocks, so every position of every block is a real token —
no padding waste, no truncation loss.

This is the piece that makes a row-per-document database loader
token-efficient enough to compare head-to-head with the classic
"tokenize offline into one giant memmapped array" pretraining flow —
while keeping what that flow destroys: per-document identity, SQL
curation filters, and instant re-slicing of the corpus.

Determinism and resume semantics (honestly stated):

- For a **fixed topology** (world_size, and num_workers=0, which packing
  requires) the packed block stream is fully deterministic — a pure
  function of the inner StreamingDataset's document stream.
- **Mid-epoch resume is exact at the same topology.** The packer pulls
  documents in whole round-robin cycles (one document per assigned
  split), so the inner dataset's ``state_dict`` is always
  cycle-aligned; together with the carried token buffer this restores
  the stream bit-exactly.
- **Across topologies** the *document* stream per global step stays
  invariant (the inner dataset's elastic guarantee), but block
  boundaries follow the per-rank stream, so packed runs refuse to
  resume across a world-size change. Making packing fully elastic
  requires packing per split inside the loader itself — see the
  follow-up proposal in the LanceDB repo (STREAMING_DATASET_FOLLOWUPS.md,
  "sequence-packing helper").
"""

from __future__ import annotations

from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info


class PackedTokenDataset(IterableDataset):
    """Wrap a StreamingDataset (yielding ``{"input_ids": list[int]}`` rows)
    into fixed-length packed blocks ready for causal-LM training.

    The inner dataset must be constructed with the default transform (rows
    as Python dicts) and iterated with ``num_workers=0``.
    """

    def __init__(self, inner, seq_len: int, eos_id: int):
        super().__init__()
        self.inner = inner
        self.seq_len = seq_len
        self.eos_id = eos_id
        # Documents per round-robin cycle for this rank. Pulling in whole
        # cycles keeps inner.state_dict() exact at every block boundary.
        self._cycle_docs = len(inner._rank_splits)
        self._buffer: list[int] = []

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if get_worker_info() is not None:
            raise RuntimeError(
                "PackedTokenDataset requires num_workers=0: the inner "
                "StreamingDataset already parallelizes I/O on threads, and "
                "worker processes would clone the packer's buffer state."
            )
        buf = self._buffer
        doc_iter = iter(self.inner)
        exhausted = False
        while not exhausted:
            # Refill with one full round-robin cycle of documents.
            for _ in range(self._cycle_docs):
                row = next(doc_iter, None)
                if row is None:
                    exhausted = True
                    break
                buf.extend(row["input_ids"])
                buf.append(self.eos_id)
            while len(buf) >= self.seq_len:
                block = torch.tensor(buf[: self.seq_len], dtype=torch.long)
                del buf[: self.seq_len]
                yield {
                    "input_ids": block,
                    "loss_mask": torch.ones(self.seq_len, dtype=torch.bool),
                }
        # A tail shorter than seq_len stays in the buffer and is carried
        # into the next epoch rather than padded or dropped.

    def state_dict(self) -> dict[str, Any]:
        return {
            "inner": self.inner.state_dict(),
            "buffer": list(self._buffer),
            "world_size": self.inner._world_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["world_size"] != self.inner._world_size:
            raise ValueError(
                "Packed runs cannot resume across a world-size change: block "
                f"boundaries follow the per-rank stream (checkpoint world_size="
                f"{state['world_size']}, current={self.inner._world_size}). "
                "Resume with the original topology, or restart the epoch."
            )
        self.inner.load_state_dict(state["inner"])
        self._buffer = list(state["buffer"])
