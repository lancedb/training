"""Standard-workflow loaders over pre-packed 1024-token blocks in Parquet.

Used by `train.py --blocks-mode parquet-random|parquet-seq` as the controls in
the loader A/B (see build_packed_datasets.py for how the blocks are made).

- ParquetRandomBlocks: a global shuffle over Parquet means random row access,
  and Parquet's unit of retrieval is the row group — each sample fetches a
  whole row group (1024 blocks = 4MB here, deliberately small).  Footers are
  opened once per worker and reused.
- ParquetSeqBlocks: the "pre-materialized shuffle" workflow — shards were
  written in a shuffled order, each rank streams its own shards sequentially.

Both work on local paths and s3:// (pyarrow S3FileSystem).
"""

from __future__ import annotations

import os
from bisect import bisect_right

import numpy as np
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

SEQ_LEN = 1024


def _fs_and_paths(root: str):
    if root.startswith("s3://"):
        fs = pafs.S3FileSystem(region=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"))
        prefix = root[5:].rstrip("/")
        paths = sorted(
            f.path for f in fs.get_file_info(pafs.FileSelector(prefix)) if f.path.endswith(".parquet")
        )
    else:
        fs = pafs.LocalFileSystem()
        paths = sorted(os.path.join(root, f) for f in os.listdir(root) if f.endswith(".parquet"))
    return fs, paths


class _Files:
    """Lazily opened ParquetFile handles (footer read once per process)."""

    def __init__(self, fs, paths):
        self.fs, self.paths, self._pf = fs, paths, {}
        self.rg_rows = None

    def pf(self, i):
        if i not in self._pf:
            self._pf[i] = pq.ParquetFile(self.fs.open_input_file(self.paths[i]))
        return self._pf[i]

    def layout(self):
        """(cum_rows_per_file, per-file cum rows per row group) from footers."""
        if self.rg_rows is None:
            per_file = []
            for i in range(len(self.paths)):
                md = self.pf(i).metadata
                per_file.append(np.cumsum([md.row_group(g).num_rows for g in range(md.num_row_groups)]))
            self.rg_rows = per_file
            self.file_cum = np.cumsum([c[-1] for c in per_file])
        return self.file_cum, self.rg_rows


def _rows(table):
    return table.column("input_ids").combine_chunks().values.to_numpy().reshape(-1, SEQ_LEN)


class ParquetRandomBlocks(IterableDataset):
    def __init__(self, root: str, rank: int, world_size: int, seed: int = 0):
        self.root, self.rank, self.world, self.seed = root, rank, world_size, seed

    def __iter__(self):
        fs, paths = _fs_and_paths(self.root)
        files = _Files(fs, paths)
        file_cum, rg_rows = files.layout()
        total = int(file_cum[-1])
        perm = np.random.default_rng(self.seed).permutation(total)[self.rank :: self.world]
        wi = get_worker_info()
        if wi is not None:
            perm = perm[wi.id :: wi.num_workers]
        for gidx in perm:
            f = bisect_right(file_cum, gidx)
            local = gidx - (file_cum[f - 1] if f else 0)
            g = bisect_right(rg_rows[f], local)
            off = local - (rg_rows[f][g - 1] if g else 0)
            rg = files.pf(f).read_row_group(g, columns=["input_ids"])  # whole row group per sample
            yield {"input_ids": torch.from_numpy(_rows(rg)[off].astype(np.int64))}


class ParquetSeqBlocks(IterableDataset):
    def __init__(self, root: str, rank: int, world_size: int):
        self.root, self.rank, self.world = root, rank, world_size

    def __iter__(self):
        fs, paths = _fs_and_paths(self.root)
        mine = paths[self.rank :: self.world]
        wi = get_worker_info()
        k = 0
        for p in mine:
            pf = pq.ParquetFile(fs.open_input_file(p))
            for g in range(pf.metadata.num_row_groups):
                k += 1
                if wi is not None and (k % wi.num_workers) != wi.id:
                    continue
                for row in _rows(pf.read_row_group(g, columns=["input_ids"])):
                    yield {"input_ids": torch.from_numpy(row.astype(np.int64))}
