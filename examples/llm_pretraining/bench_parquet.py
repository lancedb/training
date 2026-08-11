"""The standard-workflow baseline: token shards as Parquet on S3.

Reproduces what a conventional pretraining pipeline materializes — tokenized
Parquet shards — and measures the two ways to read them:

- ``bench-random``: globally-shuffled reads, implemented the *competent* way
  (row-group-granular fetches, footer caching, 32 concurrent requests), with
  read-amplification accounting (bytes downloaded vs bytes used).
- ``bench-seq``: sequential streaming, which is only shuffle-correct after
  materializing a pre-shuffled copy (``export --shuffle``).

Usage
-----
python bench_parquet.py export --db ~/blogrun/db --out s3://bucket/prefix/tokens
python bench_parquet.py export --db ~/blogrun/db --out s3://bucket/prefix/tokens-shuffled --shuffle
python bench_parquet.py bench-random --path s3://bucket/prefix/tokens --seconds 30
python bench_parquet.py bench-seq --path s3://bucket/prefix/tokens-shuffled --seconds 30
"""

from __future__ import annotations

import argparse
import bisect
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.fs as pafs
import pyarrow.parquet as pq

ROW_GROUP_ROWS = 1024  # ~4MB column chunks: generous to random access
FILE_ROWS = 131_072  # ~500MB shards


def _fs(path: str):
    if path.startswith("s3://"):
        return pafs.S3FileSystem(region="us-east-2"), path[5:]
    return pafs.LocalFileSystem(), path


def cmd_export(args) -> None:
    import lancedb

    tbl = lancedb.connect(args.db).open_table("corpus")
    filt = "NOT is_dup AND score >= 1.0 AND (id % 100 != 0)"
    t0 = time.perf_counter()
    data = tbl.search().select(["id", "input_ids"]).where(filt).limit(10**9).to_arrow()
    # int32 list offsets overflow on multi-GB takes; large_list is safe and
    # round-trips through Parquet unchanged.
    col = data.schema.get_field_index("input_ids")
    data = data.set_column(
        col, "input_ids", data.column("input_ids").cast(pa.large_list(pa.int32()))
    )
    t_read = time.perf_counter() - t0
    if args.shuffle:
        # Take the permutation in file-sized chunks: a whole-table take on a
        # ~10GB list column overflows Arrow's int32 list offsets.
        rng = random.Random(42)
        idx = list(range(data.num_rows))
        rng.shuffle(idx)

        def shuffled_batches():
            for lo in range(0, len(idx), FILE_ROWS):
                chunk = data.take(pa.array(idx[lo : lo + FILE_ROWS], pa.int64()))
                yield from chunk.to_batches()

        source = pa.RecordBatchReader.from_batches(data.schema, shuffled_batches())
    else:
        source = data
    t_shuf = time.perf_counter() - t0 - t_read
    fs, out = _fs(args.out)
    pads.write_dataset(
        source,
        out,
        filesystem=fs,
        format="parquet",
        max_rows_per_group=ROW_GROUP_ROWS,
        max_rows_per_file=FILE_ROWS,
        existing_data_behavior="overwrite_or_ignore",
    )
    dt = time.perf_counter() - t0
    print(
        f"exported {data.num_rows:,} rows ({data.nbytes / 1e9:.1f}GB) to {args.out} "
        f"in {dt:,.1f}s (read {t_read:.1f}s, shuffle {t_shuf:.1f}s, write "
        f"{dt - t_read - t_shuf:.1f}s)"
    )


def _open(path: str):
    fs, p = _fs(path)
    files = sorted(f.path for f in fs.get_file_info(pafs.FileSelector(p)) if f.path.endswith(".parquet"))
    handles = [pq.ParquetFile(f, filesystem=fs) for f in files]  # footer cached
    starts, total = [], 0
    rg_index = []  # (file_idx, rg_idx, row_start, n_rows, compressed_bytes)
    for fi, h in enumerate(handles):
        md = h.metadata
        arrow_schema = md.schema.to_arrow_schema()
        ids_col = arrow_schema.get_field_index("input_ids")
        for rg in range(md.num_row_groups):
            g = md.row_group(rg)
            comp = g.column(ids_col).total_compressed_size
            rg_index.append((fi, rg, total, g.num_rows, comp))
            starts.append(total)
            total += g.num_rows
    return handles, rg_index, starts, total


def cmd_bench_random(args) -> None:
    handles, rg_index, starts, total = _open(args.path)
    print(f"{total:,} rows in {len(handles)} files, {len(rg_index)} row groups")
    rng = random.Random(0)
    tokens = used_bytes = downloaded = rows = 0
    pool = ThreadPoolExecutor(max_workers=32)
    fs, p = _fs(args.path)
    file_paths = sorted(
        f.path
        for f in fs.get_file_info(pafs.FileSelector(p))
        if f.path.endswith(".parquet")
    )
    tls = threading.local()

    def _handle(fi: int) -> pq.ParquetFile:
        # ParquetFile.read_row_group is not thread-safe on a shared handle
        # (ReadRangeCache); give each worker thread its own handles.
        cache = getattr(tls, "handles", None)
        if cache is None:
            cache = tls.handles = {}
        if fi not in cache:
            cache[fi] = pq.ParquetFile(file_paths[fi], filesystem=fs)
        return cache[fi]

    def fetch(global_rows: list[int]):
        # Group requested rows by row group; fetch each group once.
        by_rg: dict[int, list[int]] = {}
        for r in global_rows:
            by_rg.setdefault(bisect.bisect_right(starts, r) - 1, []).append(r)

        def one(item):
            gi, rws = item
            fi, rg, row_start, _n, comp = rg_index[gi]
            t = _handle(fi).read_row_group(rg, columns=["input_ids"])
            col = t.column("input_ids")
            out = [col[r - row_start].as_py() for r in rws]
            return out, comp

        results = list(pool.map(one, by_rg.items()))
        return results

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        batch = [rng.randrange(total) for _ in range(args.batch)]
        for out, comp in fetch(batch):
            downloaded += comp
            for ids in out:
                tokens += len(ids)
                used_bytes += 4 * len(ids)
                rows += 1
    dt = time.perf_counter() - t0
    print(
        f"random: {rows / dt:,.0f} rows/s = {tokens / dt:,.0f} tok/s | "
        f"downloaded {downloaded / 1e9:.2f}GB for {used_bytes / 1e6:.1f}MB used "
        f"({downloaded / max(used_bytes, 1):,.0f}x read amplification) over {dt:.0f}s"
    )


def cmd_bench_seq(args) -> None:
    fs, p = _fs(args.path)
    dataset = pads.dataset(p, filesystem=fs, format="parquet")
    tokens = rows = 0
    t0 = time.perf_counter()
    for b in dataset.to_batches(columns=["input_ids"], batch_size=4096):
        for n in pa.compute.list_value_length(b.column("input_ids")).to_pylist():
            tokens += n
        rows += b.num_rows
        if time.perf_counter() - t0 > args.seconds:
            break
    dt = time.perf_counter() - t0
    print(f"sequential: {rows / dt:,.0f} rows/s = {tokens / dt:,.0f} tok/s over {dt:.0f}s")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["export", "bench-random", "bench-seq"])
    p.add_argument("--db", default="./lance_pretrain_db")
    p.add_argument("--out", default="")
    p.add_argument("--path", default="")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args(argv)
    {"export": cmd_export, "bench-random": cmd_bench_random, "bench-seq": cmd_bench_seq}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
