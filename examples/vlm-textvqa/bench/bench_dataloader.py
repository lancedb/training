"""1:1 throughput bench — Lance vs HuggingFace datasets vs raw FS vs WebDataset.

Same workload across all four loaders:

  * read (image_bytes, question, answer)
  * decode JPEG to PIL RGB @ IMAGE_PX x IMAGE_PX
  * yield as ``RawBatch``

We measure rows/s, batches/s, and wall-clock for N batches at matched
``batch_size`` and ``num_workers`` so the comparison isolates the
read-path cost.

Usage:

    python -m bench.bench_dataloader \\
        --db data/textvqa.lance --layout-dir data/baselines \\
        --bs 8 --nw 4 --batches 200
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm.dataloader import LanceRawLoader
from vlm.dataloader_baselines import (
    HFDatasetsLoader, WebDatasetLoader, make_raw_fs_loader,
    prepare_baseline_layouts,
)

LOG = logging.getLogger("bench.dataloader")


@contextmanager
def _stopwatch(label: str):
    t0 = time.time()
    yield (lambda: time.time() - t0)
    LOG.info("%s: %.1fs", label, time.time() - t0)


def _drain(label: str, loader, n_batches: int, batch_size: int) -> dict:
    """Drain N batches from `loader` and time it."""
    LOG.info("%s starting (batches=%d, bs=%d)", label, n_batches, batch_size)
    t0 = time.time()
    seen = 0
    samples = 0
    it = iter(loader)
    while seen < n_batches:
        try:
            b = next(it)
        except StopIteration:
            break
        samples += len(b.images)
        seen += 1
        if seen % max(n_batches // 5, 1) == 0:
            sps = samples / (time.time() - t0)
            LOG.info("  %s ... %d/%d batches  %.1f samples/s",
                     label, seen, n_batches, sps)
    elapsed = time.time() - t0
    out = {
        "loader":      label,
        "batches":     seen,
        "samples":     samples,
        "wall_s":      elapsed,
        "samples_per_s": samples / max(elapsed, 1e-6),
        "batches_per_s": seen / max(elapsed, 1e-6),
    }
    LOG.info("%s DONE: %d batches, %d samples, %.1fs -> %.2f samples/s",
             label, seen, samples, elapsed, out["samples_per_s"])
    return out


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--db",         default="data/textvqa.lance")
    p.add_argument("--layout-dir", default="data/baselines",
                   help="where to materialise raw_fs / wds / parquet layouts")
    p.add_argument("--n-rows",     type=int, default=None,
                   help="cap rows for layout export (default: full table)")
    p.add_argument("--bs",         type=int, default=8)
    p.add_argument("--nw",         type=int, default=4,
                   help="num_workers (only used by raw-FS DataLoader)")
    p.add_argument("--batches",    type=int, default=200,
                   help="number of batches to drain from each loader")
    p.add_argument("--out",        default="bench_outputs/dataloader.json")
    p.add_argument("--include",    nargs="+",
                   default=["lance", "hf", "raw_fs", "wds"],
                   help="which loaders to bench")
    p.add_argument("--skip-prep",  action="store_true",
                   help="skip layout export (use existing files)")
    p.add_argument("--no-decode",  action="store_true",
                   help="skip PIL decode; measure raw byte/string throughput only")
    args = p.parse_args()

    layout_dir = Path(args.layout_dir).resolve()
    if not args.skip_prep:
        LOG.info("preparing baseline layouts at %s", layout_dir)
        prepare_baseline_layouts(args.db, str(layout_dir), n_rows=args.n_rows)
    else:
        LOG.info("--skip-prep set, expecting layouts at %s", layout_dir)

    raw_dir     = layout_dir / "raw_fs"
    wds_dir     = layout_dir / "wds"
    parquet_dir = layout_dir / "parquet"

    results: list[dict] = []

    decode = not args.no_decode

    if "lance" in args.include:
        loader = LanceRawLoader(args.db, batch_size=args.bs, seed=0,
                                decode_images=decode)
        results.append(_drain("lance", loader, args.batches, args.bs))

    if "hf" in args.include:
        loader = HFDatasetsLoader(str(parquet_dir), batch_size=args.bs, seed=0,
                                  decode_images=decode)
        results.append(_drain("hf_datasets", loader, args.batches, args.bs))

    if "raw_fs" in args.include:
        loader = make_raw_fs_loader(str(raw_dir), batch_size=args.bs,
                                    num_workers=args.nw, decode_images=decode)
        results.append(_drain(f"raw_fs_nw{args.nw}", loader, args.batches, args.bs))

    if "wds" in args.include:
        loader = WebDatasetLoader(str(wds_dir), batch_size=args.bs, seed=0,
                                  decode_images=decode)
        results.append(_drain("webdataset", loader, args.batches, args.bs))

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config":  {"bs": args.bs, "nw": args.nw, "batches": args.batches,
                    "db": args.db, "layout_dir": str(layout_dir)},
        "results": results,
    }, indent=2))
    LOG.info("wrote %s", out_path)

    # Pretty table
    print()
    print(f"{'loader':<20}  {'bs':>4}  {'batches':>7}  {'samples':>8}  "
          f"{'wall_s':>7}  {'samples/s':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['loader']:<20}  {args.bs:>4}  {r['batches']:>7}  {r['samples']:>8}  "
              f"{r['wall_s']:>7.1f}  {r['samples_per_s']:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
